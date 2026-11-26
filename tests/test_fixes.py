"""Regression tests for the dead-square, repair, generator and solver fixes."""

import numpy as np

from levelforge.environment import SokobanEnv
from levelforge.grid import BOX_HEAVY, BOX_NORMAL, Level
from levelforge.population import random_solvable_level, repair_level
from levelforge.solvability import crush_cells
from levelforge.solver import verify_level

VERTICAL_SHAFT = """#####
#@  #
# n #
## ##
## ##
##.##
#####"""

HORIZONTAL_CORRIDOR = """#######
#@ n .#
#######"""


def test_corridors_are_not_dead_squares():
    for text in (VERTICAL_SHAFT, HORIZONTAL_CORRIDOR):
        lvl = Level.from_ascii(text)
        dead = crush_cells(lvl)
        bx, by, _ = lvl.boxes[0]
        assert (bx, by) not in dead
        assert verify_level(lvl) is not None


def test_corridor_level_solvable_in_env():
    lvl = Level.from_ascii(HORIZONTAL_CORRIDOR)
    env = SokobanEnv(lvl, max_steps=10)
    env.reset()
    for a in (3, 3, 3):  # right x3: walk, push, push onto goal
        _o, _r, done, info = env.step(a)
    assert done and env.failure is None


def test_corners_still_dead():
    lvl = Level.from_ascii("""#####
#@ .#
#   #
#####""")
    dead = crush_cells(lvl)
    assert (1, 2) in dead and (3, 2) in dead   # bottom corners without goals
    assert (3, 1) not in dead                    # goal cell is live


def test_repair_respects_box_count():
    rng = np.random.default_rng(0)
    lvl = random_solvable_level(rng, size=7, n_boxes=1)
    for _ in range(20):
        out = repair_level(lvl, rng, n_boxes=1)
        assert out is not None and len(out.boxes) == 1 and len(out.goals) == 1


def test_zero_payload_probabilities_give_plain_boxes():
    rng = np.random.default_rng(0)
    for _ in range(15):
        lvl = random_solvable_level(rng, size=7, n_boxes=2, heavy_prob=0.0, fragile_prob=0.0)
        assert all(k == BOX_NORMAL for _x, _y, k in lvl.boxes)


def test_verify_raises_energy_to_plan_cost():
    lvl = Level.from_ascii(HORIZONTAL_CORRIDOR, energy=1)
    lvl.boxes = [(3, 1, BOX_HEAVY)]          # 2 heavy pushes = 6 energy
    info = verify_level(lvl)
    assert info is not None and info["energy_cost"] == 6 and lvl.energy == 6
    assert verify_level(Level.from_ascii(HORIZONTAL_CORRIDOR), max_steps=2) is None  # needs 3 moves


def test_generated_levels_verify():
    rng = np.random.default_rng(3)
    for _ in range(10):
        lvl = random_solvable_level(rng, size=7, n_boxes=2)
        assert verify_level(lvl) is not None


def test_ga_children_keep_plain_boxes_when_kinds_disabled():
    from levelforge.genetic import next_generation

    rng = np.random.default_rng(0)
    pop = [random_solvable_level(rng, size=7, n_boxes=1, heavy_prob=0.0, fragile_prob=0.0) for _ in range(16)]
    for _ in range(3):
        pop, _a, _s = next_generation(pop, list(rng.random(len(pop))), rng, max_steps=120, allow_kinds=False)
        assert all(k == BOX_NORMAL for lvl in pop for _x, _y, k in lvl.boxes)


# --------------------------------------------------------------------------- #
# Review fixes: seeding, graded fitness, failure taxonomy, A* gate, yield
# --------------------------------------------------------------------------- #
WALLED_OFF = """#######
#@#   #
###   #
#  n .#
#     #
#######"""


def test_dqn_exploration_is_reproducible_from_seed():
    from levelforge.dqn import DQNAgent

    def actions(seed):
        agent = DQNAgent(5, 5, seed=seed)
        agent.epsilon = 1.0
        vec = np.zeros(agent.input_dim, dtype=np.float32)
        return [agent.select_action(vec, greedy=False) for _ in range(40)]

    assert actions(3) == actions(3)
    assert actions(3) != actions(4)


def test_greedy_eval_is_pass_fail_and_epsilon_eval_is_graded_but_reproducible():
    from levelforge.dqn import DQNAgent
    from levelforge.training import evaluate_levels

    level = Level.from_ascii("#######\n#@ n .#\n#     #\n#######", energy=20)
    agent = DQNAgent(level.height, level.width, seed=0)
    agent.epsilon = 0.7

    greedy, _ = evaluate_levels(agent, [level], trials=4, max_steps=30, seed=1)
    assert len({(r["solved"], r["steps"]) for r in greedy}) == 1          # identical replays

    a, _ = evaluate_levels(agent, [level], trials=6, max_steps=30, seed=1, epsilon=0.5)
    b, _ = evaluate_levels(agent, [level], trials=6, max_steps=30, seed=1, epsilon=0.5)
    assert [r["steps"] for r in a] == [r["steps"] for r in b]             # seeded
    assert len({r["steps"] for r in a}) > 1                               # trials differ
    assert agent.epsilon == 0.7                                           # restored


def test_blocked_push_is_a_counted_event_not_a_failure_type():
    from levelforge.environment import EVENT_INVALID_PUSH, FAILURE_TYPES, SokobanEnv

    assert "invalid_heavy_push" not in FAILURE_TYPES and len(FAILURE_TYPES) == 4
    env = SokobanEnv(Level.from_ascii("#####\n#.@n#\n#####", energy=10), max_steps=50)
    env.reset()
    _obs, _r, done, info = env.step(3)            # push right: box is against the wall
    assert info[EVENT_INVALID_PUSH] and not done and env.failure is None


def test_a_star_gate_rejects_level_the_quick_checker_accepts():
    from levelforge.solvability import is_solvable

    level = Level.from_ascii(WALLED_OFF, energy=20)   # box and goal fine, player walled off
    assert is_solvable(level)                         # box-only check: necessary, not sufficient
    assert verify_level(level.copy(), max_steps=120) is None


def test_random_generator_reports_its_solvability_yield():
    stats = {}
    for _ in range(5):
        random_solvable_level(np.random.default_rng(len(stats) + _), size=7, n_boxes=1,
                              heavy_prob=0.0, fragile_prob=0.0, max_steps=120, stats=stats)
    assert stats["verified"] == 5
    assert stats["candidates"] >= stats["verified"]
