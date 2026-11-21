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
