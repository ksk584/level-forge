"""Smoke tests for environment mechanics."""

import numpy as np

from levelforge.environment import FAILURE_FRAGILE, SokobanEnv
from levelforge.grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level


def make_walls():
    walls = np.zeros((5, 6), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    return walls


def test_move_into_wall_is_noop():
    level = Level(make_walls(), goals=[(4, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(1, 2), energy=10)
    env = SokobanEnv(level)
    _, reward, done, _info = env.step(1)  # down? no: 1 = down; use into wall via up
    # up from row 2 is free; instead push once and check energy bookkeeping
    env.reset()
    _, reward, _, info = env.step(2)  # left: into wall at x=0? agent at x=1 moving left -> x=0 is wall
    assert info["steps"] == 1
    assert env.level.agent == (1, 2)
    assert env.level.energy == 10


def test_push_box_to_goal_solves():
    walls = make_walls()
    level = Level(walls, goals=[(4, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(1, 2), energy=10)
    env = SokobanEnv(level)
    env.step(3)  # push right: box 2->3
    assert env.level.energy == 9
    _, reward, done, info = env.step(3)  # push right: box 3->4 (goal)
    assert done
    assert reward >= 100.0
    assert env.level.boxes_on_goals() == 1


def test_heavy_push_costs_three_energy():
    walls = make_walls()
    level = Level(walls, goals=[(5, 2)], boxes=[(2, 2, BOX_HEAVY)], agent=(1, 2), energy=10)
    env = SokobanEnv(level)
    env.step(3)
    assert env.level.energy == 7


def test_heavy_push_with_insufficient_energy_is_blocked():
    walls = make_walls()
    level = Level(walls, goals=[(5, 2)], boxes=[(2, 2, BOX_HEAVY)], agent=(1, 2), energy=2)
    env = SokobanEnv(level)
    env.step(3)
    assert env.level.boxes[0][:2] == (2, 2)  # box did not move
    assert env.level.energy == 2


def test_fragile_box_breaks_in_crush_cell():
    walls = np.zeros((5, 7), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    # (5,1) has up wall (row 0) and right wall (col 6) -> crush cell
    level = Level(walls, goals=[(1, 3)], boxes=[(4, 1, BOX_FRAGILE)], agent=(3, 1), energy=10)
    env = SokobanEnv(level)
    _, reward, done, info = env.step(3)  # push right into crush cell
    assert done
    assert env.failure == FAILURE_FRAGILE
    assert reward <= -15.0


def test_box_on_goal_counts_only_remaining():
    walls = make_walls()
    level = Level(walls, goals=[(3, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(1, 2), energy=10)
    env = SokobanEnv(level)
    env.step(3)
    assert env.level.boxes_on_goals() == 1


FREEZE = """########
#      #
# nn   #
# n n@ #
#      #
# .... #
########"""


def test_pushing_into_a_2x2_box_block_is_a_freeze_deadlock():
    from levelforge.environment import FAILURE_DEADLOCK
    from levelforge.solvability import any_frozen_off_goal, crush_cells

    level = Level.from_ascii(FREEZE, energy=30)
    dead = crush_cells(level)
    assert not any_frozen_off_goal(level.walls, level.boxes, level.goal_set(), dead)
    env = SokobanEnv(level)
    _, _r, done, info = env.step(2)                      # push (4,3) left -> completes the 2x2 block
    assert done and info["failure"] == FAILURE_DEADLOCK
    assert any_frozen_off_goal(env.level.walls, env.level.boxes, env.level.goal_set(), dead)


def test_frozen_boxes_on_goals_are_not_a_deadlock():
    from levelforge.solvability import any_frozen_off_goal, crush_cells

    level = Level.from_ascii("""######
#    #
# NN #
# NN #
#@   #
######""")
    assert not any_frozen_off_goal(level.walls, level.boxes, level.goal_set(), crush_cells(level))


def test_fragile_box_breaks_on_rough_tile_but_normal_box_and_agent_cross_it():
    from levelforge.environment import FAILURE_FRAGILE

    for ch, expected in (("f", FAILURE_FRAGILE), ("n", None)):
        env = SokobanEnv(Level.from_ascii(f"#######\n#@{ch}~ .#\n#######", energy=10))
        _, _r, done, info = env.step(3)                      # push the box onto the rough tile
        assert info["failure"] == expected
        assert done == (expected is not None)
    env = SokobanEnv(Level.from_ascii("######\n#@~  #\n# n .#\n######", energy=10))
    env.step(3)                                              # the agent walks onto rough floor
    assert env.level.agent == (2, 1) and not env.done


def test_fragile_box_cut_off_by_rough_floor_is_a_deadlock_at_once():
    from levelforge.environment import FAILURE_DEADLOCK

    # After one push the fragile box can only reach the goal by crossing '~': dead now, not at timeout.
    level = Level.from_ascii("""#######
#     #
#@f ~.#
#   ~ #
#######""", energy=20)
    env = SokobanEnv(level)
    _, _r, done, info = env.step(3)
    assert done and info["failure"] == FAILURE_DEADLOCK
    normal = SokobanEnv(Level.from_ascii(level.to_ascii().replace("f", "n"), energy=20))
    assert not normal.step(3)[2]                              # a normal box there is fine
