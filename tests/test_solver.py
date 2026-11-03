"""A* solver decides correctly on tiny handcrafted boards."""

from levelforge.grid import Level
from levelforge.solver import solve
from levelforge.environment import SokobanEnv


def tiny_level():
    """A box already resting on its goal: trivially solved."""
    import numpy as np
    walls = np.zeros((5, 6), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    return Level(walls, goals=[(2, 2)], boxes=[(2, 2, 0)], agent=(1, 2), energy=10)


def test_solver_solves_trivial_box_on_goal():
    lvl = tiny_level()
    moves, nodes, dt = solve(lvl, max_nodes=1000)
    assert moves == []


def test_solver_solves_push_puzzle():
    import numpy as np

    walls = np.zeros((5, 7), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    # push box right onto the goal: agent @ at (1,2), box n at (2,2), goal at (4,2)
    lvl = Level(walls, goals=[(4, 2)], boxes=[(2, 2, 0)], agent=(1, 2), energy=10)
    moves, nodes, dt = solve(lvl, max_nodes=10000)
    assert moves is not None
    # replay the solution through the real environment to prove validity
    env = SokobanEnv(lvl, max_steps=len(moves) + 1)
    obs, _ = env.reset()
    for a in moves:
        obs, _r, done, info = env.step(a)
    assert info["solved"]


def test_solver_reports_unsolvable():
    import numpy as np

    walls = np.zeros((5, 6), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    walls[2, 2] = True          # box stuck in corner (1,1)? Actually use corner:
    walls[0, 1] = True
    # box at (1,1) has left/up walls (border) -> corner -> unsolvable
    lvl = Level(walls, goals=[(4, 2)], boxes=[(1, 1, 0)], agent=(3, 3), energy=10)
    moves, nodes, dt = solve(lvl, max_nodes=5000)
    assert moves is None
