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


# Move-optimal plan pushes the heavy box (8 energy); a longer plan needs only 4.
ENERGY_TRAP = """######
# .###
##   #
# n  #
#@H  #
######"""


def test_verify_finds_cheaper_plan_when_move_optimal_is_over_budget():
    from levelforge.solver import plan_energy, replay_return, verify_level

    lvl = Level.from_ascii(ENERGY_TRAP, energy=4)
    fast, _n, _t = solve(lvl)
    assert plan_energy(lvl, fast) > lvl.energy           # the old check would reject this level
    info = verify_level(lvl, fix_energy=False)
    assert info is not None and info["energy_cost"] <= 4 and lvl.energy == 4
    moves, _n, _t = solve(lvl, energy_budget=4)
    assert replay_return(lvl, moves)["solved"]
    assert verify_level(Level.from_ascii(ENERGY_TRAP, energy=3), fix_energy=False) is None


def test_solver_routes_fragile_box_around_rough_tiles():
    from levelforge.solver import replay_return, verify_level

    # Straight push to the right crosses '~'; the safe route goes up a row and back down.
    lvl = Level.from_ascii("""#######
#     #
#     #
#@f~ .#
#     #
#     #
#######""", energy=20)
    moves, _n, _t = solve(lvl)
    assert moves is not None and replay_return(lvl, moves)["solved"]
    smooth = lvl.copy()
    smooth.rough = []
    assert len(moves) > len(solve(smooth)[0])                           # the detour is longer
    blocked = Level.from_ascii("#######\n#@f~ .#\n#######", energy=20)   # no way around
    assert solve(blocked)[0] is None and verify_level(blocked) is None
    normal = Level.from_ascii("#######\n#@n~ .#\n#######", energy=20)    # normal boxes don't care
    assert solve(normal)[0] is not None
