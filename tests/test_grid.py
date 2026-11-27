"""Grid serialisation, validation and solvability tests."""

from levelforge.grid import BOX_NORMAL, Level
from levelforge.solvability import is_solvable


def test_ascii_roundtrip():
    text = "######\n#    #\n#@n .#\n#    #\n######"
    level = Level.from_ascii(text, energy=12)
    assert level.agent == (1, 2)
    assert level.boxes == [(2, 2, BOX_NORMAL)]
    assert level.goals == [(4, 2)]
    again = Level.from_ascii(level.to_ascii(), energy=level.energy)
    assert again.walls.tolist() == level.walls.tolist()
    assert again.boxes == level.boxes
    assert again.goals == level.goals
    assert again.agent == level.agent


def test_validate_catches_border_violation():
    import numpy as np

    walls = np.zeros((5, 6), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    walls[0, 0] = False  # break the border
    level = Level(walls, goals=[(3, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(1, 2), energy=5)
    problems = level.validate()
    assert any("border" in p for p in problems)


def test_solvable_simple_level():
    text = "#######\n#     #\n# n . #\n#  @  #\n#######"
    level = Level.from_ascii(text, energy=20)
    assert is_solvable(level, max_nodes=5000)


def test_corner_deadlock_is_unsolvable():
    text = "#######\n#n#   #\n#     #\n#  .  #\n#  @  #\n#######"
    level = Level.from_ascii(text, energy=20)
    # Box at (1,1) sits in a corner (top + left walls) => deadlocked.
    assert not is_solvable(level, max_nodes=5000)


def test_ascii_round_trip_keeps_rough_under_agent_and_boxes():
    import numpy as np

    from levelforge.grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

    walls = np.zeros((5, 7), dtype=bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    lvl = Level(walls, [(5, 1), (5, 2), (5, 3)], [(2, 1, BOX_NORMAL), (2, 2, BOX_HEAVY), (2, 3, BOX_FRAGILE)],
                (1, 1), 20, rough=[(1, 1), (2, 1), (2, 2), (2, 3), (4, 2)])
    text = lvl.to_ascii()
    assert "&" in text and "m" in text and "k" in text and "z" in text and "~" in text
    back = Level.from_ascii(text, energy=20)
    assert (back.rough, back.boxes, back.agent, sorted(back.goals)) == (lvl.rough, lvl.boxes, lvl.agent, sorted(lvl.goals))
    bad = Level(walls, [(5, 1)], [(2, 1, BOX_NORMAL)], (1, 1), 20, rough=[(5, 1)])
    assert any("rough tile" in p for p in bad.validate())
