"""GA operator tests: repair, diversity, breeding."""

import numpy as np

from levelforge.genetic import crossover, mutate, next_generation, population_diversity
from levelforge.grid import BOX_HEAVY, BOX_NORMAL, Level
from levelforge.population import repair_level


def border_walls(h, w):
    walls = np.zeros((h, w), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    return walls


def base_level():
    walls = border_walls(6, 7)
    return Level(walls, goals=[(4, 2), (2, 3)], boxes=[(2, 2, BOX_NORMAL), (3, 2, BOX_HEAVY)], agent=(1, 1), energy=14)


def test_repair_fixes_duplicate_boxes():
    walls = border_walls(6, 7)
    bad = Level(walls, goals=[(4, 2), (2, 3)], boxes=[(2, 2, BOX_NORMAL), (2, 2, BOX_HEAVY)], agent=(1, 1), energy=10)
    repaired = repair_level(bad, np.random.default_rng(0))
    assert repaired is not None
    assert repaired.validate() == []
    positions = repaired.box_positions()
    assert len(set(positions)) == len(positions)


def test_repair_moves_agent_off_box():
    walls = border_walls(6, 7)
    bad = Level(walls, goals=[(4, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(2, 2), energy=10)
    repaired = repair_level(bad, np.random.default_rng(1))
    assert repaired is not None
    assert repaired.validate() == []


def test_signature_identical_levels_zero_distance():
    a, b = base_level(), base_level()
    assert population_diversity([a, b]) == 0.0
    assert population_diversity([a]) == 0.0


def test_mutate_keeps_level_valid():
    rng = np.random.default_rng(7)
    lvl = base_level()
    for _ in range(20):
        out = mutate(lvl, rng)
        if out is None:
            continue
        assert out.validate() == []
        lvl = out


def test_crossover_returns_valid_level():
    rng = np.random.default_rng(3)
    a, b = base_level(), base_level()
    child = crossover(a, b, rng)
    assert child is not None
    assert child.validate() == []


def test_next_generation_returns_n_children():
    rng = np.random.default_rng(11)
    pop = [base_level() for _ in range(4)]
    fit = [0.9, 0.5, 0.2, 0.1]
    children, attempts, solvable = next_generation(pop, fit, rng, elite_k=1, max_nodes=3000, max_attempts=30)
    assert len(children) == len(pop)
    assert attempts >= 0
