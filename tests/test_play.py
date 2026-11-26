import numpy as np

from levelforge.population import generate_play_level
from levelforge.solver import plan_energy, solve


def test_random_play_levels_are_valid_solvable_and_vary():
    rng = np.random.default_rng(42)
    first = generate_play_level(rng)
    second = generate_play_level(rng)

    assert first.validate() == []
    assert second.validate() == []
    assert not np.array_equal(first.signature(), second.signature())

    moves, _nodes, _elapsed = solve(first)
    assert moves is not None
    assert plan_energy(first, moves) <= first.energy
