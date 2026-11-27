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


def test_play_flags_give_heavy_and_fragile_boxes_with_rough_tiles_that_matter():
    from levelforge.grid import BOX_FRAGILE, BOX_HEAVY, Level

    rng = np.random.default_rng(3)
    for heavy, fragile, n in ((True, False, 1), (False, True, 1), (True, True, 2)):
        lvl = generate_play_level(rng, n_boxes=n, heavy=heavy, fragile=fragile)
        kinds = [b[2] for b in lvl.boxes]
        assert (BOX_HEAVY in kinds) == heavy and (BOX_FRAGILE in kinds) == fragile
        assert bool(lvl.rough) == fragile and not set(lvl.rough) & set(lvl.goals)
        moves, _n, _t = solve(lvl)
        assert moves is not None and plan_energy(lvl, moves) <= lvl.energy
        assert Level.from_dict(lvl.to_dict()).rough == lvl.rough           # saved levels keep rough tiles
    assert not generate_play_level(rng).rough                              # plain puzzles have none
    try:
        generate_play_level(rng, n_boxes=1, heavy=True, fragile=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rough_tiles_are_encoded_in_the_wall_channel():
    from levelforge.grid import Level

    lvl = Level.from_ascii("#####\n#@~n#\n#. ##\n#####")
    obs = lvl.encode()
    assert obs.shape[0] == 6 and obs[0, 1, 2] == 0.5 and obs[0, 0, 0] == 1.0 and obs[0, 1, 1] == 0.0


def test_mix_gives_varied_solvable_combinations_of_box_types():
    from levelforge.grid import BOX_FRAGILE

    rng = np.random.default_rng(5)
    combos = set()
    for _ in range(12):
        lvl = generate_play_level(rng, mix=True)
        kinds = [b[2] for b in lvl.boxes]
        assert 2 <= len(kinds) <= 3 and len(set(kinds)) >= 2           # a real mix every time
        assert bool(lvl.rough) == (BOX_FRAGILE in kinds)
        moves, _n, _t = solve(lvl)
        assert moves is not None and plan_energy(lvl, moves) <= lvl.energy
        combos.add(tuple(sorted(kinds)))
    assert len(combos) >= 3                                             # varies between puzzles
    assert len(generate_play_level(rng, n_boxes=3, mix=True).boxes) == 3
