"""Tests for the UED layer: gym wrapper, fitness, hand-made set, evaluator, stats."""

import numpy as np
import pytest

from levelforge.analysis import _spread, bootstrap_ci, permutation_p
from levelforge.fitness import SolverCache, fitness_values, novelty, per_level_stats
from levelforge.grid import Level
from levelforge.handmade import load_handmade, pad_level
from levelforge.population import random_solvable_level
from levelforge.solver import solve, verify_level

gym = pytest.importorskip("gymnasium")
from levelforge.gym_env import OBS_CHANNELS, LevelSource, SokobanGymEnv  # noqa: E402
from levelforge.ppo_agent import rollout_batch, success_rate  # noqa: E402

CORRIDOR = Level.from_ascii("""#######
#@ n .#
#######""")


class RandomActor:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic):
        return self.rng.integers(0, 4, len(obs))


class PlanActor:
    """Replays a fixed plan for every env in the batch (all envs share one level)."""

    def __init__(self, plan):
        self.plan, self.t = plan, 0

    def act(self, obs, deterministic):
        a = self.plan[min(self.t, len(self.plan) - 1)]
        self.t += 1
        return np.full(len(obs), a)


def test_gym_env_solver_plan_terminates_with_success():
    lvl = random_solvable_level(np.random.default_rng(1), size=7, n_boxes=1)
    env = SokobanGymEnv(LevelSource([lvl]), max_steps=120)
    obs, _ = env.reset()
    assert obs.shape == (OBS_CHANNELS, 7, 7)
    total = 0.0
    for a in solve(lvl)[0]:
        obs, r, term, trunc, info = env.step(a)
        total += r
    assert term and not trunc and info["is_success"]
    assert total > 9.0  # +10 solve, +1 box, small step costs


def test_gym_env_timeout_is_truncation():
    env = SokobanGymEnv(LevelSource([CORRIDOR]), max_steps=3)
    env.reset()
    for _ in range(3):
        _o, _r, term, trunc, _i = env.step(2)  # walk into the left wall
    assert trunc and not term


def test_regret_semantics():
    lvl = CORRIDOR
    solver = SolverCache(max_steps=50)
    opt = verify_level(lvl.copy())["moves"]
    recs = [
        {"level_index": 0, "solved": True, "steps": opt, "failure": None},       # optimal
        {"level_index": 0, "solved": True, "steps": 2 * opt, "failure": None},   # half efficient
        {"level_index": 0, "solved": False, "steps": 50, "failure": "timeout"},
        {"level_index": 0, "solved": False, "steps": 10, "failure": "deadlock"},
    ]
    st = per_level_stats(recs, [lvl], solver)[0]
    assert st["success"] == 0.5 and st["learnability"] == 0.25
    assert st["regret"] == pytest.approx(1 - (1 + 0.5) / 4)
    assert fitness_values([st], "failure") == [0.5]


def test_unsolvable_level_gets_zero_fitness():
    dead = Level.from_ascii("""#####
#@ .#
#n  #
#####""")
    st = per_level_stats([{"level_index": 0, "solved": False, "steps": 5, "failure": "timeout"}],
                         [dead], SolverCache(max_steps=50))[0]
    assert not st["solvable"] and fitness_values([st], "regret") == [0.0]


def test_novelty_prefers_distinct_levels():
    rng = np.random.default_rng(0)
    a = random_solvable_level(rng, size=7, n_boxes=1)
    b = random_solvable_level(rng, size=7, n_boxes=1)
    nov = novelty([a, a.copy(), b], k=1)
    assert nov[2] > nov[0] == 0.0


def test_handmade_levels_all_solvable_and_padded():
    levels = load_handmade(size=7)
    assert len(levels) >= 12
    for name, lvl in levels:
        assert lvl.walls.shape == (7, 7)
        assert verify_level(lvl, max_steps=120) is not None, name
    big = pad_level(CORRIDOR, 9)
    assert big.walls.shape == (9, 9) and big.walls[8].all()


def test_rollout_batch_records_and_plan_success():
    plan = solve(CORRIDOR)[0]
    recs = rollout_batch(PlanActor(plan), [CORRIDOR], trials=3, max_steps=20)
    assert len(recs) == 3 and success_rate(recs) == 1.0
    recs = rollout_batch(RandomActor(), [CORRIDOR] * 4, trials=2, max_steps=20)
    assert {r["level_index"] for r in recs} == {0, 1, 2, 3}
    assert all(r["steps"] <= 20 for r in recs)


def test_stats_helpers():
    assert permutation_p([1, 1, 1], [0, 0, 0]) == pytest.approx(0.1)
    assert permutation_p([0.5, 0.5], [0.5, 0.5]) == 1.0
    lo, hi = bootstrap_ci([1.0, 1.1, 0.9], [0.0, 0.1, -0.1], n=2000)
    assert 0.5 < lo < hi < 1.5
    ys = _spread([0.10, 0.11, 0.50], 0.05)
    assert ys[1] - ys[0] >= 0.05 - 1e-9 and ys[2] == 0.50


def test_handmade_file_config_loads_9x9_two_box_set():
    from levelforge.ued import UED_DEFAULTS, test_sets

    cfg = {**UED_DEFAULTS, "size": 9, "n_boxes": 2, "max_steps": 200, "heldout_size": 4,
           "handmade_file": "levels/handmade_9x9_2box.txt"}
    sets = test_sets(cfg)
    assert len(sets["handmade"]) == 12                      # every puzzle A*-verified
    assert all(l.walls.shape == (9, 9) and len(l.boxes) == 2 for l in sets["handmade"])
    assert all(len(l.boxes) == 3 for l in sets["heldout_hard"])


TINY_UED = {"size": 7, "n_boxes": 1, "heavy_prob": 0.3, "fragile_prob": 0.3, "base_pool": 8,
            "pretrain_steps": 64, "cycles": 2, "steps_per_cycle": 64, "levels_per_cycle": 2,
            "population": 4, "ga_generations": 1, "eval_trials": 1, "elite": 1, "heldout_size": 4,
            "test_trials": 1, "threads": 1, "device": "cpu",
            "ppo": {"n_envs": 2, "n_steps": 32, "batch_size": 32}}


def test_ued_pipeline_end_to_end_with_crash_resume(tmp_path, monkeypatch):
    """ued-run (crash after cycle 1, then resume), ued-attack and ued-analyze on a tiny config."""
    import json

    from levelforge import ued
    from levelforge.analysis import analyze
    from levelforge.attack import run_attack

    real_log = ued._log

    def crashing_log(out, row):
        real_log(out, row)
        if row["cycle"] == 1:
            raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(ued, "_log", crashing_log)
    with pytest.raises(KeyboardInterrupt):
        ued.run_condition(TINY_UED, "ga", 0, tmp_path)
    run = tmp_path / "ga_seed0"
    assert (run / "resume.json").exists() and not (run / "summary.json").exists()

    monkeypatch.setattr(ued, "_log", real_log)
    ued.run_condition(TINY_UED, "ga", 0, tmp_path)       # resumes at cycle 1 (row was not checkpointed)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["cycle"] for r in rows] == [-1, 0, 1]
    assert (run / "summary.json").exists() and not (run / "resume.json").exists()
    assert len(ued.load_levels(run / "archive.json")) >= 2

    ued.run_condition(TINY_UED, "random", 0, tmp_path)
    run_attack(tmp_path / "pretrain_seed0" / "model.zip", TINY_UED, tmp_path / "attack", budget=8, seeds=(0,))
    out = analyze(tmp_path)
    text = (out / "results.md").read_text(encoding="utf-8")
    assert "Held-out random levels" in text and "Failure-discovery efficiency" in text
