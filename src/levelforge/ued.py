"""Controlled UED experiment: GA adversary vs baselines, PPO agent.

Experimental design
-------------------
Every condition starts from the *same* pretrained PPO checkpoint (per seed),
receives the *same* number of PPO environment steps per cycle, and adds the
*same* number of new levels (``levels_per_cycle``) to its curriculum per
cycle.  The only thing that differs is how those new levels are chosen:

  ga             GA (tournament selection, region crossover, mutation, A*
                 verification) evolved against the current agent; the
                 population persists across cycles and keeps adapting.
  random_search  Same number of candidate levels evaluated as the GA, but
                 drawn fresh at random; top-k by the same fitness are kept.
                 Isolates the contribution of evolutionary search itself.
  random         k fresh random levels, no selection (domain randomisation:
                 the "train on random levels" baseline).
  fixed          One set of k random levels drawn once and reused every
                 cycle (the naive "small fixed training set" baseline).

Training mixture each cycle (identical for all conditions): ``mix_base`` mass
on a shared random base pool, ``mix_new`` on this cycle's new levels and
``mix_archive`` on all earlier new levels (replay against forgetting).

Evaluation each cycle (identical, shared test sets across conditions/seeds):
  heldout       same-distribution random levels (never trained on)
  heldout_hard  same generator with one extra box (harder, out-of-distribution)
  handmade      hand-designed levels (``levels/handmade_7x7.txt``)
  archive       the condition's own past curriculum levels (forgetting check)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .fitness import SolverCache, fitness_values, level_key, novelty, per_level_stats
from .genetic import next_generation, population_diversity
from .grid import Level
from .handmade import DEFAULT_FILE as HANDMADE_DEFAULT, load_handmade
from .population import random_solvable_level
from .ppo_agent import PPOAgent, rollout_batch, set_torch_threads, success_rate
from .solver import verify_level

CONDITIONS = ("ga", "random_search", "random", "fixed")

UED_DEFAULTS: Dict = {
    "size": 7, "n_boxes": 1, "wall_frac": 0.12, "heavy_prob": 0.0, "fragile_prob": 0.0,
    "max_steps": 120,
    "base_pool": 3000,
    "pretrain_steps": 400_000,
    "cycles": 10,
    "steps_per_cycle": 100_000,
    "levels_per_cycle": 16,
    "population": 32,
    "ga_generations": 4,
    "eval_trials": 4,
    "elite": 2,
    "fitness": "regret",
    "novelty_weight": 0.1,
    "mix_base": 0.4, "mix_new": 0.4, "mix_archive": 0.2,
    "heldout_size": 200,
    "test_trials": 2,
    "test_seed": 10_000,
    "handmade_file": None,   # None = levels/handmade_7x7.txt; relative paths resolve against the repo root
    "ppo": {},
    "device": "auto",
    "threads": None,
}


# ---------------------------------------------------------------------- #
# Level helpers
# ---------------------------------------------------------------------- #
def sample_levels(rng: np.random.Generator, n: int, cfg: Dict, n_boxes: Optional[int] = None) -> List[Level]:
    out: List[Level] = []
    while len(out) < n:
        try:
            out.append(random_solvable_level(
                rng, size=cfg["size"], n_boxes=n_boxes or cfg["n_boxes"], wall_frac=cfg["wall_frac"],
                heavy_prob=cfg["heavy_prob"], fragile_prob=cfg["fragile_prob"], max_steps=cfg["max_steps"]))
        except RuntimeError:
            continue
    return out


def test_sets(cfg: Dict) -> Dict[str, List[Level]]:
    """Shared across all conditions and seeds (fixed test seed)."""
    rng = np.random.default_rng(cfg["test_seed"])
    sets = {
        "heldout": sample_levels(rng, cfg["heldout_size"], cfg),
        "heldout_hard": sample_levels(rng, cfg["heldout_size"] // 2, cfg, n_boxes=cfg["n_boxes"] + 1),
    }
    hand = []
    path = cfg.get("handmade_file")
    if path:
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            path = HANDMADE_DEFAULT.parents[1] / path
    for _name, lvl in load_handmade(path or HANDMADE_DEFAULT, size=cfg["size"]):
        info = verify_level(lvl, max_steps=cfg["max_steps"])  # sets energy to plan cost
        if info is not None:
            lvl.energy = info["energy_cost"] + 6
            hand.append(lvl)
    sets["handmade"] = hand
    return sets


def save_levels(levels: Sequence[Level], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([l.to_dict() for l in levels], f)


def load_levels(path: Path) -> List[Level]:
    with open(path, "r", encoding="utf-8") as f:
        return [Level.from_dict(d) for d in json.load(f)]


def evaluate_sets(agent, sets: Dict[str, List[Level]], cfg: Dict) -> Dict[str, float]:
    out = {}
    for name, levels in sets.items():
        if not levels:
            continue
        stoch = rollout_batch(agent, levels, trials=cfg["test_trials"], max_steps=cfg["max_steps"])
        det = rollout_batch(agent, levels, trials=1, max_steps=cfg["max_steps"], deterministic=True)
        out[f"{name}_sr"] = success_rate(stoch)
        out[f"{name}_sr_det"] = success_rate(det)
    return out


def _mixture(base, archive, new, cfg) -> tuple:
    parts = [(base, cfg["mix_base"]), (archive, cfg["mix_archive"]), (new, cfg["mix_new"])]
    parts = [(lv, m) for lv, m in parts if lv and m > 0]
    total = sum(m for _lv, m in parts)
    levels, weights = [], []
    for lv, m in parts:
        levels += list(lv)
        weights += [m / total / len(lv)] * len(lv)
    return levels, weights


# ---------------------------------------------------------------------- #
# Pretraining (shared by all conditions of a seed)
# ---------------------------------------------------------------------- #
def pretrain(cfg: Dict, seed: int, exp_dir: Path) -> Path:
    path = exp_dir / f"pretrain_seed{seed}" / "model.zip"
    if path.exists():
        return path
    set_torch_threads(cfg.get("threads"))
    base = sample_levels(np.random.default_rng(seed), cfg["base_pool"], cfg)
    agent = PPOAgent(base, seed=seed, device=cfg["device"], max_steps=cfg["max_steps"], **cfg["ppo"])
    t0 = time.time()
    print(f"[ued] pretraining seed {seed} for {cfg['pretrain_steps']} steps ...", flush=True)
    agent.train(base, cfg["pretrain_steps"])
    agent.save(path)
    sets = test_sets(cfg)
    stats = {"seed": seed, "timesteps": agent.timesteps, "wall_clock_s": round(time.time() - t0, 1),
             **evaluate_sets(agent, sets, cfg)}
    with open(path.parent / "pretrain_eval.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[ued] pretrain seed {seed}: " + ", ".join(f"{k}={v:.2f}" for k, v in stats.items()
                                                       if k.endswith("_sr")), flush=True)
    return path


# ---------------------------------------------------------------------- #
# Candidate scoring
# ---------------------------------------------------------------------- #
def score_levels(agent, levels: List[Level], cfg: Dict, solver: SolverCache,
                 archive: Sequence[Level] = ()) -> tuple:
    recs = rollout_batch(agent, levels, trials=cfg["eval_trials"], max_steps=cfg["max_steps"])
    stats = per_level_stats(recs, levels, solver)
    nov = novelty(levels, archive) if cfg["novelty_weight"] else None
    fit = fitness_values(stats, cfg["fitness"], nov, cfg["novelty_weight"])
    return fit, stats


def _top_k_unique(cands: List[tuple], k: int, exclude: set) -> List[tuple]:
    """cands: (fitness, level, stats); best first, unique by layout, not in ``exclude``."""
    out, seen = [], set(exclude)
    for f, lvl, st in sorted(cands, key=lambda c: c[0], reverse=True):
        key = level_key(lvl)
        if key in seen or not st["solvable"]:
            continue
        seen.add(key)
        out.append((f, lvl, st))
        if len(out) == k:
            break
    return out


def _discovery(stats: List[Dict]) -> Dict:
    solvable = [s for s in stats if s["solvable"]]
    return {
        "candidates": len(stats),
        "breaking_frac": float(np.mean([s["success"] == 0.0 for s in solvable])) if solvable else 0.0,
        "mean_regret": float(np.mean([s["regret"] for s in solvable])) if solvable else 0.0,
    }


# ---------------------------------------------------------------------- #
# One condition x one seed
# ---------------------------------------------------------------------- #
def run_condition(cfg: Dict, condition: str, seed: int, exp_dir: Path) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    cfg = {**UED_DEFAULTS, **cfg}
    exp_dir = Path(exp_dir)
    out = exp_dir / f"{condition}_seed{seed}"
    if (out / "summary.json").exists():
        print(f"[ued] {out.name} already complete, skipping")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.jsonl").unlink(missing_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg, "condition": condition, "seed": seed}, f, indent=2)
    set_torch_threads(cfg.get("threads"))

    ckpt = pretrain(cfg, seed, exp_dir)
    base = sample_levels(np.random.default_rng(seed), cfg["base_pool"], cfg)   # same pool as pretraining
    agent = PPOAgent.load(ckpt, base, seed=seed, device=cfg["device"], max_steps=cfg["max_steps"], **cfg["ppo"])
    sets = test_sets(cfg)
    solver = SolverCache(cfg["max_steps"])
    rng = np.random.default_rng(seed * 7919 + CONDITIONS.index(condition) + 1)
    k = cfg["levels_per_cycle"]

    population = sample_levels(rng, cfg["population"], cfg) if condition == "ga" else []
    fixed_set = sample_levels(rng, k, cfg) if condition == "fixed" else []
    archive: List[Level] = []
    archive_keys: set = set()
    allow_kinds = cfg["heavy_prob"] > 0 or cfg["fragile_prob"] > 0

    row0 = {"cycle": -1, "condition": condition, "seed": seed, "timesteps": agent.timesteps,
            **evaluate_sets(agent, sets, cfg)}
    _log(out, row0)
    t0 = time.time()
    for cycle in range(cfg["cycles"]):
        row: Dict = {"cycle": cycle, "condition": condition, "seed": seed}
        # ---------------- choose this cycle's new levels ----------------
        if condition == "ga":
            cands, gens, attempts, accepted = [], [], 0, 0
            for g in range(cfg["ga_generations"]):
                fit, stats = score_levels(agent, population, cfg, solver, archive)
                cands += list(zip(fit, population, stats))
                gens.append({"gen": g, "best_fitness": float(max(fit)), "mean_fitness": float(np.mean(fit)),
                             "diversity": float(population_diversity(population)), **_discovery(stats)})
                population, a, s = next_generation(population, fit, rng, elite_k=cfg["elite"],
                                                   max_steps=cfg["max_steps"], allow_kinds=allow_kinds)
                attempts += a
                accepted += s
            chosen = _top_k_unique(cands, k, archive_keys)
            row["ga"] = gens
            row["solvability_yield"] = accepted / attempts if attempts else 1.0
            row.update({f"search_{kk}": v for kk, v in _discovery([c[2] for c in cands]).items()})
        elif condition == "random_search":
            pool = sample_levels(rng, cfg["population"] * cfg["ga_generations"], cfg)
            fit, stats = score_levels(agent, pool, cfg, solver, archive)
            cands = list(zip(fit, pool, stats))
            chosen = _top_k_unique(cands, k, archive_keys)
            row.update({f"search_{kk}": v for kk, v in _discovery(stats).items()})
        else:
            new = fixed_set if condition == "fixed" else sample_levels(rng, k, cfg)
            fit, stats = score_levels(agent, new, cfg, solver)   # measured only, not used to select
            chosen = list(zip(fit, new, stats))
        new_levels = [c[1] for c in chosen]
        row["new_levels_pre_sr"] = float(np.mean([c[2]["success"] for c in chosen]))
        row["new_levels_regret"] = float(np.mean([c[2]["regret"] for c in chosen]))
        row["new_levels_opt_moves"] = float(np.mean([c[2]["optimal_moves"] or 0 for c in chosen]))
        row["new_levels_diversity"] = float(population_diversity(new_levels))

        # ---------------- train ----------------
        levels, weights = _mixture(base, archive, new_levels, cfg)
        agent.train(levels, cfg["steps_per_cycle"], weights)
        if condition != "fixed" or not archive:
            for lvl in new_levels:
                if level_key(lvl) not in archive_keys:
                    archive_keys.add(level_key(lvl))
                    archive.append(lvl)
        save_levels(new_levels, out / "levels" / f"cycle_{cycle:02d}.json")

        # ---------------- evaluate ----------------
        post = rollout_batch(agent, new_levels, trials=cfg["eval_trials"], max_steps=cfg["max_steps"])
        row["new_levels_post_sr"] = success_rate(post)
        row["archive_size"] = len(archive)
        row.update(evaluate_sets(agent, {**sets, "archive": archive}, cfg))
        row["timesteps"] = agent.timesteps
        row["elapsed_s"] = round(time.time() - t0, 1)
        _log(out, row)
        print(f"[ued] {condition} s{seed} cycle {cycle:02d} | new pre/post SR "
              f"{row['new_levels_pre_sr']:.2f}->{row['new_levels_post_sr']:.2f} | heldout "
              f"{row['heldout_sr']:.2f} hard {row['heldout_hard_sr']:.2f} hand {row['handmade_sr']:.2f} "
              f"archive {row['archive_sr']:.2f} | {row['elapsed_s']:.0f}s", flush=True)

    agent.save(out / "model.zip")
    save_levels(archive, out / "archive.json")
    final = evaluate_sets(agent, sets, {**cfg, "test_trials": max(4, cfg["test_trials"])})
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"condition": condition, "seed": seed, "final": final, "timesteps": agent.timesteps,
                   "wall_clock_s": round(time.time() - t0, 1)}, f, indent=2)
    return out


def _log(out: Path, row: Dict) -> None:
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
