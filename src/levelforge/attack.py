"""Fixed-agent search experiment: GA vs random search at equal evaluation budget.

The agent is frozen, so this isolates the *search* question from co-evolution
dynamics: given a budget of B candidate levels (each scored with T stochastic
rollouts), how many distinct, solver-verified levels that break the agent does
each method find, and how quickly?

"Breaking" = the agent fails all T trials on a level that A* proves solvable
within the step limit.  Curves are logged after every batch of P candidates
(P = GA population size), so "levels evaluated to find N breaking levels" can
be read directly off them -- the headline efficiency number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .fitness import SolverCache
from .genetic import next_generation, population_diversity
from .ued import UED_DEFAULTS, sample_levels, save_levels, score_levels
from .ppo_agent import PPOAgent, set_torch_threads


def _run(method: str, agent, cfg: Dict, budget: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    solver = SolverCache(cfg["max_steps"])
    P = cfg["population"]
    found: Dict[str, object] = {}
    curve: List[Dict] = []
    evaluated = 0
    attempts = accepted = 0
    population = sample_levels(rng, P, cfg)
    best = 0.0
    while evaluated < budget:
        batch = population if method == "ga" else sample_levels(rng, P, cfg)
        fit, stats = score_levels(agent, batch, cfg, solver)
        evaluated += len(batch)
        for lvl, st in zip(batch, stats):
            if st["solvable"] and st["success"] == 0.0:
                # Distinct by layout: GA variants that differ only in energy count once.
                found.setdefault(lvl.to_ascii(), lvl)
        best = max(best, max(s["regret"] for s in stats))
        curve.append({
            "evaluated": evaluated, "breaking_found": len(found),
            "batch_breaking_frac": float(np.mean([s["solvable"] and s["success"] == 0.0 for s in stats])),
            "batch_mean_regret": float(np.mean([s["regret"] for s in stats])),
            "best_regret": float(best),
            "diversity": float(population_diversity(batch)),
        })
        if method == "ga":
            population, a, s = next_generation(population, fit, rng, elite_k=cfg["elite"],
                                               max_steps=cfg["max_steps"],
                                               allow_kinds=cfg["heavy_prob"] > 0 or cfg["fragile_prob"] > 0)
            attempts += a
            accepted += s
    return {"method": method, "seed": seed, "curve": curve, "breaking_levels": list(found.values()),
            "solvability_yield": (accepted / attempts) if attempts else None}


def run_attack(model_path, cfg: Dict, out_dir, budget: int = 1024, seeds=(0, 1, 2),
               methods=("ga", "random_search")) -> Path:
    # The attack's objective is to break the agent, so selection always uses regret,
    # whatever fitness the curriculum experiment (cfg) uses for training.
    cfg = {**UED_DEFAULTS, **cfg, "fitness": "regret"}
    set_torch_threads(cfg.get("threads"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dummy = sample_levels(np.random.default_rng(0), 4, cfg)
    agent = PPOAgent.load(model_path, dummy, device=cfg["device"], max_steps=cfg["max_steps"], **cfg["ppo"])
    results = []
    for seed in seeds:
        for method in methods:
            r = _run(method, agent, cfg, budget, seed)
            save_levels(r.pop("breaking_levels"), out / f"breaking_{method}_seed{seed}.json")
            results.append(r)
            last = r["curve"][-1]
            print(f"[attack] {method:13s} seed {seed}: {last['breaking_found']} breaking levels "
                  f"in {last['evaluated']} evaluated | best regret {last['best_regret']:.2f}", flush=True)
    with open(out / "attack.json", "w", encoding="utf-8") as f:
        json.dump({"model": str(model_path), "budget": budget, "config": cfg, "runs": results}, f, indent=2)
    return out
