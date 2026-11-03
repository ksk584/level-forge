"""Co-evolution loop: GA level generator x DQN agent.

One run directory layout:
    runs/<name>/
        config.json
        metrics.jsonl          # one JSON object per generation
        levels_gen_<g>.json    # best levels discovered that generation
        checkpoint_gen_<g>.pt
        summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .dqn import DQNAgent
from .genetic import next_generation, population_diversity
from .grid import Level
from .population import random_solvable_level
from .training import evaluate_levels, generate_level_pool, train_on_levels


DEFAULT_CONFIG = {
    "size": 10,
    "n_boxes": 3,
    "wall_frac": 0.12,
    "heavy_prob": 0.25,
    "fragile_prob": 0.20,
    "population_size": 24,
    "generations": 10,
    "elite_k": 2,
    "eval_trials": 2,
    "topk_retrain": 5,
    "pretrain_episodes": 300,
    "retrain_episodes": 120,
    "pool_size": 40,
    "max_steps": 300,
    "max_nodes": 30000,
    "seed": 42,
    "heldout_size": 20,
}


def _best_levels(levels: List[Level], fitnesses: List[float], k: int) -> List[Level]:
    order = np.argsort(fitnesses)[::-1]
    return [levels[i] for i in order[:k]]


def run_coevolution(
    config: Optional[Dict] = None,
    out_dir: str | Path = "runs/coevolution",
    generator: str = "ga",          # "ga" or "random" (ablation baseline)
    name: Optional[str] = None,
) -> Path:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if name:
        out_dir = Path(out_dir) / name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg, "generator": generator}, f, indent=2)

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    print(f"[levelforge] generating pool of random solvable levels (seed={seed}) ...")
    pool = generate_level_pool(
        cfg["pool_size"], seed=seed, size=cfg["size"], n_boxes=cfg["n_boxes"],
        wall_frac=cfg["wall_frac"], heavy_prob=cfg["heavy_prob"], fragile_prob=cfg["fragile_prob"],
    )
    heldout_rng = np.random.default_rng(seed + 10_000)
    heldout = [
        random_solvable_level(heldout_rng, size=cfg["size"], n_boxes=cfg["n_boxes"],
                              wall_frac=cfg["wall_frac"], heavy_prob=cfg["heavy_prob"],
                              fragile_prob=cfg["fragile_prob"])
        for _ in range(cfg["heldout_size"])
    ]

    agent = DQNAgent(cfg["size"], cfg["size"], seed=seed)

    print(f"[levelforge] pretraining DQN for {cfg['pretrain_episodes']} episodes ...")
    train_on_levels(agent, pool, episodes=cfg["pretrain_episodes"], max_steps=cfg["max_steps"],
                    epsilon_decay_episodes=max(50, cfg["pretrain_episodes"] // 2), seed=seed)
    agent.save(out_dir / "checkpoint_pretrain.pt")

    print("[levelforge] initialising population ...")
    population: List[Level] = []
    while len(population) < cfg["population_size"]:
        try:
            population.append(random_solvable_level(rng, size=cfg["size"], n_boxes=cfg["n_boxes"],
                                                    wall_frac=cfg["wall_frac"],
                                                    heavy_prob=cfg["heavy_prob"],
                                                    fragile_prob=cfg["fragile_prob"]))
        except RuntimeError:
            continue

    metrics_path = out_dir / "metrics.jsonl"
    t0 = time.time()
    for gen in range(cfg["generations"]):
        records, agg = evaluate_levels(agent, population, trials=cfg["eval_trials"],
                                       max_steps=cfg["max_steps"], seed=seed + gen)
        failures = [0] * len(population)
        for r in records:
            if not r["solved"]:
                failures[r["level_index"]] += 1
        fitnesses = [f / max(1, cfg["eval_trials"]) for f in failures]  # failure rate = fitness

        diversity = population_diversity(population)
        best = _best_levels(population, fitnesses, cfg["topk_retrain"])
        hard_records = [r for r in records if not r["solved"]]
        failure_types = sorted({r["failure"] for r in hard_records if r["failure"]})

        # Retrain agent on the hardest discovered levels + a slice of the random pool.
        weights = np.ones(len(pool), dtype=np.float64)
        retrain_levels = pool + best
        weights = np.concatenate([weights, np.full(len(best), 4.0)])
        train_on_levels(agent, retrain_levels, episodes=cfg["retrain_episodes"],
                        max_steps=cfg["max_steps"], epsilon_start=0.4, epsilon_end=0.05,
                        epsilon_decay_episodes=max(20, cfg["retrain_episodes"] // 2),
                        seed=seed + 100 * (gen + 1), level_weights=weights)
        agent.save(out_dir / f"checkpoint_gen_{gen}.pt")

        heldout_records, heldout_agg = evaluate_levels(agent, heldout, trials=1,
                                                       max_steps=cfg["max_steps"], seed=seed + 500 + gen)
        gen_stats = {
            "generation": gen,
            "generator": generator,
            "agent_success_rate_on_pop": float(np.mean([1.0 - f for f in fitnesses])),
            "population_avg_failure_rate": float(np.mean(fitnesses)),
            "population_diversity": float(diversity),
            "solvability_yield": None,   # filled by breeding step below
            "failure_types_seen": failure_types,
            "heldout_success_rate": heldout_agg["success_rate"],
            "heldout_mean_steps": heldout_agg["mean_steps"],
            "train_loss": agent.last_loss,
            "elapsed_s": round(time.time() - t0, 2),
        }

        # Breed next generation (or sample random levels for the ablation).
        if generator == "ga":
            children, attempts, solvable = next_generation(
                population, fitnesses, rng, elite_k=cfg["elite_k"], max_nodes=cfg["max_nodes"])
            yield_ = solvable / attempts if attempts else 1.0
            gen_stats["solvability_yield"] = float(yield_)
            gen_stats["offspring_attempts"] = attempts
        else:  # "random" ablation: replace breeding with fresh random levels
            children, attempts, solvable = [], 0, 0
            while len(children) < len(population):
                try:
                    children.append(random_solvable_level(rng, size=cfg["size"], n_boxes=cfg["n_boxes"],
                                                           wall_frac=cfg["wall_frac"],
                                                           heavy_prob=cfg["heavy_prob"],
                                                           fragile_prob=cfg["fragile_prob"],
                                                           max_nodes=cfg["max_nodes"]))
                    attempts += 1
                    solvable += 1
                except RuntimeError:
                    attempts += 1
            gen_stats["solvability_yield"] = 1.0
            gen_stats["offspring_attempts"] = attempts

        Ldir = out_dir / f"levels_gen_{gen}"
        Ldir.mkdir(exist_ok=True)
        for i, lvl in enumerate(best):
            lvl.save_json(Ldir / f"hard_{i}.json")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(gen_stats) + "\n")
        print(f"[levelforge] gen {gen:02d} | pop failure={gen_stats['population_avg_failure_rate']:.2f} "
              f"| diversity={diversity:.3f} | SY={gen_stats['solvability_yield']:.2f} "
              f"| heldout SR={heldout_agg['success_rate']:.2f}")
        population = children

    final_records, final_agg = evaluate_levels(agent, heldout, trials=3,
                                               max_steps=cfg["max_steps"], seed=seed + 999)
    summary = {
        "config": cfg,
        "generator": generator,
        "final_heldout": final_agg,
        "wall_clock_s": round(time.time() - t0, 2),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[levelforge] done. Summary -> {out_dir / 'summary.json'}")
    return out_dir
