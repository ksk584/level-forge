"""Calibration for the scaled-up experiment: how fast does PPO learn bigger boards / more boxes?

For each candidate setting it reports level-generation cost, optimal solution
lengths, and a PPO learning curve on held-out levels. The scaled experiment's
pretraining and per-cycle budgets are chosen from these curves: the agent must
be clearly above zero (otherwise every level has the same fitness and the GA
degenerates into random search) but far from saturated.

Usage:  PYTHONPATH=src py scripts/scale_calibration.py --variant 9x9_1box --threads 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from levelforge.ppo_agent import PPOAgent, rollout_batch, set_torch_threads, success_rate
from levelforge.solver import verify_level
from levelforge.ued import UED_DEFAULTS, sample_levels

VARIANTS = {
    "7x7_1box": {"size": 7, "n_boxes": 1, "max_steps": 120},   # main experiment, for reference
    "9x9_1box": {"size": 9, "n_boxes": 1, "max_steps": 160},
    "7x7_2box": {"size": 7, "n_boxes": 2, "max_steps": 160},
    "9x9_2box": {"size": 9, "n_boxes": 2, "max_steps": 200},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/scale_calibration")
    args = ap.parse_args()

    set_torch_threads(args.threads)
    cfg = {**UED_DEFAULTS, **VARIANTS[args.variant]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pool = sample_levels(np.random.default_rng(0), args.pool, cfg)
    gen_s = (time.time() - t0) / args.pool
    test = sample_levels(np.random.default_rng(10_000), 200, cfg)
    opt = [verify_level(l, max_steps=cfg["max_steps"])["moves"] for l in test[:100]]
    opt = [len(m) if isinstance(m, list) else m for m in opt]
    log = {"variant": args.variant, "cfg": VARIANTS[args.variant], "gen_s_per_level": round(gen_s, 3),
           "optimal_moves_mean": float(np.mean(opt)), "optimal_moves_p90": float(np.percentile(opt, 90)),
           "curve": []}
    print(f"[calib] {args.variant}: {gen_s:.3f}s/level, optimal moves mean {np.mean(opt):.1f} "
          f"p90 {np.percentile(opt, 90):.0f}", flush=True)

    agent = PPOAgent(pool, seed=0, device=cfg["device"], max_steps=cfg["max_steps"], **cfg["ppo"])
    t0 = time.time()
    while agent.timesteps < args.steps:
        agent.train(pool, args.chunk)
        sr = success_rate(rollout_batch(agent, test, trials=2, max_steps=cfg["max_steps"]))
        row = {"timesteps": agent.timesteps, "heldout_sr": sr, "elapsed_s": round(time.time() - t0, 1)}
        log["curve"].append(row)
        print(f"[calib] {args.variant} {row}", flush=True)
        (out / f"{args.variant}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
