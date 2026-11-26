"""Pick demo levels: GA-discovered levels that break the pretrained agent, before vs after training.

Takes the levels the GA found in the frozen-agent attack (the pretrained agent failed every trial on them),
re-scores them with three agents of the same seed -- pretrained, GA-curriculum-trained and
random-level-trained -- and exports:

  * "learned" levels: the pretrained agent fails, the GA-trained agent now solves them;
  * "still-unsolved" levels: every agent still fails (remaining failure modes).

Writes demo/levels/*.json, demo/levels/*.png and demo/demo_levels.md (with the watch commands).
Every agent is scored on every level, so the table also shows how the random-level agent does.

Usage:  PYTHONPATH=src py scripts/pick_demo_levels.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from levelforge.ppo_agent import PPOAgent, rollout_batch, set_torch_threads
from levelforge.solver import verify_level
from levelforge.ued import UED_DEFAULTS, load_levels
from levelforge.visualize import render_level


def per_level_success(model, levels, cfg, trials):
    agent = PPOAgent.load(model, levels[:4], device="cpu", max_steps=cfg["max_steps"], **cfg["ppo"])
    rec = rollout_batch(agent, levels, trials=trials, max_steps=cfg["max_steps"])
    solved = np.zeros(len(levels))
    for r in rec:
        solved[r["level_index"]] += r["solved"]
    return solved / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="runs/ued_7x7")
    ap.add_argument("--ga-exp", default="runs/ued_7x7_learn",
                    help="experiment whose GA agent is shown (default: the learnability GA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--n-learned", type=int, default=3)
    ap.add_argument("--n-unsolved", type=int, default=2)
    ap.add_argument("--out", default="demo")
    args = ap.parse_args()
    set_torch_threads(4)

    exp, s = Path(args.exp), args.seed
    cfg = {**UED_DEFAULTS, **json.loads((exp / f"ga_seed{s}" / "config.json").read_text(encoding="utf-8"))}
    levels = load_levels(exp / "attack" / f"breaking_ga_seed{s}.json")
    models = {
        "pretrained": exp / f"pretrain_seed{s}" / "model.zip",
        "ga_trained": Path(args.ga_exp) / f"ga_seed{s}" / "model.zip",
        "random_trained": exp / f"random_seed{s}" / "model.zip",
    }
    print(f"[demo] scoring {len(levels)} GA-discovered breaking levels with {list(models)} ...", flush=True)
    sr = {k: per_level_success(m, levels, cfg, args.trials) for k, m in models.items()}
    infos = [verify_level(l.copy(), max_steps=cfg["max_steps"]) for l in levels]
    opt = np.array([len(i["moves"]) if isinstance(i["moves"], list) else int(i["moves"]) for i in infos])

    pre, ga, rnd = sr["pretrained"], sr["ga_trained"], sr["random_trained"]
    learned = [i for i in np.argsort(-(ga - pre) * 100 - opt) if pre[i] == 0 and ga[i] >= 0.75]
    unsolved = [i for i in np.argsort(-opt) if max(pre[i], ga[i], rnd[i]) == 0]
    picks = [("learned", i) for i in learned[:args.n_learned]] + [("unsolved", i) for i in unsolved[:args.n_unsolved]]

    out = Path(args.out)
    (out / "levels").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (kind, i) in enumerate(picks, 1):
        name = f"{n:02d}_{kind}"
        lvl = levels[i]
        lvl.save_json(out / "levels" / f"{name}.json")
        render_level(lvl, str(out / "levels" / f"{name}.png"),
                     title=f"{name}: optimal {opt[i]} moves")
        rows.append((name, opt[i], pre[i], ga[i], rnd[i]))

    summary = {
        "levels_scored": len(levels),
        "pretrained_mean": float(pre.mean()), "ga_trained_mean": float(ga.mean()),
        "random_trained_mean": float(rnd.mean()),
        "learned_candidates": len(learned), "unsolved_by_all": len(unsolved),
    }
    md = ["# Demo levels", "",
          f"GA-discovered levels from `{exp}/attack/breaking_ga_seed{s}.json` (the pretrained agent failed every "
          f"trial during the attack), re-scored with {args.trials} stochastic rollouts per agent.", "",
          f"Across all {len(levels)} levels: pretrained {pre.mean():.2f}, GA-trained {ga.mean():.2f}, "
          f"random-trained {rnd.mean():.2f} mean success.", "",
          "| Level | Optimal moves | Pretrained | GA-trained | Random-trained |", "|---|---|---|---|---|"]
    md += [f"| `{n}` | {o} | {p:.2f} | {g:.2f} | {r:.2f} |" for n, o, p, g, r in rows]
    md += ["", "Watch commands (PowerShell, from the repo root, after `$env:PYTHONPATH = \"src\"`):", "", "```"]
    for n, *_ in rows:
        lv = f"{out}/levels/{n}.json"
        md += [f"py -m levelforge watch --level {lv} --tk --ppo {models['pretrained']}",
               f"py -m levelforge watch --level {lv} --tk --ppo {models['ga_trained']}",
               f"py -m levelforge watch --level {lv} --tk --solver", ""]
    md += ["```"]
    (out / "demo_levels.md").write_text("\n".join(md).replace("\\", "/"), encoding="utf-8")
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in rows:
        print("[demo]", r)


if __name__ == "__main__":
    main()
