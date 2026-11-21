"""Fitness ablation: GA with regret fitness vs GA with learnability fitness.

Compares the main experiment (runs/ued_7x7, GA uses regret) with a GA run that
uses learnability fitness p(1-p) (runs/ued_7x7_learn). Both start from the same
pretrained checkpoints and share every other setting, so the fitness function
is the only difference.

The adversarial test set is the SAME as in the main experiment: the union of
the *regret* GA archives from the other seeds. The learnability agent never
trained on any of them, so the comparison with the main table is like-for-like.

Usage:  PYTHONPATH=src py scripts/fitness_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from levelforge.analysis import (COLORS, GRID, LABELS, MUTED, TEXT, _spread, _style, bootstrap_ci,
                                 load_runs, permutation_p)
from levelforge.ued import UED_DEFAULTS, load_levels

COLORS = {**COLORS, "ga_learn": "#e87ba4"}   # categorical slot 5 (validated with slots 1-4)
LABELS = {**LABELS, "ga": "GA (regret)", "ga_learn": "GA (learnability)"}
ORDER = ["ga", "ga_learn", "random_search", "random", "fixed"]
SETS = {"heldout_sr": "Held-out random levels", "heldout_hard_sr": "Harder levels (+1 box)",
        "handmade_sr": "Hand-designed levels", "adversarial_sr": "Adversarial levels (other seeds' regret-GA)"}
DIAG = {"new_levels_pre_sr": "New levels: success before training",
        "new_levels_post_sr": "New levels: success after training",
        "new_levels_regret": "New levels: regret",
        "new_levels_opt_moves": "New levels: optimal solution length (moves)",
        "archive_sr": "Own curriculum: final success",
        "solvability_yield": "GA offspring passing A* verification"}


def adversarial(runs_learn, main_ga) -> dict:
    """Score each learnability model on the other seeds' regret-GA archives (as analysis.cross_eval)."""
    from levelforge.ppo_agent import PPOAgent, rollout_batch, success_rate

    arch = {r["seed"]: load_levels(r["dir"] / "archive.json") for r in main_ga}
    out = {}
    for r in runs_learn:
        levels = [l for s, a in arch.items() if s != r["seed"] for l in a]
        cfg = {**UED_DEFAULTS, **json.loads((r["dir"] / "config.json").read_text(encoding="utf-8"))}
        agent = PPOAgent.load(r["dir"] / "model.zip", levels[:4], device=cfg["device"],
                              max_steps=cfg["max_steps"], **cfg["ppo"])
        out[r["seed"]] = success_rate(rollout_batch(agent, levels, trials=4, max_steps=cfg["max_steps"]))
        print(f"[ablation] learnability seed {r['seed']}: adversarial_sr={out[r['seed']]:.3f}", flush=True)
    return out


def table(finals, metric) -> list:
    rows = ["| Condition | Seeds | Success rate | Δ (GA-learnability − this) | 95% CI | p |",
            "|---|---|---|---|---|---|"]
    ref = np.array(list(finals[metric]["ga_learn"].values()))
    for c in ORDER:
        v = np.array(list(finals[metric].get(c, {}).values()))
        if v.size == 0:
            continue
        if c == "ga_learn":
            rows.append(f"| **{LABELS[c]}** | {v.size} | **{v.mean():.3f} ± {v.std():.3f}** | – | – | – |")
            continue
        lo, hi = bootstrap_ci(ref, v)
        rows.append(f"| {LABELS[c]} | {v.size} | {v.mean():.3f} ± {v.std():.3f} | {ref.mean() - v.mean():+.3f} "
                    f"| [{lo:+.3f}, {hi:+.3f}] | {permutation_p(ref, v):.3f} |")
    return rows


def diagnostics(runs) -> list:
    rows = ["| Curriculum statistic (mean over cycles and seeds) | " + " | ".join(LABELS[c] for c in ORDER if c in runs) + " |",
            "|---|" + "---|" * sum(c in runs for c in ORDER)]
    for k, name in DIAG.items():
        cells = []
        for c in ORDER:
            if c not in runs:
                continue
            if k == "archive_sr":
                vals = [r["rows"][-1].get(k) for r in runs[c]]
            else:
                vals = [row[k] for r in runs[c] for row in r["rows"] if row.get(k) is not None]
            vals = [v for v in vals if v is not None]
            cells.append(f"{np.mean(vals):.2f}" if vals else "–")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return rows


def figure(runs, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [c for c in ("ga", "ga_learn", "random") if c in runs]
    panels = [("heldout_sr", "Held-out random levels", "Success rate", True),
              ("new_levels_post_sr", "New curriculum levels, after training", "Success rate", False),
              ("archive_sr", "Own curriculum (all levels so far)", "Success rate", True)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb")
    for ax, (m, title, ylab, by_steps) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for c in shown:
            rows = [[row for row in r["rows"] if row.get(m) is not None] for r in runs[c]]
            n = min(len(x) for x in rows)
            if n == 0:
                continue
            x = (np.array([row["timesteps"] for row in rows[0][:n]]) / 1e6 if by_steps
                 else np.array([row["cycle"] for row in rows[0][:n]]))
            y = np.array([[row[m] for row in rr[:n]] for rr in rows])
            mu, sd = y.mean(0), y.std(0)
            ax.plot(x, mu, color=COLORS[c], linewidth=2, marker="o", markersize=4, label=LABELS[c],
                    markeredgecolor="#fcfcfb", markeredgewidth=1)
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[c], alpha=0.12, linewidth=0)
            ends.append((x[-1], mu[-1]))
        for (xe, ye), yl in zip(ends, _spread([e[1] for e in ends], 0.06)):
            ax.annotate(f"{ye:.2f}", (xe, yl), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=TEXT, va="center", annotation_clip=False)
        _style(ax, title, ylab, "PPO environment steps (millions)" if by_steps else "Cycle")
        ax.set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("Fitness ablation: regret vs learnability (mean ± std over 3 seeds)", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    p = out / "fitness_ablation.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="runs/ued_7x7")
    ap.add_argument("--ablation", default="runs/ued_7x7_learn")
    args = ap.parse_args()
    main_dir, abl_dir = Path(args.main), Path(args.ablation)
    out = abl_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(main_dir)
    learn = load_runs(abl_dir).get("ga", [])
    for r in learn:
        fit = json.loads((r["dir"] / "config.json").read_text(encoding="utf-8")).get("fitness")
        assert fit == "learnability", f"{r['dir']} uses fitness={fit}"
    runs["ga_learn"] = learn

    finals = json.loads((main_dir / "analysis" / "finals.json").read_text(encoding="utf-8"))
    finals = {m: {c: {int(s): v for s, v in d.items()} for c, d in cs.items()} for m, cs in finals.items()}
    for m in list(SETS)[:3]:
        finals[m]["ga_learn"] = {r["seed"]: r["summary"]["final"][m] for r in learn}
    finals["adversarial_sr"]["ga_learn"] = adversarial(learn, runs["ga"])

    md = ["# Fitness ablation: regret vs learnability", "",
          f"Main experiment: `{main_dir}` (GA uses regret). Ablation: `{abl_dir}` (GA uses learnability p(1−p)).",
          "Same pretrained checkpoints, budgets, test sets and seeds; only the GA fitness differs. "
          "Adversarial set = other seeds' *regret*-GA archives, identical to the main table.", ""]
    for m, name in SETS.items():
        md += [f"## {name}", ""] + table(finals, m) + [""]
    md += ["## What each GA put into the curriculum", ""] + diagnostics(runs) + ["",
           "![fitness_ablation](fitness_ablation.png)", ""]
    (out / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    (out / "ablation_finals.json").write_text(json.dumps(finals, indent=2), encoding="utf-8")
    figure(runs, out)
    print(f"[ablation] wrote {out / 'ablation.md'}")


if __name__ == "__main__":
    main()
