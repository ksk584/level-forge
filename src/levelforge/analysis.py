"""Aggregate UED runs across seeds: tables, significance tests, figures.

Layout read: ``<exp_dir>/<condition>_seed<s>/{metrics.jsonl,summary.json,model.zip,archive.json}``
and optionally ``<exp_dir>/attack/attack.json``.

Statistics.  With few seeds per condition, parametric tests are fragile, so we
report mean +/- std across seeds plus
  * an exact two-sided permutation test on the per-seed final values, and
  * a bootstrap 95% CI for the difference in means (resampling seeds).
With 3 seeds per arm the smallest attainable permutation p-value is 0.10; use
>= 5 seeds (min p ~ 0.008) for a claim of significance.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .ued import CONDITIONS, UED_DEFAULTS, load_levels

# Reference categorical palette, fixed order: colour follows the condition.
COLORS = {"ga": "#2a78d6", "random_search": "#eb6834", "random": "#1baf7a", "fixed": "#eda100"}
LABELS = {"ga": "GA adversary", "random_search": "Random search + selection",
          "random": "Random levels", "fixed": "Fixed level set"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {"heldout_sr": "Held-out random levels", "heldout_hard_sr": "Harder levels (+1 box)",
           "handmade_sr": "Hand-designed levels", "adversarial_sr": "Adversarial levels (other seeds' GA)"}


def load_runs(exp_dir) -> Dict[str, List[Dict]]:
    runs: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    for d in sorted(Path(exp_dir).glob("*_seed*")):
        cond, _, seed = d.name.rpartition("_seed")
        if cond not in runs or not (d / "metrics.jsonl").exists():
            continue
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else None
        runs[cond].append({"seed": int(seed), "dir": d, "rows": rows, "summary": summ})
    return {c: r for c, r in runs.items() if r}


# ---------------------------------------------------------------------- #
# Statistics
# ---------------------------------------------------------------------- #
def permutation_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12
        total += 1
    return hits / total


def bootstrap_ci(a, b, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------------- #
# Cross-evaluation on adversarial levels
# ---------------------------------------------------------------------- #
def cross_eval(exp_dir, runs: Dict[str, List[Dict]]) -> Dict[str, Dict[int, float]]:
    """Evaluate every final model on GA-discovered levels from *other* seeds.

    The GA's archive for seed s was used to train the GA agent of seed s, so
    each seed is scored on the union of the remaining seeds' archives -- the
    same set for every condition of that seed.
    """
    from .ppo_agent import PPOAgent, rollout_batch, success_rate

    ga_arch = {r["seed"]: load_levels(r["dir"] / "archive.json")
               for r in runs.get("ga", []) if (r["dir"] / "archive.json").exists()}
    out: Dict[str, Dict[int, float]] = {}
    for cond, rs in runs.items():
        for r in rs:
            if not (r["dir"] / "model.zip").exists():
                continue
            levels = [l for s, arch in ga_arch.items() if s != r["seed"] for l in arch]
            if not levels:
                continue
            cfg = {**UED_DEFAULTS, **json.loads((r["dir"] / "config.json").read_text(encoding="utf-8"))}
            agent = PPOAgent.load(r["dir"] / "model.zip", levels[:4], device=cfg["device"],
                                  max_steps=cfg["max_steps"], **cfg["ppo"])
            out.setdefault(cond, {})[r["seed"]] = success_rate(
                rollout_batch(agent, levels, trials=4, max_steps=cfg["max_steps"]))
    return out


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #
def _style(ax, title, ylabel, xlabel="PPO environment steps (millions)"):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _spread(values, gap: float):
    """Nudge label y-positions apart (keeps order) so end labels never overlap."""
    order = np.argsort(values)
    ys = np.asarray(values, float)[order].copy()
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    out = np.empty_like(ys)
    out[order] = ys
    return out


def plot_curves(runs, out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    metrics = [m for m in METRICS if m != "adversarial_sr"] + ["archive_sr"]
    titles = {**METRICS, "archive_sr": "Own curriculum levels (forgetting check)"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="#fcfcfb")
    for ax, m in zip(axes.ravel(), metrics):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for cond, rs in runs.items():
            rows = [[row for row in r["rows"] if m in row] for r in rs]
            n = min(len(x) for x in rows)
            if n == 0:
                continue
            x = np.array([row["timesteps"] for row in rows[0][:n]]) / 1e6
            y = np.array([[row[m] for row in rr[:n]] for rr in rows])
            mu, sd = y.mean(0), y.std(0)
            ax.plot(x, mu, color=COLORS[cond], linewidth=2, label=LABELS[cond],
                    marker="o", markersize=4, markeredgecolor="#fcfcfb", markeredgewidth=1)
            if len(rs) > 1:
                ax.fill_between(x, mu - sd, mu + sd, color=COLORS[cond], alpha=0.12, linewidth=0)
            ends.append((x[-1], mu[-1]))
        if ends:
            for (xe, ye), yl in zip(ends, _spread([e[1] for e in ends], 0.05)):
                ax.annotate(f"{ye:.2f}", (xe, yl), xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=TEXT, va="center", annotation_clip=False)
        _style(ax, titles[m], "Success rate")
        ax.set_ylim(0, 1)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("Agent success rate during training (mean ± std over seeds)", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    p = out_dir / "success_curves.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # Search diagnostics: regret of the levels each condition adds per cycle.
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ends = []
    for cond, rs in runs.items():
        y = np.array([[row["new_levels_regret"] for row in r["rows"] if "new_levels_regret" in row] for r in rs])
        if y.size == 0:
            continue
        x = np.arange(y.shape[1])
        ax.plot(x, y.mean(0), color=COLORS[cond], linewidth=2, marker="o", markersize=4, label=LABELS[cond],
                markeredgecolor="#fcfcfb")
        ends.append((cond, x[-1], y.mean(0)[-1]))
    # Direct end labels (no legend), de-collided: GA and random search both sit at ~1.0.
    for (cond, xe, ye), yl in zip(ends, _spread([e[2] for e in ends], 0.06)):
        ax.annotate(f"{LABELS[cond]} ({ye:.2f})", (xe, yl), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, color=TEXT, va="center", annotation_clip=False)
    _style(ax, "Regret of the levels added to the curriculum", "Mean regret (1 = agent never solves)", "Cycle")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = out_dir / "curriculum_regret.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_attack(attack_json: Path, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(attack_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for method in ("ga", "random_search"):
        curves = [r["curve"] for r in data["runs"] if r["method"] == method]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        x = np.array([p["evaluated"] for p in curves[0][:n]])
        y = np.array([[p["breaking_found"] for p in c[:n]] for c in curves])
        ax.plot(x, y.mean(0), color=COLORS[method], linewidth=2, label=LABELS[method])
        if len(curves) > 1:
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=COLORS[method], alpha=0.12, lw=0)
        ax.annotate(f"{y.mean(0)[-1]:.0f}", (x[-1], y.mean(0)[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=TEXT, va="center")
    _style(ax, "Agent-breaking levels found against a frozen agent",
           "Distinct verified levels the agent always fails", "Candidate levels evaluated")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = out_dir / "attack_efficiency.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------- #
# Report
# ---------------------------------------------------------------------- #
def attack_summary(attack_json: Path, targets=(10, 25, 50)) -> List[str]:
    data = json.loads(attack_json.read_text(encoding="utf-8"))
    lines = ["| Method | Breaking levels found (mean ± std) | " +
             " | ".join(f"Evaluations to find {t}" for t in targets) + " |",
             "|---|---|" + "---|" * len(targets)]
    for method in ("ga", "random_search"):
        rs = [r for r in data["runs"] if r["method"] == method]
        if not rs:
            continue
        found = [r["curve"][-1]["breaking_found"] for r in rs]
        cells = []
        for t in targets:
            hit = [next((p["evaluated"] for p in r["curve"] if p["breaking_found"] >= t), None) for r in rs]
            ok = [h for h in hit if h is not None]
            cells.append(f"{np.mean(ok):.0f} ({len(ok)}/{len(rs)} seeds)" if ok else "not reached")
        lines.append(f"| {LABELS[method]} | {np.mean(found):.1f} ± {np.std(found):.1f} | " + " | ".join(cells) + " |")
    return lines


def analyze(exp_dir, do_cross_eval: bool = True) -> Path:
    exp_dir = Path(exp_dir)
    out = exp_dir / "analysis"
    out.mkdir(exist_ok=True)
    runs = load_runs(exp_dir)
    finals: Dict[str, Dict[str, Dict[int, float]]] = {}
    for cond, rs in runs.items():
        for r in rs:
            if r["summary"]:
                for m, v in r["summary"]["final"].items():
                    finals.setdefault(m, {}).setdefault(cond, {})[r["seed"]] = v
    if do_cross_eval and "ga" in runs:
        finals["adversarial_sr"] = cross_eval(exp_dir, runs)

    lines = ["# Results", "", f"Experiment directory: `{exp_dir}`", "",
             "Final success rates (stochastic policy), mean ± std across seeds; "
             "Δ vs GA with bootstrap 95% CI and exact permutation p-value.", ""]
    for m, title in METRICS.items():
        if m not in finals:
            continue
        lines += [f"## {title}", "", "| Condition | Seeds | Success rate | Δ (GA − this) | 95% CI | p |",
                  "|---|---|---|---|---|---|"]
        ga = list(finals[m].get("ga", {}).values())
        for cond in CONDITIONS:
            vals = list(finals[m].get(cond, {}).values())
            if not vals:
                continue
            row = f"| {LABELS[cond]} | {len(vals)} | {np.mean(vals):.3f} ± {np.std(vals):.3f} |"
            if cond != "ga" and ga and len(vals) > 1 and len(ga) > 1:
                lo, hi = bootstrap_ci(ga, vals)
                row += f" {np.mean(ga) - np.mean(vals):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {permutation_p(ga, vals):.3f} |"
            else:
                row += " – | – | – |"
            lines.append(row)
        lines.append("")

    figs = plot_curves(runs, out) if runs else []
    attack_json = exp_dir / "attack" / "attack.json"
    if attack_json.exists():
        figs.append(plot_attack(attack_json, out))
        lines += ["## Failure-discovery efficiency (frozen agent)", ""] + attack_summary(attack_json) + [""]
    lines += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "finals.json", "w", encoding="utf-8") as f:
        json.dump(finals, f, indent=2)
    text = "\n".join(lines)
    try:
        print(text)
    except UnicodeEncodeError:  # e.g. Windows cp1252 console
        print(text.encode("ascii", "replace").decode("ascii"))
    print(f"[analyze] wrote {out / 'results.md'}")
    return out
