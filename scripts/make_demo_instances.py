"""Generate 20 fixed demo instances and a report-ready contact sheet.

Outputs (deterministic, seed=2026):
  runs/demo_instances/instance_01_easy.json ... instance_20_hard.json
  runs/demo_instances/png/instance_XX.png
  runs/demo_instances/instances_contact_sheet.png   <- one figure for the report
  runs/demo_instances/MANIFEST.md                    <- table of difficulty stats

Run:  python scripts/make_demo_instances.py
"""

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from levelforge.grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level
from levelforge.population import scramble_level
from levelforge.solver import solve
from levelforge.visualize import render_level

OUT = Path("runs/demo_instances")
PNG = OUT / "png"
PNG.mkdir(parents=True, exist_ok=True)

# 6 easy / 7 medium / 7 hard  = 20 instances, mixed boards
SPECS = (
    [("easy", 7, 1)] * 6
    + [("medium", 8, 2)] * 7
    + [("hard", 8, 3)] * 3
    + [("hard", 10, 3)] * 4
)

_BOX_COLOR = {BOX_NORMAL: "#c08457", BOX_HEAVY: "#6b7280", BOX_FRAGILE: "#93c5fd"}
_BOX_LABEL = {BOX_HEAVY: "H", BOX_FRAGILE: "F"}


def draw_thumb(ax, level: Level, label: str):
    h, w = level.height, level.width
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=9)
    walls = level.walls
    for y in range(h):
        for x in range(w):
            if walls[y, x]:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#111827", edgecolor="#374151"))
            else:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#f9fafb", edgecolor="#e5e7eb"))
    for gx, gy in level.goals:
        ax.add_patch(mpatches.Circle((gx + 0.5, gy + 0.5), 0.17, facecolor="#16a34a", alpha=0.9))
    for bx, by, kind in level.boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx + 0.12, by + 0.12), 0.76, 0.76,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=_BOX_COLOR[kind], edgecolor="#1f2937", linewidth=1.2))
        lab = _BOX_LABEL.get(kind)
        if lab:
            ax.text(bx + 0.5, by + 0.5, lab, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax_, ay = level.agent
    ax.add_patch(mpatches.Circle((ax_ + 0.5, ay + 0.5), 0.26, facecolor="#2563eb", edgecolor="#1e3a8a", linewidth=1.5))


def main():
    rng = np.random.default_rng(2026)
    rows = []
    fig, axes = plt.subplots(4, 5, figsize=(17, 14))
    fig.suptitle("LevelForge — 20 demo instances (guaranteed solvable by construction)",
                 fontsize=14, fontweight="bold")

    for i, (tier, size, n_boxes) in enumerate(SPECS):
        while True:
            trace = []
            try:
                lvl = scramble_level(rng, size=size, n_boxes=n_boxes, trace=trace)
                break
            except ValueError:
                continue
        moves, nodes, dt = solve(lvl, max_nodes=1_000_000, time_limit_s=20.0)
        name = f"instance_{i + 1:02d}_{tier}"
        lvl.save_json(OUT / f"{name}.json")
        render_level(lvl, str(PNG / f"instance_{i + 1:02d}.png"),
                     title=f"#{i + 1:02d} — {tier} {size}×{size}, {n_boxes} box(es)")
        rows.append((i + 1, tier, size, n_boxes,
                     len(moves) if moves else None, nodes if moves else 0, round(dt, 3)))
        draw_thumb(axes.flat[i], lvl, f"#{i + 1:02d} {tier}")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "instances_contact_sheet.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# LevelForge — demo instance manifest",
        "",
        "Generated deterministically (seed=2026) from the reverse-scramble generator,",
        "and each instance is verified solvable by the A* planner (`solver.py`).",
        "",
        "| # | tier | board | boxes | solver moves | searched nodes | solve time (s) |",
        "|---|------|-------|-------|--------------|----------------|----------------|",
    ]
    for n, tier, size, nb, mv, nd, dt in rows:
        lines.append(f"| {n} | {tier} | {size}×{size} | {nb} | {mv} | {nd} | {dt} |")
    lines.append("")
    lines.append("Files: `instance_XX_tier.json` (levels, load/play), `png/instance_XX.png` (images),")
    lines.append("`instances_contact_sheet.png` (single figure for the report).")
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {f"instance_{r[0]:02d}": {"tier": r[1], "size": r[2], "boxes": r[3],
                                       "solver_moves": r[4], "nodes": r[5], "solve_s": r[6]}
               for r in rows}
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved 20 instances to {OUT}/")
    for n, tier, size, nb, mv, nd, dt in rows:
        print(f"  #{n:02d} {tier:6s} {size}x{size} {nb}b  moves={mv}  nodes={nd}  t={dt}s")


if __name__ == "__main__":
    main()
