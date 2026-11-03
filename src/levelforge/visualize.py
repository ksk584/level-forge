"""Matplotlib-based PNG renderer for levels (no GUI required)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

_BOX_COLORS = {BOX_NORMAL: "#c08457", BOX_HEAVY: "#6b7280", BOX_FRAGILE: "#93c5fd"}
_BOX_LABELS = {BOX_NORMAL: "", BOX_HEAVY: "H", BOX_FRAGILE: "F"}


def render_level(level: Level, path: str, title: str | None = None):
    h, w = level.height, level.width
    fig, axis = plt.subplots(figsize=(w * 0.6 + 1, h * 0.6 + 1))
    axis.set_xlim(0, w)
    axis.set_ylim(h, 0)
    axis.set_aspect("equal")
    axis.axis("off")

    for y in range(h):
        for x in range(w):
            if level.walls[y, x]:
                axis.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#111827", edgecolor="#374151"))
            else:
                axis.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="#f9fafb", edgecolor="#e5e7eb"))

    for gx, gy in level.goals:
        axis.add_patch(mpatches.Circle((gx + 0.5, gy + 0.5), 0.18, facecolor="#16a34a", alpha=0.85))

    for bx, by, kind in level.boxes:
        axis.add_patch(mpatches.FancyBboxPatch(
            (bx + 0.12, by + 0.12), 0.76, 0.76,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=_BOX_COLORS[kind], edgecolor="#1f2937", linewidth=1.5))
        label = _BOX_LABELS[kind]
        if label:
            axis.text(bx + 0.5, by + 0.5, label, ha="center", va="center",
                      color="white", fontsize=10, fontweight="bold")

    agent_x, agent_y = level.agent
    axis.add_patch(mpatches.Circle((agent_x + 0.5, agent_y + 0.5), 0.28,
                                   facecolor="#2563eb", edgecolor="#1e3a8a", linewidth=2))

    axis.set_title(title or f"energy: {level.energy}", fontsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
