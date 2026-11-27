"""Loader for hand-designed level files (ASCII, ``;`` comments, blank-line separated)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .grid import Level

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "levels" / "handmade_7x7.txt"


def pad_level(level: Level, size: int) -> Level:
    """Embed ``level`` in the top-left of a ``size`` x ``size`` all-wall board."""
    h, w = level.walls.shape
    if h > size or w > size:
        raise ValueError(f"level {h}x{w} does not fit a {size}x{size} board")
    walls = np.ones((size, size), dtype=bool)
    walls[:h, :w] = level.walls
    return Level(walls, level.goals, level.boxes, level.agent, level.energy, level.rough)


def load_handmade(path=DEFAULT_FILE, size: int | None = None) -> List[Tuple[str, Level]]:
    """Return ``[(name, level)]``.  Energy is set by the caller (``verify_level``)."""
    text = Path(path).read_text(encoding="utf-8")
    out: List[Tuple[str, Level]] = []
    name, rows = None, []

    def flush():
        nonlocal name, rows
        if rows:
            lvl = Level.from_ascii("\n".join(rows), energy=60)
            if size is not None:
                lvl = pad_level(lvl, size)
            out.append((name or f"level_{len(out) + 1:02d}", lvl))
        name, rows = None, []

    for line in text.splitlines():
        if line.startswith(";"):
            label = line[1:].strip()
            if label and label[:2].isdigit():
                flush()
                name = label.split(":")[0]
            continue
        if not line.strip():
            flush()
            continue
        rows.append(line)
    flush()
    return out
