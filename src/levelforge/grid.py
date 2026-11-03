"""Core grid model for LevelForge: levels, encodings and ASCII round-trips.

A Level is a self-contained description of one puzzle instance:
  * walls   -- (H, W) boolean array, True = wall
  * goals   -- list of (x, y) cells that must each end up covered by a box
  * boxes   -- list of (x, y, kind); kind: 0 normal, 1 heavy, 2 fragile
  * agent   -- (x, y) start cell
  * energy  -- push-energy budget (heavy pushes cost 3, others cost 1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

BOX_NORMAL = 0
BOX_HEAVY = 1
BOX_FRAGILE = 2
BOX_KINDS = (BOX_NORMAL, BOX_HEAVY, BOX_FRAGILE)
BOX_KIND_NAMES = {BOX_NORMAL: "normal", BOX_HEAVY: "heavy", BOX_FRAGILE: "fragile"}
PUSH_COST = {BOX_NORMAL: 1, BOX_HEAVY: 3, BOX_FRAGILE: 1}

# Observation channels (one-hot) + 1 scalar for remaining energy.
N_CHANNELS = 6  # wall, goal, box-normal, box-heavy, box-fragile, agent


def _freeze_boxes(boxes: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return sorted((int(x), int(y), int(k)) for x, y, k in boxes)


@dataclass
class Level:
    walls: np.ndarray                     # (H, W) bool
    goals: List[Tuple[int, int]]
    boxes: List[Tuple[int, int, int]]     # (x, y, kind)
    agent: Tuple[int, int]
    energy: int
    height: int = field(init=False)
    width: int = field(init=False)

    def __post_init__(self):
        self.walls = np.asarray(self.walls, dtype=bool)
        if self.walls.ndim != 2:
            raise ValueError("walls must be a 2D array")
        self.height, self.width = self.walls.shape
        self.goals = [(int(x), int(y)) for x, y in self.goals]
        self.boxes = _freeze_boxes(self.boxes)
        self.agent = (int(self.agent[0]), int(self.agent[1]))
        self.energy = int(self.energy)

    # ------------------------------------------------------------------ #
    def copy(self) -> "Level":
        return Level(self.walls.copy(), list(self.goals), list(self.boxes), tuple(self.agent), self.energy)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return not self.in_bounds(x, y) or bool(self.walls[y, x])

    def box_positions(self) -> List[Tuple[int, int]]:
        return [(b[0], b[1]) for b in self.boxes]

    def box_at(self, x: int, y: int):
        for i, b in enumerate(self.boxes):
            if b[0] == x and b[1] == y:
                return i, b
        return None, None

    def goal_set(self) -> set:
        return set(self.goals)

    def box_index_at(self, x: int, y: int) -> int:
        for i, b in enumerate(self.boxes):
            if b[0] == x and b[1] == y:
                return i
        return -1

    def boxes_on_goals(self) -> int:
        goals = self.goal_set()
        return sum(1 for bx, by, _ in self.boxes if (bx, by) in goals)

    def validate(self) -> List[str]:
        """Return a list of structural problems (empty list == valid)."""
        problems = []
        h, w = self.height, self.width
        if not self.walls[0].all() or not self.walls[-1].all() or not self.walls[:, 0].all() or not self.walls[:, -1].all():
            problems.append("outer border must be all walls")
        for x, y in self.goals:
            if self.is_wall(x, y):
                problems.append(f"goal at ({x},{y}) sits on a wall")
        for i, (x, y, kind) in enumerate(self.boxes):
            if self.is_wall(x, y):
                problems.append(f"box {i} at ({x},{y}) sits on a wall")
            if kind not in BOX_KINDS:
                problems.append(f"box {i} has unknown kind {kind}")
        positions = self.box_positions()
        if len(set(positions)) != len(positions):
            problems.append("two boxes occupy the same cell")
        if len(positions) != len(self.goals):
            problems.append(f"boxes ({len(positions)}) != goals ({len(self.goals)})")
        if self.is_wall(*self.agent):
            problems.append("agent sits on a wall")
        if self.agent in positions:
            problems.append("agent sits on a box")
        for x, y in self.goals:
            if not self.in_bounds(x, y):
                problems.append(f"goal at ({x},{y}) out of bounds")
        return problems

    # ------------------------------------------------------------------ #
    # ASCII serialisation (round-trippable)
    # ------------------------------------------------------------------ #
    _CHAR = {
        (BOX_NORMAL, False): "n", (BOX_NORMAL, True): "N",
        (BOX_HEAVY, False): "h",  (BOX_HEAVY, True): "H",
        (BOX_FRAGILE, False): "f", (BOX_FRAGILE, True): "F",
    }
    _FROM_CHAR = {
        "n": (BOX_NORMAL, False), "N": (BOX_NORMAL, True),
        "h": (BOX_HEAVY, False),  "H": (BOX_HEAVY, True),
        "f": (BOX_FRAGILE, False), "F": (BOX_FRAGILE, True),
    }

    def to_ascii(self) -> str:
        goals = self.goal_set()
        box_map = {(x, y): kind for x, y, kind in self.boxes}
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.walls[y, x]:
                    row.append("#")
                    continue
                if (x, y) == self.agent:
                    row.append("+" if (x, y) in goals else "@")
                    continue
                if (x, y) in box_map:
                    row.append(self._CHAR[(box_map[(x, y)], (x, y) in goals)])
                    continue
                row.append("." if (x, y) in goals else " ")
            rows.append("".join(row).rstrip() or " ")
        return "\n".join(rows)

    @classmethod
    def from_ascii(cls, text: str, energy: int = 16) -> "Level":
        lines = text.strip("\n").split("\n")
        h = len(lines)
        w = max(len(l) for l in lines)
        walls = np.zeros((h, w), dtype=bool)
        goals, boxes = [], []
        agent = None
        for y, raw in enumerate(lines):
            for x, ch in enumerate(raw):
                if ch == "#":
                    walls[y, x] = True
                elif ch == ".":
                    goals.append((x, y))
                elif ch in ("@", "+"):
                    agent = (x, y)
                    if ch == "+":
                        goals.append((x, y))
                elif ch in cls._FROM_CHAR:
                    kind, on_goal = cls._FROM_CHAR[ch]
                    boxes.append((x, y, kind))
                    if on_goal:
                        goals.append((x, y))
        if agent is None:
            raise ValueError("ASCII level has no agent ('@')")
        return cls(walls, goals, boxes, agent, energy)

    # ------------------------------------------------------------------ #
    # JSON serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "walls": ["".join("#" if v else " " for v in row) for row in self.walls],
            "goals": [list(g) for g in self.goals],
            "boxes": [list(b) for b in self.boxes],
            "agent": list(self.agent),
            "energy": self.energy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Level":
        lines = data["walls"]
        walls = np.array([[c == "#" for c in row] for row in lines], dtype=bool)
        return cls(
            walls,
            [tuple(g) for g in data["goals"]],
            [tuple(b) for b in data["boxes"]],
            tuple(data["agent"]),
            data["energy"],
        )

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path) -> "Level":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ #
    # Observation encoding
    # ------------------------------------------------------------------ #
    def encode(self) -> np.ndarray:
        """Return (N_CHANNELS, H, W) float32 one-hot observation."""
        obs = np.zeros((N_CHANNELS, self.height, self.width), dtype=np.float32)
        obs[0] = self.walls.astype(np.float32)
        for (x, y) in self.goals:
            obs[1, y, x] = 1.0
        for x, y, kind in self.boxes:
            obs[2 + kind, y, x] = 1.0
        ax, ay = self.agent
        obs[5, ay, ax] = 1.0
        return obs

    def signature(self) -> np.ndarray:
        """Fixed-length 0/1 vector used for population-diversity Hamming distance."""
        sg = np.zeros(self.height * self.width * 3, dtype=np.uint8)
        sg[0:self.height * self.width] = self.walls.astype(np.uint8).ravel()
        for (x, y) in self.goals:
            sg[self.height * self.width + y * self.width + x] = 1
        for x, y, _ in self.boxes:
            sg[2 * self.height * self.width + y * self.width + x] = 1
        return sg
