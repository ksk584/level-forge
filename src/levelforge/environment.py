"""Sokoban environment with energy budget, heavy and fragile boxes.

Rules
-----
* Move (walk)            : free; wall / blocked box => no-op, small penalty.
* Push normal box (n)    : 1 energy; box must land on a free cell.
* Push heavy box (h)     : 3 energy; cannot be pushed with insufficient energy.
* Push fragile box (f)   : 1 energy; if the box lands in a crush cell
  (corner or wall-line without a goal) it shatters and the episode fails.
* Energy reaches 0 before the puzzle is solved  => failure ``energy_exhaustion``.
* A non-goal box in a crush cell                  => failure ``deadlock``.
* All boxes on goals                               => success (+100).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .grid import BOX_FRAGILE, PUSH_COST, Level
from .solvability import box_is_deadlocked, crush_cells

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["up", "down", "left", "right"]
_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

FAILURE_DEADLOCK = "deadlock"
FAILURE_FRAGILE = "fragile_break"
FAILURE_ENERGY = "energy_exhaustion"
FAILURE_INVALID_PUSH = "invalid_heavy_push"
FAILURE_TIMEOUT = "timeout"
FAILURE_TYPES = [FAILURE_DEADLOCK, FAILURE_FRAGILE, FAILURE_ENERGY, FAILURE_INVALID_PUSH, FAILURE_TIMEOUT]


class SokobanEnv:
    def __init__(self, level: Level, max_steps: int = 300):
        self.initial_level = level.copy()
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, level: Optional[Level] = None):
        if level is not None:
            self.initial_level = level.copy()
        self.level = self.initial_level.copy()
        self.steps = 0
        self.done = False
        self.failure: Optional[str] = None
        self.box_total = len(self.level.boxes)
        self.placed_reward_given = self.level.boxes_on_goals()
        return self._obs(), self._info()

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        return self.level.encode()  # (C, H, W) float32

    def _info(self) -> dict:
        return {
            "failure": self.failure,
            "steps": self.steps,
            "boxes_on_goals": self.level.boxes_on_goals(),
            "energy": self.level.energy,
            "solved": self.failure is None and self.done and self._is_complete(),
        }

    def _is_complete(self) -> bool:
        # A broken fragile box can never reach a goal, so completion must
        # compare against the *original* box count, not the live count.
        return self.box_total > 0 and self.level.boxes_on_goals() == self.box_total

    def _box_occupied(self, x: int, y: int, ignore_index: int = -1) -> bool:
        for i, (bx, by, _) in enumerate(self.level.boxes):
            if i != ignore_index and bx == x and by == y:
                return True
        return False

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        if self.done:
            return self._obs(), 0.0, True, self._info()

        self.steps += 1
        reward = -0.1
        dx, dy = _DELTAS[int(action)]
        ax, ay = self.level.agent
        nx, ny = ax + dx, ay + dy
        crush = crush_cells(self.level)

        if self.level.is_wall(nx, ny):
            return self._after(reward, failure=None)

        box_idx, box = self.level.box_at(nx, ny)
        if box is not None:
            bx, by, kind = box
            tx, ty = nx + dx, ny + dy
            cost = PUSH_COST[kind]
            target_free = (
                self.level.in_bounds(tx, ty)
                and not self.level.walls[ty, tx]
                and not self._box_occupied(tx, ty)
            )
            # Heavy boxes additionally require enough remaining energy.
            if not target_free or self.level.energy < cost:
                reward -= 5.0
                # Invalid heavy-box push: costly no-op, counted via info['invalid_push']
                return self._after(reward, failure=None, invalid_push=True)
            self.level.energy -= cost
            boxes = list(self.level.boxes)
            boxes[box_idx] = (tx, ty, kind)
            self.level.boxes = sorted(boxes)
            self.level.agent = (nx, ny)
            # Reward for placing (newly) a box on a goal.
            placed = self.level.boxes_on_goals()
            if placed > self.placed_reward_given:
                reward += 10.0 * (placed - self.placed_reward_given)
                self.placed_reward_given = placed
            elif placed < self.placed_reward_given:
                self.placed_reward_given = placed
            # Fragile box landing in a crush cell shatters.
            if kind == BOX_FRAGILE and (tx, ty) not in self.level.goal_set() and (tx, ty) in crush:
                self.level.boxes = [b for b in self.level.boxes if not (b[0] == tx and b[1] == ty and b[2] == BOX_FRAGILE)]
                return self._after(reward - 15.0, failure=FAILURE_FRAGILE)
            # Generic deadlock check (any non-goal box irreversibly stuck).
            for bx2, by2, _k in self.level.boxes:
                if box_is_deadlocked(self.level, (bx2, by2), crush):
                    return self._after(reward - 10.0, failure=FAILURE_DEADLOCK)
            if self._is_complete():
                return self._after(reward + 100.0, failure=None, solved=True)
            if self.level.energy <= 0:
                return self._after(reward - 10.0, failure=FAILURE_ENERGY)
            return self._after(reward, failure=None)

        # Plain move.
        self.level.agent = (nx, ny)
        if self._is_complete():
            return self._after(reward + 100.0, failure=None, solved=True)
        if self.level.energy <= 0:
            return self._after(reward - 10.0, failure=FAILURE_ENERGY)
        return self._after(reward, failure=None)

    # ------------------------------------------------------------------ #
    def _after(self, reward, failure, solved=False, invalid_push=False):
        if failure is not None:
            self.done = True
            self.failure = failure
        elif solved:
            self.done = True
            self.failure = None
        elif self.steps >= self.max_steps:
            self.done = True
            self.failure = FAILURE_TIMEOUT
        info = self._info()
        info["invalid_push"] = invalid_push
        # reward shaping token used by training curves
        return self._obs(), float(reward), self.done, info

    # ------------------------------------------------------------------ #
    def render_ascii(self) -> str:
        lines = self.level.to_ascii().split("\n")
        header = f"Level | energy={self.level.energy} | step={self.steps} | boxes on goals={self.level.boxes_on_goals()}/{len(self.level.boxes)}"
        return header + "\n" + "\n".join(lines)
