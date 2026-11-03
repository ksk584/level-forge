"""Potential-based reward shaping for Sokoban.

Potential = sum over boxes of the Manhattan distance from the box to its
nearest goal (lower is better).  shaped_reward = raw_reward + F * (phi(s) - phi(s')),
so moving a box closer to a goal grants F, further away costs F.  Potential-based
shaping preserves the optimal policy (Ng, Harada, Russell 1999).
"""

from __future__ import annotations

from .grid import Level


def box_goal_potential(level: Level) -> float:
    goals = level.goals
    if not goals:
        return 0.0
    total = 0
    for bx, by, _ in level.boxes:
        total += min(abs(bx - gx) + abs(by - gy) for gx, gy in goals)
    return float(total)


def shaped_reward(raw_reward: float, level_before: Level, level_after: Level, weight: float = 0.5) -> float:
    return raw_reward + weight * (box_goal_potential(level_before) - box_goal_potential(level_after))
