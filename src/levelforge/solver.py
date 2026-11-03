"""A* / BFS planner for Sokoban — the dependable 'player' for demos and eval.

The state is (agent, frozenset(box positions)); box *kinds* are ignored for
legality except that a fragile box pushed into a crush cell is destroyed
(invalid), mirroring the environment.  Goal: every goal cell covered.

Heuristic: sum over boxes of the Manhattan distance to the nearest goal
(admissible lower bound on remaining pushes for 1-occupancy coverage).
"""

from __future__ import annotations

import heapq
import time
from typing import List, Optional, Tuple

from .environment import SokobanEnv
from .grid import Level
from .solvability import crush_cells

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _heur(box_list, goals):
    total = 0
    for bx, by in box_list:
        total += min(abs(bx - gx) + abs(by - gy) for gx, gy in goals)
    return total


def solve(level: Level, max_nodes: int = 200_000, time_limit_s: float = 20.0):
    """Return (moves, nodes, elapsed_s).  moves is a list of action indices, or None."""
    t0 = time.time()
    walls = level.walls
    goals = list(level.goals)
    initial_boxes = tuple(sorted((b[0], b[1]) for b in level.boxes))
    initial = (level.agent, initial_boxes)
    crush = crush_cells(level)

    def goal_test(boxes):
        return set(goals) <= set(boxes)  # crush pruning already removed dead states

    # Note: boxes on goals are fine; crush already excludes goal cells.
    def key(state):
        return (state[0], state[1])

    h = _heur(list(initial_boxes), goals)
    counter = 0
    heap = [(h, 0, counter, initial)]  # (priority, g, tie, state)
    best_g = {initial: 0}
    parent = {initial: None}  # state -> (prev_state, action)
    nodes = 0

    while heap:
        f, g, _c, state = heapq.heappop(heap)
        agent, boxes = state
        nodes += 1
        if nodes > max_nodes or (nodes & 0xFFF) == 0 and time.time() - t0 > time_limit_s:
            return None, nodes, time.time() - t0
        if goal_test(boxes):
            # Reconstruct path.
            path = []
            cur = state
            while parent[cur] is not None:
                pc, act = parent[cur]
                path.append(act)
                cur = pc
            path.reverse()
            return path, nodes, time.time() - t0
        box_set = set(boxes)
        for ai, (dx, dy) in enumerate(_DIRS):
            ax, ay = agent
            nx, ny = ax + dx, ay + dy
            # Wall?
            if not level.in_bounds(nx, ny) or walls[ny, nx]:
                continue
            if (nx, ny) in box_set:
                # Push: target cell must be free.
                tx, ty = nx + dx, ny + dy
                if not level.in_bounds(tx, ty) or walls[ty, tx] or (tx, ty) in box_set:
                    continue
                rest = box_set - {(nx, ny)}
                new_boxes = tuple(sorted(rest | {(tx, ty)}))
                # Pruning: any non-goal box stuck in crush -> dead.
                if any(b in crush for b in new_boxes):
                    continue
                # Fragile box shattered?  For goal-only solve we forbid crush cells entirely,
                # so a fragile box can never be "both legally placed and broken" here.
                new_state = ((nx, ny), new_boxes)
            else:
                new_state = ((nx, ny), boxes)
            ng = g + 1
            if new_state in best_g and best_g[new_state] <= ng:
                continue
            best_g[new_state] = ng
            parent[new_state] = (state, ai)
            counter += 1
            nf = ng + _heur(list(new_state[1]), goals)
            heapq.heappush(heap, (nf, ng, counter, new_state))
    return None, nodes, time.time() - t0
