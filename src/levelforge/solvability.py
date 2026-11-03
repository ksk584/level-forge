"""Deadlock detection and bounded reverse-BFS solvability checking.

Sokoban feasibility is PSPACE-complete in general, so this module uses a
cheap but effective combination:

  1. Local crush rules (corner / wall-line deadlock) to reject hopeless boxes.
  2. A node-capped reverse BFS from the solved configuration (boxes on goals)
     towards the initial configuration.  In reverse we *pull* boxes.  Node
     states that contain a box stuck in a non-goal crush cell are pruned.
  3. Static free-cell checks for pulls.  Agent reachability inside the free
     space is not modelled (a deliberate, documented approximation) and the
     search is capped by ``max_nodes`` so the checker always terminates.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Set, Tuple

from .grid import Level


def crush_cells(level: Level) -> Set[Tuple[int, int]]:
    """Cells where a non-goal box can never be pushed to safety again."""
    h, w = level.height, level.width
    walls = level.walls
    goals = level.goal_set()
    crush: Set[Tuple[int, int]] = set()

    def is_wall(x, y):
        return x < 0 or y < 0 or x >= w or y >= h or bool(walls[y, x])

    def has_goal_along_line(x, y, dx, dy):
        """Scan one direction until a wall; True if a goal lies in the unbroken run."""
        xx, yy = x + dx, y + dy
        while level.in_bounds(xx, yy) and not walls[yy, xx]:
            if (xx, yy) in goals:
                return True
            xx += dx
            yy += dy
        return False

    for y in range(h):
        for x in range(w):
            walls_row = walls[y, x]
            if walls_row or (x, y) in goals:
                continue
            left, right = is_wall(x - 1, y), is_wall(x + 1, y)
            up, down = is_wall(x, y - 1), is_wall(x, y + 1)
            # Corner deadlock: two perpendicular sides blocked.
            if (left or right) and (up or down):
                crush.add((x, y))
                continue
            # Wall-line deadlock: run of walls (or unrescuable segment) with no goal.
            if left and right and not (has_goal_along_line(x, y, -1, 0) or has_goal_along_line(x, y, 1, 0)):
                crush.add((x, y))
            elif up and down and not (has_goal_along_line(x, y, 0, -1) or has_goal_along_line(x, y, 0, 1)):
                crush.add((x, y))
    return crush


def box_is_deadlocked(level: Level, pos: Tuple[int, int], crush: Set[Tuple[int, int]] | None = None) -> bool:
    goals = level.goal_set()
    if pos in goals:
        return False
    if crush is None:
        crush = crush_cells(level)
    return pos in crush


def _free(level: Level, x: int, y: int, box_set: Set[Tuple[int, int]]) -> bool:
    return level.in_bounds(x, y) and not level.walls[y, x] and (x, y) not in box_set


def is_solvable(level: Level, max_nodes: int = 30000) -> bool:
    """Return True if the level appears solvable (bounded reverse BFS)."""
    if not level.boxes:
        return True
    if level.validate():
        return False

    crush = crush_cells(level)
    goals = level.goal_set()
    initial = tuple(sorted(level.box_positions()))
    start = tuple(sorted(goals))
    if initial == start:
        return True
    # Any box already stuck in a non-goal crush cell => hopeless.
    for bx, by, _ in level.boxes:
        if (bx, by) in crush:
            return False

    visited: Set[Tuple[Tuple[int, int], ...]] = {start}
    queue: deque[Tuple[Tuple[int, int], ...]] = deque([start])
    nodes = 0
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        state = queue.popleft()
        nodes += 1
        if nodes > max_nodes:
            return False  # inconclusive -> treat as unsolvable (caller may raise cap)
        box_set = set(state)
        n = len(state)
        for i in range(n):
            bx, by = state[i]
            for dx, dy in directions:
                # Reverse "pull".  A forward push moves the box from (bx,by) to
                # (qx,qy) with the agent standing at (zx,zy) behind it, so in
                # reverse the box moves there and (qx,qy), (zx,zy) must both be free.
                qx, qy = bx + dx, by + dy
                zx, zy = bx + 2 * dx, by + 2 * dy
                rest = box_set - {(bx, by)}
                if not _free(level, qx, qy, rest):
                    continue
                if not _free(level, zx, zy, rest):
                    continue
                new_state = tuple(sorted(rest | {(qx, qy)}))
                # Prune states with an unrescuable box.
                if any(cell in crush for cell in new_state):
                    continue
                if new_state in visited:
                    continue
                if new_state == initial:
                    return True
                visited.add(new_state)
                queue.append(new_state)
    return False
