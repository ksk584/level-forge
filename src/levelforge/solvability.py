"""Deadlock detection and bounded reverse-BFS solvability checking.

Sokoban feasibility is PSPACE-complete in general, so this module uses a
cheap but effective combination:

  1. Dead squares (``crush_cells``) and freeze deadlocks (``is_frozen``) to reject
     hopeless boxes.
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


def crush_cells(level: Level, fragile: bool = False) -> Set[Tuple[int, int]]:
    """Dead squares: free cells from which a lone box can never reach any goal.

    Standard "simple deadlock" detection.  Starting with a box on every goal we
    repeatedly *pull* it backwards: a box at ``c`` can have been pushed there
    from ``p = c - d`` iff ``p`` and the agent's cell ``p - d`` are both free.
    Every cell reached this way is live; all other free cells are dead.

    This subsumes corner deadlocks and wall-line deadlocks, and -- unlike a
    local "two walls opposite each other" rule -- never flags 1-wide corridors
    that a box can legitimately travel along.  Depends only on walls + goals,
    so callers should compute it once per level and reuse it.

    ``fragile=True`` gives the dead squares of a fragile box, which may never be pushed onto
    rough floor: a rough cell is live (the box can be pushed off it) but nothing is reached
    through it.  The agent may still stand on rough floor.
    """
    h, w = level.height, level.width
    walls = level.walls
    rough = set(level.rough) if fragile else set()

    def free(x, y):
        return 0 <= x < w and 0 <= y < h and not walls[y, x]

    live: Set[Tuple[int, int]] = set(g for g in level.goals if free(*g) and g not in rough)
    stack = list(live)
    while stack:
        cx, cy = stack.pop()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            px, py = cx - dx, cy - dy          # box's previous cell
            ax, ay = px - dx, py - dy          # where the agent stood to push
            if (px, py) not in live and free(px, py) and free(ax, ay):
                live.add((px, py))
                if (px, py) not in rough:      # a box can only be pushed *from* a rough cell
                    stack.append((px, py))
    return {(x, y) for y in range(h) for x in range(w) if not walls[y, x] and (x, y) not in live}


def is_frozen(walls, pos: Tuple[int, int], box_set: Set[Tuple[int, int]], dead: Set[Tuple[int, int]],
              _assumed: frozenset = frozenset()) -> bool:
    """Freeze deadlock: the box at ``pos`` can never move again.

    A box is stuck along an axis if a wall (or a box already assumed stuck) is on
    either side, if both sides are dead squares, or if a neighbouring box on that
    axis is itself frozen.  Frozen = stuck on both axes.  Boxes under examination
    are treated as walls to break cycles (standard Sokoban freeze test).
    """
    h, w = walls.shape
    assumed = _assumed | {pos}
    x, y = pos

    def wall(c):
        return not (0 <= c[0] < w and 0 <= c[1] < h) or walls[c[1], c[0]] or c in assumed

    def stuck(a, b):
        if wall(a) or wall(b) or (a in dead and b in dead):
            return True
        return any(c in box_set and is_frozen(walls, c, box_set, dead, assumed) for c in (a, b))

    return stuck((x - 1, y), (x + 1, y)) and stuck((x, y - 1), (x, y + 1))


def any_frozen_off_goal(walls, boxes, goal_set, dead) -> bool:
    """True if some box off a goal is frozen -- the level can no longer be solved."""
    box_set = {(b[0], b[1]) for b in boxes}
    return any(p not in goal_set and is_frozen(walls, p, box_set, dead) for p in box_set)


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
