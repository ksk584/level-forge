"""Procedural generation of random (and repaired) levels."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_KINDS, BOX_NORMAL, PUSH_COST, Level
from .solvability import is_solvable


def _components(walls: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = walls.shape
    seen = np.zeros_like(walls, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if walls[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            comp: List[Tuple[int, int]] = []
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not walls[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def scramble_level(
    rng: np.random.Generator,
    size: int = 10,
    n_boxes: int = 3,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
    scramble_steps: int = 250,
    min_pulls: int = 4,
    energy_slack: int = 8,
    trace: Optional[list] = None,
) -> Level:
    """Generate a *guaranteed solvable* level by reverse Sokoban play.

    Starting from the solved state (boxes on goals, agent anywhere free), a
    random agent performs a sequence of legal reverse moves -- walks and
    pulls (the exact inverse of a push).  Replaying the sequence backwards is
    a legal forward solution, so the generated puzzle is solvable by
    construction.  Unlike naive box-shuffling, the agent/box kinematics are
    fully modelled, so the reversed solution can never get stuck.

    Energy is set to the exact cost of the constructing solution plus
    ``energy_slack``, so every generated level is solvable within budget.
    If ``trace`` is provided, the reverse-step direction per move (as an
    action index, ``0=up 1=down 2=left 3=right``) is appended to it — use
    ``expert.forward_actions()`` to obtain the matching forward actions.
    """
    h = w = size
    walls = np.zeros((h, w), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    interior = rng.random((h, w)) < wall_frac
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    walls |= interior
    free_cells = [(x, y) for y in range(h) for x in range(w) if not walls[y, x]]
    interior_cells = (h - 2) * (w - 2)
    if len(free_cells) < max(2 * n_boxes + 6, int(interior_cells * 0.45)):
        raise ValueError("too few free cells")

    goals_idx = rng.choice(len(free_cells), size=n_boxes, replace=False)
    goals = [free_cells[i] for i in goals_idx]
    box_pos = list(goals)

    # Assign payload kinds (at most one heavy and one fragile for balance).
    kinds = [BOX_NORMAL] * n_boxes
    if rng.random() < heavy_prob + 0.25:
        heavy_idx = int(rng.integers(0, n_boxes))
        kinds[heavy_idx] = BOX_HEAVY
    if rng.random() < fragile_prob + 0.25:
        fragile_candidates = [i for i, k in enumerate(kinds) if k == BOX_NORMAL]
        if fragile_candidates:
            fragile_idx = int(rng.choice(fragile_candidates))
            kinds[fragile_idx] = BOX_FRAGILE

    def cell_free(x, y, exclude):
        return 0 <= x < w and 0 <= y < h and not walls[y, x] and (x, y) not in exclude

    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    dir_to_action = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

    # Start the agent adjacent to a box (or the chain stalls immediately).
    adj_cells = []
    for bx, by in box_pos:
        for dx, dy in dirs:
            ax, ay = bx + dx, by + dy
            if cell_free(ax, ay, set(box_pos)):
                adj_cells.append((ax, ay))
    if not adj_cells:
        raise ValueError("no cell adjacent to any box")
    agent = adj_cells[int(rng.integers(0, len(adj_cells)))]
    pulls = 0
    push_energy = 0

    # Boxes must never occupy non-goal crush cells (corners / wall-lines):
    # the forward env treats those as deadlocks (and fragile boxes shatter),
    # which would break the double-entry guarantee of the scramble.
    from .solvability import crush_cells
    crush = crush_cells(Level(walls, list(goals), [], goals[0], 0))

    # --- BFS from the agent to any cell from which a pull is guaranteed legal. ---
    def cell_wall(x, y):
        return x < 0 or y < 0 or x >= w or y >= h or walls[y, x]

    def pull_targets(box_set):
        """Map agent-position -> (box index, pull direction f).

        A pull in direction f is legal with the agent at box - f, provided the
        cell on the far side (box - 2f) is in the board, free of walls and of
        other boxes, and the destination cell (box, where the box lands... the
        box moves into the agent's old cell) is not a crush cell.
        """
        targets = {}
        for i, (bx, by) in enumerate(box_pos):
            for f in dirs:
                px, py = bx - f[0], by - f[1]           # agent stands here
                zx, zy = bx - 2 * f[0], by - 2 * f[1]   # pull destination (box goes to agent's old cell)
                if cell_wall(px, py) or cell_wall(zx, zy):
                    continue
                if (px, py) in box_set:
                    continue
                if (zx, zy) in box_set:
                    continue
                # The box moves into the agent's cell; that destination must
                # not be a non-goal crush cell.
                if (px, py) in crush:
                    continue
                targets[(px, py)] = (i, f)
        return targets

    def bfs(start, goals_set):
        """Find a path (list of action indices) from start to any cell in goals_set."""
        from collections import deque
        box_set = set(box_pos)
        prev = {start: None}  # cell -> (prev_cell, action)
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in goals_set:
                path = []
                node = cur
                while prev[node] is not None:
                    pc, act = prev[node]
                    path.append(act)
                    node = pc
                return list(reversed(path)), cur
            for dx, dy in dirs:
                nxt = (cur[0] + dx, cur[1] + dy)
                if cell_wall(nxt[0], nxt[1]) or nxt in box_set or nxt in prev:
                    continue
                prev[nxt] = (cur, dir_to_action[(dx, dy)])
                q.append(nxt)
        return None, None

    # Reverse Sokoban play (from the solved state): walk to any cell where a
    # pull is legal, execute the pull, repeat.  Guaranteed productive.
    for _ in range(scramble_steps):
        if pulls >= scramble_steps:
            break
        box_set = set(box_pos)
        targets = pull_targets(box_set)
        if not targets:
            import os
            if os.environ.get("LEVELFORGE_DEBUG"):
                print(f"[debug] pulls={pulls} box_pos={box_pos} goals={goals}")
                print(walls.astype(int))
                print("crush:", crush)
            raise ValueError("no pull is legal anywhere")
        path, dest = bfs(agent, set(targets))
        if dest is None:
            raise ValueError("pull positions are unreachable")
        for act in path:
            dx, dy = dirs[act]
            agent = (agent[0] + dx, agent[1] + dy)
            if trace is not None:
                trace.append(act)
        i, f = targets[dest]
        bx, by = box_pos[i]
        zx, zy = dest[0] - f[0], dest[1] - f[1]
        box_pos[i] = dest  # box moves into the agent's vacated cell
        agent = (zx, zy)
        push_energy += PUSH_COST[kinds[i]]
        pulls += 1
        if trace is not None:
            # The agent stepped opposite to f.
            trace.append(dir_to_action[(-f[0], -f[1])])
    if pulls < min_pulls:
        import os
        if os.environ.get("LEVELFORGE_DEBUG"):
            print(f"[debug] pulls={pulls} agent={agent} boxes={box_pos} walls=\\n{walls.astype(int)}")
        raise ValueError("scramble did not displace boxes enough")
    if set(box_pos) == set(goals):
        raise ValueError("scramble left every box on a goal")

    boxes = [(x, y, k) for (x, y), k in zip(box_pos, kinds)]
    energy = push_energy + energy_slack
    return Level(walls, list(goals), boxes, agent, energy)


def random_level(
    rng: np.random.Generator,
    size: int = 10,
    n_boxes: int = 3,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
) -> Level:
    """Generate a structurally valid random board (not guaranteed solvable)."""
    h = w = size
    walls = np.zeros((h, w), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    interior = rng.random((h, w)) < wall_frac
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    walls |= interior

    comps = _components(walls)
    if not comps:
        raise ValueError("no free cells")
    main = comps[0]
    main_set = set(main)
    # Wall off disconnected pockets so every free cell is reachable.
    for y in range(h):
        for x in range(w):
            if not walls[y, x] and (x, y) not in main_set:
                walls[y, x] = True
    free = [c for c in main]
    interior = (h - 2) * (w - 2)
    if len(free) < max(2 * n_boxes + 6, int(interior * 0.45)):
        raise ValueError("too few free cells")
    if len(free) < n_boxes * 4 + 1:
        raise ValueError("not enough room for boxes and goals")

    cells = list(free)
    rng.shuffle(cells)
    agent = cells[0]
    positions = cells[1:]

    boxes: List[Tuple[int, int, int]] = []
    for i in range(n_boxes):
        bx, by = positions[i]
        r = rng.random()
        if r < fragile_prob:
            kind = BOX_FRAGILE
        elif r < fragile_prob + heavy_prob:
            kind = BOX_HEAVY
        else:
            kind = BOX_NORMAL
        boxes.append((bx, by, kind))
    goal_cells = list(positions[n_boxes:])
    rng.shuffle(goal_cells)
    goals = goal_cells[:n_boxes]
    energy = 10 + 4 * n_boxes + int(rng.integers(0, 6))
    return Level(walls, goals, boxes, agent, energy)


def random_solvable_level(
    rng: np.random.Generator,
    size: int = 10,
    n_boxes: int = 3,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
    max_tries: int = 60,
    max_nodes: int = 30000,
) -> Level:
    """Generate a solvable random level via the reverse-scramble construction.

    (Rejection sampling of random box/goal placements is almost never solvable
    for Sokoban, so generation always starts from the solved state.)
    """
    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            level = scramble_level(rng, size, n_boxes, wall_frac, heavy_prob, fragile_prob)
        except ValueError as e:
            last_err = e
            continue
        if level.validate():
            continue
        return level
    raise RuntimeError(f"could not generate a solvable level in {max_tries} tries ({last_err})")


def repair_level(level: Level, rng: np.random.Generator, max_tries: int = 20) -> Optional[Level]:
    """Fix structural problems produced by crossover/mutation.

    Returns a Level with borders enforced, no overlaps, and matching
    box/goal counts, or None if repair fails.
    """
    h, w = level.height, level.width
    walls = level.walls.copy()
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    level = Level(walls, level.goals, level.boxes, level.agent, level.energy)

    free_cells = [(x, y) for y in range(h) for x in range(w) if not walls[y, x]]
    if len(free_cells) < 8:
        return None

    free_set = set(free_cells)

    def random_free(exclude: set) -> Tuple[int, int]:
        candidates = [c for c in free_cells if c not in exclude]
        return candidates[int(rng.integers(0, len(candidates)))] if candidates else (-1, -1)

    used: set = set()

    # Goals: drop invalid/duplicated, keep count sane.
    goals: List[Tuple[int, int]] = []
    for g in level.goals:
        x, y = g
        if (x, y) in free_set and (x, y) not in goals:
            goals.append((x, y))
    # Boxes: drop on-wall/out-of-grid duplicates; relocate overflow.
    boxes: List[Tuple[int, int, int]] = []
    for x, y, kind in level.boxes:
        if (x, y) in free_set and all(b[:2] != (x, y) for b in boxes):
            boxes.append((x, y, kind))
    while len(boxes) < 3:
        c = random_free(used | set(goals) | {b[:2] for b in boxes})
        if c == (-1, -1):
            return None
        kind = int(rng.choice(BOX_KINDS))
        boxes.append((c[0], c[1], kind))
    # Match box/goal counts by trimming/extending goals.
    while len(goals) > len(boxes):
        goals.pop()
    while len(goals) < len(boxes):
        c = random_free(used | set(goals) | {b[:2] for b in boxes})
        if c == (-1, -1):
            return None
        goals.append(c)
    # Agent placement.
    ax, ay = level.agent
    occupied = {b[:2] for b in boxes}
    if (ax, ay) not in free_set or (ax, ay) in occupied:
        c = random_free(used | occupied | set(goals))
        if c == (-1, -1):
            return None
        ax, ay = c
    energy = int(np.clip(level.energy, 6, 60))
    try:
        repaired = Level(walls, goals, boxes, (ax, ay), energy)
    except ValueError:
        return None
    if repaired.validate():
        # last resort: nudge agent/boxes off marker cells
        return None if max_tries <= 0 else _resolve_overlap(repaired, rng)
    return repaired


def _resolve_overlap(level: Level, rng: np.random.Generator) -> Optional[Level]:
    """Final cleanup: drop invalid boxes/goals until perfectly clean."""
    goals = list(level.goals)
    boxes = list(level.boxes)
    free = [(x, y) for y in range(level.height) for x in range(level.width) if not level.walls[y, x]]
    free_set = set(free)
    boxes = [b for b in boxes if b[:2] in free_set]
    goals = [g for g in goals if g in free_set]
    # dedupe boxes
    seen = set()
    deduped = []
    for b in boxes:
        if b[:2] not in seen:
            seen.add(b[:2])
            deduped.append(b)
    boxes = deduped
    while len(goals) > len(boxes):
        goals.pop()
    while len(goals) < len(boxes):
        candidates = [c for c in free if c not in goals and c not in {b[:2] for b in boxes}]
        if not candidates:
            return None
        goals.append(candidates[int(rng.integers(0, len(candidates)))])
    ax, ay = level.agent
    occupied = {b[:2] for b in boxes}
    if (ax, ay) not in free_set or (ax, ay) in occupied:
        candidates = [c for c in free if c not in occupied and c not in goals]
        if not candidates:
            return None
        ax, ay = candidates[int(rng.integers(0, len(candidates)))]
    try:
        out = Level(level.walls, goals, boxes, (ax, ay), level.energy)
    except ValueError:
        return None
    return out if not out.validate() else None
