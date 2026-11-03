"""Procedural generation of random (and repaired) levels."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_KINDS, BOX_NORMAL, Level
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
) -> Level:
    """Generate a *guaranteed solvable* level by reverse Sokoban play.

    Starting from the solved state (boxes on goals, agent anywhere free), a
    random agent performs a sequence of legal reverse moves -- walks and
    pulls (the exact inverse of a push).  Replaying the sequence backwards is
    a legal forward solution, so the generated puzzle is solvable by
    construction.  Unlike naive box-shuffling, the agent/box kinematics are
    fully modelled, so the reversed solution can never get stuck.
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
    if len(free_cells) < max(4 * n_boxes + 6, int(interior_cells * 0.60)):
        raise ValueError("too few free cells")

    goals_idx = rng.choice(len(free_cells), size=n_boxes, replace=False)
    goals = [free_cells[i] for i in goals_idx]
    box_pos = list(goals)

    def cell_free(x, y, exclude):
        return 0 <= x < w and 0 <= y < h and not walls[y, x] and (x, y) not in exclude

    agent = free_cells[int(rng.integers(0, len(free_cells)))]
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    pulls = 0
    # A reverse move pair is the exact inverse of a forward push:
    #   push:  agent A, box A+v, target A+2v   ->  agent A+v, box A+2v
    #   pull:  agent at X, box at X+d (d = -v); if Z = X-d is free, the push is
    #          undone: agent -> Z, box -> X.
    for _ in range(scramble_steps):
        box_set = set(box_pos)
        # Enumerate legal reverse moves.
        pulls_available, walks_available = [], []
        for dx, dy in dirs:
            # Pull: box adjacent in direction d; agent steps away, box follows.
            bx, by = agent[0] + dx, agent[1] + dy
            zx, zy = agent[0] - dx, agent[1] - dy
            if (bx, by) in box_set and cell_free(zx, zy, box_set):
                pulls_available.append((dx, dy))
            # Walk: adjacent cell free of walls and boxes.
            wx, wy = agent[0] + dx, agent[1] + dy
            if cell_free(wx, wy, box_set):
                walks_available.append((dx, dy))
        do_pull = pulls_available and (not walks_available or rng.random() < 0.50)
        if do_pull:
            dx, dy = pulls_available[int(rng.integers(0, len(pulls_available)))]
            bx, by = agent[0] + dx, agent[1] + dy
            zx, zy = agent[0] - dx, agent[1] - dy
            idx = box_pos.index((bx, by))
            box_pos[idx] = agent  # box moves into the agent's vacated cell
            agent = (zx, zy)
            pulls += 1
        elif walks_available:
            dx, dy = walks_available[int(rng.integers(0, len(walks_available)))]
            agent = (agent[0] + dx, agent[1] + dy)
        else:
            break
    if pulls < min_pulls:
        raise ValueError("scramble did not displace boxes enough")
    if set(box_pos) == set(goals):
        raise ValueError("scramble left every box on a goal")

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
    boxes = [(x, y, k) for (x, y), k in zip(box_pos, kinds)]

    heavy_count = sum(1 for k in kinds if k == BOX_HEAVY)
    energy = 12 + 5 * n_boxes + 2 * heavy_count
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
    if len(free) < max(4 * n_boxes + 6, int(interior * 0.70)):
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
