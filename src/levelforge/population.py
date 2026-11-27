"""Procedural generation of random (and repaired) levels."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_KINDS, BOX_NORMAL, PUSH_COST, Level


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


def _random_kinds(rng: np.random.Generator, n_boxes: int, heavy_prob: float, fragile_prob: float) -> List[int]:
    """At most one heavy and one fragile box (for balance); the rest are normal."""
    kinds = [BOX_NORMAL] * n_boxes
    if rng.random() < heavy_prob:
        kinds[int(rng.integers(0, n_boxes))] = BOX_HEAVY
    if rng.random() < fragile_prob:
        fragile_candidates = [i for i, k in enumerate(kinds) if k == BOX_NORMAL]
        if fragile_candidates:
            kinds[int(rng.choice(fragile_candidates))] = BOX_FRAGILE
    return kinds


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
    rough_frac: float = 0.15,
    kinds: Optional[List[int]] = None,
) -> Level:
    """Generate a *guaranteed solvable* level by reverse Sokoban play.

    ``kinds`` fixes the box types (one per box, any mix); otherwise ``heavy_prob`` /
    ``fragile_prob`` give at most one heavy and one fragile box.

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

    If the level has a fragile box, ``rough_frac`` of the free non-goal cells become rough
    floor; the reverse play never moves the fragile box across them, so a safe route exists.
    (No rough tiles and no extra random draws without a fragile box, so plain-box levels are
    unchanged for a given seed.)
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

    if kinds is not None:
        if len(kinds) != n_boxes:
            raise ValueError(f"kinds has {len(kinds)} entries for {n_boxes} boxes")
        kinds = [int(k) for k in kinds]
    else:
        kinds = _random_kinds(rng, n_boxes, heavy_prob, fragile_prob)

    rough: set = set()
    if BOX_FRAGILE in kinds and rough_frac > 0:
        open_cells = [c for c in free_cells if c not in set(goals)]
        n_rough = int(round(rough_frac * len(open_cells)))
        rough = {open_cells[i] for i in rng.choice(len(open_cells), size=n_rough, replace=False)}

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
                if kinds[i] == BOX_FRAGILE and (px, py) in rough:
                    continue  # forward play would push the fragile box onto rough floor
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
        raise ValueError("scramble did not displace boxes enough")
    if set(box_pos) == set(goals):
        raise ValueError("scramble left every box on a goal")

    boxes = [(x, y, k) for (x, y), k in zip(box_pos, kinds)]
    energy = push_energy + energy_slack
    return Level(walls, list(goals), boxes, agent, energy, sorted(rough))


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
    max_steps: int = 300,
    energy_slack: int = 6,
    stats: Optional[dict] = None,
    rough_frac: float = 0.15,
    kinds: Optional[List[int]] = None,
) -> Level:
    """Generate a solvable random level via the reverse-scramble construction.

    (Rejection sampling of random box/goal placements is almost never solvable
    for Sokoban, so generation always starts from the solved state.)

    Every level is then verified with A*: it must be winnable within
    ``max_steps``, and its energy budget is set to the optimal plan's push cost
    plus ``energy_slack``.  (The scramble's own energy tally counts every pull,
    including ones that cancel out, which made budgets ~40x too loose.)

    If ``stats`` is given, ``stats["candidates"]`` counts every candidate produced and
    ``stats["verified"]`` the ones that passed A* -- the generator's solvability yield,
    measured the same way as the GA's (verified children / children produced).
    """
    from .solver import verify_level

    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + 1
        try:
            level = scramble_level(rng, size, n_boxes, wall_frac, heavy_prob, fragile_prob, rough_frac=rough_frac,
                                   kinds=kinds)
        except ValueError as e:
            last_err = e
            continue
        if level.validate():
            continue
        info = verify_level(level, max_steps=max_steps, max_nodes=max_nodes)
        if info is None:
            continue
        level.energy = info["energy_cost"] + energy_slack
        if stats is not None:
            stats["verified"] = stats.get("verified", 0) + 1
        return level
    raise RuntimeError(f"could not generate a solvable level in {max_tries} tries ({last_err})")


def generate_play_level(rng: np.random.Generator, size: int = 7, n_boxes: int = 1,
                        heavy: bool = False, fragile: bool = False, mix: bool = False) -> Level:
    """Generate an easy, solver-verified puzzle for interactive play.

    ``heavy`` / ``fragile`` guarantee one heavy / one fragile box; they need
    ``n_boxes >= heavy + fragile``.  ``mix`` instead draws a random combination of plain, heavy
    and fragile boxes (at least two different types; ``n_boxes`` boxes if >= 2, else 2 or 3).
    With a fragile box the level also has rough floor tiles, and puzzles are redrawn until the
    rough tiles block the shortest route, so the player has to find a safe one.
    """
    from .solver import solve

    kinds = None
    if mix:
        if heavy or fragile:
            raise ValueError("use either mix or heavy/fragile")
        n_boxes = n_boxes if n_boxes >= 2 else int(rng.integers(2, 4))
        kinds = [int(k) for k in rng.integers(0, 3, n_boxes)]
        while len(set(kinds)) < 2:
            kinds = [int(k) for k in rng.integers(0, 3, n_boxes)]
        fragile = BOX_FRAGILE in kinds
    if n_boxes < int(heavy) + int(fragile) and not mix:
        raise ValueError("--heavy and --fragile together need at least 2 boxes")
    # shortcut: up to 30 redraws; after that a puzzle whose rough tiles don't matter is accepted.
    for _ in range(30 if fragile else 1):
        level = random_solvable_level(
            rng,
            size=size,
            n_boxes=n_boxes,
            wall_frac=0.08,
            heavy_prob=1.0 if heavy else 0.0,
            fragile_prob=1.0 if fragile else 0.0,
            max_steps=500,
            energy_slack=8,
            rough_frac=0.25,
            kinds=kinds,
        )
        if not fragile:
            return level
        smooth = level.copy()
        smooth.rough = []
        safe, unsafe = solve(level)[0], solve(smooth)[0]
        if safe is not None and unsafe is not None and len(safe) > len(unsafe):
            return level
    return level


def repair_level(level: Level, rng: np.random.Generator, max_tries: int = 20,
                 n_boxes: Optional[int] = None, allow_kinds: bool = True) -> Optional[Level]:
    """Fix structural problems produced by crossover/mutation.

    Returns a Level with borders enforced, no overlaps, and matching
    box/goal counts, or None if repair fails.  ``n_boxes`` fixes the box count
    (default: keep the incoming count, but at least 1).
    """
    h, w = level.height, level.width
    walls = level.walls.copy()
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    level = Level(walls, level.goals, level.boxes, level.agent, level.energy, level.rough)

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
    target = n_boxes if n_boxes is not None else max(1, len(level.boxes))
    if len(boxes) > target:
        order = rng.permutation(len(boxes))[:target]
        boxes = [boxes[i] for i in sorted(order)]
    while len(boxes) < target:
        c = random_free(used | set(goals) | {b[:2] for b in boxes})
        if c == (-1, -1):
            return None
        kind = int(rng.choice(BOX_KINDS)) if allow_kinds else BOX_NORMAL
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
        rough = [c for c in level.rough if c in free_set and c not in goals]
        repaired = Level(walls, goals, boxes, (ax, ay), energy, rough)
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
        out = Level(level.walls, goals, boxes, (ax, ay), level.energy, level.rough)
    except ValueError:
        return None
    return out if not out.validate() else None
