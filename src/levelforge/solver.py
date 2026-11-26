"""A* planner for Sokoban -- the reference "near-optimal player".

Used for three things:
  1. Demos (``watch --solver``).
  2. Verifying GA offspring: a child is accepted only if A* finds a plan that
     fits the step cap and the energy budget (``verify_level``).
  3. Regret-based fitness: the solver's return on a level is the reference
     against which the agent's return is compared (PAIRED-style regret with
     an exact planner standing in for the antagonist).

State = (agent, sorted tuple of (x, y, kind) boxes).  Kinds are tracked so the
energy cost of a plan (heavy push = 3) can be computed exactly.  Boxes may
never enter dead squares (``crush_cells``), which mirrors the environment's
deadlock/fragile-break rules, so any plan found is legal in ``SokobanEnv``.

Heuristic: sum over boxes of the Manhattan distance to the nearest goal --
a lower bound on the remaining pushes, hence on remaining moves (admissible),
so returned plans are move-optimal.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional

from .grid import PUSH_COST, Level
from .solvability import crush_cells

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _heur(boxes, goals) -> int:
    return sum(min(abs(bx - gx) + abs(by - gy) for gx, gy in goals) for bx, by, _k in boxes)


def solve(level: Level, max_nodes: int = 200_000, time_limit_s: float = 20.0):
    """Return (moves, nodes, elapsed_s).  ``moves`` is a list of action indices, or None."""
    t0 = time.time()
    walls = level.walls
    h, w = walls.shape
    goals = list(level.goals)
    goal_set = set(goals)
    crush = crush_cells(level)
    initial = (level.agent, tuple(sorted(level.boxes)))
    if any((x, y) in crush for x, y, _k in initial[1]):
        return None, 0, time.time() - t0

    counter = 0
    heap = [(_heur(initial[1], goals), 0, counter, initial)]
    best_g = {initial: 0}
    parent: Dict = {initial: None}
    nodes = 0

    while heap:
        _f, g, _c, state = heapq.heappop(heap)
        if g > best_g.get(state, g):
            continue  # stale heap entry
        agent, boxes = state
        nodes += 1
        if nodes > max_nodes or ((nodes & 0xFFF) == 0 and time.time() - t0 > time_limit_s):
            return None, nodes, time.time() - t0
        if all((x, y) in goal_set for x, y, _k in boxes):
            path = []
            cur = state
            while parent[cur] is not None:
                cur, act = parent[cur]
                path.append(act)
            path.reverse()
            return path, nodes, time.time() - t0

        box_at = {(x, y): k for x, y, k in boxes}
        ax, ay = agent
        for ai, (dx, dy) in enumerate(_DIRS):
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < w and 0 <= ny < h) or walls[ny, nx]:
                continue
            if (nx, ny) in box_at:
                tx, ty = nx + dx, ny + dy
                if not (0 <= tx < w and 0 <= ty < h) or walls[ty, tx] or (tx, ty) in box_at:
                    continue
                if (tx, ty) in crush:
                    continue  # deadlock / fragile break
                kind = box_at[(nx, ny)]
                new_boxes = tuple(sorted([b for b in boxes if (b[0], b[1]) != (nx, ny)] + [(tx, ty, kind)]))
                new_state = ((nx, ny), new_boxes)
            else:
                new_state = ((nx, ny), boxes)
            ng = g + 1
            if ng >= best_g.get(new_state, 1 << 30):
                continue
            best_g[new_state] = ng
            parent[new_state] = (state, ai)
            counter += 1
            heapq.heappush(heap, (ng + _heur(new_state[1], goals), ng, counter, new_state))
    return None, nodes, time.time() - t0


def plan_energy(level: Level, moves: List[int]) -> int:
    """Total push-energy a move sequence spends on ``level``."""
    agent = level.agent
    box_at = {(x, y): k for x, y, k in level.boxes}
    cost = 0
    for a in moves:
        dx, dy = _DIRS[a]
        nx, ny = agent[0] + dx, agent[1] + dy
        if (nx, ny) in box_at:
            k = box_at.pop((nx, ny))
            box_at[(nx + dx, ny + dy)] = k
            cost += PUSH_COST[k]
        agent = (nx, ny)
    return cost


def replay_return(level: Level, moves: List[int], max_steps: int = 300) -> Dict:
    """Play ``moves`` in the real environment; returns its return and outcome."""
    from .environment import SokobanEnv

    env = SokobanEnv(level, max_steps=max_steps)
    env.reset()
    total = 0.0
    done = False
    for a in moves:
        _obs, r, done, _info = env.step(a)
        total += r
        if done:
            break
    return {"return": total, "solved": done and env.failure is None, "steps": env.steps}


def verify_level(level: Level, max_steps: int = 300, max_nodes: int = 60_000,
                 time_limit_s: float = 5.0, fix_energy: bool = True) -> Optional[Dict]:
    """Prove a level is winnable inside the environment's limits.

    Returns ``{"moves", "pushes", "energy_cost", "solver_return"}`` or None.
    If ``fix_energy`` is set and the plan needs more energy than the level
    grants, the level's energy is raised *in place* to exactly the plan cost
    (so the GA cannot win by starving the agent of energy).  A plan longer than
    ``max_steps`` means the level is unwinnable under the step cap -> None.
    """
    moves, _nodes, _t = solve(level, max_nodes=max_nodes, time_limit_s=time_limit_s)
    if moves is None or len(moves) == 0 or len(moves) > max_steps:
        return None
    cost = plan_energy(level, moves)
    if cost > level.energy:
        if not fix_energy:
            return None
        level.energy = cost
    rep = replay_return(level, moves, max_steps=max_steps)
    if not rep["solved"]:  # should never happen; guards against rule drift
        return None
    pushes = 0
    agent = level.agent
    box_cells = {(x, y) for x, y, _k in level.boxes}
    for a in moves:
        dx, dy = _DIRS[a]
        nxt = (agent[0] + dx, agent[1] + dy)
        if nxt in box_cells:
            box_cells.remove(nxt)
            box_cells.add((nxt[0] + dx, nxt[1] + dy))
            pushes += 1
        agent = nxt
    return {"moves": len(moves), "pushes": pushes, "energy_cost": cost,
            "solver_return": rep["return"]}
