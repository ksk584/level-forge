"""Genetic algorithm: tournament selection, region-wise crossover, mutation.

Levels double as genomes (a Level instance).  Child levels are always passed
through ``repair_level`` and the solvability checker; offspring that fail the
check are counted towards the solvability yield (SY) metric and discarded.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .grid import BOX_KINDS, Level
from .population import repair_level
from .solvability import is_solvable


def tournament_select(population: List[Level], fitnesses: List[float], rng: np.random.Generator, k: int = 3) -> Level:
    idxs = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[int(best)]


def crossover(a: Level, b: Level, rng: np.random.Generator) -> Optional[Level]:
    """Region-wise crossover: 2x2 block mixing of walls; goals/boxes inherited
    from a randomly chosen parent with per-item coin flips; then repaired."""
    h, w = a.walls.shape
    if (b.height, b.width) != (h, w):
        return None
    child_walls = a.walls.copy()
    hy, wy = h // 2, w // 2
    for y0, y1, x0, x1 in ((0, hy, 0, wy), (0, hy, wy, w), (hy, h, 0, wy), (hy, h, wy, w)):
        if rng.random() < 0.5:
            child_walls[y0:y1, x0:x1] = b.walls[y0:y1, x0:x1]

    goals = []
    for g in a.goals:
        if rng.random() < 0.5:
            goals.append(g)
    for g in b.goals:
        if rng.random() < 0.5 and g not in goals:
            goals.append(g)

    boxes = []
    for bx, by, kind in a.boxes:
        if rng.random() < 0.5:
            boxes.append((bx, by, kind))
    for bx, by, kind in b.boxes:
        if rng.random() < 0.5 and all(p[:2] != (bx, by) for p in boxes):
            boxes.append((bx, by, kind))

    agent = a.agent if rng.random() < 0.5 else b.agent
    energy = int(round(0.5 * a.energy + 0.5 * b.energy + rng.integers(-2, 3)))
    child = Level(child_walls, goals, boxes, agent, energy)
    return repair_level(child, rng)


def mutate(level: Level, rng: np.random.Generator,
           p_wall: float = 0.06,
           p_box_shift: float = 0.10,
           p_goal_shift: float = 0.10,
           p_kind: float = 0.08,
           p_agent: float = 0.10,
           p_energy: float = 0.25) -> Optional[Level]:
    walls = level.walls.copy()
    h, w = walls.shape
    # Flip random interior wall cells (keep border solid).
    interior = [(x, y) for y in range(1, h - 1) for x in range(1, w - 1)]
    for x, y in interior:
        if rng.random() < p_wall:
            walls[y, x] = not walls[y, x]

    boxes = list(level.boxes)
    # Shift boxes.
    boxes = [(bx + int(rng.integers(-1, 2)) if rng.random() < p_box_shift else bx,
              by + int(rng.integers(-1, 2)) if rng.random() < p_box_shift else by,
              kind) for bx, by, kind in boxes]
    # Change box kind.
    boxes = [(bx, by, int(rng.choice(BOX_KINDS)) if rng.random() < p_kind else kind) for bx, by, kind in boxes]
    # Shift goals.
    goals = [(gx + int(rng.integers(-1, 2)) if rng.random() < p_goal_shift else gx,
              gy + int(rng.integers(-1, 2)) if rng.random() < p_goal_shift else gy) for gx, gy in level.goals]
    # Move agent.
    ax, ay = level.agent
    if rng.random() < p_agent:
        ax += int(rng.integers(-2, 3))
        ay += int(rng.integers(-2, 3))
    # Energy budget tweak.
    energy = level.energy
    if rng.random() < p_energy:
        energy = int(np.clip(energy + rng.integers(-3, 4), 6, 60))

    child = Level(walls, goals, boxes, (ax, ay), energy)
    return repair_level(child, rng)


def population_diversity(levels: List[Level]) -> float:
    """Average pairwise Hamming distance over cell signatures (0..1 normalised)."""
    if len(levels) < 2:
        return 0.0
    sigs = [lvl.signature() for lvl in levels]
    total, count = 0, 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += int(np.count_nonzero(sigs[i] != sigs[j]))
            count += sigs[i].size
    return (total / count) if count else 0.0


def produce_offspring(a: Level, b: Level, rng: np.random.Generator, mutate_prob: float = 0.8) -> Optional[Level]:
    child = crossover(a, b, rng)
    if child is None:
        return None
    if rng.random() < mutate_prob:
        child = mutate(child, rng)
    return child


def next_generation(
    population: List[Level],
    fitnesses: List[float],
    rng: np.random.Generator,
    elite_k: int = 2,
    max_nodes: int = 30000,
    max_attempts: int = 200,
) -> Tuple[List[Level], int, int]:
    """Breed the next generation.

    Returns (children, attempts, solvable_children) -- SY = solvable_children / attempts.
    """
    n = len(population)
    order = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in order[:elite_k]]

    children: List[Level] = list(elites)
    attempts = 0
    solvable = 0
    while len(children) < n and attempts < max_attempts:
        attempts += 1
        p1 = tournament_select(population, fitnesses, rng, k=3)
        p2 = tournament_select(population, fitnesses, rng, k=3)
        child = produce_offspring(p1, p2, rng)
        if child is None:
            continue
        if not is_solvable(child, max_nodes=max_nodes):
            continue
        solvable += 1
        children.append(child)
    # Extreme fallback: top up with elites so the loop always returns n levels.
    i = 0
    while len(children) < n and elites:
        children.append(elites[i % len(elites)].copy())
        i += 1
    return children, attempts, solvable
