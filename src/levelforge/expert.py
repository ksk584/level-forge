"""Expert trajectories from the generator's own solution or A* planner.

The A* planner provides optimal, loop-free demonstrations that enable reliable
imitation learning without the conflicting state labels inherent in random-walk
reverse scrambles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dqn import flatten_obs
from .environment import SokobanEnv
from .grid import Level
from .solver import solve

_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def forward_actions(trace: List[int]) -> List[int]:
    """Convert a recorded reverse-step direction list to forward actions."""
    return [_OPPOSITE[d] for d in reversed(trace)]


def collect_bc_samples(level: Level, trace: List[int]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Replay the expert solve on the level and return (states, actions, solved)."""
    actions = forward_actions(trace)
    return collect_solver_samples(level, actions)


def collect_solver_samples(level: Level, moves: List[int]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Replay forward moves on the level and return (states, actions, solved)."""
    env = SokobanEnv(level, max_steps=len(moves) + 5)
    obs, _ = env.reset()
    states, labels = [], []
    for a in moves:
        states.append(flatten_obs(obs, env.level.energy))
        labels.append(a)
        obs, _r, done, _info = env.step(a)
        if done:
            break
    solved = env.failure is None and env._is_complete()
    if not states:
        return np.empty((0, env.level.encode().size + 1), dtype=np.float32), np.empty(0, dtype=np.int64), solved
    return np.stack(states).astype(np.float32), np.asarray(labels, dtype=np.int64), solved


def augment_d4(X: np.ndarray, y: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Augment state-action pairs with 8-fold dihedral symmetries (rotations and reflections).

    Valid when height == width. Rotates the observation grid channels and maps the action
    indices accordingly: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    """
    if height != width or len(X) == 0:
        return X, y
    n_ch = (X.shape[1] - 1) // (height * width)
    x_aug, y_aug = [], []
    for vec, act in zip(X, y):
        grid = vec[:-1].reshape(n_ch, height, width)
        energy = vec[-1:]
        # 1. rot0 (identity)
        x_aug.append(vec)
        y_aug.append(act)
        # 2. rot90 cw
        r1 = np.rot90(grid, -1, (1, 2))
        x_aug.append(np.concatenate([r1.reshape(-1), energy]))
        y_aug.append({0: 3, 1: 2, 2: 0, 3: 1}[act])
        # 3. rot180
        r2 = np.rot90(grid, 2, (1, 2))
        x_aug.append(np.concatenate([r2.reshape(-1), energy]))
        y_aug.append({0: 1, 1: 0, 2: 3, 3: 2}[act])
        # 4. rot270 cw
        r3 = np.rot90(grid, 1, (1, 2))
        x_aug.append(np.concatenate([r3.reshape(-1), energy]))
        y_aug.append({0: 2, 1: 3, 2: 1, 3: 0}[act])
        # 5. horizontal flip
        hf = np.flip(grid, 2)
        x_aug.append(np.concatenate([hf.reshape(-1), energy]))
        y_aug.append({0: 0, 1: 1, 2: 3, 3: 2}[act])
        # 6. vertical flip
        vf = np.flip(grid, 1)
        x_aug.append(np.concatenate([vf.reshape(-1), energy]))
        y_aug.append({0: 1, 1: 0, 2: 2, 3: 3}[act])
        # 7. transpose (diagonal flip)
        tr = np.transpose(grid, (0, 2, 1))
        x_aug.append(np.concatenate([tr.reshape(-1), energy]))
        y_aug.append({0: 2, 1: 3, 2: 0, 3: 1}[act])
        # 8. anti-diagonal flip
        adr = np.flip(np.transpose(grid, (0, 2, 1)), (1, 2))
        x_aug.append(np.concatenate([adr.reshape(-1), energy]))
        y_aug.append({0: 3, 1: 2, 2: 1, 3: 0}[act])
    return np.array(x_aug, dtype=np.float32), np.array(y_aug, dtype=np.int64)


def generate_expert_dataset(
    n_levels: int,
    size: int,
    n_boxes: int,
    seed: int = 0,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
    use_solver: bool = True,
    multi_start: int = 2,
    include_toys: bool = True,
    mix_boxes: bool = False,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect expert samples for behaviour cloning."""
    from .population import random_solvable_level, scramble_level

    rng = np.random.default_rng(seed)
    xs, ys = [], []

    # Optional: include matching handcrafted toy levels
    if include_toys:
        toy_paths = sorted(Path("levels").glob("toy_*.json"))
        for p in toy_paths:
            try:
                lvl = Level.load_json(p)
                if lvl.height == size and lvl.width == size:
                    if not mix_boxes and len(lvl.boxes) > n_boxes:
                        continue
                    moves, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
                    if moves:
                        states, labels, solved = collect_solver_samples(lvl, moves)
                        if solved and len(states) > 0:
                            xs.append(states)
                            ys.append(labels)
            except Exception:
                pass

    kept = 0
    tries = 0
    max_tries = n_levels * 10
    while kept < n_levels and tries < max_tries:
        tries += 1
        trace: List[int] = []
        cur_boxes = n_boxes
        if mix_boxes:
            cur_boxes = int(rng.choice([1, 2])) if size >= 6 else 1

        try:
            if use_solver:
                level = random_solvable_level(rng, size=size, n_boxes=cur_boxes,
                                              wall_frac=wall_frac, heavy_prob=heavy_prob,
                                              fragile_prob=fragile_prob)
                moves, _, _ = solve(level, max_nodes=50000, time_limit_s=2.0)
                if not moves:
                    continue
                states, labels, solved = collect_solver_samples(level, moves)
                if not solved or len(states) == 0:
                    continue
                xs.append(states)
                ys.append(labels)
                kept += 1

                # Multi-start augmentation: solve from other reachable free cells
                if multi_start > 0:
                    box_coords = set((b[0], b[1]) for b in level.boxes)
                    free = [c for y in range(size) for c in [(x, y) for x in range(size)]
                            if not level.walls[c[1], c[0]] and c not in box_coords and c != level.agent]
                    rng.shuffle(free)
                    for alt_agent in free[:multi_start]:
                        alt_lvl = Level(level.walls, level.goals, level.boxes, alt_agent, level.energy)
                        alt_moves, _, _ = solve(alt_lvl, max_nodes=10000, time_limit_s=0.5)
                        if alt_moves:
                            st, lb, sol = collect_solver_samples(alt_lvl, alt_moves)
                            if sol and len(st) > 0:
                                xs.append(st)
                                ys.append(lb)
            else:
                level = scramble_level(rng, size=size, n_boxes=cur_boxes,
                                       wall_frac=wall_frac, heavy_prob=heavy_prob,
                                       fragile_prob=fragile_prob, trace=trace)
                states, labels, solved = collect_bc_samples(level, trace)
                if not solved or len(states) == 0:
                    continue
                xs.append(states)
                ys.append(labels)
                kept += 1
        except (ValueError, RuntimeError):
            continue

    if not xs:
        raise RuntimeError("could not collect any expert samples")
    X, y = np.concatenate(xs), np.concatenate(ys)
    if augment and size == size:
        X, y = augment_d4(X, y, size, size)
    return X, y
