"""Expert trajectories from the generator's own solution.

The reverse-scramble construction knows a construct solution for every level
it creates.  Replaying the recorded reverse moves backwards yields a legal
forward solve; these expert trajectories drive imitation learning (BC) so the
agent starts from a sensible policy instead of pure chance.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .dqn import flatten_obs
from .environment import SokobanEnv
from .grid import Level

_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def forward_actions(trace: List[int]) -> List[int]:
    """Convert a recorded reverse-step direction list to forward actions."""
    return [_OPPOSITE[d] for d in reversed(trace)]


def collect_bc_samples(level: Level, trace: List[int]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Replay the expert solve on the level and return (states, actions, solved)."""
    actions = forward_actions(trace)
    env = SokobanEnv(level, max_steps=len(actions) + 1)
    obs, _ = env.reset()
    states, labels = [], []
    for a in actions:
        states.append(flatten_obs(obs, env.level.energy))
        labels.append(a)
        obs, _r, _done, _info = env.step(a)
    solved = env.failure is None and env._is_complete()
    if not states:
        return np.empty((0, env.level.encode().size + 1), dtype=np.float32), np.empty(0, dtype=np.int64), solved
    return np.stack(states).astype(np.float32), np.asarray(labels, dtype=np.int64), solved


def generate_expert_dataset(
    n_levels: int,
    size: int,
    n_boxes: int,
    seed: int = 0,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect BC samples; skips (and retries) any level whose trace replay fails."""
    from .population import scramble_level

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    kept = 0
    tries = 0
    while kept < n_levels and tries < n_levels * 5:
        tries += 1
        trace: List[int] = []
        try:
            level = scramble_level(rng, size=size, n_boxes=n_boxes,
                                   wall_frac=wall_frac, heavy_prob=heavy_prob,
                                   fragile_prob=fragile_prob, trace=trace)
        except ValueError:
            continue
        states, labels, solved = collect_bc_samples(level, trace)
        if not solved or len(states) == 0:
            continue
        xs.append(states)
        ys.append(labels)
        kept += 1
    if not xs:
        raise RuntimeError("could not collect any expert samples")
    return np.concatenate(xs), np.concatenate(ys)
