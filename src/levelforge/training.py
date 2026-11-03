"""Shared training/evaluation loops for the DQN agent."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .dqn import DQNAgent, flatten_obs
from .environment import SokobanEnv
from .grid import Level


def generate_level_pool(
    n: int,
    seed: int,
    size: int = 10,
    n_boxes: int = 3,
    wall_frac: float = 0.12,
    heavy_prob: float = 0.25,
    fragile_prob: float = 0.20,
) -> List[Level]:
    """Pre-generate a pool of random solvable levels (cached, reused for training)."""
    from .population import random_solvable_level

    rng = np.random.default_rng(seed)
    pool: List[Level] = []
    attempts = 0
    while len(pool) < n and attempts < n * 20:
        attempts += 1
        try:
            pool.append(random_solvable_level(rng, size=size, n_boxes=n_boxes,
                                              wall_frac=wall_frac,
                                              heavy_prob=heavy_prob,
                                              fragile_prob=fragile_prob))
        except RuntimeError:
            continue
    if not pool:
        raise RuntimeError("could not generate any solvable levels; loosen generation constraints")
    return pool


def episode_stats(agent: DQNAgent, env: SokobanEnv, train: bool = False,
                  shaping: bool = False, shaping_weight: float = 0.5) -> Dict:
    from .shaping import shaped_reward

    obs, _ = env.reset()
    total_reward = 0.0
    invalid_pushes = 0
    done = False
    while not done:
        vec = flatten_obs(obs, env.level.energy)
        action = agent.select_action(vec, greedy=not train)
        if train:
            level_before = env.level.copy() if shaping else None
            next_obs, reward, done, info = env.step(action)
            if shaping:
                reward = shaped_reward(reward, level_before, env.level, shaping_weight)
            agent.buffer.push(vec, action, reward, flatten_obs(next_obs, env.level.energy), done)
            agent.train_step()
            obs = next_obs
        else:
            obs, reward, done, info = env.step(action)
        total_reward += reward
        invalid_pushes += int(info.get("invalid_push", False))
    solved = env.failure is None
    return {
        "solved": solved,
        "failure": env.failure,
        "steps": env.steps,
        "total_reward": total_reward,
        "invalid_pushes": invalid_pushes,
        "loss": agent.last_loss,
        "energy_left": env.level.energy,
        "boxes_on_goals": env.level.boxes_on_goals(),
    }


def train_on_levels(
    agent: DQNAgent,
    levels: List[Level],
    episodes: int,
    max_steps: int = 300,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 200,
    seed: int = 0,
    level_weights: Optional[np.ndarray] = None,
    shaping: bool = False,
    shaping_weight: float = 0.5,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(episodes):
        agent.decay_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        if level_weights is not None:
            level = levels[int(rng.choice(len(levels), p=level_weights / level_weights.sum()))]
        else:
            level = levels[int(rng.integers(0, len(levels)))]
        env = SokobanEnv(level, max_steps=max_steps)
        stats = episode_stats(agent, env, train=True, shaping=shaping, shaping_weight=shaping_weight)
        stats["episode"] = ep
        stats["epsilon"] = agent.epsilon
        history.append(stats)
    return history


def evaluate_levels(
    agent: DQNAgent,
    levels: List[Level],
    trials: int = 3,
    max_steps: int = 300,
    seed: int = 0,
) -> Tuple[List[Dict], Dict]:
    """Greedy evaluation; returns per-episode records and an aggregate."""
    rng = np.random.default_rng(seed)
    records: List[Dict] = []
    for i, level in enumerate(levels):
        for t in range(trials):
            env = SokobanEnv(level, max_steps=max_steps)
            stats = episode_stats(agent, env, train=False)
            stats["level_index"] = i
            stats["trial"] = t
            records.append(stats)
    solved = sum(r["solved"] for r in records)
    agg = {
        "success_rate": solved / max(1, len(records)),
        "episodes": len(records),
        "mean_reward": float(np.mean([r["total_reward"] for r in records])),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
    }
    failures: Dict[str, int] = {}
    for r in records:
        if not r["solved"]:
            failures[r["failure"] or "unknown"] = failures.get(r["failure"] or "unknown", 0) + 1
    agg["failures"] = failures
    return records, agg
