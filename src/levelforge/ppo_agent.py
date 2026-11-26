"""PPO agent (Stable-Baselines3) with a small-grid CNN and a batched evaluator.

Why PPO rather than the from-scratch DQN in ``dqn.py``: in our pilot runs the
DQN stayed at or below random-policy success on 6x6 one-box boards (greedy
policies collapse into move loops, and per-sample updates are CPU-bound),
which leaves the GA's fitness signal flat.  PPO with parallel environments is
the standard, well-tested choice for discrete grid puzzles (and the one used
in most UED work); the DQN is kept as a documented comparison.

The batched evaluator steps many episodes in lock-step so that one network
forward pass serves the whole batch -- this is what makes it affordable to
score a GA population of dozens of levels x several trials every generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .environment import SokobanEnv
from .grid import Level
from .gym_env import LevelSource, SokobanGymEnv, encode_obs

PPO_DEFAULTS = dict(
    n_envs=16, n_steps=128, batch_size=256, n_epochs=4, learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_steps=120, shaping_weight=0.1, reward_scale=10.0, reward_mode="boxoban", features_dim=256,
)


class GridCNN(nn.Module):
    """3x3 conv stack sized for tiny grids (SB3's NatureCNN needs >= 36x36 inputs)."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__()
        c, h, w = observation_space.shape
        self.features_dim = features_dim
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, features_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def _make_extractor_class():
    # Imported lazily so the rest of the package works without SB3 installed.
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class GridCNNExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            self.cnn = GridCNN(observation_space, features_dim)

        def forward(self, obs):
            return self.cnn(obs)

    return GridCNNExtractor


class PPOAgent:
    """Thin wrapper: train on a (mutable) level source, evaluate on level lists."""

    def __init__(self, init_levels: Sequence[Level], seed: int = 0, device: str = "auto", **overrides):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        self.cfg = {**PPO_DEFAULTS, **overrides}
        self.seed = seed
        self.source = LevelSource(init_levels, seed=seed)
        c = self.cfg
        self.vec_env = DummyVecEnv([
            (lambda: SokobanGymEnv(self.source, max_steps=c["max_steps"], reward_scale=c["reward_scale"],
                                   shaping_weight=c["shaping_weight"], reward_mode=c["reward_mode"]))
            for _ in range(c["n_envs"])
        ])
        self.model = PPO(
            "CnnPolicy", self.vec_env, seed=seed, device=device, verbose=0,
            n_steps=c["n_steps"], batch_size=c["batch_size"], n_epochs=c["n_epochs"],
            learning_rate=c["learning_rate"], gamma=c["gamma"], gae_lambda=c["gae_lambda"],
            clip_range=c["clip_range"], ent_coef=c["ent_coef"], vf_coef=c["vf_coef"],
            policy_kwargs=dict(features_extractor_class=_make_extractor_class(),
                               features_extractor_kwargs=dict(features_dim=c["features_dim"]),
                               normalize_images=False, net_arch=dict(pi=[128], vf=[128])),
        )
        self.timesteps = 0

    # ------------------------------------------------------------------ #
    def train(self, levels: Sequence[Level], timesteps: int, weights: Optional[Sequence[float]] = None) -> None:
        self.source.set(levels, weights)
        self.model.learn(total_timesteps=int(timesteps), reset_num_timesteps=False, progress_bar=False)
        self.timesteps = int(self.model.num_timesteps)

    def act(self, obs_batch: np.ndarray, deterministic: bool) -> np.ndarray:
        actions, _ = self.model.predict(obs_batch, deterministic=deterministic)
        return np.asarray(actions)

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path, init_levels: Sequence[Level], seed: int = 0, device: str = "auto", **overrides) -> "PPOAgent":
        from stable_baselines3 import PPO

        agent = cls(init_levels, seed=seed, device=device, **overrides)
        agent.model = PPO.load(str(path), env=agent.vec_env, device=device,
                               custom_objects={"policy_kwargs": agent.model.policy_kwargs})
        agent.timesteps = int(agent.model.num_timesteps)
        return agent


def rollout_batch(agent, levels: Sequence[Level], trials: int = 1, max_steps: int = 120,
                  deterministic: bool = False) -> List[Dict]:
    """Run ``len(levels) * trials`` episodes in lock-step; one record per episode.

    Records: level_index, trial, solved, failure, steps, return (raw, unscaled,
    unshaped environment reward), boxes_on_goals, n_boxes.
    ``agent`` needs only ``act(obs_batch, deterministic) -> actions``.
    """
    envs, meta = [], []
    for i, lvl in enumerate(levels):
        for t in range(trials):
            envs.append(SokobanEnv(lvl, max_steps=max_steps))
            meta.append((i, t))
    returns = np.zeros(len(envs))
    active = list(range(len(envs)))
    obs = {k: encode_obs(envs[k].level) for k in active}
    while active:
        batch = np.stack([obs[k] for k in active])
        actions = agent.act(batch, deterministic)
        still = []
        for k, a in zip(active, actions):
            _o, r, done, _info = envs[k].step(int(a))
            returns[k] += r
            if done:
                continue
            obs[k] = encode_obs(envs[k].level)
            still.append(k)
        active = still
    out = []
    for k, env in enumerate(envs):
        i, t = meta[k]
        out.append({
            "level_index": i, "trial": t,
            "solved": env.failure is None and env.done,
            "failure": env.failure, "steps": env.steps, "return": float(returns[k]),
            "boxes_on_goals": env.level.boxes_on_goals(), "n_boxes": env.box_total,
        })
    return out


def success_rate(records: List[Dict]) -> float:
    return float(np.mean([r["solved"] for r in records])) if records else 0.0


def set_torch_threads(n: Optional[int] = None) -> None:
    torch.set_num_threads(n or max(1, min(8, torch.get_num_threads())))
