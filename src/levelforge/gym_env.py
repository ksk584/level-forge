"""Gymnasium adapter around ``SokobanEnv`` plus the level sources that feed it.

The co-evolution loop changes *which* levels the agent trains on between
cycles.  Each vectorised sub-environment therefore holds a reference to one
shared ``LevelSource``; replacing the source's contents re-targets training
without rebuilding the agent or the vector env (``DummyVecEnv`` keeps all
sub-envs in-process, which is the right choice here: an environment step costs
~10 microseconds, so the policy network -- not the simulator -- is the
bottleneck and subprocess IPC would only add overhead).

Observation: float32 tensor (7, H, W) = the 6 one-hot channels of
``Level.encode`` plus a constant plane holding the normalised energy budget.
Training reward (``reward_mode``):
  * ``"boxoban"`` (default) -- the standard Sokoban RL reward (Racaniere et
    al. 2017; gym-sokoban): -0.1 per step, +1 / -1 for placing / removing a box
    on a goal, +10 for solving, and -1 when an episode is lost (deadlock,
    fragile break, energy exhaustion).  Pilot runs showed the environment's
    native reward (-5 for a blocked push) drives PPO into a "never touch a
    box" local optimum, below random-policy success even on 10 levels.
  * ``"native"`` -- ``SokobanEnv``'s own reward divided by ``reward_scale``.
Plus optional potential-based distance shaping (Ng et al. 1999).
Evaluation metrics never use this reward -- only success / steps.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np

from .environment import SokobanEnv
from .grid import N_CHANNELS, Level
from .shaping import box_goal_potential

OBS_CHANNELS = N_CHANNELS + 1
MAX_ENERGY = 60.0


def encode_obs(level: Level) -> np.ndarray:
    grid = level.encode()
    energy = np.full((1, level.height, level.width), min(max(level.energy, 0), MAX_ENERGY) / MAX_ENERGY,
                     dtype=np.float32)
    return np.concatenate([grid, energy], axis=0)


class LevelSource:
    """A weighted bag of levels that sub-environments sample from on reset."""

    def __init__(self, levels: Sequence[Level], weights: Optional[Sequence[float]] = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.set(levels, weights)

    def set(self, levels: Sequence[Level], weights: Optional[Sequence[float]] = None) -> None:
        if not levels:
            raise ValueError("LevelSource needs at least one level")
        self.levels: List[Level] = list(levels)
        w = np.ones(len(levels)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()

    def sample(self) -> Level:
        return self.levels[int(self.rng.choice(len(self.levels), p=self.p))]


class SokobanGymEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, source: LevelSource, max_steps: int = 120, reward_scale: float = 10.0,
                 shaping_weight: float = 0.1, reward_mode: str = "boxoban"):
        super().__init__()
        self.reward_mode = reward_mode
        self.source = source
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.shaping_weight = shaping_weight
        lvl = source.levels[0]
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_CHANNELS, lvl.height, lvl.width), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.env: Optional[SokobanEnv] = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        level = (options or {}).get("level") or self.source.sample()
        self.env = SokobanEnv(level, max_steps=self.max_steps)
        self._phi = box_goal_potential(self.env.level)
        return encode_obs(self.env.level), {}

    def step(self, action):
        on_before = self.env.level.boxes_on_goals()
        _obs, reward, done, info = self.env.step(int(action))
        solved = done and self.env.failure is None
        if self.reward_mode == "boxoban":
            reward = -0.1 + (self.env.level.boxes_on_goals() - on_before) + (10.0 if solved else 0.0)
            if done and self.env.failure not in (None, "timeout"):
                reward -= 1.0
        else:
            reward = reward / self.reward_scale
        if self.shaping_weight:
            phi = box_goal_potential(self.env.level)
            reward += self.shaping_weight * (self._phi - phi)
            self._phi = phi
        timeout = done and self.env.failure == "timeout"
        info = dict(info)
        info["is_success"] = bool(solved)
        return encode_obs(self.env.level), float(reward), done and not timeout, timeout, info

    def render(self):
        return self.env.render_ascii() if self.env else ""
