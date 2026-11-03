"""From-scratch DQN: network, replay buffer, agent. PyTorch only."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .grid import N_CHANNELS

ACTION_DIM = 4


def state_dim(height: int, width: int) -> int:
    return N_CHANNELS * height * width + 1  # +1 for normalised energy


def build_q_network(height: int, width: int, arch: str = "mlp", hidden: int = 64):
    if arch == "cnn":
        return QNetworkCNN(height, width, hidden_fc=hidden)
    if arch == "mlp":
        return QNetwork(state_dim(height, width), hidden=hidden)
    raise ValueError(f"unknown arch: {arch}")


class QNetworkCNN(nn.Module):
    """CNN over the 6-channel one-hot grid + energy scalar.

    Sokoban is spatial: a 3x3 receptive field already matches 'am I pushing into something'.
    Accepts the same flattened observation vector as QNetwork (adapter reshapes inside).
    """

    def __init__(self, height: int, width: int, channels: int = N_CHANNELS, hidden_fc: int = 64):
        super().__init__()
        self.height, self.width, self.channels = height, width, channels
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * height * width + 1, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, ACTION_DIM)

    def forward(self, flat):
        x = flat[:, : self.channels * self.height * self.width]
        x = x.view(-1, self.channels, self.height, self.width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = torch.cat([x, flat[:, -1:]], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class QNetwork(nn.Module):
    """MLP: input -> 64 -> ReLU -> 64 -> ReLU -> 4 Q-values (synopsis 5.3)."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def flatten_obs(obs: np.ndarray, energy: int, max_energy: int = 60) -> np.ndarray:
    flat = obs.reshape(-1).astype(np.float32)
    energy_norm = np.float32(min(max(energy, 0), max_energy) / max_energy)
    return np.concatenate([flat, np.array([energy_norm], dtype=np.float32)])


class ReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )


class DQNAgent:
    def __init__(
        self,
        height: int,
        width: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 20000,
        batch_size: int = 64,
        target_sync_steps: int = 500,
        device: Optional[str] = None,
        seed: int = 0,
        arch: str = "mlp",
        hidden: int = 64,
    ):
        self.height, self.width = height, width
        self.arch, self.hidden = arch, hidden
        self.input_dim = state_dim(height, width)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy_net = build_q_network(height, width, arch=arch, hidden=hidden).to(self.device)
        self.target_net = build_q_network(height, width, arch=arch, hidden=hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self.epsilon = 1.0
        self._train_steps = 0
        self.last_loss = 0.0

    # ------------------------------------------------------------------ #
    def select_action(self, state_vec: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            q = self.policy_net(t)
            return int(q.argmax(dim=1).item())

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(states).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        a = torch.from_numpy(actions).to(self.device)
        r = torch.from_numpy(rewards).to(self.device)
        d = torch.from_numpy(dones).to(self.device)

        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1).values
            target = r + self.gamma * next_q * (1.0 - d)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self._train_steps += 1
        if self._train_steps % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def decay_epsilon(self, episode: int, start: float = 1.0, end: float = 0.05, decay_episodes: int = 200):
        frac = min(1.0, episode / max(1, decay_episodes))
        self.epsilon = start + frac * (end - start)

    # ------------------------------------------------------------------ #
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "input_dim": self.input_dim,
                "height": self.height,
                "width": self.width,
                "epsilon": self.epsilon,
                "arch": self.arch,
                "hidden": self.hidden,
            },
            path,
        )

    @classmethod
    def init_from(cls, path, height: int, width: int,
                  device: Optional[str] = None, seed: int = 0) -> "DQNAgent":
        """Construct a fresh agent whose weights are copied from an existing checkpoint
        (e.g. a behaviour-cloned policy) — used to warm-start DQN training."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        h = int(ckpt.get("height", height))
        w = int(ckpt.get("width", width))
        arch = ckpt.get("arch", "mlp")
        hidden = int(ckpt.get("hidden", 64))
        agent = cls(h, w, device=device, seed=seed, arch=arch, hidden=hidden)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
        agent.target_net.eval()
        agent.epsilon = ckpt.get("epsilon", 0.05)
        return agent

    @classmethod
    def load(cls, path, device: Optional[str] = None, seed: int = 0) -> "DQNAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ckpt["height"], ckpt["width"], device=device, seed=seed,
                    arch=ckpt.get("arch", "mlp"), hidden=int(ckpt.get("hidden", 64)))
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        agent.target_net.eval()
        agent.epsilon = ckpt.get("epsilon", 0.05)
        return agent

    def evaluate(self, envs, greedy: bool = True) -> Dict[str, float]:
        """Average success rate / steps across a list of (level, max_steps) envs."""
        solved = 0
        total_steps = []
        failures: Dict[str, int] = {}
        for env in envs:
            obs, _ = env.reset()
            done = False
            while not done:
                vec = flatten_obs(obs, env.level.energy)
                action = self.select_action(vec, greedy=greedy)
                obs, _r, done, info = env.step(action)
            if env._is_complete():
                solved += 1
            else:
                failures[info["failure"] or "unknown"] = failures.get(info["failure"] or "unknown", 0) + 1
            total_steps.append(env.steps)
        n = max(1, len(envs))
        return {
            "success_rate": solved / n,
            "mean_steps": float(np.mean(total_steps)) if total_steps else 0.0,
            **{f"fail_{k}": v for k, v in failures.items()},
        }
