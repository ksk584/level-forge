"""Learning from human players in ``play``.

Every puzzle a player solves without agent help is appended to ``<learn_dir>/games.jsonl``.
Right after each such solve, the PPO policy is fine-tuned by behaviour cloning on *all*
recorded games (maximise the log-probability of the player's actions) and saved to
``<learn_dir>/model.zip``; the next ``play`` session starts from that model.

The policy was trained by reinforcement learning (``ued-*`` commands); this adds imitation of
players on top of it.  Wall bumps and other moves that change nothing are not imitated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .environment import SokobanEnv
from .grid import Level
from .gym_env import encode_obs


def load_model(path):
    """Load a PPO model.zip for inference / fine-tuning (no environment needed)."""
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


def record_game(path, level: Level, actions: Sequence[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"level": level.to_dict(), "actions": [int(a) for a in actions]}) + "\n")


def load_games(path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def demonstrations(games: Sequence[Dict], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """(observations, actions, games used) from games on ``shape`` (H, W) boards that replay to a solve."""
    obs, acts, used = [], [], 0
    for g in games:
        level = Level.from_dict(g["level"])
        if level.walls.shape != tuple(shape):
            continue
        env = SokobanEnv(level, max_steps=len(g["actions"]) + 1)
        o, a = [], []
        for act in g["actions"]:
            before = (env.level.agent, tuple(env.level.boxes))
            x = encode_obs(env.level)
            env.step(act)
            if (env.level.agent, tuple(env.level.boxes)) != before:   # skip wall bumps / blocked pushes
                o.append(x)
                a.append(act)
        if env.done and env.failure is None and a:
            obs += o
            acts += a
            used += 1
    if not acts:
        return np.zeros((0,)), np.zeros((0,), dtype=np.int64), 0
    return np.stack(obs), np.asarray(acts, dtype=np.int64), used


def learn_from_players(model, games: Sequence[Dict], epochs: int = 10) -> Optional[Dict]:
    """Behaviour-clone ``model``'s policy on the player games; returns stats, or None if none fit the model."""
    shape = model.observation_space.shape[1:]
    obs, acts, used = demonstrations(games, shape)
    if not used:
        return None
    policy = model.policy
    x = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    y = torch.as_tensor(acts, device=policy.device)
    policy.set_training_mode(True)
    # shortcut: full-batch updates on every recorded game; switch to minibatches past ~10k moves.
    for _ in range(epochs):
        _values, log_prob, _entropy = policy.evaluate_actions(x, y)
        loss = -log_prob.mean()
        policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()
    policy.set_training_mode(False)
    return {"games": used, "moves": int(len(acts)), "loss": loss.item()}
