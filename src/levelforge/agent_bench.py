"""Agent comparison: from-scratch DQN vs PPO at an equal environment-step budget.

Supports the report's "why PPO" decision with data rather than assertion.
Both agents train on the same random level pool and are evaluated with the
same batched evaluator on the same held-out / hand-designed sets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from .dqn import DQNAgent
from .gym_env import MAX_ENERGY
from .ppo_agent import PPOAgent, rollout_batch, set_torch_threads, success_rate
from .training import train_on_levels
from .ued import UED_DEFAULTS, sample_levels, test_sets


class DQNActor:
    """Adapts a DQNAgent to the batched evaluator's ``act`` interface."""

    def __init__(self, agent: DQNAgent, epsilon: float = 0.05):
        self.agent, self.epsilon = agent, epsilon
        self.rng = np.random.default_rng(0)

    def act(self, obs_batch: np.ndarray, deterministic: bool) -> np.ndarray:
        import torch

        grid = obs_batch[:, :-1].reshape(len(obs_batch), -1)
        # dqn.flatten_obs normalises energy by 60 -- identical to the gym plane.
        energy = obs_batch[:, -1, 0, 0:1] * (MAX_ENERGY / 60.0)
        x = torch.from_numpy(np.concatenate([grid, energy], axis=1)).float().to(self.agent.device)
        with torch.no_grad():
            a = self.agent.policy_net(x).argmax(1).cpu().numpy()
        if not deterministic:
            explore = self.rng.random(len(a)) < self.epsilon
            a[explore] = self.rng.integers(0, 4, explore.sum())
        return a


def _eval(actor, sets, cfg) -> Dict[str, float]:
    return {f"{k}_sr": success_rate(rollout_batch(actor, v, trials=2, max_steps=cfg["max_steps"]))
            for k, v in sets.items()}


def run_agent_bench(cfg: Dict, out_dir, total_steps: int = 400_000, checkpoints: int = 8, seed: int = 0) -> Path:
    cfg = {**UED_DEFAULTS, **cfg}
    set_torch_threads(cfg.get("threads"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = sample_levels(np.random.default_rng(seed), cfg["base_pool"], cfg)
    sets = test_sets(cfg)
    chunk = total_steps // checkpoints
    log = {"ppo": [], "dqn": []}

    t0 = time.time()
    ppo = PPOAgent(pool, seed=seed, device=cfg["device"], max_steps=cfg["max_steps"], **cfg["ppo"])
    for _ in range(checkpoints):
        ppo.train(pool, chunk)
        log["ppo"].append({"timesteps": ppo.timesteps, "elapsed_s": round(time.time() - t0, 1), **_eval(ppo, sets, cfg)})
        print("[bench] ppo", log["ppo"][-1], flush=True)

    t0 = time.time()
    dqn = DQNAgent(cfg["size"], cfg["size"], seed=seed, arch="cnn", hidden=128)
    steps = 0
    for c in range(checkpoints):
        target = (c + 1) * chunk
        while steps < target:
            hist = train_on_levels(dqn, pool, episodes=20, max_steps=cfg["max_steps"], seed=seed + steps,
                                   epsilon_start=max(0.05, 1.0 - steps / (0.5 * total_steps)),
                                   epsilon_end=max(0.05, 1.0 - (steps + 20 * cfg["max_steps"]) / (0.5 * total_steps)),
                                   epsilon_decay_episodes=20, shaping=True)
            steps += sum(h["steps"] for h in hist)
        log["dqn"].append({"timesteps": steps, "elapsed_s": round(time.time() - t0, 1),
                           **_eval(DQNActor(dqn), sets, cfg)})
        print("[bench] dqn", log["dqn"][-1], flush=True)

    with open(out / "agent_bench.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "seed": seed, "log": log}, f, indent=2)
    return out
