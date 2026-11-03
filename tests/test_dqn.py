"""Minimal DQN smoke tests (CPU)."""

import numpy as np

from levelforge.dqn import DQNAgent, flatten_obs, state_dim
from levelforge.grid import BOX_NORMAL, Level
from levelforge.environment import SokobanEnv


def tiny_level():
    walls = np.zeros((5, 6), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    return Level(walls, goals=[(4, 2)], boxes=[(2, 2, BOX_NORMAL)], agent=(1, 2), energy=10)


def test_state_dim_and_flatten():
    level = tiny_level()
    obs = level.encode()
    vec = flatten_obs(obs, level.energy)
    assert vec.shape == (state_dim(5, 6),)


def test_agent_selects_valid_action():
    agent = DQNAgent(5, 6, seed=0)
    vec = flatten_obs(tiny_level().encode(), 10)
    assert agent.select_action(vec) in (0, 1, 2, 3)
    assert agent.select_action(vec, greedy=True) in (0, 1, 2, 3)


def test_train_step_runs_after_enough_samples():
    agent = DQNAgent(5, 6, seed=0, batch_size=8)
    env = SokobanEnv(tiny_level(), max_steps=20)
    for _ in range(16):
        obs, _ = env.reset()
        done = False
        while not done:
            vec = flatten_obs(obs, env.level.energy)
            a = agent.select_action(vec)
            next_obs, r, done, _ = env.step(a)
            agent.buffer.push(vec, a, r, flatten_obs(next_obs, env.level.energy), done)
            obs = next_obs
    loss = agent.train_step()
    assert loss >= 0.0
