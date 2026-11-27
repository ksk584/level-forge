"""Learning from players: recording, demonstration extraction, behaviour-cloning update."""

import numpy as np
import pytest
import torch

from levelforge.grid import Level
from levelforge.population import random_solvable_level
from levelforge.solver import solve

pytest.importorskip("stable_baselines3")
from levelforge.player_learning import (  # noqa: E402
    demonstrations, learn_from_players, load_games, load_model, record_game)
from levelforge.ppo_agent import PPOAgent  # noqa: E402


def _player_game(seed):
    level = random_solvable_level(np.random.default_rng(seed), size=7, n_boxes=1, heavy_prob=0, fragile_prob=0)
    moves, _n, _t = solve(level)
    return level, moves


def test_record_learn_save_reload(tmp_path):
    level, moves = _player_game(1)
    path = tmp_path / "games.jsonl"
    record_game(path, level, moves)
    small = Level.from_ascii("#####\n#@n.#\n#####")                    # other board size: skipped
    record_game(path, small, [3])
    games = load_games(path)
    assert len(games) == 2

    obs, acts, used = demonstrations(games, (7, 7))
    assert used == 1 and len(acts) == len(moves) and obs.shape == (len(moves), 7, 7, 7)

    model = PPOAgent([level], seed=0, device="cpu", n_envs=1, n_steps=32, batch_size=32).model
    x, y = torch.as_tensor(obs), torch.as_tensor(acts)

    def player_logp():
        with torch.no_grad():
            return model.policy.evaluate_actions(x, y)[1].mean().item()

    before = player_logp()
    info = learn_from_players(model, games, epochs=20)
    assert info["games"] == 1 and info["moves"] == len(moves)
    assert player_logp() > before                                     # policy moved towards the player

    model.save(str(tmp_path / "model.zip"))
    again = load_model(tmp_path / "model.zip")
    with torch.no_grad():
        assert again.policy.evaluate_actions(x, y)[1].mean().item() == pytest.approx(player_logp(), abs=1e-5)


def test_wall_bumps_are_not_imitated_and_unsolved_games_are_ignored():
    level, moves = _player_game(2)
    bump = next(a for a in range(4) if _blocked(level, a))
    games = [{"level": level.to_dict(), "actions": [bump] + moves},
             {"level": level.to_dict(), "actions": moves[:-1]}]          # never finished
    _obs, acts, used = demonstrations(games, (7, 7))
    assert used == 1 and list(acts) == list(moves)


def _blocked(level, action):
    from levelforge.environment import SokobanEnv

    env = SokobanEnv(level)
    before = (env.level.agent, tuple(env.level.boxes))
    env.step(action)
    return (env.level.agent, tuple(env.level.boxes)) == before
