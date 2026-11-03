"""Headless agent replay: prints board frames to the terminal.

Used when the Qt window is unsuitable (no desktop session, or a crashing
viewer).  Usage:  python -m levelforge watch --level L.json --checkpoint C.pt --text
"""

from __future__ import annotations

import os
import time

from .dqn import DQNAgent, flatten_obs
from .environment import SokobanEnv
from .grid import Level


def run_text(level: Level, checkpoint: str, delay_ms: int = 250, max_steps: int = 500,
             clear_screen: bool = True, explore: float = 0.0, seed: int = 0) -> str:
    try:
        agent = DQNAgent.load(checkpoint, device="cpu")
    except Exception as e:
        print(f"[levelforge] could not load checkpoint: {checkpoint}\n           {e}")
        raise SystemExit(2) from e
    agent.epsilon = max(0.0, min(1.0, explore))

    env = SokobanEnv(level, max_steps=max_steps)
    obs, _ = env.reset()
    try:
        import random
        random.seed(seed)
        while not env.done:
            if clear_screen:
                os.system("cls" if os.name == "nt" else "clear")
            print(env.render_ascii())
            vec = flatten_obs(obs, env.level.energy)
            action = agent.select_action(vec, greedy=(explore <= 0.0))
            obs, _r, _done, _info = env.step(action)
            time.sleep(max(0.0, delay_ms / 1000.0))
    except KeyboardInterrupt:
        print("\n[levelforge] interrupted by user")
        outcome = "INTERRUPTED"
    else:
        outcome = "SOLVED" if env.failure is None else f"FAILED ({env.failure})"
    print(env.render_ascii())
    print(f"[levelforge] replay finished: {outcome} after {env.steps} steps")
    return outcome
