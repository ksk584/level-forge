"""Tkinter viewer — zero additional dependencies (stdlib).

Launch:
    python -m levelforge play  --level L.json --tk                     (you play)
    python -m levelforge watch --level L.json --checkpoint C.pt --tk    (agent replay)

Play mode has three buttons: Player (you), RL agent (the PPO policy) and A* solver; the
agents continue from the current position.  Puzzles you solve unaided are saved and the RL
agent learns from them (``player_learning.py``).

Controls (both modes): Arrows/WASD move (human) | R restart | Esc quit
Agent mode: Space pause | +/- speed | R restart
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

CELL = 44

_COLORS = {
    "wall": "#111827",
    "floor": "#f9fafb",
    "rough": "#d6b98c",
    "goal": "#16a34a",
    "agent": "#2563eb",
    f"box_{BOX_NORMAL}": "#c08457",
    f"box_{BOX_HEAVY}": "#6b7280",
    f"box_{BOX_FRAGILE}": "#93c5fd",
}
_LABEL = {BOX_HEAVY: "H", BOX_FRAGILE: "F"}
_DIR = {"Up": 0, "w": 0, "Down": 1, "s": 1, "Left": 2, "a": 2, "Right": 3, "d": 3}


class _PPOActor:
    """Drives the viewer with a loaded PPO model (sampled actions, as in evaluation, unless deterministic)."""

    def __init__(self, model, deterministic: bool = False):
        self.model, self.deterministic = model, deterministic

    def act(self, level: Level) -> int:
        from .gym_env import encode_obs

        action, _ = self.model.predict(encode_obs(level)[None], deterministic=self.deterministic)
        return int(action[0])


class _SolverActor:
    def __init__(self, moves: list[int]):
        self._moves = list(moves)

    def select_action(self, _vec=None, greedy: bool = True):
        return self._moves.pop(0)

    @property
    def has_moves(self) -> bool:
        return bool(self._moves)


def _run(level: Level, checkpoint: str | None, delay_ms: int, explore: float, seed: int,
         solver_moves: list[int] | None = None, ppo_model: str | None = None,
         deterministic: bool = False, max_steps: int | None = None,
         level_factory: Callable[[], Level] | None = None, switchable: bool = False,
         learn_dir: str | None = None):
    try:
        import tkinter as tk
    except ImportError as e:  # pragma: no cover
        print("[levelforge] tkinter is not available in this Python; use --text instead.")
        raise SystemExit(2) from e

    from .environment import SokobanEnv
    from .dqn import DQNAgent, flatten_obs

    if solver_moves is not None:
        class SolverAgent:
            def __init__(self, moves):
                self._moves = list(moves)
                self.epsilon = 0.0
            def select_action(self, vec, greedy: bool = True):
                return self._moves.pop(0) if self._moves else 0
        agent = SolverAgent(solver_moves)
        mode = "A* solver replay"
    elif ppo_model and not switchable:
        import torch

        from .player_learning import load_model

        torch.manual_seed(seed)
        agent = _PPOActor(load_model(ppo_model), deterministic)
        mode = f"PPO agent: {ppo_model}"
    else:
        agent = None
        if checkpoint:
            import random
            try:
                agent = DQNAgent.load(checkpoint, device="cpu")
            except Exception as e:
                print(f"[levelforge] could not load checkpoint: {checkpoint}\n           {e}")
                raise SystemExit(2) from e
            agent.epsilon = max(0.0, min(1.0, explore))
            agent.rng = random.Random(seed)  # seeds --explore sampling
            mode = "DQN replay"
        else:
            mode = "You are playing"
    max_steps = max_steps or (500 if agent else 10**9)
    state = {
        "level": level,
        "agent": agent,
        "env": SokobanEnv(level, max_steps=max_steps),
        "delay": max(30, int(delay_ms)),
        "paused": False,
        "obs": None,
        "after_id": None,
        "notice": "",
        "mode": "player",       # play mode: player | rl | astar
        "model": None,          # PPO model, loaded on first use
        "moves": [],            # the player's actions since the level (re)started
        "assisted": False,      # an agent moved during this attempt -> not recorded
    }

    root = tk.Tk()
    root.title(f"LevelForge — {mode}")
    root.resizable(False, False)
    width, height = level.width * CELL, level.height * CELL
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0)
    canvas.pack()
    status = tk.Label(root, text="", anchor="w", bg="#0f172a", fg="#e2e8f0",
                      font=("Segoe UI", 11), padx=10, pady=6)
    status.pack(fill="x")
    controls = tk.Frame(root)
    controls.pack(fill="x")
    if level_factory is not None:
        tk.Button(controls, text="New random puzzle", command=lambda: new_level()).pack(side="left", padx=6, pady=6)
    mode_buttons = {}
    if switchable:
        for key, text in (("player", "Player"), ("rl", "RL agent"), ("astar", "A* solver")):
            mode_buttons[key] = tk.Button(controls, text=text, command=lambda k=key: set_mode(k))
            mode_buttons[key].pack(side="left", padx=6, pady=6)

    def statustext():
        env = state["env"]
        lv = env.level
        if env.done:
            outcome = "SOLVED!" if env.failure is None else f"FAILED ({env.failure})"
            text = f"{outcome} after {env.steps} steps   |   R restart   Esc quit"
        else:
            current_agent = state["agent"]
            extra = (f"   |   speed {state['delay']} ms" if current_agent
                 else "   |   WASD/arrows move, R reset, Esc quit")
            text = f"energy={lv.energy}   boxes on goals={lv.boxes_on_goals()}/{env.box_total}   step={env.steps}{extra}"
        if state["notice"]:
            text += f"   |   {state['notice']}"
        return text

    def draw():
        env = state["env"]
        lv = env.level
        canvas.delete("all")
        goals = set(lv.goals)
        rough = set(lv.rough)
        box_map = {(x, y): k for x, y, k in lv.boxes}
        for y in range(lv.height):
            for x in range(lv.width):
                x0, y0, x1, y1 = x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL
                if lv.walls[y, x]:
                    canvas.create_rectangle(x0, y0, x1, y1, fill=_COLORS["wall"], outline="#374151")
                    continue
                canvas.create_rectangle(x0, y0, x1, y1, outline="#e5e7eb",
                                        fill=_COLORS["rough" if (x, y) in rough else "floor"])
                if (x, y) in rough:  # gravel dots: fragile boxes break here
                    for dx, dy in ((0.25, 0.3), (0.6, 0.2), (0.4, 0.65), (0.75, 0.7), (0.2, 0.8)):
                        cx, cy = x0 + dx * CELL, y0 + dy * CELL
                        canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#8b6b3e", outline="")
                if (x, y) in goals:
                    r = CELL // 5
                    cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
                    canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=_COLORS["goal"], outline="")
                if (x, y) in box_map:
                    kind = box_map[(x, y)]
                    pad = 6
                    canvas.create_rectangle(x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                                            fill=_COLORS[f"box_{kind}"], outline="#1f2937", width=2)
                    label = _LABEL.get(kind)
                    if label:
                        canvas.create_text(x * CELL + CELL // 2, y * CELL + CELL // 2, text=label,
                                           fill="white", font=("Segoe UI", 16, "bold"))
        ax_, ay_ = lv.agent
        pad = 3
        canvas.create_oval(ax_ * CELL + pad, ay_ * CELL + pad,
                           (ax_ + 1) * CELL - pad, (ay_ + 1) * CELL - pad,
                           fill=_COLORS["agent"], outline="#1e3a8a", width=2)
        status.config(text=statustext())
        for key, button in mode_buttons.items():
            button.config(relief="sunken" if state["mode"] == key else "raised")

    def cancel_agent_tick():
        if state["after_id"] is not None:
            root.after_cancel(state["after_id"])
            state["after_id"] = None

    def start_solver_agent():
        from .solver import plan_energy, solve

        current_level = state["env"].level.copy()
        moves, _nodes, _elapsed = solve(current_level, max_nodes=500_000, time_limit_s=10.0)
        if moves is None:
            state["notice"] = "No solution found from this position"
            draw()
            return
        if plan_energy(current_level, moves) > current_level.energy:
            state["notice"] = "Not enough energy for the agent to finish"
            draw()
            return
        if len(moves) + state["env"].steps > state["env"].max_steps:
            state["notice"] = "Agent solution exceeds the step limit"
            draw()
            return
        state["agent"] = _SolverActor(moves)
        state["mode"] = "astar"
        state["notice"] = f"A* solving in {len(moves)} moves"
        state["obs"] = None
        state["paused"] = False
        draw()
        state["after_id"] = root.after(state["delay"], agent_tick)

    def get_model():
        if state["model"] is None and ppo_model and Path(ppo_model).exists():
            from .player_learning import load_model
            state["model"] = load_model(ppo_model)
        return state["model"]

    def model_fits(model) -> bool:
        return tuple(model.observation_space.shape[1:]) == state["env"].level.walls.shape

    def start_rl_agent():
        model = get_model()
        if model is None:
            state["notice"] = "No RL model found: pass --ppo <model.zip>"
        elif not model_fits(model):
            h, w = model.observation_space.shape[1:]
            state["notice"] = f"RL agent was trained on {w}x{h} boards only"
        else:
            state["agent"] = _PPOActor(model)
            state["mode"] = "rl"
            state["notice"] = "RL agent playing"
            state["paused"] = False
            state["after_id"] = root.after(state["delay"], agent_tick)
        draw()

    def set_mode(mode):
        cancel_agent_tick()
        state["agent"] = None
        state["paused"] = False
        state["mode"] = "player"
        if mode == "player":
            state["notice"] = "Player mode"
        elif state["env"].done:
            state["notice"] = "Puzzle is over: press R to restart"
        else:
            state["notice"] = "A* planning..." if mode == "astar" else "Loading RL agent..."
            draw()
            root.update_idletasks()
            start_solver_agent() if mode == "astar" else start_rl_agent()
            return
        draw()

    def on_player_solved():
        from .player_learning import learn_from_players, load_games, record_game

        games_path = Path(learn_dir) / "games.jsonl"
        record_game(games_path, state["level"], state["moves"])
        model = get_model()
        if model is None:
            state["notice"] = f"Game saved to {games_path} (no RL model loaded to learn from it)"
            return
        if not model_fits(model):
            state["notice"] = "Game saved (the RL agent only learns from boards of its own size)"
            return
        state["notice"] = "Game saved - RL agent is learning from your solve..."
        draw()
        root.update_idletasks()
        info = learn_from_players(model, load_games(games_path))
        model.save(str(Path(learn_dir) / "model.zip"))
        state["notice"] = f"RL agent learned from {info['games']} player game(s)"

    def new_level():
        if level_factory is None:
            return
        cancel_agent_tick()
        state["level"] = level_factory()
        state["env"] = SokobanEnv(state["level"], max_steps=max_steps)
        new_width = state["level"].width * CELL
        new_height = state["level"].height * CELL
        canvas.configure(width=new_width, height=new_height)
        root.update_idletasks()
        root.geometry(
            f"+{root.winfo_screenwidth() // 2 - new_width // 2}"
            f"+{root.winfo_screenheight() // 2 - new_height // 2}"
        )
        state["agent"] = None
        state["mode"] = "player"
        state["moves"], state["assisted"] = [], False
        state["obs"] = None
        state["paused"] = False
        state["notice"] = "New random puzzle"
        draw()

    def agent_tick():
        env = state["env"]
        current_agent = state["agent"]
        state["after_id"] = None
        if current_agent is None:
            return
        if env.done:
            draw()
            return
        if state["paused"]:
            state["after_id"] = root.after(state["delay"], agent_tick)
            return
        if isinstance(current_agent, _SolverActor):
            if not current_agent.has_moves:
                state["agent"] = None
                state["mode"] = "player"
                state["notice"] = "Agent stopped before solving"
                draw()
                return
            action = current_agent.select_action()
        elif isinstance(current_agent, _PPOActor):
            action = current_agent.act(env.level)
        else:
            vec = flatten_obs(state["obs"], env.level.energy)
            action = current_agent.select_action(vec, greedy=(explore <= 0.0))
        state["assisted"] = True
        state["obs"], _r, _done, _info = env.step(action)
        state["notice"] = ""
        draw()
        if not env.done and state["agent"] is not None:
            state["after_id"] = root.after(state["delay"], agent_tick)

    def restart():
        current_agent = state["agent"]
        cancel_agent_tick()
        state["env"] = SokobanEnv(state["level"], max_steps=max_steps)
        state["notice"] = ""
        state["moves"], state["assisted"] = [], False
        if switchable and state["mode"] != "player":
            state["agent"] = None
            start_solver_agent() if state["mode"] == "astar" else start_rl_agent()
        elif current_agent:
            state["obs"], _ = state["env"].reset()
            state["after_id"] = root.after(state["delay"], agent_tick)
        draw()

    def on_key(event):
        key = event.keysym
        if key == "Escape":
            root.destroy()
            return
        if key in ("r", "R"):
            restart()
            return
        if state["agent"]:
            if key == "space":
                state["paused"] = not state["paused"]
                draw()
                return
            if key in ("plus", "equal", "KP_Add"):
                state["delay"] = max(30, int(state["delay"] / 1.5)); draw(); return
            if key in ("minus", "underscore", "KP_Subtract"):
                state["delay"] = min(5000, int(state["delay"] * 1.5)); draw(); return
            return
        if key in _DIR and not state["env"].done:
            env = state["env"]
            state["notice"] = ""
            env.step(_DIR[key])
            state["moves"].append(_DIR[key])
            if switchable and env.done and env.failure is None and not state["assisted"]:
                on_player_solved()
            draw()

    root.bind("<Key>", on_key)
    root.update_idletasks()
    root.geometry(f"+{root.winfo_screenwidth() // 2 - width // 2}+{root.winfo_screenheight() // 2 - height // 2}")
    draw()
    if state["agent"]:
        state["obs"], _ = state["env"].reset()
        state["after_id"] = root.after(state["delay"], agent_tick)
    root.mainloop()


def run_solver(level: Level, moves: list[int], delay_ms: int = 250):
    _run(level, checkpoint=None, delay_ms=delay_ms, explore=0.0, seed=0, solver_moves=moves)


def run(level: Level):
    _run(level, checkpoint=None, delay_ms=0, explore=0.0, seed=0)


def run_game(level: Level, generate_next: Callable[[], Level], delay_ms: int = 250,
             ppo_model: str | None = None, learn_dir: str = "runs/player_learning"):
    _run(level, checkpoint=None, delay_ms=delay_ms, explore=0.0, seed=0, ppo_model=ppo_model,
         max_steps=500, level_factory=generate_next, switchable=True, learn_dir=learn_dir)


def run_agent(level: Level, checkpoint: str, delay_ms: int = 350, explore: float = 0.0, seed: int = 0):
    _run(level, checkpoint=checkpoint, delay_ms=delay_ms, explore=explore, seed=seed)


def run_ppo(level: Level, model: str, delay_ms: int = 250, deterministic: bool = False, seed: int = 0,
            max_steps: int = 120):
    _run(level, checkpoint=None, delay_ms=delay_ms, explore=0.0, seed=seed, ppo_model=model,
         deterministic=deterministic, max_steps=max_steps)
