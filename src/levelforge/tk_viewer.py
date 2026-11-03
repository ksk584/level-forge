"""Tkinter viewer — zero additional dependencies (stdlib).

Launch:
    python -m levelforge play  --level L.json --tk                     (you play)
    python -m levelforge watch --level L.json --checkpoint C.pt --tk    (agent replay)

Controls (both modes): Arrows/WASD move (human) | R restart | Esc quit
Agent mode: Space pause | +/- speed | R restart
"""

from __future__ import annotations

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

CELL = 44

_COLORS = {
    "wall": "#111827",
    "floor": "#f9fafb",
    "goal": "#16a34a",
    "agent": "#2563eb",
    f"box_{BOX_NORMAL}": "#c08457",
    f"box_{BOX_HEAVY}": "#6b7280",
    f"box_{BOX_FRAGILE}": "#93c5fd",
}
_LABEL = {BOX_HEAVY: "H", BOX_FRAGILE: "F"}
_DIR = {"Up": 0, "w": 0, "Down": 1, "s": 1, "Left": 2, "a": 2, "Right": 3, "d": 3}


def _run(level: Level, checkpoint: str | None, delay_ms: int, explore: float, seed: int):
    try:
        import tkinter as tk
    except ImportError as e:  # pragma: no cover
        print("[levelforge] tkinter is not available in this Python; use --text instead.")
        raise SystemExit(2) from e

    from .environment import SokobanEnv
    from .dqn import DQNAgent, flatten_obs

    agent = None
    if checkpoint:
        import random
        try:
            agent = DQNAgent.load(checkpoint, device="cpu")
        except Exception as e:
            print(f"[levelforge] could not load checkpoint: {checkpoint}\n           {e}")
            raise SystemExit(2) from e
        agent.epsilon = max(0.0, min(1.0, explore))
        random.seed(seed)

    mode = "DQN replay" if agent else "You are playing"
    max_steps = 500 if agent else 10**9
    state = {
        "env": SokobanEnv(level, max_steps=max_steps),
        "delay": max(30, int(delay_ms)),
        "paused": False,
        "obs": None,
        "after_id": None,
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

    def statustext():
        env = state["env"]
        lv = env.level
        if env.done:
            outcome = "SOLVED!" if env.failure is None else f"FAILED ({env.failure})"
            return f"{outcome} after {env.steps} steps   |   R restart   Esc quit"
        extra = (f"   |   speed {state['delay']} ms" if agent
                 else "   |   WASD/arrows move, R reset, Esc quit")
        return f"energy={lv.energy}   boxes on goals={lv.boxes_on_goals()}/{env.box_total}   step={env.steps}{extra}"

    def draw():
        env = state["env"]
        lv = env.level
        canvas.delete("all")
        goals = set(lv.goals)
        box_map = {(x, y): k for x, y, k in lv.boxes}
        for y in range(lv.height):
            for x in range(lv.width):
                x0, y0, x1, y1 = x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL
                if lv.walls[y, x]:
                    canvas.create_rectangle(x0, y0, x1, y1, fill=_COLORS["wall"], outline="#374151")
                    continue
                canvas.create_rectangle(x0, y0, x1, y1, fill=_COLORS["floor"], outline="#e5e7eb")
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

    def agent_tick():
        env = state["env"]
        if env.done:
            draw()
            return
        if state["paused"]:
            state["after_id"] = root.after(state["delay"], agent_tick)
            return
        vec = flatten_obs(state["obs"], env.level.energy)
        action = agent.select_action(vec, greedy=(explore <= 0.0))
        state["obs"], _r, _done, _info = env.step(action)
        draw()
        state["after_id"] = root.after(state["delay"], agent_tick)

    def restart():
        state["env"] = SokobanEnv(level, max_steps=max_steps)
        if agent:
            state["obs"], _ = state["env"].reset()
            if state["after_id"] is not None:
                root.after_cancel(state["after_id"])
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
        if agent:
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
            state["env"].step(_DIR[key])
            draw()

    root.bind("<Key>", on_key)
    root.update_idletasks()
    root.geometry(f"+{root.winfo_screenwidth() // 2 - width // 2}+{root.winfo_screenheight() // 2 - height // 2}")
    draw()
    if agent:
        state["obs"], _ = state["env"].reset()
        state["after_id"] = root.after(state["delay"], agent_tick)
    root.mainloop()


def run(level: Level):
    _run(level, checkpoint=None, delay_ms=0, explore=0.0, seed=0)


def run_agent(level: Level, checkpoint: str, delay_ms: int = 350, explore: float = 0.0, seed: int = 0):
    _run(level, checkpoint=checkpoint, delay_ms=delay_ms, explore=explore, seed=seed)
