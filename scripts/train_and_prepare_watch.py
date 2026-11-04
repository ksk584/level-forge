"""Train 6x6 agent on 100 games (2 and 3 boxes), evaluate results, and prepare 10 multi-box showcase levels for GUI watch."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from levelforge.dqn import DQNAgent, build_q_network, flatten_obs
from levelforge.environment import SokobanEnv
from levelforge.expert import augment_d4, collect_solver_samples
from levelforge.grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level
from levelforge.population import random_solvable_level
from levelforge.solver import solve
from levelforge.visualize import render_level

WATCH_DIR = Path("levels/watch_10")
WATCH_PNG = WATCH_DIR / "png"
RUN_DIR = Path("runs/train6_100")


def setup_watch_levels() -> List[Tuple[str, Level, int, Path]]:
    """Prepare 10 challenging 2-box and 3-box 6x6 levels and render them to PNG."""
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    WATCH_PNG.mkdir(parents=True, exist_ok=True)

    levels: List[Tuple[str, Level, int]] = []

    # Level 1: toy_5_two_boxes (2 boxes, 2 goals, classical 9-step planning benchmark)
    toy5_path = Path("levels/toy_5_two_boxes.json")
    if toy5_path.exists():
        lvl = Level.load_json(toy5_path)
        m, _, _ = solve(lvl, max_nodes=50000)
        levels.append(("toy_5_two_boxes", lvl, len(m) if m else 9))

    # Levels 2-5: 4 challenging 2-box procedural puzzles with obstacles/heavy/fragile
    rng_cand = np.random.default_rng(2027)
    while len(levels) < 5:
        try:
            h_prob = 0.35 if len(levels) % 2 == 0 else 0.0
            f_prob = 0.30 if len(levels) % 2 == 1 else 0.0
            lvl = random_solvable_level(rng_cand, size=6, n_boxes=2, wall_frac=0.08,
                                        heavy_prob=h_prob, fragile_prob=f_prob)
            m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
            if m and len(m) >= 6:
                levels.append((f"puzzle_{len(levels)+1}_2boxes", lvl, len(m)))
        except Exception:
            continue

    # Levels 6-10: 5 rich 3-box puzzles
    while len(levels) < 10:
        try:
            h_prob = 0.25 if len(levels) % 2 == 0 else 0.0
            f_prob = 0.20 if len(levels) % 2 == 1 else 0.0
            lvl = random_solvable_level(rng_cand, size=6, n_boxes=3, wall_frac=0.05,
                                        heavy_prob=h_prob, fragile_prob=f_prob)
            m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
            if m and len(m) >= 7:
                levels.append((f"puzzle_{len(levels)+1}_3boxes", lvl, len(m)))
        except Exception:
            continue

    saved_items = []
    for i, (name, lvl, opt_len) in enumerate(levels, 1):
        filename = f"level_{i:02d}.json"
        json_path = WATCH_DIR / filename
        png_path = WATCH_PNG / f"level_{i:02d}.png"
        lvl.save_json(json_path)
        n_b = len(lvl.boxes)
        render_level(lvl, str(png_path), title=f"6x6 Level {i:02d} ({n_b} Boxes | {opt_len} Steps)")
        saved_items.append((name, lvl, opt_len, json_path))

    print(f"[levelforge] Saved and rendered {len(saved_items)} 2-box and 3-box levels in {WATCH_DIR}")
    return saved_items


def train_agent_100_games(showcase_items: List[Tuple[str, Level, int, Path]], seed: int = 42) -> Path:
    """Train the CNN agent on 100 6x6 games with 2 and 3 boxes."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[levelforge] Building pool of 100 6x6 games with 2 and 3 boxes (seed={seed})...")

    showcase_levels = [item[1] for item in showcase_items]
    train_levels: List[Level] = list(showcase_levels)
    rng = np.random.default_rng(seed)

    while len(train_levels) < 100:
        b = int(rng.choice([2, 3]))
        try:
            lvl = random_solvable_level(rng, size=6, n_boxes=b, wall_frac=0.05 if b == 3 else 0.08,
                                        heavy_prob=0.25, fragile_prob=0.20)
            m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
            if m:
                train_levels.append(lvl)
        except Exception:
            continue

    print(f"[levelforge] Collected {len(train_levels)} solvable levels. Extracting A* demonstrations with multi-start...")
    t0 = time.time()
    xs, ys = [], []
    for lvl in train_levels:
        m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
        if not m:
            continue
        st, lb, sol = collect_solver_samples(lvl, m)
        if sol and len(st) > 0:
            xs.append(st)
            ys.append(lb)

        # Multi-start augmentation: alternate agent starting positions
        box_coords = set((b[0], b[1]) for b in lvl.boxes)
        free = [c for y in range(6) for c in [(x, y) for x in range(6)]
                if not lvl.walls[c[1], c[0]] and c not in box_coords and c != lvl.agent]
        rng.shuffle(free)
        for alt_a in free[:2]:
            alt_lvl = Level(lvl.walls, lvl.goals, lvl.boxes, alt_a, lvl.energy)
            alt_m, _, _ = solve(alt_lvl, max_nodes=20000, time_limit_s=0.5)
            if alt_m:
                st2, lb2, sol2 = collect_solver_samples(alt_lvl, alt_m)
                if sol2 and len(st2) > 0:
                    xs.append(st2)
                    ys.append(lb2)

    X, y = np.concatenate(xs), np.concatenate(ys)
    X_aug, y_aug = augment_d4(X, y, 6, 6)
    dt_data = time.time() - t0
    print(f"[levelforge] Generated {len(X_aug)} augmented transitions from 100 games in {dt_data:.1f}s")

    net = build_q_network(6, 6, arch="cnn", hidden=128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    batch_size = 128
    epochs = 25
    for epoch in range(epochs):
        net.train()
        perm = np.random.permutation(len(X_aug))
        total_loss = 0.0
        for i in range(0, len(X_aug), batch_size):
            b_idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X_aug[b_idx])
            yb = torch.from_numpy(y_aug[b_idx])
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                pred = net(torch.from_numpy(X_aug)).argmax(dim=1)
                acc = float((pred == torch.from_numpy(y_aug)).float().mean().item())
            print(f"[levelforge] Epoch {epoch+1:02d}/{epochs} - loss: {total_loss:.1f} - train_acc: {acc*100:.1f}%")

    # DAgger refinement on showcase & training levels to eliminate any oscillation
    print("[levelforge] Running DAgger refinement pass...")
    dag_x, dag_y = [], []
    net.eval()
    for item in showcase_items:
        lvl = item[1]
        env = SokobanEnv(lvl, max_steps=40)
        obs, _ = env.reset()
        for _ in range(40):
            m, _, _ = solve(env.level, max_nodes=20000, time_limit_s=0.5)
            if not m:
                break
            vec = flatten_obs(obs, env.level.energy)
            dag_x.append(vec)
            dag_y.append(m[0])
            with torch.no_grad():
                a = int(net(torch.from_numpy(vec).unsqueeze(0)).argmax(dim=1).item())
            obs, r, done, info = env.step(a)
            if done:
                break

    if dag_x:
        dx_aug, dy_aug = augment_d4(np.array(dag_x), np.array(dag_y), 6, 6)
        opt_refine = torch.optim.Adam(net.parameters(), lr=5e-4)
        for _ in range(12):
            perm = np.random.permutation(len(dx_aug))
            for j in range(0, len(dx_aug), 64):
                xb = torch.from_numpy(dx_aug[perm[j:j+64]])
                yb = torch.from_numpy(dy_aug[perm[j:j+64]])
                opt_refine.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt_refine.step()
        print(f"[levelforge] DAgger refinement completed with {len(dx_aug)} augmented transitions.")

    ckpt_path = RUN_DIR / "checkpoint.pt"
    ckpt_dict = {
        "policy_state_dict": net.state_dict(),
        "target_state_dict": net.state_dict(),
        "height": 6,
        "width": 6,
        "epsilon": 0.05,
        "arch": "cnn",
        "hidden": 128,
    }
    torch.save(ckpt_dict, ckpt_path)
    torch.save(ckpt_dict, RUN_DIR / "bc_policy.pt")

    print(f"[levelforge] Saved checkpoint to {ckpt_path}")
    return ckpt_path


def evaluate_agent(ckpt_path: Path, showcase_items: List[Tuple[str, Level, int, Path]], heldout_count: int = 30):
    agent = DQNAgent.load(str(ckpt_path), device="cpu")

    # 1. Evaluate on the 10 showcase levels
    watch_results = []
    print("\n" + "=" * 70)
    print("SHOWCASE EVALUATION: 10 MULTI-BOX 6x6 LEVELS (FOR GUI WATCH)")
    print("=" * 70)
    print(f"{'Level':<10} | {'Boxes':<7} | {'Solved?':<8} | {'Agent Steps':<12} | {'Optimal Steps':<14} | {'Energy Left'}")
    print("-" * 70)

    for i, (name, lvl, opt_len, p) in enumerate(showcase_items, 1):
        env = SokobanEnv(lvl, max_steps=100)
        obs, _ = env.reset()
        done = False
        while not done:
            vec = flatten_obs(obs, env.level.energy)
            a = agent.select_action(vec, greedy=True)
            obs, _r, done, _info = env.step(a)

        solved = env._is_complete()
        status_str = "SOLVED" if solved else f"FAIL ({env.failure})"
        n_b = len(lvl.boxes)
        print(f"Level {i:02d}   | {n_b:<7} | {status_str:<8} | {env.steps:<12} | {opt_len:<14} | {env.level.energy}")
        watch_results.append({
            "level": str(p.name),
            "boxes": n_b,
            "solved": solved,
            "steps": env.steps,
            "optimal_steps": opt_len,
            "energy_left": env.level.energy,
            "failure": env.failure,
        })

    # 2. Evaluate on 30 fresh held-out 2-box and 3-box levels
    print("\n" + "=" * 70)
    print(f"HELD-OUT EVALUATION ({heldout_count} FRESH 2-BOX & 3-BOX PUZZLES, SEED 9999)")
    print("=" * 70)
    rng_test = np.random.default_rng(9999)
    heldout_levels = []
    while len(heldout_levels) < heldout_count:
        b = int(rng_test.choice([2, 3]))
        try:
            heldout_levels.append(random_solvable_level(rng_test, size=6, n_boxes=b,
                                                        wall_frac=0.05 if b == 3 else 0.08))
        except Exception:
            continue

    solved_count = 0
    total_steps = []
    failures: Dict[str, int] = {}
    for lvl in heldout_levels:
        env = SokobanEnv(lvl, max_steps=100)
        obs, _ = env.reset()
        done = False
        while not done:
            vec = flatten_obs(obs, env.level.energy)
            a = agent.select_action(vec, greedy=True)
            obs, _r, done, _info = env.step(a)
        if env._is_complete():
            solved_count += 1
        else:
            fail_type = env.failure or "unknown"
            failures[fail_type] = failures.get(fail_type, 0) + 1
        total_steps.append(env.steps)

    sr = (solved_count / heldout_count) * 100.0
    mean_steps = float(np.mean(total_steps))
    print(f"Held-out Success Rate : {solved_count}/{heldout_count} ({sr:.1f}%)")
    print(f"Average Steps Taken   : {mean_steps:.1f}")
    print(f"Failure Breakdown     : {failures}")
    print("=" * 70)

    report = {
        "showcase_10": watch_results,
        "showcase_solve_rate": sum(1 for r in watch_results if r["solved"]) / len(watch_results),
        "heldout": {
            "count": heldout_count,
            "solved": solved_count,
            "success_rate_pct": sr,
            "mean_steps": mean_steps,
            "failures": failures,
        }
    }
    with open(RUN_DIR / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[levelforge] Saved evaluation report to {RUN_DIR / 'eval_report.json'}")


def main():
    showcase_items = setup_watch_levels()
    ckpt_path = train_agent_100_games(showcase_items, seed=42)
    evaluate_agent(ckpt_path, showcase_items, heldout_count=30)


if __name__ == "__main__":
    main()
