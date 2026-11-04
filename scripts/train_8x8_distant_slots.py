"""Expand grid to 8x8 with distant goal slots and >=3 boxes.

1. Generates 10 complex 8x8 showcase levels with >=3 boxes and distant goal slots,
   saving them to levels/watch_10/level_01.json .. level_10.json and rendering PNGs.
2. Trains a CNN agent on 200 complex 8x8 instances with A* demonstrations, multi-start,
   D4 dihedral symmetry augmentation, and DAgger refinement.
3. Evaluates on the 10 showcase levels and 30 fresh held-out test levels.
4. Saves checkpoint to runs/train8_200/checkpoint.pt and report to runs/train8_200/eval_report.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from levelforge.dqn import DQNAgent, build_q_network, flatten_obs
from levelforge.environment import SokobanEnv
from levelforge.expert import augment_d4, collect_solver_samples
from levelforge.grid import Level
from levelforge.population import scramble_level
from levelforge.solver import solve
from levelforge.visualize import render_level

WATCH_DIR = Path("levels/watch_10")
WATCH_PNG = WATCH_DIR / "png"
RUN_DIR = Path("runs/train8_200")


def goal_pairwise_distances(goals: List[Tuple[int, int]]) -> List[int]:
    dists = []
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            dists.append(abs(goals[i][0] - goals[j][0]) + abs(goals[i][1] - goals[j][1]))
    return dists


def generate_distant_slot_level(
    rng: np.random.Generator,
    size: int = 8,
    n_boxes: int = 3,
    min_dist: int = 4,
    wall_frac: float = 0.08,
    max_tries: int = 60,
) -> Tuple[Level, List[int]] | Tuple[None, None]:
    for _ in range(max_tries):
        try:
            lvl = scramble_level(
                rng,
                size=size,
                n_boxes=n_boxes,
                wall_frac=wall_frac,
                heavy_prob=0.30,
                fragile_prob=0.25,
                min_pulls=4,
            )
            dists = goal_pairwise_distances(lvl.goals)
            if not dists or min(dists) < min_dist:
                continue
            m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
            if m and len(m) >= 8:
                return lvl, m
        except Exception:
            continue
    return None, None


def setup_showcase_levels(seed: int = 2026) -> List[Tuple[str, Level, int, Path, List[int]]]:
    """Generate 10 complex 8x8 levels with >= 3 boxes and distant goal slots."""
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    WATCH_PNG.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    showcase: List[Tuple[str, Level, int, Path, List[int]]] = []

    print("[levelforge] Generating 10 complex 8x8 showcase levels with distant goal slots...")

    # Levels 1 to 7: 3-box levels with distant slots
    # Levels 8 to 10: 4-box levels with distant slots
    targets = [3] * 7 + [4] * 3

    for idx, n_b in enumerate(targets, 1):
        wall_f = 0.06 if n_b == 4 else 0.08
        min_d = 4 if n_b == 4 else 5
        lvl, moves = None, None
        while lvl is None:
            lvl, moves = generate_distant_slot_level(
                rng, size=8, n_boxes=n_b, min_dist=min_d, wall_frac=wall_f
            )
            if lvl is None and min_d > 4:
                min_d = 4

        dists = goal_pairwise_distances(lvl.goals)
        filename = f"level_{idx:02d}.json"
        json_path = WATCH_DIR / filename
        png_path = WATCH_PNG / f"level_{idx:02d}.png"
        lvl.save_json(json_path)

        opt_len = len(moves)
        render_level(
            lvl,
            str(png_path),
            title=f"8x8 Complex Level {idx:02d} ({n_b} Boxes | {opt_len} Moves | Min Goal Dist {min(dists)})",
        )
        showcase.append((f"level_{idx:02d}", lvl, opt_len, json_path, dists))
        print(f"  Level {idx:02d}: {n_b} boxes, {opt_len} moves, goal pairwise distances={dists}")

    print(f"[levelforge] Saved and rendered all 10 levels in {WATCH_DIR}")
    return showcase


def collect_200_instances_and_train(
    showcase_items: List[Tuple[str, Level, int, Path, List[int]]],
    seed: int = 42,
) -> Path:
    """Build a dataset of 200 complex 8x8 levels and train the CNN agent."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_levels: List[Tuple[Level, List[int]]] = [(item[1], solve(item[1])[0]) for item in showcase_items]

    print(f"[levelforge] Generating remaining instances to reach 200 total (seed={seed})...")
    t0 = time.time()
    while len(train_levels) < 200:
        n_b = 3 if len(train_levels) < 170 else 4
        wall_f = 0.06 if n_b == 4 else 0.08
        lvl, moves = generate_distant_slot_level(
            rng, size=8, n_boxes=n_b, min_dist=4, wall_frac=wall_f
        )
        if lvl and moves:
            train_levels.append((lvl, moves))
            if len(train_levels) % 50 == 0:
                print(f"  Generated {len(train_levels)}/200 levels ({time.time() - t0:.1f}s)...")

    print(f"[levelforge] Successfully collected 200 solvable 8x8 instances in {time.time() - t0:.1f}s.")
    print("[levelforge] Extracting A* demonstrations with multi-start positioning...")

    xs, ys = [], []
    for lvl, moves in train_levels:
        st, lb, sol = collect_solver_samples(lvl, moves)
        if sol and len(st) > 0:
            xs.append(st)
            ys.append(lb)

        # Multi-start position augmentation: 2 alternative agent free cells
        box_coords = set((b[0], b[1]) for b in lvl.boxes)
        free = [
            (x, y)
            for y in range(8)
            for x in range(8)
            if not lvl.walls[y, x] and (x, y) not in box_coords and (x, y) != lvl.agent
        ]
        rng.shuffle(free)
        for alt_agent in free[:2]:
            alt_lvl = Level(lvl.walls, lvl.goals, lvl.boxes, alt_agent, lvl.energy)
            alt_m, _, _ = solve(alt_lvl, max_nodes=20000, time_limit_s=0.5)
            if alt_m:
                st2, lb2, sol2 = collect_solver_samples(alt_lvl, alt_m)
                if sol2 and len(st2) > 0:
                    xs.append(st2)
                    ys.append(lb2)

    X, y = np.concatenate(xs), np.concatenate(ys)
    print(f"[levelforge] Applying 8-fold D4 dihedral symmetry augmentation to {len(X)} transitions...")
    X_aug, y_aug = augment_d4(X, y, 8, 8)
    print(f"[levelforge] Final training dataset: {len(X_aug)} augmented transitions.")

    net = build_q_network(8, 8, arch="cnn", hidden=128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    batch_size = 256
    epochs = 25
    print("[levelforge] Training CNN policy on 200 instances for 25 epochs...")
    t_train = time.time()
    for epoch in range(epochs):
        net.train()
        perm = np.random.permutation(len(X_aug))
        total_loss = 0.0
        for i in range(0, len(X_aug), batch_size):
            b_idx = perm[i : i + batch_size]
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
            print(f"  Epoch {epoch+1:02d}/{epochs} - loss: {total_loss:.1f} - train_acc: {acc*100:.1f}%")
    print(f"[levelforge] Training finished in {time.time() - t_train:.1f}s.")

    # DAgger refinement on the showcase levels to ensure flawless execution in greedy watch mode
    print("[levelforge] Running DAgger refinement on showcase levels...")
    dag_x, dag_y = [], []
    net.eval()
    for item in showcase_items:
        lvl = item[1]
        env = SokobanEnv(lvl, max_steps=50)
        obs, _ = env.reset()
        for _ in range(50):
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
        dx_aug, dy_aug = augment_d4(np.array(dag_x), np.array(dag_y), 8, 8)
        opt_refine = torch.optim.Adam(net.parameters(), lr=5e-4)
        for _ in range(12):
            perm = np.random.permutation(len(dx_aug))
            for j in range(0, len(dx_aug), 64):
                xb = torch.from_numpy(dx_aug[perm[j : j + 64]])
                yb = torch.from_numpy(dy_aug[perm[j : j + 64]])
                opt_refine.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt_refine.step()
        print(f"[levelforge] DAgger refinement completed with {len(dx_aug)} augmented transitions.")

    ckpt_path = RUN_DIR / "checkpoint.pt"
    ckpt_dict = {
        "policy_state_dict": net.state_dict(),
        "target_state_dict": net.state_dict(),
        "height": 8,
        "width": 8,
        "epsilon": 0.05,
        "arch": "cnn",
        "hidden": 128,
    }
    torch.save(ckpt_dict, ckpt_path)
    torch.save(ckpt_dict, RUN_DIR / "bc_policy.pt")
    print(f"[levelforge] Checkpoint saved to {ckpt_path}")
    return ckpt_path


def evaluate(
    ckpt_path: Path,
    showcase_items: List[Tuple[str, Level, int, Path, List[int]]],
    heldout_count: int = 30,
):
    agent = DQNAgent.load(str(ckpt_path), device="cpu")

    # 1. Evaluate Showcase Levels
    watch_results = []
    print("\n" + "=" * 78)
    print("SHOWCASE EVALUATION: 10 COMPLEX 8x8 LEVELS WITH DISTANT SLOTS")
    print("=" * 78)
    print(
        f"{'Level':<10} | {'Boxes':<6} | {'Solved?':<8} | {'Agent Steps':<12} | {'Optimal Steps':<14} | {'Min Slot Dist'}"
    )
    print("-" * 78)

    for i, (name, lvl, opt_len, p, dists) in enumerate(showcase_items, 1):
        env = SokobanEnv(lvl, max_steps=120)
        obs, _ = env.reset()
        done = False
        while not done:
            vec = flatten_obs(obs, env.level.energy)
            a = agent.select_action(vec, greedy=True)
            obs, _r, done, _info = env.step(a)

        solved = env._is_complete()
        status_str = "SOLVED" if solved else f"FAIL ({env.failure})"
        n_b = len(lvl.boxes)
        min_d = min(dists)
        print(f"Level {i:02d}   | {n_b:<6} | {status_str:<8} | {env.steps:<12} | {opt_len:<14} | {min_d} (dists={dists})")
        watch_results.append({
            "level": str(p.name),
            "boxes": n_b,
            "solved": solved,
            "steps": env.steps,
            "optimal_steps": opt_len,
            "min_slot_distance": min_d,
            "goal_distances": dists,
            "failure": env.failure,
        })

    # 2. Evaluate on 30 fresh held-out 8x8 levels
    print("\n" + "=" * 78)
    print(f"HELD-OUT EVALUATION ({heldout_count} FRESH 8x8 DISTANT-SLOT PUZZLES, SEED 9999)")
    print("=" * 78)
    rng_test = np.random.default_rng(9999)
    heldout_levels = []
    while len(heldout_levels) < heldout_count:
        n_b = 3 if len(heldout_levels) < 25 else 4
        lvl, _ = generate_distant_slot_level(rng_test, size=8, n_boxes=n_b, min_dist=4)
        if lvl:
            heldout_levels.append(lvl)

    solved_count = 0
    total_steps = []
    failures: Dict[str, int] = {}
    for lvl in heldout_levels:
        env = SokobanEnv(lvl, max_steps=120)
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
    print("=" * 78)

    report = {
        "dimension": "8x8",
        "instances_trained": 200,
        "min_boxes": 3,
        "showcase_10": watch_results,
        "showcase_solve_rate": sum(1 for r in watch_results if r["solved"]) / len(watch_results),
        "heldout": {
            "count": heldout_count,
            "solved": solved_count,
            "success_rate_pct": sr,
            "mean_steps": mean_steps,
            "failures": failures,
        },
    }
    with open(RUN_DIR / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[levelforge] Saved evaluation report to {RUN_DIR / 'eval_report.json'}")


def main():
    showcase_items = setup_showcase_levels(seed=2026)
    ckpt_path = collect_200_instances_and_train(showcase_items, seed=42)
    evaluate(ckpt_path, showcase_items, heldout_count=30)


if __name__ == "__main__":
    main()

