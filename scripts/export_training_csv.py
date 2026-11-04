"""Train on 200 complex 8x8 levels and record per-epoch results to CSV.

Produces:
    runs/train8_200/training_epochs.csv
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from levelforge.dqn import DQNAgent, build_q_network, flatten_obs
from levelforge.environment import SokobanEnv
from levelforge.expert import augment_d4, collect_solver_samples
from levelforge.grid import Level
from levelforge.population import scramble_level
from levelforge.solver import solve

WATCH_DIR = Path("levels/watch_10")
RUN_DIR = Path("runs/train8_200")
CSV_PATH = RUN_DIR / "training_epochs.csv"
DATASET_CACHE = RUN_DIR / "dataset_8x8_200.npz"


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


def get_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_CACHE.exists():
        print(f"[levelforge] Loading cached dataset from {DATASET_CACHE}...")
        data = np.load(DATASET_CACHE)
        return data["X"], data["y"]

    print("[levelforge] Loading 10 showcase levels...")
    showcase_levels = []
    for idx in range(1, 11):
        lvl_p = WATCH_DIR / f"level_{idx:02d}.json"
        lvl = Level.load_json(lvl_p)
        m, _, _ = solve(lvl, max_nodes=50000, time_limit_s=2.0)
        showcase_levels.append((lvl, m))

    rng = np.random.default_rng(seed)
    train_levels = list(showcase_levels)
    print(f"[levelforge] Generating 200 distant-slot 8x8 levels (seed={seed})...")
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

    print(f"[levelforge] Extracting solver trajectories with multi-start...")
    xs, ys = [], []
    for lvl, moves in train_levels:
        st, lb, sol = collect_solver_samples(lvl, moves)
        if sol and len(st) > 0:
            xs.append(st)
            ys.append(lb)

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
    print(f"[levelforge] Applying 8-fold D4 symmetry augmentation...")
    X_aug, y_aug = augment_d4(X, y, 8, 8)
    print(f"[levelforge] Saving dataset cache to {DATASET_CACHE} ({len(X_aug)} transitions)...")
    np.savez_compressed(DATASET_CACHE, X=X_aug, y=y_aug)
    return X_aug, y_aug


def train_and_log_epochs(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_aug, y_aug = get_dataset(seed=seed)
    n_samples = len(X_aug)

    net = build_q_network(8, 8, arch="cnn", hidden=128)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    batch_size = 256
    epochs = 25
    n_batches = int(np.ceil(n_samples / batch_size))

    print(f"\n[levelforge] Starting 25 epochs training logging every epoch to {CSV_PATH}...")
    print(f"{'Epoch':<6} | {'Total Loss':<12} | {'Avg Batch Loss':<16} | {'Train Acc (%)':<14} | {'Epoch Time (s)'}")
    print("-" * 65)

    csv_rows = []
    fieldnames = [
        "epoch",
        "total_loss",
        "avg_batch_loss",
        "train_accuracy_pct",
        "learning_rate",
        "batch_size",
        "num_batches",
        "num_samples",
        "epoch_duration_s",
        "cumulative_time_s",
    ]

    total_training_start = time.time()
    t_aug_tensor = torch.from_numpy(X_aug)
    y_aug_tensor = torch.from_numpy(y_aug)

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        net.train()
        perm = np.random.permutation(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            b_idx = perm[i : i + batch_size]
            xb = torch.from_numpy(X_aug[b_idx])
            yb = torch.from_numpy(y_aug[b_idx])
            opt.zero_grad()
            out = net(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        epoch_duration = time.time() - t_epoch_start
        cumulative_time = time.time() - total_training_start
        avg_batch_loss = total_loss / n_batches

        # Evaluate full dataset accuracy
        net.eval()
        with torch.no_grad():
            correct = 0
            eval_bs = 2048
            for j in range(0, n_samples, eval_bs):
                preds = net(t_aug_tensor[j : j + eval_bs]).argmax(dim=1)
                correct += int((preds == y_aug_tensor[j : j + eval_bs]).sum().item())
            acc_pct = (correct / n_samples) * 100.0

        row = {
            "epoch": epoch,
            "total_loss": round(total_loss, 4),
            "avg_batch_loss": round(avg_batch_loss, 5),
            "train_accuracy_pct": round(acc_pct, 2),
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_batches": n_batches,
            "num_samples": n_samples,
            "epoch_duration_s": round(epoch_duration, 2),
            "cumulative_time_s": round(cumulative_time, 2),
        }
        csv_rows.append(row)

        print(
            f"{epoch:02d}/{epochs:<3} | {total_loss:<12.2f} | {avg_batch_loss:<16.4f} | {acc_pct:<14.2f} | {epoch_duration:.2f}s"
        )

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[levelforge] Successfully saved epoch training CSV to {CSV_PATH}")


if __name__ == "__main__":
    train_and_log_epochs(seed=42)

