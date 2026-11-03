"""LevelForge command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .coevolution import DEFAULT_CONFIG, run_coevolution
from .dqn import DQNAgent
from .evaluation import failure_breakdown, summarise_run
from .grid import Level
from .training import evaluate_levels, generate_level_pool, train_on_levels
from .visualize import render_level


def cmd_train_baseline(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = generate_level_pool(args.pool, seed=args.seed, size=args.size, n_boxes=args.n_boxes,
                               wall_frac=args.wall_frac)
    if args.init_from:
        agent = DQNAgent.init_from(args.init_from, args.size, args.size, seed=args.seed)
        print(f"[levelforge] warm-starting from {args.init_from}")
    else:
        agent = DQNAgent(args.size, args.size, seed=args.seed)
    history = train_on_levels(agent, pool, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
                              shaping=args.shaping, shaping_weight=args.shaping_weight)
    agent.save(out / "baseline.pt")
    with open(out / "baseline_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    solved = sum(1 for h in history if h["solved"])
    print(f"[levelforge] baseline trained: {solved}/{len(history)} episodes solved (see {out})")


def cmd_evolve(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "size": args.size,
        "n_boxes": args.n_boxes,
        "population_size": args.population,
        "generations": args.generations,
        "pretrain_episodes": args.pretrain_episodes,
        "retrain_episodes": args.retrain_episodes,
        "pool_size": args.pool,
        "eval_trials": args.eval_trials,
        "topk_retrain": args.topk,
        "heldout_size": args.heldout_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "init_from": args.init_from,
    })
    run_coevolution(cfg, out_dir=args.out, generator=args.generator, name=args.name)


def cmd_pretrain_bc(args):
    """Behaviour-cloning warm start: train the Q-network as a classifier on expert traces."""
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn

    from .dqn import DQNAgent, build_q_network
    from .expert import generate_expert_dataset
    from .population import random_solvable_level
    from .training import evaluate_levels

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y = generate_expert_dataset(args.levels, size=args.size, n_boxes=args.n_boxes,
                                   seed=args.seed, wall_frac=args.wall_frac,
                                   heavy_prob=args.heavy_prob, fragile_prob=args.fragile_prob)
    print(f"[levelforge] collected {len(X)} expert steps from {args.levels} levels in {time.time() - t0:.1f}s")

    net = build_q_network(args.size, args.size, arch=args.arch, hidden=args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    counts = np.bincount(y, minlength=4).astype(np.float64)
    print(f"[levelforge] action distribution (0 up 1 down 2 left 3 right): {counts.astype(int)}")
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    cut = max(1, int(0.9 * len(X)))
    tr_x, tr_y = X[idx[:cut]], y[idx[:cut]]
    va_x, va_y = X[idx[cut:]], y[idx[cut:]]

    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(tr_x))
        total = 0.0
        for i in range(0, len(tr_x), args.batch_size):
            j = perm[i:i + args.batch_size]
            xb = torch.from_numpy(tr_x[j])
            yb = torch.from_numpy(tr_y[j])
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(va_x)).argmax(dim=1)
            acc = float((pred == torch.from_numpy(va_y)).float().mean().item())
            tpred = net(torch.from_numpy(tr_x)).argmax(dim=1)
            tacc = float((tpred == torch.from_numpy(tr_y)).float().mean().item())
        history.append(acc)
        print(f"[levelforge] epoch {epoch + 1}/{args.epochs}  train_loss~{total:.1f}  train_acc={tacc:.3f}  val_action_acc={acc:.3f}")

    ckpt_path = out / "bc_policy.pt"
    torch.save({"policy_state_dict": net.state_dict(),
                "target_state_dict": net.state_dict(),
                "height": args.size,
                "width": args.size,
                "epsilon": 0.05,
                "arch": args.arch,
                "hidden": args.hidden}, ckpt_path)

    # Greedy held-out evaluation with the imitation policy.
    agent = DQNAgent.load(ckpt_path, seed=args.seed + 1)
    held = [random_solvable_level(rng, size=args.size, n_boxes=args.n_boxes,
                                  wall_frac=args.wall_frac, heavy_prob=args.heavy_prob,
                                  fragile_prob=args.fragile_prob)
            for _ in range(args.eval_levels)]
    _records, held_agg = evaluate_levels(agent, held, trials=1,
                                         max_steps=args.max_steps, seed=args.seed + 2)
    print("[levelforge] held-out evaluation with BC policy:")
    print(json.dumps(held_agg, indent=2))
    with open(out / "bc_report.json", "w", encoding="utf-8") as f:
        json.dump({"val_action_acc_history": history, "heldout": held_agg,
                   "n_samples": int(len(X))}, f, indent=2)
    print(f"[levelforge] saved {ckpt_path}")


def cmd_evaluate(args):
    out = Path(args.out)
    heldout = [Level.load_json(p) for p in sorted(out.glob("heldout_*.json"))]
    if not heldout:
        print("[levelforge] no heldout_*.json found; generating a fresh held-out set")
        rng_pool = generate_level_pool(args.heldout_size, seed=args.seed + 7777, size=args.size, n_boxes=args.n_boxes)
        heldout = rng_pool
    agent = DQNAgent.load(args.checkpoint)
    records, agg = evaluate_levels(agent, heldout, trials=args.trials, max_steps=args.max_steps, seed=args.seed)
    agg["failure_breakdown_pct"] = failure_breakdown(records)
    print(json.dumps(agg, indent=2))
    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "records": records}, f, indent=2)


def cmd_render(args):
    level = Level.load_json(args.level)
    render_level(level, args.out, title=args.title)
    print(f"[levelforge] wrote {args.out}")


def cmd_watch(args):
    level = Level.load_json(args.level)
    if args.solver:
        from .solver import solve
        from .tk_viewer import run_solver
        print("[levelforge] solving...")
        moves, nodes, dt = solve(level, max_nodes=2_000_000, time_limit_s=60.0)
        if moves is None:
            print(f"[levelforge] solver could not solve it (nodes={nodes}, t={dt:.2f}s)")
            return
        print(f"[levelforge] solved in {len(moves)} moves ({nodes} explored nodes, {dt:.2f}s)")
        run_solver(level, moves, delay_ms=args.delay)
        return
    if not args.checkpoint:
        print("[levelforge] --checkpoint is required unless --solver is set")
        return
    if args.tk:
        from .tk_viewer import run_agent
        run_agent(level, args.checkpoint, delay_ms=args.delay, explore=args.explore, seed=args.seed)
        return
    if args.text:
        from .replay import run_text
        run_text(level, args.checkpoint, delay_ms=args.delay, max_steps=args.max_steps,
                 explore=args.explore, seed=args.seed)
        return
    from .qt_viewer import run_agent
    run_agent(level, args.checkpoint, delay_ms=args.delay, explore=args.explore, seed=args.seed)


def cmd_play(args):
    level = Level.load_json(args.level)
    if args.tk:
        from .tk_viewer import run
        run(level)
        return
    from .qt_viewer import run
    run(level)


def cmd_solve(args):
    from .solver import solve
    level = Level.load_json(args.level)
    moves, nodes, dt = solve(level, max_nodes=args.max_nodes, time_limit_s=args.time_limit)
    if moves is None:
        print(f"failed (nodes={nodes}, t={dt:.2f}s, max_nodes={args.max_nodes})")
        return
    print(f"solved in {len(moves)} moves (nodes={nodes}, t={dt:.2f}s)")


def cmd_summarise(args):
    print(json.dumps(summarise_run(args.metrics), indent=2))


def cmd_demo(args):
    """Tiny end-to-end run to verify the pipeline on CPU."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "size": args.size,
        "population_size": 6,
        "generations": 2,
        "pretrain_episodes": 20,
        "retrain_episodes": 10,
        "pool_size": 8,
        "heldout_size": 3,
        "eval_trials": 1,
        "topk_retrain": 2,
        "max_steps": 120,
        "max_nodes": 10000,
        "seed": args.seed,
    })
    run_coevolution(cfg, out_dir=args.out, generator="ga", name="demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="levelforge", description="LevelForge: GA + DQN adversarial level generation for Sokoban")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train-baseline", help="Train the DQN agent on random solvable levels")
    t.add_argument("--episodes", type=int, default=300)
    t.add_argument("--pool", type=int, default=40)
    t.add_argument("--size", type=int, default=10)
    t.add_argument("--n-boxes", type=int, default=3)
    t.add_argument("--wall-frac", type=float, default=0.12)
    t.add_argument("--max-steps", type=int, default=300)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", default="runs/baseline")
    t.add_argument("--init-from", default=None, help="warm-start weights from a BC/DQN checkpoint (e.g. runs/bc/bc_policy.pt)")
    t.add_argument("--shaping", action="store_true", help="use potential-based distance shaping (recommended)")
    t.add_argument("--shaping-weight", type=float, default=0.5)
    t.set_defaults(func=cmd_train_baseline)

    e = sub.add_parser("evolve", help="Run the GA + DQN co-evolution loop")
    e.add_argument("--size", type=int, default=10)
    e.add_argument("--n-boxes", type=int, default=3)
    e.add_argument("--population", type=int, default=24)
    e.add_argument("--generations", type=int, default=10)
    e.add_argument("--pretrain-episodes", type=int, default=300)
    e.add_argument("--retrain-episodes", type=int, default=120)
    e.add_argument("--pool", type=int, default=40)
    e.add_argument("--eval-trials", type=int, default=2)
    e.add_argument("--topk", type=int, default=5)
    e.add_argument("--heldout-size", type=int, default=20)
    e.add_argument("--max-steps", type=int, default=300)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", default="runs")
    e.add_argument("--name", default="coevolution")
    e.add_argument("--generator", choices=["ga", "random"], default="ga",
                   help="'random' is the ablation baseline: fresh random levels instead of GA breeding")
    e.add_argument("--init-from", default=None, help="warm-start agent from a BC policy (e.g. runs/bc/bc_policy.pt)")
    e.set_defaults(func=cmd_evolve)

    v = sub.add_parser("evaluate", help="Evaluate a checkpoint on held-out levels")
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--out", required=True, help="run directory containing heldout_*.json or for saving eval_report.json")
    v.add_argument("--trials", type=int, default=3)
    v.add_argument("--size", type=int, default=10)
    v.add_argument("--n-boxes", type=int, default=3)
    v.add_argument("--heldout-size", type=int, default=20)
    v.add_argument("--max-steps", type=int, default=300)
    v.add_argument("--seed", type=int, default=42)
    v.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("render", help="Render a level JSON to PNG")
    r.add_argument("--level", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--title", default=None)
    r.set_defaults(func=cmd_render)

    p = sub.add_parser("play", help="Interactive viewer — you play the level")
    p.add_argument("--level", required=True)
    p.add_argument("--tk", action="store_true", help="use the Tkinter viewer (no Qt required, recommended)")
    p.set_defaults(func=cmd_play)

    w = sub.add_parser("watch", help="Watch the trained DQN agent replay a level (PyQt5)")
    w.add_argument("--level", required=True)
    w.add_argument("--checkpoint", default=None, help="path to a checkpoint .pt written by evolve/train-baseline (not needed with --solver)")
    w.add_argument("--delay", type=int, default=350, help="milliseconds between agent moves (30-5000); +/- keys adjust live")
    w.add_argument("--text", action="store_true", help="headless mode: print board frames to the terminal instead of opening a window")
    w.add_argument("--max-steps", type=int, default=500, help="step cap for --text mode")
    w.add_argument("--explore", type=float, default=0.0, help="exploration epsilon for replay (0 = pure greedy). Try 0.2-0.3 to see the agent move when undertrained")
    w.add_argument("--seed", type=int, default=0, help="RNG seed for --explore sampling")
    w.add_argument("--tk", action="store_true", help="use the Tkinter viewer (no Qt required, recommended)")
    w.add_argument("--solver", action="store_true", help="replay the A* solver solution instead of a DQN checkpoint (no checkpoint required)")
    w.set_defaults(func=cmd_watch)

    s = sub.add_parser("summarise", help="Summarise a metrics.jsonl file")
    s.add_argument("--metrics", required=True)
    s.set_defaults(func=cmd_summarise)

    b = sub.add_parser("pretrain-bc", help="Behaviour-cloning warm start from generator traces (recommended before evolve/train-baseline)")
    b.add_argument("--size", type=int, default=8)
    b.add_argument("--n-boxes", type=int, default=2)
    b.add_argument("--levels", type=int, default=150)
    b.add_argument("--epochs", type=int, default=15)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--hidden", type=int, default=256)
    b.add_argument("--arch", choices=["cnn", "mlp"], default="cnn", help="network architecture (cnn recommended for Sokoban)")
    b.add_argument("--wall-frac", type=float, default=0.12)
    b.add_argument("--heavy-prob", type=float, default=0.25)
    b.add_argument("--fragile-prob", type=float, default=0.20)
    b.add_argument("--eval-levels", type=int, default=30)
    b.add_argument("--max-steps", type=int, default=300)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="runs/bc")
    b.set_defaults(func=cmd_pretrain_bc)

    s = sub.add_parser("solve", help="Solve a level with A* and print stats")
    s.add_argument("--level", required=True)
    s.add_argument("--max-nodes", type=int, default=1_000_000)
    s.add_argument("--time-limit", type=float, default=30.0)
    s.set_defaults(func=cmd_solve)

    d = sub.add_parser("demo", help="Tiny end-to-end run that verifies the whole pipeline on CPU")
    d.add_argument("--size", type=int, default=8)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", default="runs")
    d.set_defaults(func=cmd_demo)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
