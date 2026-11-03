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
    agent = DQNAgent(args.size, args.size, seed=args.seed)
    history = train_on_levels(agent, pool, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
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
    })
    run_coevolution(cfg, out_dir=args.out, generator=args.generator, name=args.name)


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
    if args.text:
        from .replay import run_text
        run_text(level, args.checkpoint, delay_ms=args.delay, max_steps=args.max_steps,
                 explore=args.explore, seed=args.seed)
        return
    from .qt_viewer import run_agent
    run_agent(level, args.checkpoint, delay_ms=args.delay, explore=args.explore, seed=args.seed)


def cmd_play(args):
    from .qt_viewer import run
    level = Level.load_json(args.level)
    run(level)


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

    p = sub.add_parser("play", help="Interactive viewer (PyQt5) — you play the level")
    p.add_argument("--level", required=True)
    p.set_defaults(func=cmd_play)

    w = sub.add_parser("watch", help="Watch the trained DQN agent replay a level (PyQt5)")
    w.add_argument("--level", required=True)
    w.add_argument("--checkpoint", required=True, help="path to a checkpoint .pt written by evolve/train-baseline")
    w.add_argument("--delay", type=int, default=350, help="milliseconds between agent moves (30-5000); +/- keys adjust live")
    w.add_argument("--text", action="store_true", help="headless mode: print board frames to the terminal instead of opening a window")
    w.add_argument("--max-steps", type=int, default=500, help="step cap for --text mode")
    w.add_argument("--explore", type=float, default=0.0, help="exploration epsilon for replay (0 = pure greedy). Try 0.2-0.3 to see the agent move when undertrained")
    w.add_argument("--seed", type=int, default=0, help="RNG seed for --explore sampling")
    w.set_defaults(func=cmd_watch)

    s = sub.add_parser("summarise", help="Summarise a metrics.jsonl file")
    s.add_argument("--metrics", required=True)
    s.set_defaults(func=cmd_summarise)

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
