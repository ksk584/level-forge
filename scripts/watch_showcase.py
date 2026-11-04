"""Convenience script to watch the 8x8 distant slot showcase levels in GUI or text mode.

Usage:
    python scripts/watch_showcase.py --level 1
    python scripts/watch_showcase.py --level 8 --delay 0.3
    python scripts/watch_showcase.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path

WATCH_DIR = Path("levels/watch_10")
CHECKPOINT = Path("runs/train8_200/checkpoint.pt")


def main():
    parser = argparse.ArgumentParser(description="Watch trained agent play 8x8 showcase levels.")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=list(range(1, 11)),
        help="Showcase level number (1-10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=300,
        help="Delay between moves in ms (e.g. 300) or seconds (e.g. 0.3)",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use terminal ASCII text animation instead of Tkinter GUI",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Play through all 10 showcase levels sequentially",
    )
    args = parser.parse_args()

    # Convert seconds (<= 10.0) to integer milliseconds
    delay_ms = int(args.delay * 1000) if args.delay <= 10.0 else int(args.delay)

    levels = list(range(1, 11)) if args.all else [args.level]

    for lvl_idx in levels:
        lvl_path = WATCH_DIR / f"level_{lvl_idx:02d}.json"
        if not lvl_path.exists():
            print(f"Error: {lvl_path} does not exist.")
            continue

        cmd = [
            sys.executable,
            "-m",
            "levelforge",
            "watch",
            "--level",
            str(lvl_path),
            "--checkpoint",
            str(CHECKPOINT),
            "--delay",
            str(delay_ms),
        ]
        if args.text:
            cmd.append("--text")
        else:
            cmd.append("--tk")

        print(f"\n=======================================================")
        print(f" Watching Showcase Level {lvl_idx:02d} ({'Text' if args.text else 'Tkinter GUI'})")
        print(f" Delay: {delay_ms} ms per step | File: {lvl_path.name}")
        print(f" Command: {' '.join(cmd)}")
        print(f"=======================================================\n")
        try:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"[watch] exited with code {res.returncode}")
        except KeyboardInterrupt:
            print("\n[watch] Interrupted by user.")
            break


if __name__ == "__main__":
    main()
