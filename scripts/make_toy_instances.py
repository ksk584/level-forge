"""Hand-authored toy instances for the phase-1 demo (no training needed).

Writes theory tiny boards to levels/*.json and renders levels/png/*.png.
Each level is validated and A*-solved (so the demo never shows a broken board).

Note: strict 4x4 with border walls is unplayable (no cell behind a box is free
for any push).  5x5/6x6 toys are the smallest real Sokoban boards.
"""

from pathlib import Path

from levelforge.grid import Level
from levelforge.solver import solve
from levelforge.visualize import render_level

OUT = Path("levels")
PNG = OUT / "png"
PNG.mkdir(parents=True, exist_ok=True)

TOYS = {
    # push the normal box straight down onto the goal (1 push)
    "toy_1": ("""#####
# @ #
# n #
# . #
#####""", 10),
    # push it twice — watch energy drop by 2
    "toy_2": ("""######
# @  #
# n  #
#    #
# .  #
######""", 10),
    # heavy box costs 3 energy per push — watch the budget matter
    "toy_3_heavy": ("""######
# @  #
# h  #
#    #
# .  #
######""", 12),
    # fragile box — a legal solve is trivial; push it into any corner to see it shatter
    "toy_4_fragile": ("""######
# @  #
# f  #
#    #
# .  #
######""", 12),
    # two boxes, two goals, four pushes minimum
    "toy_5_two_boxes": ("""######
# @  #
# n .#
# n .#
#    #
######""", 10),
}


def main():
    for name, (ascii_art, energy) in TOYS.items():
        lvl = Level.from_ascii(ascii_art, energy=energy)
        problems = lvl.validate()
        assert not problems, (name, problems)
        moves, nodes, dt = solve(lvl, max_nodes=100_000, time_limit_s=5.0)
        print(f"{name}: solved in {len(moves)} moves (nodes={nodes}, t={dt:.3f}s)")
        assert moves is not None, f"{name} is unsolvable!"
        lvl.save_json(OUT / f"{name}.json")
        render_level(lvl, str(PNG / f"{name}.png"), title=name)


if __name__ == "__main__":
    main()
