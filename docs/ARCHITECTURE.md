# LevelForge — Architecture Notes

Design decisions, rule definitions and pointers for the final report.

## 1. Board model (`grid.py`)

`Level` holds:

* `walls` — `(H, W)` `bool`, border is always walls (validated).
* `goals` — list of `(x, y)`. Exactly one box per goal is required (`validate()`).
* `boxes` — `(x, y, kind)`; `kind`: 0 normal, 1 heavy, 2 fragile.
* `agent` — `(x, y)`.
* `energy` — push budget. Push costs: normal/fragile 1, heavy 3.

ASCII round-trip (`n/N/h/H/f/F` = box kind, upper-case = on goal) and JSON serialisation
get levels on/off disk and into tests. Observation = 6 one-hot channels (wall, goal,
normal box, heavy box, fragile box, agent) → `state_dim = 6·H·W + 1` with a normalised
energy scalar appended.

## 2. Environment semantics (`environment.py`)

Actions: 0 up, 1 down, 2 left, 3 right.

| Event | Energy | Reward | Terminal |
|---|---|---|---|
| Move into wall / blocked | 0 | −0.1 (or −5.1 for invalid push) | no |
| Push normal / fragile | 1 | −0.1 (+10 if it newly completes a goal) | fragile → crush ⇒ **fail** (`fragile_break`) |
| Push heavy | 3 | −0.1 (+10 likewise) | insufficient energy ⇒ push refused |
| Puzzle solved | — | +100 | **done** (success) |
| Box in non-goal crush cell (corner / wall-line) | — | −10 | **fail** (`deadlock`) |
| Energy hits 0 before completion | — | −10 | **fail** (`energy_exhaustion`) |
| 300 steps elapsed | — | — | **fail** (`timeout`) |

Reward shaping details: the +10 is granted only when the count of boxes-on-goals
*increases* after a push (undoing a placement removes the credit). Completion is checked
against the *original* box count — a fragile box that shatters deducts from it, so a board
with a broken box can never report `solved`.

Failure taxonomy feeds both evaluation and GA fitness. `invalid_heavy_push` is a *counted
non-terminal event* (info flag `invalid_push`), not a terminal failure — it contributes to
the failure-type statistics if it occurs in greedy evaluation runs only as an event, or may
be promoted to a metric in the report.

## 3. Deadlocks & solvability (`solvability.py`)

* `crush_cells`: a cell is a crush cell if boxed in by two perpendicular blocked sides
  (corner), or by walls/out-of-play along an unbroken horizontal/vertical run that contains
  no goal (wall-line). Boxes there can never be freed.
* `is_solvable`: node-capped reverse BFS from the solved state. Boxes move via pulls; a box
  moving from `X` to `X + d` additionally requires `X + d` and `X + 2d` free (the latter is
  where the agent stood during the forward push). States containing a non-goal box in a crush
  cell are pruned. Returns False on node exhaustion (inconclusive ⇒ treated as unusable).
* Complexity caveat (documented in the report): Sokoban is PSPACE-complete; the checker is a
  filter, not a complete decision procedure. Raising `max_nodes` trades CPU for recall.

## 4. Guaranteed-solvable generation (`population.py::scramble_level`)

Naive random placement is essentially never solvable with ≥2 boxes. Instead:

1. Place walls (border + random interior, `wall_frac`), keep boxes/goals off walls.
2. Initialise boxes on goals (solved state); agent anywhere free.
3. Simulate reverse play: walks + pulls (pull = agent steps *away* from an adjacent box,
   box follows into the agent's vacated cell; the far cell must be free).
4. Replaying the recorded sequence backwards yields a legal forward solution —
   **solvable by construction**, regardless of wall topology.

At least 4 pulls are required (else the puzzle is re-drawn). Payload kinds: at most one
heavy and one fragile for balance; energy = 12 + 5·n_boxes + 2×(#heavy) — deliberately tight
so that the energy budget is a meaningful constraint for both the agent and the GA.

GA children from crossover/mutation are filtered by `is_solvable` before acceptance and may
also be repaired (`repair_level`): borders enforced, boxes/goals deduplicated and relocated,
agent re-sited, box/goal counts re-matched. Children failing repair or the solvability
filter are discarded; the discard rate is logged as Solvability Yield (SY).

## 5. DQN (`dqn.py`)

Per synopsis §5.3: fully-connected, two hidden layers of 64 (ReLU), one Q-value per action.
MSE TD loss, Adam (1e-3), γ 0.99, ε 1.0 → 0.05, experience replay (cap 20 000, batch 64),
target network synced every 500 gradient steps, gradient-norm clipping at 10. State = flat
one-hot grid + normalised energy. Greedy evaluation for fitness; ε-greedy during training.

## 6. Co-evolution orchestration (`coevolution.py`)

```
for each generation:
    evaluate population (greedy, eval_trials episodes/level)
    fitness(level) = failure_rate = 1 − mean success
    retrain agent on (random pool + top-k hardest ×4 weight), 120 episodes
    evaluate on untouched held-out set (every generation, 1 trial)
    breed next population via GA (or random ablation), log SY
    save checkpoint + hardest levels + metrics.jsonl row
```

Run dir: `config.json`, `metrics.jsonl`, `checkpoint_pretrain.pt`, `checkpoint_gen_*.pt`,
`levels_gen_*/hard_*.json`, `summary.json`. `--generator random` performs identical steps
but replaces GA breeding with freshly sampled solvable levels — the ablation required by
the synopsis for comparing failure-discovery efficiency.

## 7. Metrics (synopsis §7)

| Metric | Implementation |
|---|---|
| Agent Success Rate (held-out) | `evaluate_levels`, logged per generation |
| Failure-Discovery Efficiency | `evaluation.failure_discovery_index` (episodes to k distinct failure types) |
| Failure-Type Breakdown | `evaluation.failure_breakdown` (% per category) |
| Population Diversity | `genetic.population_diversity` (normalised mean pairwise Hamming) |
| Solvability Yield | `children accepted / children attempted`, logged per generation |
| Training Stability | rolling loss logged; target-vs-policy Q gap available in `DQNAgent` |
| Co-Evolution Sample Efficiency | episodes-to-target derivable from `metrics.jsonl` series |

Suggested reporting: table per seed × condition (GA vs random), plus curves for held-out SR
vs generation; Welch's t-test / Mann–Whitney U on final SR and FDE.

## 8. Known simplifications

* Boxes are interchangeable for feasibility (any box may fill any goal) — standard Sokoban
  convention, and the reason the checker only tracks positions.
* The reverse-BFS checker ignores agent reachability within free space (it enforces the
  geometry of pulls but not corridor traversability); it is a filter. Generation correctness
  does **not** depend on it (reverse-play generator is exact by construction).
* Heavy/fragile payload mechanics do not change structural solvability; they shape rewards
  and failure categories (used by the GA fitness and evaluation).
* 10×10 is the default board as per the synopsis's 10–15×15 plan; `--size` is exposed.
* One CPU process trains everything (no distributed sampling) — adequate for 8–10×10; use
  Colab T4 for the full protocol.

## 9. Mapping to report sections

| Report section | Source of material |
|---|---|
| §2 Project Description | README + this file §§1–6 |
| §5 Dataset / generation | `population.py` (reverse-play construction, why not rejection sampling) |
| §6 Methodology | §§3–6 above |
| §7 Results | `runs/*/metrics.jsonl`, `evaluation.py` |
| §8 Conclusion | constraints of CPU training; UED connection; future work (regret approximations, diffusion generators) |
