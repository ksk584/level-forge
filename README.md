# LevelForge

**Genetic-Algorithm Adversarial Co-Evolution for Robust Sokoban Agents**

LevelForge is an Unsupervised Environment Design (UED) study. A **genetic algorithm** invents
Sokoban levels that a **PPO agent** currently fails on; the agent then trains on them; the GA
searches for the next weakness — and so on. We measure whether this adversarial curriculum
produces more robust agents than training on the same number of random levels, and whether the
GA finds agent-breaking levels more efficiently than random search.

> **Research question.** Can a classical GA, used as an adversarial level generator in a
> co-evolutionary loop with an RL agent, discover agent failure modes more efficiently than random
> search, and yield agents that generalise better to unseen levels than equal-budget training on
> random levels?

---

## System overview

| Component | What it does | Code |
|---|---|---|
| Sokoban environment | walls/boxes/goals/agent, push mechanics, deadlock & failure taxonomy, optional heavy/fragile boxes and energy budget | `environment.py`, `grid.py` |
| Dead-square analysis | cells from which a lone box can never reach a goal (reverse-pull reachability) | `solvability.py` |
| A* solver | move-optimal plans; proves levels are winnable within the step cap and energy budget | `solver.py` |
| Level generator | reverse-play scrambling from the solved state (solvable by construction) + A* verification | `population.py` |
| RL agent | PPO (Stable-Baselines3) with a small-grid CNN; batched evaluator | `ppo_agent.py`, `gym_env.py` |
| GA adversary | tournament selection, region-wise crossover, mutation, repair, A* verification | `genetic.py` |
| Fitness | solver-referenced **regret** (default), failure rate, learnability; optional novelty bonus | `fitness.py` |
| Co-evolution orchestrator | four controlled conditions, shared pretraining, shared test sets | `ued.py` |
| Frozen-agent search experiment | GA vs random search at an equal evaluation budget | `attack.py` |
| Analysis | mean ± std over seeds, permutation tests, bootstrap CIs, figures, `results.md` | `analysis.py` |
| Agent comparison | from-scratch DQN vs PPO at equal environment steps | `agent_bench.py`, `dqn.py` |
| Hand-designed test set | 16 original 7×7 puzzles, each targeting one skill | `levels/handmade_7x7.txt` |

## Experimental design

All conditions start from the **same pretrained PPO checkpoint** (per seed), get the **same PPO
step budget** per cycle, and add the **same number of new levels** (16) to their curriculum per
cycle. Only the way the new levels are chosen differs:

| Condition | How the 16 new levels per cycle are chosen | Role |
|---|---|---|
| `ga` | GA evolved against the current agent (population persists across cycles) | **our method** |
| `random_search` | same number of candidates scored as the GA, but drawn at random; keep top-16 | isolates the value of evolution itself |
| `random` | 16 fresh random levels, no selection | the "train on random levels" baseline |
| `fixed` | one set of 16 levels drawn once, reused every cycle | the naive baseline |

Training mixture each cycle (identical for every condition): 40 % shared random base pool, 40 %
this cycle's new levels, 20 % replay of earlier new levels.

**Evaluation sets** (shared by all conditions and seeds; never trained on):

* **held-out** — 200 random levels from the training distribution;
* **harder** — 100 random levels with one extra box (out-of-distribution);
* **hand-designed** — the 16 puzzles in `levels/handmade_7x7.txt`;
* **adversarial** — GA-discovered levels from *other* seeds' runs (cross-evaluation);
* **own curriculum** — each run's own past curriculum levels (forgetting check).

**Fitness — regret against an exact solver.** Every candidate level is first proven winnable by A*
(within the step cap and the energy budget). The solver's normalised score is therefore 1; the
agent's per-episode score is `solved × min(1, optimal_moves / agent_moves)`. Regret = 1 − mean agent
score over 4 stochastic rollouts. This is PAIRED's regret with an exact planner standing in for the
antagonist, and it cannot be gamed with impossible levels — the classic failure mode of raw
failure-rate fitness. A small novelty bonus (k-nearest-neighbour Hamming distance) counters
collapse onto one trick.

**Frozen-agent experiment.** With the agent fixed, GA and random search each score 1024 candidate
levels; we count distinct, solver-verified levels the agent fails on every trial. This yields the
headline efficiency number ("levels evaluated to find N breaking levels").

**Statistics.** Mean ± std across seeds, exact permutation test and bootstrap 95 % CI for GA vs each
baseline. With 3 seeds the minimum attainable p is 0.10 — use 5 seeds for a significance claim.

---

## Quickstart (Windows / Linux, Python 3.10+)

```powershell
pip install -e .            # numpy, torch, matplotlib, gymnasium, stable-baselines3
pip install pytest
python -m pytest -q          # 49 tests (~20 s, includes a tiny end-to-end UED run)

# 3-minute end-to-end smoke run of the whole pipeline (numbers are meaningless)
python -m levelforge ued-run --exp runs/smoke --config configs/ued_smoke.json --seeds 0
```

## Running the experiment

```powershell
$CFG = "configs/ued_7x7.json"; $EXP = "runs/ued_7x7"

python -m levelforge agent-bench  --exp $EXP --config $CFG --steps 400000          # DQN vs PPO (why PPO)
python -m levelforge ued-pretrain --exp $EXP --config $CFG --seeds 0 1 2           # shared starting agents
python -m levelforge ued-run      --exp $EXP --config $CFG --seeds 0 1 2 --condition all
python -m levelforge ued-attack   --exp $EXP --config $CFG --seeds 0 1 2 --budget 1024
python -m levelforge ued-analyze  --exp $EXP                                        # tables + figures
```

Every command is **resumable**: finished runs are skipped, and an interrupted `ued-run` restarts
after its last completed cycle (`resume.json` + `resume_model.zip` in the run directory; deleted
when the run finishes). A resumed run is statistically equivalent, not bit-identical, because
SB3's internal RNG restarts. Budget: ~4–5 h for 4 conditions × 3 seeds on an 8-core CPU; faster
on a Colab T4 (`notebooks/LevelForge_UED_Colab.ipynb`, which writes results to Google Drive).

`configs/ued_7x7_payload.json` turns on heavy and fragile boxes, with rough floor (energy budget
and fragile-break rules). Run it with `bash scripts/run_payload.sh` (3 seeds in parallel, then the
attack and analysis; results in `runs/ued_7x7_payload/analysis/`).

Outputs in `runs/ued_7x7/`:

```
pretrain_seed<s>/model.zip, pretrain_eval.json
<condition>_seed<s>/config.json, metrics.jsonl, summary.json, model.zip, archive.json, levels/cycle_*.json
attack/attack.json, breaking_<method>_seed<s>.json
agent_bench/agent_bench.json
analysis/results.md, finals.json, success_curves.png, curriculum_regret.png, attack_efficiency.png
```

`metrics.jsonl` (one row per cycle) logs, among others: success on every test set (stochastic and
greedy), the regret and pre/post-training success of the new curriculum levels, per-generation GA
best/mean fitness, diversity, solvability yield and the fraction of candidates that break the agent.

Play a fresh, random solvable puzzle with `python -m levelforge play`. Use the arrow keys or
WASD to move. Three buttons choose who plays from the current position: **Player** (you),
**RL agent** (the PPO policy) and **A\* solver**; **New random puzzle** generates another level.

**Learning from players.** Every puzzle you solve without agent help is saved to
`runs/player_learning/games.jsonl`. Straight after each solve, the RL agent is fine-tuned by
behaviour cloning on all saved games and written to `runs/player_learning/model.zip`, which the
next `play` session loads. The agent's skill comes from reinforcement learning (`ued-*`); this
adds imitation of players on top. It copies your moves, including roundabout ones, and only
learns from boards of its own training size (7×7 for the default model). Choose the starting
model with `--ppo <model.zip>` and the folder with `--learn-dir`.

**Heavy and fragile boxes in play.** `--heavy` puts one heavy box (**H**, a push costs 3 energy) in
every random puzzle. `--fragile` puts in one fragile box (**F**) plus rough floor tiles (sandy,
dotted). The player and other boxes cross rough tiles freely, but a fragile box pushed onto one
breaks and the puzzle is lost. A fragile box pushed where it could only reach a goal across rough
floor ends the puzzle at once as `deadlock`. ASCII levels: `~` rough floor, `&` player on rough,
`m`/`k`/`z` normal/heavy/fragile box on rough (rough tiles are never goals). Fragile puzzles are redrawn until the rough
tiles block the shortest route, so you must find a safe one; A* verifies that one exists. Both
flags together need `--boxes 2` (set automatically). The RL agent was never trained with these
boxes or tiles; once `scripts/run_payload.sh` has trained one, `play` uses
`runs/ued_7x7_payload/ga_seed0/model.zip` by default (one-box puzzles; it was not trained on two).
With `--level`, the flags apply only to puzzles from **New random puzzle**.

**Mixed boxes.** `--mix` gives every new puzzle a random combination of plain, heavy and fragile
boxes: 2 or 3 boxes (or `--boxes N`), at least two different types, and possibly two heavy or two
fragile boxes. Fragile boxes bring rough floor with them. `--mix` replaces `--heavy`/`--fragile`.
The RL agent was trained on one-box puzzles, so expect it to struggle with these. Press R to restart and Esc to quit. Set difficulty
with `--size` and `--boxes`, or start from an existing JSON level with `--level <file.json>`.

Inspect any level: `python -m levelforge render --level <file.json> --out x.png`,
`python -m levelforge watch --level <file.json> --solver --tk`, `python -m levelforge solve --level <file.json>`.

**Quick toy demo** (no training; Tkinter window), good for a viva:

```powershell
python -m levelforge play --level levels/toy_1.json --tk            # push once onto the goal
python -m levelforge play --level levels/toy_3_heavy.json --tk      # heavy box costs 3 energy
python -m levelforge play --level levels/toy_5_two_boxes.json --tk  # two boxes, two goals
python scripts/make_toy_instances.py                                 # regenerate/verify the toys
```

Prefer `--tk` (Tkinter, standard library): pygame has no Python 3.14 wheel, and Qt windows may
hide behind other windows; the Tk window is centred on screen with focus.

---

## Engineering decisions worth knowing (and defending in a viva)

* **PPO, not the from-scratch DQN.** In pilot runs the DQN stayed at or below random-policy success
  on 6×6 one-box boards (greedy policies loop; per-sample CNN updates take ~15 ms on CPU). With the
  agent failing everything, every level gets fitness 1.0 and the GA degenerates into random search.
  `agent-bench` reproduces the comparison. The DQN is kept as a documented baseline.
* **Standard Sokoban training reward.** The environment's native reward (−5 for a blocked push)
  pushed PPO into a "never touch a box" optimum — below random-policy success even on 10 levels.
  The Gym wrapper trains with the standard Boxoban reward (−0.1/step, ±1 box on/off goal, +10 solve,
  −1 for a lost episode) plus light potential-based shaping. After 100k steps on the same 10 levels, PPO
  solved 15 % of them with the native reward and 67 % with the standard one. Evaluation never uses rewards — only success and step counts.
* **Dead squares, not local wall rules.** The original deadlock rule flagged every 1-wide corridor
  as a deadlock (a level as simple as `#@ n .#` was "unsolvable"). Replaced by reverse-pull
  reachability from the goals, which is exact for single-box deadlocks.
* **Every level is A*-verified**, generated or evolved, and its energy budget is set from the
  optimal plan's cost. Previously budgets were ~40× too loose and GA mutations could make levels
  unwinnable by starving energy.
* **Stochastic evaluation for fitness.** A greedy policy in a deterministic environment replays the
  same episode every trial, so multi-trial fitness was a single 0/1 sample. The PPO pipeline scores
  levels with 4 sampled rollouts and a continuous, step-efficiency-weighted regret. The legacy
  `evolve` pipeline now scores with seeded ε-exploration (`--eval-epsilon 0.1`, 4 trials); with
  `--eval-epsilon 0` its fitness is pass/fail and must be reported as such.
* **Solvability yield is measured the same way for every generator**: A*-verified candidates /
  candidates produced (GA children or random scrambles). It used to be hard-coded to 1.0 for the
  random generator.

## Which agent produced which result

Several agents exist in this repository. The report must name the one behind each result.

| Result | Agent | How it was trained | Command / location |
|---|---|---|---|
| **All UED results**: main experiment, fitness and mixture ablations, 7×7 two-box and 9×9 scale-ups, frozen-agent attack, demo kit | **PPO** (Stable-Baselines3) with the `GridCNN` feature extractor | Reinforcement learning from scratch: 0.4–1.5 M pretraining steps on random levels, then each condition's curriculum (Boxoban reward) | `ued-*`; `runs/ued_*`, `docs/results/` |
| DQN vs PPO comparison | DQN from scratch (CNN) vs PPO | RL, equal environment steps | `agent-bench`. Code provided; the DQN finding quoted above comes from pilot runs, and `agent-bench` was not part of the final experiment runs |
| Legacy co-evolution (`evolve`, `train-baseline`, `demo`) | DQN: 2-hidden-layer MLP (64-64) in `evolve`; `train-baseline` defaults to a CNN (`--arch`) | ε-greedy RL, optional BC warm start | legacy pipeline (below) |
| BC warm start (`pretrain-bc`) | DQN Q-network (CNN by default) trained as a classifier | Behaviour cloning on **A\* solver demonstrations** by default (`--no-solver`: the generator's reverse-scramble traces), multi-start, optional D4 augmentation; no DAgger | `pretrain-bc` |
| 8×8 showcase (`levels/watch_10/`, `scripts/train_8x8_distant_slots.py`, `scripts/watch_showcase.py`, `scripts/export_training_csv.py`) | CNN policy | **Imitation learning**: A\* demonstrations on 200 levels, multi-start, D4 symmetry augmentation and DAgger | `runs/train8_200/` |
| 6×6 multi-box showcase (`scripts/train_and_prepare_watch.py`) | CNN policy | Imitation learning on 100 games (2–3 boxes) | `runs/train6_100/` |

The showcase agents learn by **imitating the A\* solver**, not by reinforcement learning. They are
neither the DQN nor the PPO agent and play no part in the UED experiments; label them as
imitation-learning results wherever they appear (e.g. "CNN trained on A\* demonstrations with
DAgger"). Their success shows that the policy *class* can represent good Sokoban play when given
expert labels; it says nothing about the GA curriculum.

## Related work and positioning

* **PAIRED** (Dennis et al., 2020) — regret-based UED with a learned adversary. We use the same
  regret idea with an exact planner as the reference.
* **POET** (Wang et al., 2019) — open-ended co-evolution of environments and agents.
* **PLR / Robust PLR** (Jiang et al., 2021) — curate replayed levels by estimated regret.
* **ACCEL** (Parker-Holder et al., 2022) — evolves levels by small edits selected by regret. **This
  is the closest prior work**: our contribution is a classical GA with *crossover*, an exact
  solver-referenced regret and hard solvability verification, studied on a combinatorial puzzle
  domain with controlled baselines (including random search with identical selection).
* Racanière et al., 2017 (Sokoban generation and reward); Culberson, 1997 (Sokoban is PSPACE-complete).

## Limitations

* 7×7 boards with one box are small; the "+1 box" test set probes harder generalisation, and the
  config can be scaled (`n_boxes: 2`, longer pretraining) on a GPU.
* With 3 seeds, permutation tests cannot reach p < 0.1; report effect sizes and CIs, or run 5 seeds.
* Deadlock detection covers dead squares and freeze deadlocks (boxes jamming each other against
  walls), not every multi-box deadlock (e.g. corral deadlocks); those episodes still end by timeout.
  The freeze check was added after the experiments in `docs/results/` were run: in those runs a
  frozen two-box state ended by timeout, not by an explicit deadlock.
* Heavy/fragile boxes and rough floor are exercised by one experiment only
  (`docs/results/ued_7x7_payload/`: 7×7, one box, learnability fitness, 3 seeds). Rough tiles are
  drawn only when a level has a fragile box, so plain-box experiment levels are unchanged. Its
  hand-designed test set has plain boxes only, and its adversarial set comes from learnability
  GAs like the one the GA agent trained against.
* The frozen-agent attack counts breaking levels as distinct by layout (ignoring energy). The
  attack numbers in `docs/results/` predate this and counted energy-only variants separately,
  which inflated the GA's count by about 5 % (7×7, one box).

## Legacy DQN pipeline

The original from-scratch pipeline remains available: `train-baseline`, `evolve`
(`--generator ga|random`), `pretrain-bc`, `evaluate`, `summarise`, `demo`, and
`notebooks/LevelForge_Colab.ipynb`. See `docs/ARCHITECTURE.md` for its design notes.

**Its results are not used in the report.** With CPU-scale training the DQN solves almost no
levels, so every level gets the same fitness and GA selection has nothing to act on: "GA vs
random levels" cannot show a difference there. `evolve` now detects this: each generation logs
`fitness_std` and `fitness_distinct_values`, and prints a warning when all fitness values are
equal. Fitness is scored with seeded ε-exploration (`--eval-epsilon`, default 0.1) so it is graded
rather than pass/fail, the random-generator ablation measures its solvability yield instead of
recording 1.0, and DQN exploration uses a seeded generator, so the same `--seed` reproduces a run.
