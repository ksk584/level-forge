# LevelForge

**A Genetic Algorithm and Deep Q-Network Framework for Adversarial Game Level Generation**

IS353IA – Artificial Intelligence and Machine Learning (Lab Part B) · RV College of Engineering
Team: **Mrinal S (1RV24IS075)** · **Kiran S K (1RV24IS057)**

LevelForge evolves *Sokoban-style* puzzle levels with a genetic algorithm (GA) and trains a
custom Deep Q-Network (DQN) agent on them. A level's fitness is the **failure rate of the
current agent**, so the GA actively searches for layouts that expose the agent's weaknesses
rather than merely producing varied content. The hardest evolved levels are fed back into the
agent's training curriculum — a lightweight co-evolutionary loop inspired by Unsupervised
Environment Design (UED), but discrete, interpretable and built from first principles.

---

## What's implemented

| Synopsis component | Status | Location |
|---|---|---|
| Grid-based Sokoban environment (walls, boxes, goals, agent, energy budget) | ✔ | `src/levelforge/environment.py` |
| Heavy (push = 3 energy) & fragile (breaks in crush cells) boxes | ✔ | `grid.py`, `environment.py` |
| Fixed-size genome + evolutionary operators (tournament selection, region-wise crossover, mutation) | ✔ | `genetic.py` |
| Solvability filter (corner / wall-line deadlock rules + bounded reverse BFS) | ✔ | `solvability.py` |
| Guaranteed-solvable level generation (reverse Sokoban play from the solved state) | ✔ | `population.py` |
| Custom DQN: 2×64 MLP, experience replay, target network, ε-greedy, Adam | ✔ | `dqn.py` |
| Co-evolutionary loop (fitness = agent failure rate; top-k hardest levels → retraining) | ✔ | `coevolution.py` |
| Evaluation: held-out SR, failure-discovery efficiency, breakdown, diversity, solvability yield | ✔ | `evaluation.py`, `training.py` |
| Rendering (PNG) + interactive PyQt5 viewer | ✔ | `visualize.py`, `qt_viewer.py` |
| Watch mode: greedy DQN replay with pause/restart | ✔ | `python -m levelforge watch` |
| Test suite (19 tests) | ✔ | `tests/` |

No pre-built RL / evolutionary-computation libraries are used — the algorithms are implemented
directly on top of PyTorch, NumPy and (optionally) PyQt5.

---

## Quickstart (Windows, Python 3.10+)

```powershell
# 1. Create a virtual environment (recommended)
cd "C:\Users\krish\OneDrive\Documents\Default Project\LevelForge"
python -m venv .venv
.venv\Scripts\activate

# 2. Install (editable) — PyTorch CPU is enough
pip install -e .
pip install pytest            # for the test suite

# 3. Verify the pipeline end-to-end (a tiny run, finishes in ~1 minute)
python -m levelforge demo

# 4. Run the test suite
python -m pytest -q
```

### Full training (what the synopsis evaluates)

```powershell
# Train a DQN baseline on random solvable levels
python -m levelforge train-baseline --episodes 2000 --pool 60 --size 10

# Run the co-evolution loop (GA generator)
python -m levelforge evolve --generations 12 --population 24 --size 10 --name ga_run

# Ablation: same loop but levels are freshly random (not GA-bred)
python -m levelforge evolve --generations 12 --population 24 --size 10 --generator random --name random_run

# Evaluate a checkpoint on the held-out set
python -m levelforge evaluate --checkpoint runs/ga_run/checkpoint_gen_11.pt --out runs/ga_run

# Turn any saved level into a picture / play it / watch the agent
python -m levelforge render --level runs/ga_run/levels_gen_9/hard_0.json --out hard_0.png
python -m levelforge play   --level runs/ga_run/levels_gen_9/hard_0.json   # you control the agent
python -m levelforge watch  --level runs/ga_run/levels_gen_9/hard_0.json --checkpoint runs/ga_run/checkpoint_gen_11.pt

# Summarise a run
python -m levelforge summarise --metrics runs/ga_run/metrics.jsonl
```

Runs write to `runs/<name>/`: `config.json`, `metrics.jsonl`, per-generation checkpoints, and
the hardest evolved levels (`levels_gen_*/hard_*.json`) so the adversarial cases can be re-used
or inspected. The conservative CPU settings above are a starting point; serious training (e.g.
2000+ episodes, 10×10 boards, multiple seeds) belongs on a GPU or Colab T4, per Section 5.6 of
the synopsis.

---

## How it works

**Environment.** 10×10 (configurable) bordered grids. The agent moves in four directions. Pushes
spend energy: normal = 1, fragile = 1, heavy = 3. Energy exhaustion before solving, any
non-goal box in a crush cell, a shattered fragile box, or the step budget elapsing all end the
episode with those failure categories (`deadlock`, `fragile_break`, `energy_exhaustion`,
`invalid_heavy_push`, `timeout`).

**Observation & DQN.** One-hot grid channels (wall; goal; box kind ×3; agent) plus a normalised
energy scalar, flattened and fed to an MLP: input → 64 → ReLU → 64 → ReLU → 4 Q-values.
Mean-squared TD error, Adam, γ = 0.99, ε decayed from 1.0 to 0.05, a target network synchronised
periodically, and a 20 000-frame experience-replay buffer.

**Guaranteed-solvable generation.** Uniform-random box/goal placement is almost never solvable,
so levels are generated by **reverse Sokoban play** from the solved state: a random agent walks
around and pulls boxes (the exact inverse of a push). Replaying the sequence backwards is a
legal forward solution, so generated levels are solvable by construction. Candidate children
from the GA are additionally filtered by a deadlock-aware, node-capped reverse BFS
(`solvability.py`); rejected children count towards the Solvability-Yield metric.

**GA loop.** Population → evaluate each level with the current agent (failure rate = fitness)
→ tournament selection (k = 3) → 2×2 region-wise wall crossover + per-item goal/box inheritance
→ mutation (wall flips, box/goal shifts, payload kind, energy, agent move) → solvability filter
→ top-k hardest *valid* levels added to the retraining set with ×4 weight. The random ablation
(`--generator random`) swaps GA breeding for fresh random levels to quantify the GA's
contribution.

**Metrics (Section 7 of synopsis).** Per generation we log:
`heldout_success_rate`, `population_avg_failure_rate`, `population_diversity`
(mean pairwise Hamming distance over cell signatures), `solvability_yield`,
`failure_types_seen`, training loss and wall-clock time. `evaluation.py` additionally computes
the failure breakdown and Failure-Discovery Efficiency (episodes to find k distinct failure
types); three independent seeds (`--seed`) should be run and compared with Welch's t-test or
Mann–Whitney U for the headline comparison.

---

## Repository layout

```
LevelForge/
├── pyproject.toml            # package + console script (levelforge)
├── requirements.txt
├── README.md
├── configs/                  # (optional) config snapshots
├── docs/ARCHITECTURE.md      # design notes, rule definitions, report pointers
├── runs/                     # experiment outputs (git-ignored)
├── src/levelforge/
│   ├── grid.py               # Level model, ASCII/JSON serialisation, observation encoding
│   ├── environment.py        # SokobanEnv (step semantics, rewards, failure categories)
│   ├── solvability.py        # deadlock rules + reverse-BFS solvability checker
│   ├── population.py         # reverse-play generator, random levels, repair operators
│   ├── genetic.py            # tournament selection, crossover, mutation, diversity
│   ├── dqn.py                # QNetwork, ReplayBuffer, DQNAgent
│   ├── training.py           # level pools, episode loop, held-out evaluation
│   ├── coevolution.py        # full co-evolution orchestrator + run directory layout
│   ├── evaluation.py         # SR / FDE / breakdown / SY metrics
│   ├── visualize.py          # PNG renderer
│   ├── qt_viewer.py          # interactive viewer (PyQt5)
│   └── cli.py                # command line interface
└── tests/                    # 19 tests: mechanics, serialisation, solvability, GA, DQN
```

---

## Verification performed

* `python -m pytest -q` → **19 passed**
* `python -m levelforge demo` → end-to-end CPU run (pool generation, pretraining, 2 GA
  generations, held-out evaluation, checkpointing) completes and writes a consistent
  `runs/demo/summary.json` — e.g. solvability yield 0.67/0.40 across generations.
* Generated boards verified structurally valid (`Level.validate()`).
* Renderer verified visually (`runs/demo/hard_0.png`).

Honest limitation: with only a few hundred CPU episodes the DQN reaches 0 % held-out solves —
that is expected for Sokoban. The *framework*, *metrics* and *adversarial-curriculum protocol*
are the contribution; meaningful success-rate differences appear with full-length training on a
GPU and ≥ 3 seeds per condition.

---

## Research protocol (for the report)

1. Fixed DQN architecture, training budget and evaluation protocol for both generators.
2. Held-out 20-level set generated with an unused seed; never used for fitness or retraining.
3. ≥ 3 seeds per condition; report mean ± std; as appropriate, Welch's t-test / Mann–Whitney U.
4. Record population/evaluation budgets and wall-clock time for reproducibility.
5. Compare conditions: GA-evolved vs random-level ablation (`--generator {ga,random}`).

## References

See the synopsis document. Core methodological anchors: Mnih et al. 2015 (DQN), Dennis et al.
2020 (PAIRED / UED), Pereira et al. 2021 (evolutionary PCG), Kalafatis et al. 2025 (GA content ×
DRL testing agents), Culberson (Sokoban complexity / deadlocks), Procgen (generalisation in PCG
environments).
