# LevelForge demo (about 10 minutes)

Run everything from the `level-forge` folder in PowerShell. Do this once at the start:

```powershell
$env:PYTHONPATH = "src"
```

Each `watch` command opens a small window. Keys: **R** restart, **+/-** speed, **Esc** close.

---

## 1. The idea (1 min, no commands)

> "An RL agent learns to play Sokoban. A genetic algorithm acts as its opponent: it evolves new levels that
> the agent fails on. Every level is checked by an exact A* solver, so the GA cannot cheat with impossible
> levels. I compare it with random search and random-level training across 3 seeds, 4 conditions and
> 3 board settings."

## 2. Live: the GA finds weaknesses faster than random search (1 min)

```powershell
py -m levelforge ued-attack --exp runs/demo_live --config configs/ued_7x7.json --model runs/ued_7x7/pretrain_seed0/model.zip --seeds 0 --budget 160 --threads 6
```

It runs in about 10 s. Expected result: **GA ≈ 82 breaking levels vs random search ≈ 55** from the same
160 tries. Then open the full-scale version of this result, 1024 tries × 3 seeds, where the GA finds
**1.73×** as many:
`docs/results/ued_7x7/attack_efficiency.png`

## 3. Watch: a level the GA discovered, before and after training (3 min)

Level `demo/levels/01_learned.json` is one the GA invented. The pretrained agent failed it every time.

```powershell
# Before: the pretrained agent wanders and runs out of steps
py -m levelforge watch --level demo/levels/01_learned.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip

# After: the agent trained on the GA curriculum solves it
py -m levelforge watch --level demo/levels/01_learned.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip

# Reference: the A* solver's optimal solution (18 moves)
py -m levelforge watch --level demo/levels/01_learned.json --tk --solver
```

Levels `02_learned` and `03_learned` work the same way.

**Say this honestly:** these three levels were *chosen* as clear examples. Over all 583 levels the GA found,
the GA-trained agent solves 34 % and the random-trained agent 32 % (`demo/demo_levels.md`). The GA curriculum
does not beat random training overall, and that finding is part of the result (step 5).

## 4. A failure mode that remains (1 min)

```powershell
py -m levelforge watch --level demo/levels/04_unsolved.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/04_unsolved.json --tk --solver
```

The agent starts next to the goal, but the box is far away and has to go up a narrow column first. Every
agent fails this (181 of the 583 GA levels are like this). This is the "qualitative failure analysis" part
of the project.

## 5. Results (3 min), from `docs/RESULTS_DRAFT.md`

Show the figures in this order:

1. `docs/results/ued_7x7/attack_efficiency.png`: **the GA is a better adversary** (1.73×, then 1.40× with
   two boxes and 1.11× on 9×9). The advantage is biggest when failures are rare.
2. `docs/results/ued_7x7/curriculum_regret.png` and `success_curves.png`: **but a good adversary is not
   automatically a good teacher.** Regret picks levels the agent cannot learn from (0 % before training,
   only 55 % after).
3. `docs/results/ued_7x7_learn/fitness_ablation.png`: **the fix**. Learnability fitness picks levels at
   about 50 % success; the agent masters them (99 %), and robustness on adversarial levels recovers.
4. Two boxes (`docs/results/ued_7x7_2box/results.md`): the one setting with a clear robustness gain
   (+0.09 on other seeds' GA levels).

## 6. Optional: let your teacher try a GA level (1 min)

Arrow keys or WASD; R resets.

```powershell
py -m levelforge play --level demo/levels/04_unsolved.json --tk
```

People usually find it harder than it looks, which makes the point that the GA finds genuinely tricky
levels and not broken ones.

---

## Likely questions

| Question | Short answer |
|---|---|
| Why a GA, not a neural generator (PAIRED)? | Crossover recombines level parts, every child is A*-verified, and it needs no extra training. It found 1.73× more breaking levels than random search with the same budget. |
| How is this different from ACCEL? | ACCEL edits levels with small mutations only. We use a full GA with crossover, an exact solver as the regret reference, hard solvability checks, and a random-search control with identical selection. |
| Why PPO and not your DQN? | The DQN never beat a random policy. If the agent fails everything, every level gets the same fitness and the GA turns into random search. |
| Is the result significant? | With 3 seeds the smallest possible p is 0.10. I report effect sizes and bootstrap CIs. The attack result is consistent on every seed in every setting. |
| Why didn't the GA curriculum win? | Regret selects levels beyond the agent's learning frontier. The learnability ablation confirms this. On small boards, random levels are already close to the frontier. |
| What would you do next? | 9×9 with two boxes on a GPU (config ready), 5+ seeds, and evolve by regret but train by learnability. |
