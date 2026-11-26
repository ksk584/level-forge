# Demo levels

GA-discovered levels from `runs/ued_7x7/attack/breaking_ga_seed0.json` (the pretrained agent failed every trial during the attack), re-scored with 8 stochastic rollouts per agent.

Across all 583 levels: pretrained 0.03, GA-trained 0.34, random-trained 0.32 mean success.

| Level | Optimal moves | Pretrained | GA-trained | Random-trained |
|---|---|---|---|---|
| `01_learned` | 18 | 0.00 | 1.00 | 0.00 |
| `02_learned` | 14 | 0.00 | 1.00 | 0.12 |
| `03_learned` | 13 | 0.00 | 1.00 | 0.12 |
| `04_unsolved` | 26 | 0.00 | 0.00 | 0.00 |
| `05_unsolved` | 24 | 0.00 | 0.00 | 0.00 |

Watch commands (PowerShell, from the repo root, after `$env:PYTHONPATH = "src"`):

```
py -m levelforge watch --level demo/levels/01_learned.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip
py -m levelforge watch --level demo/levels/01_learned.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/01_learned.json --tk --solver

py -m levelforge watch --level demo/levels/02_learned.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip
py -m levelforge watch --level demo/levels/02_learned.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/02_learned.json --tk --solver

py -m levelforge watch --level demo/levels/03_learned.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip
py -m levelforge watch --level demo/levels/03_learned.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/03_learned.json --tk --solver

py -m levelforge watch --level demo/levels/04_unsolved.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip
py -m levelforge watch --level demo/levels/04_unsolved.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/04_unsolved.json --tk --solver

py -m levelforge watch --level demo/levels/05_unsolved.json --tk --ppo runs/ued_7x7/pretrain_seed0/model.zip
py -m levelforge watch --level demo/levels/05_unsolved.json --tk --ppo runs/ued_7x7_learn/ga_seed0/model.zip
py -m levelforge watch --level demo/levels/05_unsolved.json --tk --solver

```