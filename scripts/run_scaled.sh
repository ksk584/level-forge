#!/usr/bin/env bash
# Scaled-up experiments, two chains in parallel (3 threads each). Every step is resumable.
cd "$(dirname "$0")/.." && export PYTHONPATH=src
T=3
chain_a() {  # more boxes: 7x7, 2 boxes
  E=runs/ued_7x7_2box; C=configs/ued_7x7_2box.json; mkdir -p $E
  for s in 0 1 2; do for c in ga random_search random; do
    py -m levelforge ued-run --exp $E --config $C --seeds $s --condition $c --threads $T || return 1
  done; done
  py -m levelforge ued-attack --exp $E --config $C --seeds 0 1 2 --budget 1024 --threads $T || return 1
  py -m levelforge ued-analyze --exp $E
}
chain_b() {  # curriculum-share ablation, then larger board: 9x9, 1 box
  E=runs/ued_7x7_mix70; C=configs/ued_7x7_mix70.json
  for s in 0 1 2; do for c in ga random; do
    py -m levelforge ued-run --exp $E --config $C --seeds $s --condition $c --threads $T || return 1
  done; done
  py -m levelforge ued-analyze --exp $E --no-cross-eval
  E=runs/ued_9x9_1box; C=configs/ued_9x9_1box.json; mkdir -p $E
  for s in 0 1 2; do for c in ga random; do
    py -m levelforge ued-run --exp $E --config $C --seeds $s --condition $c --threads $T || return 1
  done; done
  py -m levelforge ued-attack --exp $E --config $C --seeds 0 1 2 --budget 1024 --threads $T || return 1
  py -m levelforge ued-analyze --exp $E
}
chain_a > runs/chain_a.log 2>&1 & A=$!
chain_b > runs/chain_b.log 2>&1 & B=$!
wait $A; echo "chain A exit=$?"; wait $B; echo "chain B exit=$?"
