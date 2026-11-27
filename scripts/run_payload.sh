#!/usr/bin/env bash
# Payload experiment (heavy + fragile boxes, rough floor): one chain per seed in parallel
# (2 threads each), then the frozen-agent attack and the analysis. Every step is resumable.
cd "$(dirname "$0")/.." && export PYTHONPATH=src
E=runs/ued_7x7_payload; C=configs/ued_7x7_payload.json; T=2
mkdir -p $E
seed_chain() {
  py -m levelforge ued-pretrain --exp $E --config $C --seeds $1 --threads $T || return 1
  for c in ga random_search random fixed; do
    py -m levelforge ued-run --exp $E --config $C --seeds $1 --condition $c --threads $T || return 1
  done
}
for s in 0 1 2; do seed_chain $s > $E/seed$s.log 2>&1 & pids[$s]=$!; done
for s in 0 1 2; do wait ${pids[$s]}; echo "seed $s exit=$?"; done
py -m levelforge ued-attack --exp $E --config $C --seeds 0 1 2 --budget 1024 --threads 6 > $E/attack.log 2>&1
echo "attack exit=$?"
py -m levelforge ued-analyze --exp $E > $E/analyze.log 2>&1
echo "analyze exit=$?"
