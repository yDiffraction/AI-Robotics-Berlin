#!/bin/bash
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${1:-8}
for f in 0 1; do
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/v2a --seed 0 --resume runs/whfor_s0 --lr 5e-4 --bs 24 --minutes $M > v2/logs/v2a_f$f.log 2>&1 &
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/v2b --seed 1 --resume runs/whfor_s1 --lr 5e-4 --bs 24 --minutes $M > v2/logs/v2b_f$f.log 2>&1 &
done
wait
for r in v2a v2b; do $PY -W ignore ctx2.py stream --model-dir v2/runs/$r --out v2/sstreams/$r > v2/logs/stream_$r.log 2>&1 & done
wait
$PY -W ignore ctx2.py score --streams v2/sstreams/v2a v2/sstreams/v2b --json v2/score2_val.json > v2/logs/score2_val.log 2>&1
echo ALLDONE >> v2/logs/score2_val.log
