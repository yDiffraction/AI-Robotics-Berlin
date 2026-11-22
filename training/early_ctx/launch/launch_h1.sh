#!/bin/bash
# one quick hires run (both CV folds in parallel), capped at half the A100 (2 x 0.24 of GPU memory)
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${1:-20}; R=h1
for f in 0 1; do
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/$R --seed 1 --bs 12 --minutes $M --mem-frac 0.24 \
      --init $HOME/snn_pretrain/runs/sweep_hires/best.pt > v2/logs/${R}_f$f.log 2>&1 &
done
wait
$PY -W ignore ctx2.py stream --model-dir v2/runs/$R --out v2/sstreams/$R --chunk 400 --mem-frac 0.45 > v2/logs/stream_$R.log 2>&1
$PY -W ignore ctx2.py score --streams v2/sstreams/$R v2/sstreams/v3_t8_s1 --json v2/score2_$R.json > v2/logs/score2_$R.log 2>&1
echo ALLDONE >> v2/logs/score2_$R.log
