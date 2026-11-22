#!/bin/bash
# fold 1 restart of h1 (fold 0 keeps running), then stream + score once both models exist
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=h1
$PY -W ignore ctx2.py train --fold 1 --out v2/runs/$R --seed 1 --bs 12 --minutes ${1:-20} --mem-frac 0.24 \
    --init $HOME/snn_pretrain/runs/sweep_hires/best.pt > v2/logs/${R}_f1.log 2>&1
while [ ! -f v2/runs/$R/model_f0.pt ]; do sleep 10; done; sleep 5
$PY -W ignore ctx2.py stream --model-dir v2/runs/$R --out v2/sstreams/$R --chunk 400 --mem-frac 0.45 > v2/logs/stream_$R.log 2>&1
$PY -W ignore ctx2.py score --streams v2/sstreams/$R v2/sstreams/v3_t8_s1 --json v2/score2_$R.json > v2/logs/score2_$R.log 2>&1
echo ALLDONE >> v2/logs/score2_$R.log
