#!/bin/bash
# longer follow-up: clean runs from DVS-Gesture init with all fixes, 2 seeds at tau<=8, 1 seed at tau<=16
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${1:-40}
for f in 0 1; do
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/v3_t8_s0  --seed 0 --bs 32 --minutes $M > v2/logs/v3_t8_s0_f$f.log 2>&1 &
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/v3_t8_s1  --seed 1 --bs 32 --minutes $M > v2/logs/v3_t8_s1_f$f.log 2>&1 &
  $PY -W ignore ctx2.py train --fold $f --out v2/runs/v3_t16_s0 --seed 0 --bs 32 --minutes $M --tau-max 16 > v2/logs/v3_t16_s0_f$f.log 2>&1 &
done
wait
for r in v3_t8_s0 v3_t8_s1 v3_t16_s0; do $PY -W ignore ctx2.py stream --model-dir v2/runs/$r --out v2/sstreams/$r > v2/logs/stream_$r.log 2>&1 & done
wait
$PY -W ignore ctx2.py score --streams v2/sstreams/v3_t8_s0 v2/sstreams/v3_t8_s1 v2/sstreams/v3_t16_s0 --json v2/score2_v3.json > v2/logs/score2_v3.log 2>&1
echo ALLDONE >> v2/logs/score2_v3.log
