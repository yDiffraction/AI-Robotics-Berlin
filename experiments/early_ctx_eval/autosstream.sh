#!/bin/bash
cd ~/early_ctx; r=$1
while [ ! -f runs/$r/model_f0.pt ] || [ ! -f runs/$r/model_f1.pt ]; do sleep 10; done; sleep 5
for c in 0 1; do ~/snn_pretrain/.venv/bin/python -W ignore ctx.py stream --model-dir runs/$r --out sstreams/${r}_c$c --clamp $c > logs/sstream_${r}_c$c.log 2>&1 & done; wait
