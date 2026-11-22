#!/bin/bash
# wait for both fold checkpoints of a run, then stream all scored recordings through them
cd ~/early_ctx
r=$1
while [ ! -f runs/$r/model_f0.pt ] || [ ! -f runs/$r/model_f1.pt ]; do sleep 10; done
sleep 5
~/snn_pretrain/.venv/bin/python -W ignore ctx.py stream --model-dir runs/$r --out streams/$r > logs/stream_$r.log 2>&1
echo done >> logs/stream_$r.log
