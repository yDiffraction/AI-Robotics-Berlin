#!/bin/bash
# usage: launch_ext.sh <banks dir> <tag> <extra early_cv args...>
# One background process per fold (2 folds), 800 steps, target on 500 ms before release.
# This is how every run in jobs_ext.txt was started on the A100 (~/early). ~10 GB GPU memory each.
cd ~/early
banks=$1; tag=$2; shift 2
mkdir -p runs_ext/$tag
for k in 0 1; do
  OMP_NUM_THREADS=2 nohup ~/snn_pretrain/.venv/bin/python -W ignore early_cv.py --banks $banks --out runs_ext \
    --tag $tag --folds 2 --only-fold $k --steps 800 --onset 500 --dc 200 "$@" > runs_ext/$tag/f$k.log 2>&1 &
done
