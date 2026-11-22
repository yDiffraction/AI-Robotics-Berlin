#!/bin/bash
# usage: run_grid.sh <jobs.txt> <parallel>   jobs.txt lines: <tag> <robust_cv args...>
# Each config runs as 6 processes (one per fold, both CV seeds); skips finished folds.
cd ~/robust
jobs=$1; par=${2:-24}
grep -v '^#' "$jobs" | grep -v '^\s*$' | while read -r tag args; do
  for k in $(seq 0 $((${FOLDS:-6}-1))); do echo "$tag|$k|$args"; done
done | xargs -P "$par" -I{} bash -c '
  IFS="|" read -r tag k args <<< "{}"
  [ -f runs_robust/$tag/result_f$k.json ] && exit 0
  mkdir -p runs_robust/$tag
  ~/snn_pretrain/.venv/bin/python -W ignore robust_cv.py --tag $tag --only-fold $k $args > runs_robust/$tag/f$k.log 2>&1'
echo GRID_DONE "$jobs"
