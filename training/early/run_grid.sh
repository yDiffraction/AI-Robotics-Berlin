#!/bin/bash
# usage: run_grid.sh <jobs.txt> <parallel>   jobs.txt lines: <tag> <early_cv args...>
# One process per (config, fold); skips finished folds.  FOLDS (default 4).
cd ~/early
jobs=$1; par=${2:-8}
grep -v '^#' "$jobs" | grep -v '^\s*$' | while read -r tag args; do
  for k in $(seq 0 $((${FOLDS:-4}-1))); do echo "$tag|$k|$args"; done
done | xargs -P "$par" -I{} bash -c '
  IFS="|" read -r tag k args <<< "{}"
  [ -f runs_early/$tag/result_f$k.json ] && exit 0
  mkdir -p runs_early/$tag
  OMP_NUM_THREADS=2 ~/snn_pretrain/.venv/bin/python -W ignore early_cv.py --tag $tag --only-fold $k $args > runs_early/$tag/f$k.log 2>&1'
echo GRID_DONE "$jobs"
