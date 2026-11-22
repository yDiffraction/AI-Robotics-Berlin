#!/usr/bin/env bash
# Runs eval_readout.py on every finished run (all fold models present, no readout_sweep.json yet)
# and rebuilds ~/throwdet/summary.md.
cd ~/robotics_hackathon
while true; do
  for d in ~/throwdet/runs/*/; do
    tag=$(basename $d)
    [ -f $d/readout_sweep.json ] && continue
    grep -q "OWN out-of-fold" $d/log.txt 2>/dev/null || continue
    .venv/bin/python throwdet/eval_readout.py $d > $d/readout.txt 2>&1 && echo "$(date -u +%T) evaluated $tag"
  done
  .venv/bin/python throwdet/summary.py > ~/throwdet/summary.md 2>/dev/null
  sleep 60
done
