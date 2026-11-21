#!/usr/bin/env bash
# Queue worker: runs lines "tag|train.py args" from ~/throwdet/queue.txt, one at a time.
# A job is claimed by creating ~/throwdet/runs/<tag>/ (mkdir is atomic). Touch ~/throwdet/PAUSE to hold.
cd ~/robotics_hackathon
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/throwdet/runs
while true; do
  [ -f ~/throwdet/PAUSE ] && { sleep 30; continue; }
  job=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    tag=${line%%|*}
    if mkdir ~/throwdet/runs/$tag 2>/dev/null; then job=$line; break; fi
  done < ~/throwdet/queue.txt
  if [ -z "$job" ]; then sleep 30; continue; fi
  tag=${job%%|*}; args=${job#*|}
  echo "$(date -u +%T) start $tag"
  uv run python throwdet/train.py --tag $tag $args > ~/throwdet/runs/$tag/stdout.txt 2>&1
  echo "$(date -u +%T) done $tag ($?)"
done
