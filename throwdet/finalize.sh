#!/usr/bin/env bash
# Export a final model (trained with --folds 0) with the readout picked on the out-of-fold sweep of
# its CV run, at three false-alarm budgets: strict (1/min), default (2/min), sensitive (4/min).
#   bash throwdet/finalize.sh <final_run_tag> <cv_run_tag> <name>
set -e
cd ~/robotics_hackathon
final=~/throwdet/runs/$1; cv=~/throwdet/runs/$2; out=~/throwdet/deliver/$3
mkdir -p $out
pick() { python3 -c "
import json,sys
d=json.load(open('$cv/readout_sweep.json'))
r=[x for x in d['rows'] if x['budget_fa_per_min']==$1][0]
print(r['alpha'], r['hold'], r['thr'])"; }
read a h t < <(pick 2.0)
.venv/bin/python throwdet/export.py $final/final.pt --alpha $a --hold $h --thr $t --n-check 160 --out $out > $out/export_log.txt 2>&1
for b in 1.0 4.0; do
  read a2 h2 t2 < <(pick $b)
  python3 -c "
import json; r=json.load(open('$out/readout.json')); r.update(alpha=$a2, threshold=$t2, hold_frames=$h2)
json.dump(r, open('$out/readout_fa$b.json','w'), indent=1)"
done
cp $out/readout.json $out/readout_fa2.0.json
cp $final/final.pt $out/model.pt
cp $cv/readout.txt $out/cv_readout.txt
echo "exported $3 ($1, readout from $2): alpha $a hold $h thr $t"
