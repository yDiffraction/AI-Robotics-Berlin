#!/usr/bin/env bash
# Adapt a delivered model to a new site in ~5-10 min on the A100.
#   bash throwdet/adapt.sh <delivered model dir, e.g. ~/throwdet/deliver/A_neg> <clips dir from cut_recording.py> <name> [steps]
# Fine-tunes the model on the new clips (40 % of every batch) mixed with the original training data,
# then exports it with the same readout (alpha, hold, threshold) into ~/throwdet/deliver/<name>.
set -e
src=$1; new=$2; name=$3; steps=${4:-300}
cd ~/robotics_hackathon
.venv/bin/python throwdet/prep.py --noah ~/throwdet/noah/dataset --build dataset/build --dd ~/ext/dailydvs \
    --own-new $new --out ~/throwdet/data/table_$name.npz | tail -4
sh="own_new:1=0.2,own_new:0=0.2,own:1=0.12,own:0=0.18,own_aug:1=0.03,own_aug:0=0.03,sim:1=0.06,sim:0=0.06,dd:1=0.05,dd:0=0.05"
.venv/bin/python throwdet/train.py --tag adapt_$name --table ~/throwdet/data/table_$name.npz --resume $src/model.pt \
    --folds 0 --steps $steps --lr 5e-4 --density-match 0 --bg 0.3 --shares $sh | grep -vE "^  step [0-9]*[1-9]00/" | tail -3
read a h t < <(python3 -c "import json; r=json.load(open('$src/readout.json')); print(r['alpha'], r['hold_frames'], r['threshold'])")
out=~/throwdet/deliver/$name
.venv/bin/python throwdet/export.py ~/throwdet/runs/adapt_$name/final.pt --alpha $a --hold $h --thr $t --n-check 120 --out $out | tail -3
for f in $src/readout_fa*.json; do cp $f $out/; done
cp ~/throwdet/runs/adapt_$name/final.pt $out/model.pt
echo "adapted model in $out"
