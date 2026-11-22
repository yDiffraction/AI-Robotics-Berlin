#!/bin/bash
# Full external-positive build on a GPU box (A100). Local runs only did a sample (see README).
#
#   OWN=~/robotics_hackathon/dataset/_dataset01_01020304 OUT=~/extpos JOBS=32 bash run_cloud.sh
#
# Rough times on an A100 box: DailyDVS fetch ~15 min (10.3 GB, range requests), match ~5 min;
# HMDB fetch ~2 min (0.5 GB), v2e with SuperSloMo on the GPU ~30-60 min for ~1,500 clips.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OWN=${OWN:-}                       # our clip dataset (labels.csv); optional if own_profile.json exists
OUT=${OUT:-$HOME/extpos}
JOBS=${JOBS:-32}
GPU_JOBS=${GPU_JOBS:-4}
mkdir -p "$OUT"
cd "$HERE"

# --- environments: v2e needs Python 3.14 (v2e 1.8.5), kept apart from the training venv
if [ ! -x "$OUT/v2eenv/bin/python" ]; then
  uv venv -q --python 3.14 "$OUT/v2eenv"
  uv pip install -q --python "$OUT/v2eenv/bin/python" v2e scikit-image
fi
PY=$OUT/v2eenv/bin/python  # --- 1. targets from our own clips (or a profile copied from the laptop)
if [ ! -f "$OUT/own_profile.json" ]; then
  [ -f own_profile.json ] && cp own_profile.json "$OUT/" || $PY profile_own.py "$OWN" "$OUT/own_profile.json"
fi

# --- 2. DailyDVS-200: real DVS, throw classes + same-pipeline negatives + amb (network/CPU bound)
dd() {
  [ -f "$OUT/dailydvs/index_all.csv" ] || $PY fetch_dailydvs.py index --out "$OUT/dailydvs"
  $PY fetch_dailydvs.py fetch --out "$OUT/dailydvs" --workers 16
  $PY match.py --src dd --raw "$OUT/dailydvs" --profile "$OUT/own_profile.json"       --out "$OUT/clips_dd" --variants-pos 3 --neg-random --jobs "$JOBS"
  $PY to_bank.py "$OUT/clips_dd" "$OUT/banks/dd.npz" --src dd
}

# --- 3. HMDB51 via v2e (SuperSloMo on the GPU; camera-motion filter inside v2e_convert)
hmdb() {
  $PY fetch_hmdb.py --out "$OUT/hmdb"
  $PY v2e_convert.py --manifest "$OUT/hmdb/manifest.csv" --videos "$OUT/hmdb/videos"       --out "$OUT/hmdb/raw" --interp slomo --roles pos --jobs "$GPU_JOBS" --threads 4
  $PY v2e_convert.py --manifest "$OUT/hmdb/manifest.csv" --videos "$OUT/hmdb/videos"       --out "$OUT/hmdb/raw" --interp slomo --roles neg,amb --limit-per-class "${HMDB_NEG_PER_CLASS:-40}"       --skip-bad --jobs "$GPU_JOBS" --threads 4
  $PY match.py --src hmdb --raw "$OUT/hmdb/raw" --profile "$OUT/own_profile.json"       --out "$OUT/clips_hmdb" --variants-pos 4 --time-jitter-us 33333 --jobs "$JOBS"
  $PY to_bank.py "$OUT/clips_hmdb" "$OUT/banks/hmdb.npz" --src hmdb
}

dd > "$OUT/dd.log" 2>&1 &
hmdb > "$OUT/hmdb.log" 2>&1 &
wait

# --- 4. shortcut probe (needs our clips)
[ -n "$OWN" ] && $PY probe.py --own "$OWN" --ext "$OUT/clips_dd" --ext "$OUT/clips_hmdb" --out "$OUT/probe.json"
d" --ext "$OUT/clips_hmdb" --out "$OUT/probe.json"
echo EXTPOS_DONE
