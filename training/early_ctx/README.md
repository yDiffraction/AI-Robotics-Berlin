# Early throw-motion detector: whole-recording training (GOAL B)

Trains, streams and scores the early-detection SNN on **whole recordings** (pauses, walking, ball pick-up and
camera handling are real negatives; throws and feints are positive). Runs on the A100 in `~/early_ctx/`;
this folder is a copy of that code. The exported board models are in `live_early/`.

| file | what |
|---|---|
| `ctx2.py` | **current pipeline**: `train` / `stream` / `score` (v2 fixes, see its docstring) |
| `lif_triton.py` | fused Triton PLIF time loop used by `ctx2.py train` (~5x faster, checked against riverops at start of every run) |
| `export_early.py` | checkpoint → board `weights.txt` (arch v1, padded to 11 outputs, Q2.6 / Q4.12, per-layer scaling, float-vs-fixed check) |
| `ctx.py` | Noah's original pipeline (first round, REPORT on the noah branch); `ctx2.py` imports its helpers |
| `prep.py` | whole recordings → `frames/*.npy` (bit-packed 2×80×80 per 10 ms) + `labels.json` |
| `labels.json` | events per recording (throw / feint / idle / invalid / end, release times) for 02/03/04 |
| `tim/` | unchanged copies of Tim's `early_cv.py`, `snn_model.py`, `speedpatch.py` |
| `launch/` | the exact launch scripts of the A100 runs (`launch_val.sh` v2a/v2b, `launch_v3.sh`, `launch_h1*.sh` hires) |
| `results/` | `score2_*.json` of the old models, v2a/v2b and v3 |
| `table.py`, `oracle_chance.py`, `trainsweep.py` | Noah's helper scripts (tables, oracle chance baseline, training-θ sweep) |

## Run

```bash
python prep.py                       # needs dataset/0[234]/samples/*.bin and dataset/_dataset01_01020304/labels.csv
                                     # (noah branch; the raw 02/03 recordings are not in git)
PY=~/snn_pretrain/.venv/bin/python   # on the A100 (CUDA, riverops, triton)
$PY ctx2.py train --fold 0 --out runs/x --minutes 40 --bs 32                       # v1 (default init sweep_v1)
$PY ctx2.py train --fold 0 --out runs/h --minutes 20 --bs 12 --mem-frac 0.24 \
    --init ~/snn_pretrain/runs/sweep_hires/best.pt                                  # hires, half the GPU with 2 folds
$PY ctx2.py stream --model-dir runs/x --out sstreams/x
$PY ctx2.py score --streams sstreams/x --json score_x.json
python export_early.py --ckpt runs/x/model_f0.pt --out export/x --theta 0.85     # v1 only: board firmware is v1
```

Train both folds (`--fold 0` and `--fold 1`) before `stream`. `--fold -1` trains on all sessions (no held-out data).
Scores: the pre-registered θ (chosen on the training sessions), a cross-fold θ, and a θ-free comparison
(best held-out recall at ≤ 1/2/3 pause false alarms per min with its chance baseline; θ picked on test, so it
compares models and is not an operating point).
