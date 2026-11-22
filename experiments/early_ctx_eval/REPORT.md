# GOAL B – Early detection on whole recordings (experiments/early_ctx_eval)

Status 22 Nov 2026, 12:10. **Gate verdict: FAILED.** No variant reaches ≥ 60 % of throw motions on held-out whole recordings at ≤ 2–3 pause false alarms (FA) per minute with a lead ≤ −200 ms. That holds even when only hits above the chance baseline are counted.

## 1. What I changed and why

**Evaluation** (`ctx.py score`, replaces `tools/early_stream.py`):
- **Stateful, no reset.** The whole recording runs as *one* pass, reset only at the start of the file (`stateful_forward`). Equality with the full-sequence forward is checked: 0 differing frames out of 1169.
  - My first evaluation still used the 30 s reset of `early_stream.py`. I discarded those numbers.
- **Pause time as the union of masks**, not as a sum. The recording is cut at the `end` label.
- **Every alarm lands in exactly one category:**
  - Hit: the first unused alarm in the window [peak − 1.0 s, peak + 0.3 s], matched 1:1.
  - Hard-negative FA: within ±1 s of `idle`/`invalid` labels, i.e. walking, picking up the ball, handling the camera.
  - FA near a swing: ±1.5 s around a swing, including double alarms.
  - Pause FA: everything else.
- **Only fresh alarms count.** An alarm is an upward crossing of θ with 1 s refractory time. A score already above θ is not a hit. The column "y ≥ θ at window start" shows how often that happened.
- **Chance baseline:** the same alarms, with all label times of a recording shifted circularly by a random offset, 100 draws. Mean and 95th percentile of the hits are given.
- **Lead:** against `release_ms` where available (33/42 throws in 02/03), otherwise against the peak, in separate columns. Negative means early.
- **θ is chosen per fold on that fold's training sessions only** (02/03, without 215305). Rule, fixed before any results: maximum recall with pause FA ≤ 1.0/min. Where that is never reached, the smallest FA applies.
  - As a sensitivity check, clearly *post hoc*: the same rule with ≤ 2.5/min.
- **A review found 3 errors in my first evaluation, all fixed:**
  - Alarms near swings were not counted anywhere.
  - Hard negatives next to swings were masked out.
  - One alarm could hit two swings.
- **Folds unchanged** (f0 = 170441, 201703, 215305; f1 = 195808, 200001, 200828, 201452, 201622). Only this keeps the baselines held out. 211821 is used only as a training negative and is never scored.

**Data and training** (`ctx.py train`, Tim's v1 architecture, 1 output, DVS-Gesture init, τ ≤ 8, EMA loss as in Tim's runs):
- Training uses **windows (3 s + 0.6 s warm-up) from the whole recordings** of the training sessions instead of spliced clips.
  - That makes real pauses, walking, picking up and camera handling in the throw scene into negatives.
  - `idle`/`invalid` get weight 2 (4 for `whneg`).
  - The target is 1 on [release − 0.5 s, release + 0.3 s], or on the peak where no release is known.
  - 211821 is added as a calm negative. Light augmentation: flip, x shift, thinning, polarity swap, noise.
  - This tests cause 1 (missing pause negatives) and cause 2 (learning on spliced streams).
- The frames come from `prep.py`, which bins bit-identically to `ball_live.frames_from_file`.
- **Cause 3 (preprocessing) was checked and ruled out:** Tim's bank clips and the live binning agree to 96–99.9 % (IoU, offset 0, 12 clips).
- **Variants**, one hypothesis each:
  - `whole`: whole recordings only. 2 seeds (s0, s1).
  - `whfor`: like `whole`, but 25 % of every batch are foreign negatives from Tim's banks (mb/dd/dvsg). Seeds s0, s1, see below.
  - `whneg`: `whole_s0` trained on for 600 steps with stronger negative weighting (p_pos 0.3, hn_w 4, pos_w 1). Hypothesis: the pauses are underfitted, because even in-sample the pause FA at θ 0.8 was still 5.7/min.
  - `_c1`: **post-hoc bias clamp against tonic channels.** The BN-folded bias is limited to ≤ 0.9·v_th/τ, without retraining. Afterwards no channel is tonic any more (count from `tonic_report`).
- **Readouts:**
  - `abs`: EMA with α 0.85 (Tim's).
  - `slow`: EMA with α 0.93.
  - `rel`: y minus its own mean over [t − 1.7 s, t − 0.2 s], i.e. the rise.

## 2. Results (stateful, 02/03 held out, 107 swings = 42 throws + 65 feints)

Common to all rows:
- Real pause time 3.02 min, hard-negative time 2.87 min, total duration 10.2 min.
- Hit window [−1.0, +0.3] s around the peak.
- FA upper bound: one-sided 95 %, Poisson.
- 215305 is reported separately, with the fold-0 model, key-press labels and an invisible ball.

### 2a. Main protocol (θ rule FA ≤ 1/min on training, fixed in advance)

| Model | Readout | θ f0/f1 | Throws | Feints | Total | Chance (mean, p95) | Pause FA (/min, ≤95 %) | Hard-neg FA /min | FA near swing | Lead vs release (median, n) | Lead vs peak | y≥θ at window start | 215305 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2_old_fa (baseline) | abs | .99/.99 | 22/42 | 38/65 | **60/107 (56 %)** | 40 (46) | 28 → **9.27** (≤12.7) | 18.8 | 32 | −80 (13) | −475 | 17 | 5/65, 1.8 FA/min |
| r2_ext_fa (baseline) | abs | .99/.99 | 17/42 | 28/65 | **45/107 (42 %)** | 22 (27) | 9 → **2.98** (≤5.2) | 10.8 | 8 | **+145** (10) | +20 | 5 | 4/65, 1.1 |
| r2_old_fa + clamp | abs | .99/.97 | 12/42 | 29/65 | 41/107 (38 %) | 29 (35) | 14 → 4.64 | 15.3 | 31 | −365 (8) | −520 | 9 | 1/65 |
| r2_ext_fa + clamp | abs | .99/.99 | 18/42 | 22/65 | 40/107 (37 %) | 22 (28) | 6 → 1.99 | 11.2 | 14 | +190 (11) | +140 | 3 | 5/65 |
| whole_s0 | abs | .99/.97 | 3/42 | 6/65 | 9/107 (8 %) | 6 (9) | 0 → 0.00 (≤1.0) | 4.2 | 3 | −515 (2) | −570 | 0 | 0/65, 0 FA |
| whole_s1 | abs | .99/.97 | 7/42 | 11/65 | 18/107 (17 %) | 9 (13) | 1 → 0.33 | 4.5 | 4 | −470 (2) | −320 | 0 | 0/65 |
| whole_s1 | rel | .90/.70 | 8/42 | 17/65 | 25/107 (23 %) | 7 (11) | 3 → 0.99 (≤2.6) | 1.7 | 0 | −410 (7) | −430 | 0 | 0/65 |
| whfor_s0 | abs | .99/.90 | 11/42 | 16/65 | 27/107 (25 %) | 12 (17) | 3 → 0.99 (≤2.6) | 5.6 | 3 | −290 (7) | −160 | 0 | 0/65 |
| whfor_s0 | slow | .99/.75 | 13/42 | 15/65 | 28/107 (26 %) | 11 (15) | 3 → 0.99 | 4.5 | 4 | +60 (7) | −285 | 0 | 0/65 |
| whole_s0 + clamp | abs | .99/.80 | 12/42 | 12/65 | 24/107 (22 %) | 11 (15) | 1 → 0.33 | 5.6 | 6 | −370 (7) | +20 | 1 | 0/65 |
| whneg_s0 | abs | .99/.95 | 5/42 | 5/65 | 10/107 (9 %) | 5 (7) | 0 → 0.00 (≤1.0) | 1.7 | 2 | +45 (4) | −440 | 0 | 0/65 |
| whneg_s0 | slow | .97/.65 | 19/42 | 16/65 | 35/107 (33 %) | 16 (21) | 3 → 0.99 (≤2.6) | 8.7 | 7 | +65 (14) | −30 | 2 | 0/65 |
| whneg_s0 + clamp | rel | .90/.55 | 17/42 | 17/65 | 34/107 (32 %) | 12 (16) | 5 → 1.66 (≤3.5) | 4.2 | 5 | −435 (12) | −410 | 1 | 0/65 |
| whfor_s1 | abs | .99/.98 | 2/42 | 2/65 | 4/107 (4 %) | 3 (5) | 0 → 0.00 (≤1.0) | 1.4 | 0 | −450 (2) | −90 | 0 | 0/65 |
| whfor_s1 | rel | .90/.70 | 2/42 | 9/65 | 11/107 (10 %) | 5 (8) | 1 → 0.33 | 2.4 | 3 | +200 (1) | −330 | 0 | 0/65 |

The `rel` readout of the baselines gives 1/107 (the θ choice on training collapses).

### 2b. Sensitivity, post hoc: θ rule pause FA ≤ 2.5/min on training

| Model | Readout | θ f0/f1 | Total | Chance (mean, p95) | Pause FA /min (≤95 %) | Hard-neg FA /min | Lead vs release |
|---|---|---|---|---|---|---|---|
| r2_ext_fa | abs | .99/.97 | 57/107 (53 %) | 30 (35) | 3.64 (≤6.0) | 16.0 | +80 |
| whole_s1 | rel | .80/.65 | 33/107 (31 %) | 11 (15) | 2.32 (≤4.4) | 3.1 | −415 |
| whole_s1 | abs | .99/.90 | 40/107 (37 %) | 17 (23) | 0.99 (≤2.6) | 7.7 | +80 |
| **whfor_s0** | **rel** | .80/.60 | **40/107 (37 %)** | 14 (19) | **1.66** (≤3.5) | 4.2 | **−505** (n=16) |
| whole_s0 | abs | .99/.85 | 33/107 (31 %) | 16 (21) | 0.99 | 9.8 | +75 |
| whfor_s1 (seed 1) | rel | .85/.70 | 12/107 (11 %) | 6 (9) | 0.33 | 2.4 | −240 (n=2) |
| whfor_s1 (seed 1) | abs | .97/.90 | 19/107 (18 %) | 9 (12) | 0.00 | 3.8 | −135 |
| whneg_s0 | slow | .80/.65 | 35/107 (33 %) | 16 (21) | 0.99 | 8.7 | +50 |

For context, reported only and not used for selection: the best point with θ chosen on the *test* data (oracle) is whfor_s0 abs θ 0.70 with 68/107 (64 %) at 2.65 pause FA/min and a lead of −445 ms. But it has 16.4 hard-negative FA/min and 34 FA near swings, and the chance baseline is 34 (p95 42). That is +34 hits over chance, more than any baseline (+20 to +23). **Seed 1 confirms it:** whfor_s1 abs θ 0.70 reaches 66/107 at chance 31 (p95 37), 3.3 pause FA/min, 12.6 hard-negative FA/min and a lead of −290 ms. The gap shows that choosing θ on the noisy training sessions (03, much camera handling) is a bottleneck here. `oracle_chance.py` reproduces the number.

## 3. Gate verdict: failed

Measured as hits above the chance mean, all variants are at about +20 to +26 out of 107. That is 19–24 %, far below 60 %.

**Baselines:**
- Run statefully, r2_old_fa does reach 56 %, but at 9.3 pause FA/min. 40 of those hits are chance.
- r2_ext_fa fires after the release in the median (+145 ms).

**New variants:**
- **Training on whole recordings** clearly lowers pause and hard-negative FA: 0–1.7 instead of 3–9 pause FA/min, 2–6 instead of 11–19 hard-negative FA/min.
- The leads are consistently early: −400 to −500 ms with `rel`.
- **But recall collapses.** With the θ rule fixed in advance they reach 8–26 % against 42–56 % for the baselines.
- Best honest point (2b, post hoc): whfor_s0 rel with 37 %, about 2.7× chance, 1.7 pause FA/min and −505 ms lead. That does not pass the gate and **does not replicate with seed 1** (12/107). The θ choice is unstable across seeds.

**Further observations:**
- The **bias clamp** (tonic channels → 0) lowers the FA and chance rate of the baselines, but does not raise recall over chance.
  - The drift test (60 s of noise) still gives 12 instead of 55 output spikes after the clamp for r2_old_fa f1, but 56 instead of 16 for whole_s0 f1. So the clamp alone is not enough.
  - The new models also have tonic channels (L2 up to 22/64), because Tim's training does not penalise them.
- **215305** (fold-0 models, trained on 03): the new models are silent there, 0/65 at 0 FA. The baselines reach 4–5/65. That fits "different setting, ball invisible, labels by key press".
- **Fold 0 has hardly any test swings** (17 in 02/03). The θ choice for fold 1 rests on only 0.4 min of pause (170441 + 201703).

**Diagnosis:**
- The real negatives of the throw scene are the right lever against "a person is moving".
- But a network with τ ≤ 80 ms and 800 steps does not yet separate the wind-up from walking or picking up. Even in-sample the pause FA stays high.
- The only positive signal, with oracle θ only and therefore not a metric: whfor (whole recordings plus foreign negatives) at a fixed θ 0.70 is at +34/+35 hits over chance in both seeds (66–68/107, lead −290 to −445 ms). The baselines reach +20 to +23. The price is 2.7–3.3 pause FA/min and 12–16 hard-negative FA/min. An honest test would need a separate validation session for choosing θ.

**Next steps, not today:**
- More steps.
- A bias penalty in training instead of a post-hoc clamp.
- More pause clips from 02/03, i.e. `_dataset01_010203`.
- Longer time constants, if exportable.

## 4. Checkpoints, θ, readout

- **Local:** `experiments/early_ctx_eval/runs/<variant>/model_f{0,1}.pt`, i.e. whole_s0, whole_s1, whfor_s0, whfor_s1 and whneg_s0, plus the training logs `trainlog_f*.json`.
- **Outputs:**
  - stateful streams: `experiments/early_ctx_eval/sstreams/<variant>_c{0,1}/f{fold}_{rec}.npz`, with `info.json` (tonic channels, drift, sanity)
  - scores: `score_stateful.json` (2a) and `score_stateful_fa25.json` (2b)
- **GPU machine:** `~/early_ctx/runs/<variant>/model_f{0,1}.pt`, streams under `~/early_ctx/sstreams/`, logs under `~/early_ctx/logs/`. Tim's files are unchanged; `~/early_ctx/tim/` holds copies only.
- **If anything is followed up at all:** whfor_s0 with the `rel` readout (rise over the mean of [t − 1.7, t − 0.2] s of the EMA with α 0.85), θ 0.80 (fold-0 model) or 0.60 (fold-1 model), refractory time 1 s.
  - This is **not a released demo model.** Nothing was exported.

## 5. Reproduce

```bash
# locally (main checkout): frames + labels.json
python3 experiments/early_ctx_eval/prep.py
rsync -az experiments/early_ctx_eval/{frames,labels.json,ctx.py,table.py} <user>@<gpu-host>:early_ctx/
# GPU machine (once: copies of Tim's code, change nothing)
mkdir -p ~/early_ctx/tim && cp ~/early/early_cv.py ~/robust/snn_model.py ~/robust/speedpatch.py ~/early_ctx/tim/
cd ~/early_ctx; PY=~/snn_pretrain/.venv/bin/python
for f in 0 1; do
  $PY -W ignore ctx.py train --fold $f --out runs/whole_s0 --seed 0
  $PY -W ignore ctx.py train --fold $f --out runs/whole_s1 --seed 1
  $PY -W ignore ctx.py train --fold $f --out runs/whfor_s0 --seed 0 --foreign 0.25
  $PY -W ignore ctx.py train --fold $f --out runs/whfor_s1 --seed 1 --foreign 0.25
  $PY -W ignore ctx.py train --fold $f --out runs/whneg_s0 --seed 0 --resume runs/whole_s0 --steps 600 --lr 5e-4 \
      --p-pos 0.3 --hn-w 4 --pos-w 1.0 --early-w 1.5
done
for r in whole_s0 whole_s1 whfor_s0 whfor_s1 whneg_s0; do for c in 0 1; do
  $PY -W ignore ctx.py stream --model-dir runs/$r --out sstreams/${r}_c$c --clamp $c; done; done
for r in r2_old_fa r2_ext_fa; do for c in 0 1; do
  $PY -W ignore ctx.py stream --model-dir ~/early/runs_ext/$r --out sstreams/${r}_c$c --clamp $c; done; done
$PY -W ignore ctx.py score --n-chance 100 --streams sstreams/* --json score_stateful.json            # 2a
$PY -W ignore ctx.py score --n-chance 100 --fa-target 2.5 --streams sstreams/* --json score_stateful_fa25.json  # 2b
$PY table.py score_stateful.json
```

## 6. Limitations

- **Seeds:** whole has 2 seeds, and they scatter widely (abs 9 vs. 18 hits, `rel` 5 vs. 25). whfor has 2 seeds: s0 reaches 25–37 % with the chosen θ, s1 4–18 %. At a fixed θ 0.70 both are similar (68 and 66/107, chance 34 and 31). whneg has only 1 seed.
  - Differences below about 10 hits are not reliable.
- **Pause time:** only 3.0 min of real pause in 02/03. The Poisson upper bound at 0 FA is therefore already 1.0/min, and the FA rates are coarse.
  - The fold-1 θ choice rests on 0.4 min of pause.
- **In-sample:** θ is chosen on the training sessions the models have seen. That is optimistic for FA, and the model over-reacts there, which is why θ mostly ends up at 0.99. There was no separate validation session; there are too few sessions for that.
  - 2b is post hoc (the FA limit was raised after looking at the test results). The oracle number is explicitly not a metric.
- **Incomplete:** `sstreams/whfor_s0_c1` and `sstreams/whole_s1_c1` were stopped for lack of time (3 files each) and are not scored. For `score`, list the directories individually instead of `sstreams/*`. `obsolete_segmented/` holds the discarded first round with the 30 s reset.
- **Labels:** 215305 has key-press labels. The `idle`/`invalid` windows are ±1 s around point labels, not real segments. 9 of the 42 throws have no `release_ms` and only count towards the peak lead.
- **Chance baseline:** computed only for the chosen θ. The circular shift keeps the alarm rate but not the coupling between activity and swing, so it is a lower bound for what a pure motion detector would hit.
