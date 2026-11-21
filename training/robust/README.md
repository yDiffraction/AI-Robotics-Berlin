# Robust throw / feint training on very few real recordings

`robust_cv.py` is a training and evaluation pipeline for the current data: 24 usable
recordings (13 throw, 11 feint) plus idle stretches cut from the same recordings.
All generated data is derived from these recordings. Nothing is synthesized from scratch.

## Why the old pipeline over-fits early

Leave-one-recording-out logistic regression on trivial features
(`bank.pt`, windows aligned at the swing peak):

| Feature | Time | Accuracy |
|---|---|---|
| active-cell count | first 10 ms frame (−700 ms) | 0.67 |
| active-cell count | first 100 ms | 0.75 |
| active-cell count | −300…−200 ms | 0.83 |
| 8×8 spatial activity map | −300…−200 ms | **0.92** |
| 8×8 spatial activity map | +100…+300 ms (after release) | 0.54 |

The **pre-release activity level and position** separate the classes better than anything
after the release. There are two reasons:

1. Throws and feints were recorded in separate blocks (13:05–13:08 vs 13:08–13:14).
   Throw takes have much more motion before the wind-up: 100–2900 events in the first
   10 ms, against 4–50 for most feints.
2. The auto label uses the post/pre activity ratio, so pre-release activity is partly
   part of the label definition.

A network fits this cue first ("early wrong fitting") and still scores well in CV, but the
cue will not transfer to new people or sessions.

## What the pipeline changes

**Input.** Windows are cut on the fly from the full recordings (`prep_raw.py` → `bank.pt`),
so time jitter uses real context instead of zero padding. Idle windows are quiet stretches
played back and forth, so there are no empty tail frames. Options:

- `--pre/--T`: window length, e.g. −300…+500 ms
- `--pool 2`: 40×40 input
- `--arch tiny`: ~6k-weight scratch model

**Generation from real recordings** (`--gen`):

| Method | What it does | Targets |
|---|---|---|
| `thin` | global event thinning, 30–100 % kept | activity level stops being a label |
| `swap` | context swap: everything before a cut 120–450 ms before release comes from another recording (either class), aligned to the body position | prefix carries no label |
| `overlay` | OR-in a thinned, shifted chunk of another recording | distractor, level noise |
| `mix` | EventMix-style space-time cutmix with soft labels | EventMix (Shen et al.) |
| `extra` | the existing offline `aug_train` + `splice_train` windows, derivatives of held-out recordings removed | baseline for the old generator |

**Loss.**

- `--loss ts`: cross-entropy on the running spike rate at every frame after
  release + 50 ms. Idle clips are scored at every frame.
- `--sym`: prefix symmetry. Before the release, the throw and feint running logits must
  be equal, because the decision cannot be observed yet. This is a direct penalty on early
  wrong fitting, inspired by ALBAR's static-clip penalty and TRT's stronger
  regularization of early time steps.
- AdamW with wd 1e-2, cosine LR, and EMA weights with BN re-estimation. There is no
  early stopping or model selection on test folds.

**Shortcut metrics** are reported next to accuracy:

- `prefix_tf@-200ms`: throw-vs-feint accuracy from the prefix only. It should be ≈ 50 %.
- `swap_tf`: accuracy when the first part of a held-out clip is replaced by an
  opposite-class recording. It should match `acc_tf` if the model follows the motion.

## Run

```bash
python -I prep_raw.py ../../dataset/01/samples bank.pt                       # 67 MB
python -I prep_raw.py ../../dataset/01/samples bank_extra.pt --extra ../../dataset/build aug_train,splice_train
python robust_cv.py --bank bank.pt --tag thin_swap_sym --init <dvsg v1 best.pt> --loss ts --gen thin,swap --sym 0.5 --folds 4
python robust_cv.py --bank bank.pt --tag final --final --steps 600 --init <dvsg v1 best.pt> --loss ts --gen thin,swap,overlay --sym 0.5   # best config, all data
FOLDS=4 ./run_grid.sh jobs_h.txt 11   # A100: ~6 GB per process, 11 in parallel
python aggregate.py runs_robust
```

## Results (21 Nov 2026, A100, 1-hour budget)

**Setup.**

- Grouped 4-fold CV by recording (6 held-out recordings per fold, 24 throw/feint test
  windows in total, plus idle stretches).
- 300 steps, batch 32, v1 backbone pretrained on DVS Gesture, one fold assignment.
- `acc_tf` is the 2-way throw-vs-feint accuracy at the window end. The 95 % CI is a
  bootstrap over recordings, so differences under ~10 points are noise at n = 24.

| Config | acc_tf | 95 % CI | prefix@−200 ms ↓ | swap_tf ↑ | acc3 | idle recall |
|---|---|---|---|---|---|---|
| `h_base_rate`: old pipeline (rate MSE, basic aug) | 0.75 | 0.54–0.88 | 0.71 | 0.60 | 0.50 | 0.36 |
| `h_extra`: + old offline aug_train/splice | 0.79 | 0.58–0.96 | 0.67 | 0.64 | 0.73 | 0.95 |
| `h_ts_sym`: ts loss + prefix symmetry, no generation | 0.71 | 0.52–0.88 | 0.54 | 0.68 | 0.80 | 0.98 |
| `h_thin_swap`: generation, no symmetry | 0.83 | 0.67–0.96 | 0.71 | 0.73 | 0.79 | 0.98 |
| `h_thin_swap_sym` | 0.83 | 0.67–0.96 | 0.54 | 0.76 | 0.74 | 1.00 |
| **`h_tso_sym`: thin + swap + overlay + sym** | **0.85** | 0.71–0.98 | 0.56 | **0.85** | 0.82 | 0.95 |
| `h_tsom_sym`: + EventMix | 0.81 | 0.63–0.94 | 0.63 | 0.81 | 0.83 | 0.95 |
| `h_pre30`: window −300…+500 ms (T = 80) | 0.81 | 0.65–0.94 | 0.48 | 0.81 | 0.82 | 1.00 |
| `h_pool2`: 40×40 input | 0.73 | 0.54–0.90 | 0.67 | 0.72 | 0.79 | 0.86 |
| `h_tiny`: 6k weights, scratch | 0.69 | 0.49–0.85 | 0.56 | 0.60 | 0.79 | 0.91 |

**Findings.**

- **The old pipeline decides from the prefix.** At −200 ms it is already 71 % "right", and
  when the prefix comes from the other class it drops to 60 %.
- **The prefix-symmetry loss removes that.** Prefix accuracy falls to chance (0.54), but
  the loss alone does not raise accuracy.
- **Generation from real recordings raises accuracy** (0.83 with thin + swap), but alone it
  still leaves the prefix shortcut in place (0.71).
- **Combined: thin + swap + overlay + sym.** It has the best accuracy and is robust to
  context swaps (0.85 / 0.85).
- **EventMix did not help here.** Body parts of two people mixed into one frame are not
  plausible for this task.
- **The old offline `aug_train`/`splice` set helps only a little** (0.79), and it keeps the
  shortcut.
- **Input and model:**
  - The −300…+500 ms window is as good as −700…+500 ms (0.81 / 0.81) and costs ⅔ of the
    compute. Use it on the board if latency or compute matters.
  - 40×40 input and the tiny scratch model are worse. Keep the pretrained v1 backbone at
    80×80.
- **Caveat: idle at window end.** With 3 classes, 33–67 % of throw/feint windows end
  with "idle" leading (`tf_as_idle`). The scene is quiet again 500 ms after a throw.
  Read throw-vs-feint 2-way, or at +200 ms, once a swing is detected, instead of taking
  the 3-way argmax at the window end.

**Final model (old data):** an `h_tso_sym` model trained on all 24 recordings. It was
deleted when the old dataset was dropped; see the new-dataset section below. Export path as
in `pretrain/README.md` (QAT, fuse, `export_for_river`).

**Next, when new recordings arrive:**

- Record throws and feints *interleaved*, not in blocks.
- Label from video or a spoken note, not from the post/pre ratio.
- Re-run `prep_raw.py` and `jobs_h.txt`.
- Then use `--cv-seeds 0,1,2`. With more recordings the CIs shrink enough to separate the
  top configs.

## New dataset: `_dataset01_010203` (noah branch, 21 Nov 2026 evening)

The old 24-recording bank is no longer used.

**Data.** 372 clips of 1.2 s: 49 throw, 68 feint, 255 idle. The release is at 700 ms.
They come from 8 recording sessions: dataset 01 counts as one session, plus 7 long
recordings in which throws and feints are interleaved.

```bash
python -I prep_raw.py dataset/_dataset01_010203 bank_clips.pt --clips
FOLDS=3 ./run_grid.sh jobs_l.txt 12
python aggregate.py runs_robust --groups
```

**Changes for clip data.**

- Time jitter pads with empty frames, and idle clips are used whole.
- `--group session` keeps whole recording sessions together in one fold. This is the
  honest test for new sessions. `--group rec` gives clip-level stratified folds, where the
  same sessions appear in train and test, so it is optimistic.
- All throw/feint metrics are now **balanced** (mean of per-class recall), because there
  are more feints than throws.
- Test-time stress tests were added: 50 % thinning, noise, 10 px shift, ±100 ms shift,
  mirror.
- `overlay` now uses idle material only. Throw clips contain a visible ball, which would
  leak into feints.

**Data-level probes** (logistic regression, balanced accuracy).

| Feature | Leave-one-session-out | Clip-level 5-fold |
|---|---|---|
| active-cell count, first 100 ms | 0.57 | 0.61 |
| active-cell count, −300…−200 ms | 0.33 | 0.43 |
| spatial map, first 100 ms | 0.55 | 0.71 |
| **active cells in the top 8 rows, 0…+100 ms (ball streak)** | **0.91** | |

- The old activity-level shortcut is gone, because the recordings are interleaved now.
- Spatial maps still carry a per-session position cue (0.71 within sessions, 0.55 across).
- The real signal is the **ball leaving the top edge right after release**. It is visible
  in every session except dataset 01, where the ball is invisible, so throw/feint there is
  near chance for every model.

**SNN results** (balanced throw/feint accuracy at window end; 95 % CI from bootstrapping
clips).

| Config | CV | Steps | acc_tf | 95 % CI | prefix@−200 | swap | train acc |
|---|---|---|---|---|---|---|---|
| `l_base_rate`: old pipeline | session | 1200 | 0.50 | 0.48–0.53 | 0.51 | 0.51 | 0.93 |
| `l_tso_sym`: thin + swap + overlay + sym | session | 1200 | 0.56 | 0.51–0.62 | 0.50 | 0.56 | 0.75–0.92 |
| `l_tso_ball`: same, ±2 px, ±3 % scale, thin ≥ 0.6 | session | 1200 | 0.53 | 0.47–0.59 | 0.46 | 0.54 | 0.98 |
| `n_pre30`: −300…+500 ms window | session | 300 | 0.68 | 0.61–0.74 | 0.47 | 0.64 | |
| `n_tso_sym` seed 0 / seed 1 | session | 300 | 0.64 / 0.52 | | 0.49 | 0.62 / 0.54 | 0.64–0.72 |
| **`lc_tso_ball`** | **clip** | 1200 | **0.89** | 0.84–0.93 | 0.52 | **0.91** | 0.95–0.99 |

The 300-step `n_tso*` rows used the old overlay, which could leak the ball, and they are
underfit. Stress tests for `lc_tso_ball`, balanced acc_tf:

| thin 50 % | noise | 10 px shift | t+100 ms | t−100 ms | mirror |
|---|---|---|---|---|---|
| 0.80 | 0.91 | 0.71 | 0.87 | 0.85 | 0.88 |

**Verdict on robustness.**

- **Shortcut-free: yes.** Prefix accuracy is at chance in every pipeline run (0.46–0.52),
  and the swap accuracy equals the plain accuracy. The model does not use the pre-release
  part.
- **Within sessions it works.** On clip-level folds it reaches 0.89, is robust to noise,
  time shifts, mirroring and thinning, and is weaker to a 10 px shift (0.71).
- **Across sessions it does not work yet.** Every SNN reaches 0.95–0.98 train accuracy but
  scores 0.50–0.56 on an unseen session, while one hand-made ball feature gets 0.91. The
  network memorizes session-specific appearance: where the thrower stands, and so where
  the ball leaves the frame. The conv head has position-specific weights, so a ball at a
  new x position is out of distribution.
- **Seed variance is ±0.1.** One 3-fold run cannot rank configs that differ by less than
  that.

**Horizontal-shift augmentation** (`jobs_x.txt`: `--shift-x 25 --shift-y 2 --scale 0.03
--thin-min 0.6`, session CV, 1200 steps):

| Seed | acc_tf | 95 % CI | prefix@−200 | swap | train acc |
|---|---|---|---|---|---|
| 0 | 0.69 | 0.61–0.74 | 0.52 | 0.69 | 0.81–0.95 |
| 1 | 0.62 | 0.58–0.66 | 0.57 | 0.63 | 0.86–0.91 |

- **This is the first configuration that beats chance on unseen sessions with both seeds**
  (mean 0.65, against 0.56 for `l_tso_sym` and 0.50 for the old pipeline).
- It is still well below the 0.91 of the one-feature ball probe.
- Dataset 01, where the ball is invisible, rises to 0.50–0.67.
- With 3 classes at the window end, 82–93 % of throw/feint clips are read as idle, so read
  throw-vs-feint 2-way.

Other options:

- crop or re-centre on the thrower (body centroid) before the network;
- more sessions or people: 8 sessions is the real limit for cross-session generalization;
- a head that pools over x. This needs an exportable pooling layer, so check riverops.

## Binary throw / no-throw on `_dataset01_01020304` (current)

**Task change (21 Nov, 22:30).** The classifier now outputs only **throw vs no throw**.
Feint and idle are merged into no throw. Earlier datasets are no longer used.

**Dataset.** `dataset/_dataset01_01020304` on the `noah` branch, commit `c936f55`.

| Sub-type | Clips | Source |
|---|---|---|
| throw | 77 | 01–03 plus `04/215305` |
| feint | 104 | 01–03 plus `04/215305` |
| idle | 288 | only the 5:47 min idle take `04/211821` |

- Idle clips are pre-thinned to one event per 10 ms per 4×4 block.
- The dataset files are unchanged: the merge happens in the script.
- `prep_raw.py --clips` stores the sub-type, and `robust_cv.py` maps throw → class 0 and
  feint/idle → class 1.
- Keeping the sub-type allows sampling feints as hard negatives and reporting false
  triggers separately for feints and for idle.

```bash
python -I prep_raw.py dataset/_dataset01_01020304 bank04.pt --clips
FOLDS=4 ./run_grid.sh jobs_b1.txt 12
python aggregate.py runs_robust --groups
```

**What changed in the pipeline.**

| Part | Now |
|---|---|
| Output | 2 neurons (throw, no throw). On the board, `config.yaml` output is 2×1×1. |
| Bank | binary cells (`clamp(max=1)`). Idle clips are pre-thinned, so raw counts would differ by class and event thinning would hit idle harder, which would be a free class cue. |
| Batches | `--p-throw 0.5` throws; the rest split `--feint-share 0.5` feint and 0.5 idle |
| Loss `ts` | throw and feint scored from release + 50 ms, idle at every frame |
| Prefix symmetry `--sym` | throw and feint clips must stay at throw = no throw before release (the outcome is not visible yet). `--sym-win N` limits this to the last N frames before release; idle-like standing at the clip start is then not forced to 50/50. |
| `swap` | unchanged: the label follows the post-release part |
| `overlay` | unchanged: idle material only, which is now the no-throw class |
| Folds `--group session` | throw/feint sessions stay whole. The single idle take is cut into contiguous time blocks, one per fold, so every model trains on idle and is tested on idle it has not seen. Idle results are therefore within-session. |

**Metrics.**

| Metric | Meaning |
|---|---|
| `bal_acc` (primary) | mean of throw recall and no-throw specificity, with feint and idle weighted equally |
| `recall_throw` | share of throws detected |
| `fp_feint`, `fp_idle` | false-trigger rates at the window end |
| `acc_tf` | throw vs feint, balanced (the hard part) |
| prefix, swap, stress | as before, on throw/feint clips |

**Iteration 1** (`jobs_b1.txt`, session CV, 1000 steps):

- `b_x`: the horizontal-shift config
- `b_x_symwin`: symmetry only over the last 300 ms
- `b_x_pre30`: −300…+500 ms window

Results below.

## Sources

- EventDrop: https://arxiv.org/abs/2106.05836
- EventMix: https://arxiv.org/abs/2205.12054
- NDA, Neuromorphic Data Augmentation for SNNs (ECCV 2022): https://arxiv.org/abs/2203.06145
- ESTF, spatiotemporal fragments: https://arxiv.org/abs/2207.11659
- TRT, temporal regularization training for SNNs: https://arxiv.org/abs/2506.19256
- ALBAR, static-bias penalty for action recognition: https://arxiv.org/abs/2502.00156
- Static/dynamic disentangling: https://arxiv.org/abs/2509.23009
- Shortcut survey: https://arxiv.org/abs/2412.05152
- JTT: https://arxiv.org/abs/2107.09044
