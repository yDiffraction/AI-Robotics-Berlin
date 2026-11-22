# Throw detection on the GenX320: everything tried so far

Status: 22 Nov 2026, late morning. Written to merge ideas across the team and pick the next move.
It covers this session's work and what earlier sessions and branches left behind.

## How to read this

Two different goals were worked on. Every section is tagged:

| tag | goal | positive | negative |
|---|---|---|---|
| **[GOAL A: real vs fake]** (earlier) | Tell a real throw (ball leaves the hand) from a fake throw / feint. Later narrowed to binary *throw vs no-throw* (feint + idle = no-throw). | real throw | feint (+ idle) |
| **[GOAL B: early throw intention]** (current) | Fire as early as possible when someone *starts a throw motion* and stay silent for anything else, on a chip that runs forever. | real throw **and** feint (any throw-like wind-up) | everything else: idle, walking, waving, other gestures |
| **[SHARED]** | Infrastructure both goals use (board format, pretraining, datasets) | | |

The difference matters for every result below. Under GOAL A a feint is a *negative*; under GOAL B
it is a *positive*. Numbers from the two goals are not comparable.

---

## 0. TL;DR

### [GOAL B] (current)

- **New dataset.** External throw *positives* now look like our GenX320 data
  (`experiments/early_v1/extpos/`). Before, external data was almost only negatives, so a network
  could learn "foreign look = no throw".
  - A logistic-regression probe tells external clips from ours with 0.90 balanced accuracy
    (naive conversion) and 0.55 after matching (0.5 = indistinguishable).
  - External positives against our positives: 0.99 → 0.58.
- **The earlier finding "our throws must be told apart from dataset throws" was a
  preprocessing artefact.**
  - With the old DailyDVS import, a model trained with DailyDVS throws as positives still
    detected only 0.28 of them (earlier session, e1_ddpos), and the default model 0.00.
  - With matched data they are learnable: 46–66 % detected on unseen people. Even a model
    trained only on our own positives detects 16–38 % of them.
- **Best checkpoints**, trained on all data:
  - `experiments/early_v1/models/final_ext_fa.pt`, the new data plus a false-alarm recipe.
    Recommended when the next recordings have people moving around.
  - `final_old_fa.pt`, the same recipe on the old banks. Higher recall in clean sessions.
- **Current numbers.** Every one of our 181 throw-motion clips was tested once by a model that
  did not see its session (2-fold session CV). FA = false alarms per minute on our held-out
  idle footage.

  | model (2-fold CV twin of the final) | threshold | throws | feints | all | ordinary sessions* | FA/min own idle | median alarm time vs release |
  |---|---|---|---|---|---|---|---|
  | old banks, previous recipe (`x_old`) | 0.95 | 49/77 | 69/104 | 118/181 | 98/107 | 6.0 | −470 ms |
  | old banks + FA recipe (`r2_old_fa`) | 0.95 | 40/77 | 66/104 | 106/181 | 98/107 | 2.0 | −390 ms |
  | **new banks + FA recipe (`r2_ext_fa`)** | 0.80 | 43/77 | 59/104 | 102/181 | 83/107 | 1.5 | −375 ms |
  | | 0.95 | 32/77 | 47/104 | 79/181 | 72/107 | **0.0** | −80 ms |

  \*Ordinary sessions = all own sessions except `215305` (labels from key presses, ball rarely
  visible) and `d01` (ball invisible). Those two hold 74 of the 181 clips and most of the
  misses for every model.
- **Trade-off.**
  - The new-data model is the only one that reaches 0 false alarms on our busy idle segment
    (people walking through). It also does better on the hard session 215305.
  - It catches fewer throws in clean sessions and fires later at strict thresholds.
  - It fires 10–18 times per minute on DailyDVS arm actions (punch, wave): it learned "arm swing ≈ throw".
- **HMDB51 video (v2e) failed the quality gate** and is not used. Events still bunch at the
  30 fps video frames; it needs ≥ 120 fps source video.

### [GOAL A] (earlier)

On unseen *sessions*, throw-vs-feint never got clearly above chance with the SNN (best balanced
throw-vs-no-throw: 0.65). A single hand-made feature, "ball streak in the top rows right after
release", reaches 0.91, so the information is in the data. The networks memorised
session-specific appearance instead.

---

## 1. [SHARED] Board format and hardware

| | |
|---|---|
| Sensor | Prophesee GenX320 (320×320) on a Kria KV260, despot `downsampling: 4`, `frequency: 100` |
| SNN input | binary `[T, 2, 80, 80]`: 10 ms frames; a 4×4 pixel block is 1 if any event of that polarity fell in it |
| Network | riverops spiking conv only (`SpikingConvBN2d` ×4, 16-32-64-64, stride 2; head `SpikingConv2d` → `[T, K, 1, 1]`), ~74–79 k weights |
| Output (GOAL B) | 1 neuron (training/early) or 2 neurons (throwdet). The host computes an EMA readout of the output spikes (α = 0.85 per frame, about 65 ms) and alarms when it crosses θ. |
| Board noise filter | despot `noise_filtering_factor: 0.5`. Our training recordings (`river-tools capture`) are probably unfiltered. |
| Export | QAT 8-bit weights, Q4.12 potentials (`scaling: 4096`), `fuse_model` → `export_for_river` (training/pretrain/README, throwdet/export.py) |

Because the network only ever sees this binary grid, every "does the data look like ours"
question below is measured on this grid (`extpos/cellstats.py`).

## 2. [SHARED] Data inventory

### Our own recordings (GenX320 on the Kria, 21 Nov 2026)

| set | where | content | labels |
|---|---|---|---|
| `dataset/01` | data-pipeline / noah branch | 25 short files, one person facing the camera, ball invisible to the sensor | auto labels from the event-rate profile, reviewed |
| `_dataset01_010203` | noah branch | 372 clips of 1.2 s: 49 throw, 68 feint, 255 idle (pauses between swings); 8 sessions | Noah's label tool |
| **`_dataset01_01020304`** (current) | noah branch, commit `c936f55` | **469 clips of 1.2 s: 77 throw, 104 feint, 288 idle.** Idle comes only from the 5:47 min idle take `04/211821` (people walk through in the first ~130 s and last ~40 s). Release / swing peak at 700 ms. | throw/feint from Noah's tool; `04/215305` from key presses (Space = throw, k = feint) matched to swings by agents, so some are uncertain |

Sessions in `_dataset01_01020304` (throw + feint clips): `d01` 10, `170441` 15,
`195808` 10, `200001` 25, `200828` 46, `201452` 5, `201622` 4, `201703` 2, `215305` 64.
Idle: `211821` 288 (pre-thinned to one event per 10 ms / polarity / 4×4 block).

### Generated from our recordings ([GOAL A] era, data-pipeline branch, `dataset/build/`)

- `aug_train`: augmented real windows (speed, mirror, scale, shift, rotation, thinning, shake,
  distractor person, noise, refractory, time jitter).
- `splice_train`: synthetic feints, a real throw's wind-up cross-faded into a real feint's return.
- `synth_train`: procedural 3-D arm renderer plus the repo's own event simulator
  (`tools/event_sim.py`, `tools/synth_throw.py`), v2. A v3 was designed but not generated.
- `video_train`: 5 Wikimedia phone videos → events (`tools/video_to_events.py`, own simulator),
  52 clips.
- Probe results there (generated vs real, balanced accuracy):
  - splice feints 85 %
  - aug throws / feints 62 / 70 %
  - synth v2 throws / feints 83 / 86 %
  - All remained detectable.

### External

| dataset | sensor | used for | notes |
|---|---|---|---|
| DVS Gesture | DVS128 | pretraining; negatives in GOAL B | 11 hand/arm gestures, 29 users |
| motion_buddy (Durance) | GenX320 | negatives | sit / lay down / hello / dance / no action, 14 sessions; proprietary, never commit. Noise-filtered (noise floor 0) and 5–8× denser than ours. |
| DailyDVS-200 | DVXplorer Lite 320×240 | GOAL B negatives (old) → now positives and negatives (new) | 200 actions, 47 people, MIT licence, HF mirror `Charlieqaq/DailyDVS-200` (91 GB zip, read by range requests) |
| HMDB51 | RGB video | tried for GOAL B via v2e, rejected | HF mirror `divm/hmdb51` (per-clip mp4 with camera / view / quality tags) |

## 3. [SHARED] DVS Gesture pretraining (`training/pretrain/`)

- **Backbone.** "v1" in the board format, trained on DVS Gesture (128 → 80 px rescale, binary
  10 ms cells, ±6 px shift, 10 % dropout, noise cells).
- **Accuracy.** 87.9 % per 0.64 s window and 91.3 % per recording on held-out users 24–29.
  Fused and float predictions are identical.
- **Use.** Every SNN below starts from this backbone (`~/snn_pretrain/runs/sweep_v1/best.pt`
  on the A100), unless it says scratch.
- **Not measured.** Quantised accuracy.

---

## 4. [GOAL A: real vs fake] History

### 4.1 First fine-tune (`training/pretrain/train_throw.py`, 38 real windows, 3 classes throw / feint / idle)

5-fold CV grouped by recording, from the "hires" backbone:

| training data | acc (3-class) | throw-vs-feint | note |
|---|---|---|---|
| real only (`s0`) | 0.76 [0.62–0.89] | 0.78 | |
| real + 1,050 synth (`s1050`) | 0.63 [0.47–0.78] | 0.70 | synthetic data hurt |
| synth only, test on real (TSTR) | 0.18–0.37 | 0.09–0.13 | synthetic data does not transfer |

`eval_input_stats`: synthetic clips have about half the event rate of real ones and are
separable (permutation p = 0). This is where "don't generate from scratch" came from.

### 4.2 Earlier session (pasted transcript) and `experiments/robust_release/`

The task was: 25 terrible real examples, separable at 75 % from the first frame; adapt training
to bad data; research early wrong fitting; generate data from the real recordings, not from
scratch; A100 runs inside a 1 h budget.

**Diagnosis (old 24-recording set).** Leave-one-recording-out logistic regression:

| feature | time | accuracy |
|---|---|---|
| active-cell count | first 10 ms frame | 0.67 |
| active-cell count | first 100 ms | 0.75 |
| active-cell count | −300…−200 ms | 0.83 |
| 8×8 spatial map | −300…−200 ms | 0.92 |
| 8×8 spatial map | +100…+300 ms (after release) | 0.54 |

Throws and feints were recorded in separate blocks, and the auto label used the post/pre
activity ratio. So pre-release activity was a shortcut.

**Pipeline built (`experiments/robust_release/robust_cv.py`, `prep_raw.py`).**

- Windows are cut on the fly from the full recordings.
- Generation from real recordings:
  - `thin`: event thinning
  - `swap`: context swap before release
  - `overlay`: OR in idle material
  - `mix`: EventMix
- Losses:
  - `ts`: running-rate cross-entropy after release
  - `--sym`: prefix symmetry, so throw and feint outputs must be equal before release
- AdamW, EMA weights, no early stopping.
- Shortcut metrics: `prefix_tf@-200ms` (should be 0.5) and `swap_tf`.

**Results, old 24 recordings (4-fold by recording, 300 steps).** acc_tf = throw vs feint,
n = 24, CI about ±0.15:

| config | acc_tf | prefix@−200 ↓ | swap_tf ↑ | idle recall |
|---|---|---|---|---|
| h_base_rate (old pipeline) | 0.75 | 0.71 | 0.60 | 0.36 |
| h_extra (+ offline aug/splice) | 0.79 | 0.67 | 0.64 | 0.95 |
| h_ts_sym | 0.71 | 0.54 | 0.68 | 0.98 |
| h_thin_swap | 0.83 | 0.71 | 0.73 | 0.98 |
| h_thin_swap_sym | 0.83 | 0.54 | 0.76 | 1.00 |
| **h_tso_sym** (thin + swap + overlay + sym) | **0.85** | 0.56 | **0.85** | 0.95 |
| h_tsom_sym (+ EventMix) | 0.81 | 0.63 | 0.81 | 0.95 |
| h_pre30 (−300…+500 ms) | 0.81 | 0.48 | 0.81 | 1.00 |
| h_pool2 (40×40) | 0.73 | 0.67 | 0.72 | 0.86 |
| h_tiny (6 k weights, scratch) | 0.69 | 0.56 | 0.60 | 0.91 |

**`_dataset01_010203` (8 sessions, interleaved throws and feints, balanced acc).**

- The activity shortcut is gone.
- **Ball streak in the top 8 rows, 0…+100 ms: 0.91** leave-one-session-out, from one feature.
- SNNs on unseen sessions:

  | config | CV | acc_tf |
  |---|---|---|
  | `l_base_rate` | session | 0.50 |
  | `l_tso_sym` | session | 0.56 |
  | x-shift ±25 px, 2 seeds | session | 0.69 / 0.62 |
  | `lc_tso_ball` | clip-level (same sessions in train and test) | 0.89 |

- The networks reach 0.95–0.98 train accuracy and memorise where the thrower stands.

**Binary throw / no-throw on `_dataset01_01020304`** (4-fold session CV, 1,000 steps;
`results/summary_binary_it1.json`):

| config | bal_acc | recall throw | fp feint | fp idle | acc_tf | prefix@−200 |
|---|---|---|---|---|---|---|
| b_x (x-shift recipe) | 0.62 | 0.44 | 0.34 | 0.08 | 0.55 | 0.51 |
| **b_x_symwin** (symmetry only over the last 300 ms) | **0.65** | 0.51 | 0.37 | 0.035 | 0.57 | 0.48 |
| b_x_pre30 | 0.55 | 0.29 | 0.34 | 0.07 | 0.48 | 0.51 |

- **Session 215305.** Recall 0.04–0.23 there in every config. The same session is hard in GOAL B.
- **Final model.** `experiments/robust_release/results/b_x_symwin_final.pt`, on all clips.
- **Iteration 2 was killed half way.** Only 2 of 4 folds finished per config, and different
  folds per config, so these are **not comparable**:
  - `b2_fs7` (feint share 0.7): 0.82 bal_acc on its 2 folds
  - `b2_long` (1,600 steps): 0.54
  - `bc_symwin` (clip-level CV): 0.73
- **DANN.** Written, never tested.

### 4.3 What GOAL A taught

1. Every SNN found a session or position shortcut before the real cue. Prefix symmetry plus
   context swap removes the pre-release shortcut, but cross-session accuracy stays near chance.
2. The real cue (the ball after release) is small and position-specific. A position-specific
   conv head does not generalise it.
3. Synthetic-from-scratch data hurts. Generation from real recordings (thin, swap, overlay)
   helps a little.

---

## 5. [GOAL B: early throw intention] History

### 5.1 throw-detector branch (`throwdet/`, teammate line, parallel to this one)

- **Task.** Binary throw motion (throw + feint = 1) vs none.
- **Network.** 2-neuron output; the host readout uses EMA plus hold frames.
- **Data (`prep.py`, 5,615 clips).**
  - our own 01–04
  - own aug / splice
  - synth v2 + video_train
  - DailyDVS throw classes as positives (old importer: crop + rescale only, no noise or
    density matching)
- **Anti-shortcut measures.**
  - balanced sampling per (domain, target)
  - DailyDVS thinned to our density (in some runs)
  - our idle OR-ed under half the external clips
- **Evaluation.**
  - per-clip, 4-fold by recording
  - an alarm counts within −250…+300 ms of the swing peak
  - FA/min on the held-out idle stream
- **Results (recall at 2 FA/min).**
  - e10: AUC 0.865–0.924, recall 0.54–0.79
  - e26 (more DailyDVS): AUC 0.88–0.92, recall 0.58–0.82
  - without synth / video, AUC drops to 0.83–0.85
  - seed variance up to ±0.06 AUC
- **Timing.** Alarm about 150 ms before the swing peak.
- **Delivered.** Models A–E in board format plus `live_readout.py`, `replay.py`, `adapt.sh`
  (site fine-tuning).
- **Not comparable with section 5.3.** The evaluation protocol differs: per-clip window, and
  peak instead of release.

### 5.2 training/early, iteration 1 (previous session; 4-fold, old banks)

- **Trainer.** `early_cv.py`: 1 output neuron, multi-second training *streams* stitched from
  clips of all sources with quiet gaps, per-frame target, EMA readout loss, PLIF τ capped at
  80 ms so the net cannot integrate for minutes.
- **Anti-shortcut set.**
  - density normalisation
  - a faint foreign "bed" layer
  - polarity swap
  - flip, shift, zoom, time stretch
  - noise and hot pixels
- **Evaluation.**
  - 60 s streams per negative source
  - positives after 3 s of held-out context
  - detection within −800…+300 ms of release
- **Results.** At θ = 0.95 (FA ≤ 1/min was never reached):

| config | throw | feint | DailyDVS throws | latency | FA/min: own idle / mb / DailyDVS / DVSG |
|---|---|---|---|---|---|
| e1_base | 0.61 | 0.68 | 0.01 | −430 ms | 11 / 6.8 / 1.5 / 0 |
| e1_noaug | 0.60 | 0.68 | 0.00 | −445 ms | 4.8 / 3.0 / 0 / 0 |
| e1_ddpos (DailyDVS throws as positives) | 0.29 | 0.44 | 0.28 | −320 ms | 5 / 2.8 / 16 / 0 |
| e1_onset500 | 0.69 | 0.71 | 0.02 | −470 ms | 14 / 7 / 1.3 / 0 |

- **Conclusions then.**
  - Early detection works, about 0.45 s before release.
  - False alarms on our own scene and motion_buddy are the blocker.
  - The net learned "a throw in Noah's setup". DailyDVS throws were not detected, and adding
    them hurt.
  - **This session shows the last point came from preprocessing, not from different throws.**

### 5.3 This session: matched external positives (`experiments/early_v1/extpos/`)

#### Label policy (`extpos/labels.py`)

| source | pos (throw motion) | neg | amb (never a negative) |
|---|---|---|---|
| DailyDVS-200 | throwing ball (147), throw things (37), pitch (187) | punch, wave hands, point, rotate arm, stretch, surrender, salute, bounce ball; yawn, hands in pockets, cross arms, phone, nod, shake head, check time, walk, touch nose, stand up | swing badminton racket, hit people with things, throw hats in the air, throw rubbish, bounce volleyball |
| HMDB51 | throw | wave, punch, clap, walk, stand, turn, pick, shake hands, kick (ball), jump, run, brush hair, drink, talk, smoke, push, climb stairs | catch, shoot ball, swing baseball, golf, hit, fencing, sword, sword exercise, draw sword |
| motion_buddy, DVS Gesture | none (no throw-like class) | all | |

**Why amb.** Under GOAL B a feint (overhand swing, no release) is positive. A badminton smash
or hitting with an object is the same arm motion, so labelling it negative contradicts our own
feint labels. The old pipelines used these as negatives.

#### Preprocessing (`extpos/match.py`): identical steps for pos, neg and amb

Targets are sampled per clip from our 181 throws and feints (`own_profile.json` from
`profile_own.py`):

| | median | 10–90 % |
|---|---|---|
| active cells per frame in the swing window (−500…+300 ms) | 90 | 50–210 |
| ON share of cells | 0.48 | |
| motion height | 38 of 80 cells | |
| centre | x ≈ 36, y ≈ 41 | |
| isolated cells | 0.035 | |
| noise floor (10th-percentile frame) | 36 cells | |

Steps:

1. **Background-activity filter.** An event needs 2 more events within ±1 px and ±3 ms bins.
2. **Cell-level cleaning.** Cells with fewer than 2 active neighbours (3×3, ±1 frame) are
   removed. Without it, DVXplorer residual noise drew the source field of view as a visible
   rectangle.
3. **Actor mask.** A 7×7-smoothed activity map, cut at 8 % of its maximum, keeps the largest
   component (grown by 2 cells). Only the actor is taken from the foreign recording.
   Positives with less than 75 % of their activity on one actor are dropped as unusable
   (about 25 % of DailyDVS throw clips).
4. **Anchor.** The event-rate peak (20 ms bins, 3-tap smoothing) goes to 700 ms, our labelling
   rule. Negatives are anchored at their own busiest moment too, so "burst at 700 ms" is not a
   label cue. `--neg-random` adds an unanchored window per negative.
5. **Geometry.** Motion height is scaled to a height drawn from our clips (×0.85–1.15,
   s ∈ [0.3, 2.5]) and the centre moved to ours (±6 / ±3 cells), with sub-pixel jitter.
   Source frame edges that land inside the image are faded out, so there is no straight cut
   through the body.
6. **Polarity.** The ON share is matched by raising the dominant polarity's threshold
   (bisection). DVXplorer is ON-heavy: 0.55.
7. **Density.** A per-pixel threshold increase (keep every k-th event per pixel and polarity,
   random phase, bisection on k) matches the swing-window density to a target drawn from
   ours. It thins weak edges first, like a less sensitive sensor; random dropout left speckle.
8. **Noise.** Our sensor's per-cell noise map is OR-ed in, measured on 92 empty-scene idle
   clips (0.4 cells per frame on average, hottest cell 0.095), with gain exp(N(0.7, 0.5)).
9. **Output.** GenX320 `.bin`, 1.2 s, release at 700 ms, plus `labels.csv` in our schema
   (+ src, role, cls, group, variant, params, stats) and preview PNGs.

#### Video → events (`extpos/v2e_convert.py`, HMDB51 only)

- **Simulator.** v2e 1.8.5 (`pip install v2e`, Python 3.14; the SuperSloMo checkpoint
  downloads automatically).
- **Camera motion.** Clips with a moving camera are rejected: LK feature tracking plus a RANSAC
  similarity fit, p90 background shift > 0.5 px/frame. Calibrated on the HMDB tags: keeps 72 %
  of "static" and 10 % of "moving" clips. Phase correlation and border-change tests did not
  work (the moving person dominates).
- **Window and crop.** Motion-energy peak −1.1…+0.8 s; square crop of the frame height around
  the motion; 320×320.
- **Interpolation.** SuperSloMo ×10 (30 → 300 fps) on the GPU (about 3 s per clip on the
  A100). `--interp dis` is a CPU optical-flow fallback, 2.5 s per clip against 500 s for
  SuperSloMo on the laptop.
- **v2e pixel model.** Per clip: thresholds 0.15–0.25, σ 0.02–0.05, photoreceptor cutoff
  below the sub-frame rate (IIR stability), refractory 0.1–1 ms, **no noise** (ours is added
  in match.py).

#### Quality gate (`extpos/probe.py`)

Logistic regression with grouped CV on board-grid features, balanced accuracy (0.5 = cannot
tell). Feature sets:

- **sensor**: noise floor, isolated cells, ON share, edge run length, temporal autocorrelation,
  empty frames
- **level**: density, peakiness, extent, centre
- **map**: 8×8 position map

| question | features | DailyDVS naive (old importer) | DailyDVS matched (full: 366 pos / 2,906 neg) | HMDB matched (172 pos / 82 neg) |
|---|---|---|---|---|
| external pos vs our pos | sensor | 0.99 | 0.60 | 0.94 |
| | all | 0.99 | **0.58** | 0.83 |
| external pos vs external neg | sensor | 0.61 | 0.72 | 0.66 |
| any external vs any own | sensor | 0.94 | **0.53** | 0.79 |
| | all | 0.90 | **0.55** | 0.73 |
| | map | 0.56 | 0.48 | 0.50 |

Medians (ours / DailyDVS matched / HMDB matched):

| | ours | DailyDVS matched | HMDB matched |
|---|---|---|---|
| cells per frame | 90 | 83 | 97 |
| ON share | 0.48 | 0.47 | 0.52 |
| isolated cells | 0.035 | 0.030 | 0.041 |
| 30 ms autocorrelation | 0.78 | 0.84 | **0.39** |
| motion width (cells) | 16 | 26 | 29 |

- **DailyDVS passes.** What remains is motion content: DailyDVS throws are wider and partly lateral.
- **HMDB fails.** Its activity still pulses at 30 fps even after SuperSloMo. The 30 ms
  autocorrelation went from 0.14 to 0.39 with one frame of timestamp jitter. It is excluded
  from training.
- **External pos vs external neg (0.72 on "sensor").** Within DailyDVS the negatives
  (nod, yawn, …) are quieter. That is motion level, not sensor, and both went through the
  same steps.

**DailyDVS full build.** 2,063 recordings without camera motion; 32 of 3,114 fetches lost to
HTTP 429 (re-run `fetch` to fill). Result: **366 positive clips** (about 120 recordings × 3
variants, after the actor gate), **2,906 negatives** (×2 windows), **356 amb**.

#### Training changes (`experiments/early_v1/early_cv.py`, `aggregate_early.py`)

- **Sources.** `SRCS` adds `hmdb`; missing banks are skipped.
- **Positives.**
  - `--pos-w-src noah:0.6,dd:0.4`: per-source positive shares.
  - `--pos-srcs noah`: ablation, external positives not trained.
- **Negatives.** `--w-hmdb`.
- **Generic sources.** Evaluation and folds work for any number of sources.
- **Aggregation.** `aggregate_early.py --fa-src noah` picks θ on **our own idle stream**, the
  deployment criterion, instead of the mean over all sources.
- **Banks.** `to_bank.py` writes match.py output as a bank and puts amb in `*_amb.npz` (never loaded).
- **Old trainer** kept as `~/early/early_cv.pre_extpos.py` on the A100.

#### Experiments (2-fold session CV as requested, 800 steps, onset 500 ms, A100)

Old banks = noah, mb, dvsg + DailyDVS from the old importer (its throws are "heldpos", never
trained). New banks = the same noah, mb, dvsg + the matched DailyDVS bank.
LIGHT = `--p-bed 0.3 --dnorm 0 --thin-min 0.6 --noise 3e-4 --hot 1` (lighter
anti-shortcut augmentation, since the data now matches).
FA = `--pos-w 1.5 --w-noah 0.45 --w-mb 0.2 --w-dd 0.25 --w-dvsg 0.1` (less positive
weight, more of our idle).
Exact list: `experiments/early_v1/jobs_ext.txt`.

- **Detected**: the readout crosses θ within −800…+300 ms of release and did not already fire
  in the 3 s context.
- **Ordinary**: sessions except 215305 and d01.
- **lead**: median first crossing relative to release.

Full table: `python experiments/early_v1/report_ext.py experiments/early_v1/results_ext/runs_ext`.

| run | banks / changes | θ | throws /77 | feints /104 | ordinary /107 | DailyDVS pos | lead | FA/min own idle | motion_buddy | DailyDVS neg | DVSG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x_old | old, e1_onset500 recipe | 0.95 | 49 | 69 | 98 | 4/590 | −470 | 6.0 | 8.5 | 0.0 | 0 |
| | | 0.99 | 43 | 67 | 100 | 0/590 | −365 | 3.0 | 5.0 | 0.0 | 0 |
| x_ext | new, pos noah:0.6 dd:0.4 | 0.95 | 34 | 53 | 74 | 197/366 | −260 | 2.0 | 6.0 | 17.0 | 0 |
| x_ext_light | new + LIGHT | 0.95 | 36 | 61 | 76 | 242/366 | −370 | 2.0 | 1.5 | 21.5 | 0 |
| x_own_light | new + LIGHT, own positives only | 0.95 | 29 | 43 | 68 | 59/366 | −295 | 2.0 | 3.5 | 7.0 | 0 |
| | | 0.80 | 37 | 61 | 83 | 139/366 | −495 | 3.0 | 5.5 | 13.5 | 0 |
| **r2_old_fa** | old + LIGHT + FA | 0.90 | 44 | 71 | 104 | 1/590 | −490 | 2.5 | 2.5 | 0.0 | 0 |
| | | 0.95 | 40 | 66 | 98 | 1/590 | −390 | 2.0 | 1.5 | 0.0 | 0 |
| | | 0.99 | 30 | 48 | 77 | 0/590 | −225 | 1.5 | 0.0 | 0.0 | 0 |
| **r2_ext_fa** | new + LIGHT + FA | 0.80 | 43 | 59 | 83 | 230/366 | −375 | 1.5 | 3.5 | 18.0 | 0 |
| | | 0.90 | 36 | 52 | 74 | 200/366 | −235 | 1.5 | 1.0 | 13.0 | 0 |
| | | 0.95 | 32 | 47 | 72 | 169/366 | −80 | **0.0** | 0.5 | 10.5 | 0 |
| r3_ext_own80 | as r2_ext_fa, pos noah:0.8 dd:0.2 | 0.90 | 34 | 54 | 77 | 187/366 | −255 | 2.0 | 1.5 | 12.5 | 0 |
| | | 0.95 | 29 | 46 | 69 | 140/366 | −140 | 1.5 | 0.5 | 9.0 | 0 |

Per session at about 2 FA/min on own idle (`aggregate_early.py --fa-src noah --fa 2 --groups`):

| session (clips) | x_old (θ 0.99, 3 FA/min) | r2_old_fa (θ 0.95) | r2_ext_fa (θ 0.80) | r3_ext_own80 (θ 0.85) |
|---|---|---|---|---|
| d01 (10) | 0.30 | 0.20 | 0.10 | 0.10 |
| 170441 (15) | 0.93 | 0.87 | 0.73 | 0.80 |
| 195808 (10) | 1.00 | 0.90 | 0.80 | 0.80 |
| 200001 (25) | 0.96 | 0.96 | 0.72 | 0.80 |
| 200828 (46) | 0.91 | 0.91 | 0.80 | 0.74 |
| 201452 (5) | 0.80 | 0.80 | 1.00 | 0.80 |
| 201622 (4) | 1.00 | 1.00 | 0.75 | 0.75 |
| 201703 (2) | 1.00 | 1.00 | 0.50 | 1.00 |
| 215305 (64, key-press labels) | 0.11 | 0.09 | **0.28** | 0.19 |

False alarms on our idle per fold, by threshold:

| run | fold | θ 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 0.97–0.99 |
|---|---|---|---|---|---|---|---|
| r2_ext_fa | 0 (busy idle block, people walking) | 5 | 3 | 3 | 3 | **0** | 0 |
| r2_ext_fa | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| r2_old_fa | 0 | 8 | 8 | 5 | 5 | 4 | **3** |
| r2_old_fa | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

#### What the experiments say

1. **Matched DailyDVS throws are learnable.** 0.00 → 0.46–0.66 on unseen people (fold
   holdout by person). The old "they look different" result was the old importer.
2. **The FA recipe** (positive weight 1.5, 45 % own idle) is what makes our own idle quiet,
   for old and new banks alike. Without it, nothing reaches 2 FA/min.
3. **New data buys quiet and transfer, and costs own recall.**
   - Only the new-data model reaches 0 FA/min on the busy idle block (people walking).
   - It is better on the worst session (215305).
   - In clean sessions it detects less, and at strict thresholds it fires later (−80 ms at θ 0.95).
   - It fires on DailyDVS arm actions (10–18 FA/min). It generalised "big overhand arm motion".
   - Cutting the DailyDVS positive share to 20 % (r3) did not recover own recall.
4. **Noise level.** One seed, 2 folds, n = 181. Differences under about 0.05–0.1 in recall or
   about 1 FA/min are within noise, and the FA numbers move in steps of 1 alarm per fold minute.
5. **The two bad sessions** (215305, d01: 74 of 181 clips) dominate the misses for every
   model. Their labels or visibility are the problem as much as the models.

#### Checkpoints (all on the PC, `experiments/early_v1/`)

| file | what | config |
|---|---|---|
| `models/final_ext_fa.pt` | **recommended for new own recordings with people moving.** All data, r2_ext_fa recipe, 800 steps | `config` inside: arch v1, 1 output, EMA α 0.85, τ list |
| `models/final_old_fa.pt` | runner-up: old banks, same recipe; higher recall in clean scenes, more FA with walking people | |
| `results_ext/runs_ext/r2_*_fa/model_f{0,1}.pt` | the CV fold models behind the tables (fold 0 holds out d01, 170441, 201703, 215305 + the busy idle block) | |
| `results_ext/runs_ext/*/result_f*.json` | raw per-clip and per-stream evaluation of every run | |
| `results_ext/probe_dd.json`, `probe_hmdb.json` | shortcut probe outputs | |

- **Not done.** No QAT or export to `weights.txt` for these yet.
- **Thresholds.** θ from the CV twins: about 0.80–0.90 for final_ext_fa (about 1.5 FA/min,
  alarm about 0.25–0.4 s before release) or 0.95 (quiet, late). Re-check on a site recording;
  the final model is not the CV model.

---

## 6. Reproduce this session

Paths: PC repo `<path-to-repo>`; A100 `<user>@<gpu-host>`
(key `~/.ssh/<gpu-key>`), work dirs `~/extpos` and `~/early`.
Nothing of this session is committed yet.

### Environments

- **Training.** The project `.venv` (CPU torch) for local probes; `~/snn_pretrain/.venv` on the A100.
- **Simulator / extpos.** Python 3.14 venv with `v2e` (pulls torch, opencv, dv-processing):
  `uv venv --python 3.14 v2eenv && uv pip install --python v2eenv v2e`.
  On the A100: `~/extpos/v2eenv` (CUDA torch).

### Data build (A100; about 1 h, most of it downloads and SuperSloMo)

```bash
# our clips (labels.csv + bins): git archive origin/noah dataset/_dataset01_01020304 | tar -x
python experiments/early_v1/extpos/profile_own.py <dataset/_dataset01_01020304> own_profile.json
# copy extpos/*.py + own_profile.json to ~/extpos/code, then:
OUT=~/extpos OWN=~/extpos/own_clips JOBS=24 GPU_JOBS=4 bash run_cloud.sh
#   DailyDVS: fetch_dailydvs.py index / fetch --workers 16, then
#             match.py --src dd --variants-pos 3 --neg-random, to_bank.py -> banks/dd.npz
#   HMDB (run here as two steps after rate limits): fetch_hmdb.py (or upload videos),
#             v2e_convert.py --interp slomo --roles pos;
#             --roles neg,amb --limit-per-class 15 --skip-bad;
#             match.py --src hmdb --variants-pos 4 --time-jitter-us 33333; to_bank.py
python extpos/probe.py --own <own clips> --ext ~/extpos/clips_dd --ext ~/extpos/clips_hmdb
```

Hugging Face rate-limits bursts (HTTP 429). Both fetchers resume; re-run with fewer workers.
A local sample run (8 clips per DailyDVS class, all throws, HMDB with `--interp dis`) takes
about 10 minutes on the laptop; see `extpos/README.md`.

### Banks and training (A100, `~/early`)

```bash
mkdir banks_ext && ln -s ../banks/{noah,mb,dvsg}.npz banks_ext/ && ln -s ~/extpos/banks/dd.npz banks_ext/
bash launch_ext.sh banks_ext r2_ext_fa --pos-w-src noah:0.6,dd:0.4 \
     --p-bed 0.3 --dnorm 0 --thin-min 0.6 --noise 3e-4 --hot 1 \
     --pos-w 1.5 --w-noah 0.45 --w-mb 0.2 --w-dd 0.25 --w-dvsg 0.1     # ~20 min, 2 procs, ~10 GB each
python aggregate_early.py runs_ext --fa-src noah --fa 2 --groups
python report_ext.py runs_ext                                           # tables above
# final model: same args with --final instead of the fold loop (early_cv.py ... --final --out runs_ext_final)
```

All run arguments are in `jobs_ext.txt`. Each result JSON also stores its full `args`.

---

## 7. Open problems and candidate next moves (for discussion)

Ordered by expected value for GOAL B (detect on our next recordings):

1. **Fix or quarantine the two bad sessions.** 215305 (key-press labels, ball rarely
   visible) and d01 (ball invisible) hold 41 % of the positives and most misses in every
   model and both goals. Either re-verify those labels or report them separately. As things
   stand, they make every comparison noisy.
2. **Record own hard negatives at the demo site.** People walking, waving, stretching,
   picking things up, arm swings without throw intent. This attacks the trade-off directly:
   the new-data model's weakness is "arm swing = throw", and our own idle has only walking.
   It also gives a real FA test. throwdet's `cut_recording.py --idle` + `adapt.sh` already does
   site fine-tuning.
3. **One evaluation protocol for both lines.** throwdet (per-clip AUC, 4-fold, peak-anchored)
   and training/early (streams, 2-fold, release-anchored) cannot be compared today. Run
   `final_ext_fa`, `final_old_fa` and throwdet's D/E through one replay on the same held-out
   stream (throwdet `replay.py`), and pick by FA/min plus lead time.
4. **Lead time vs false alarms.** The quiet model fires late at θ 0.95.
   - Try the readout with hold frames, as throwdet does.
   - Try a 600 ms onset target.
   - Calibrate θ for "≤ 1 FA/min and lead ≤ −200 ms" instead of FA only.
   - The dodge needs about 0.45–0.6 s (sim results).
5. **Seeds.** One seed per config, 2 folds. Before trusting the old/new ranking, run 2–3
   seeds of r2_old_fa and r2_ext_fa (about 20 min per seed pair on the A100).
6. **Use the amb classes deliberately.**
   - Option 1: train them as positives. They are feint-like overhand swings, which may make
     detection less dependent on our scene.
   - Option 2: keep them as a separate eval set to see what the model calls a throw.
7. **HMDB / video positives.** Only with ≥ 120 fps source video. Our own phone slow-motion
   takes plus `tools/video_to_events.py`, or v2e with `--interp dis` on 240 fps footage. The
   30 fps comb is the blocker, not the simulator.
8. **Export.** QAT + `export_for_river` for the chosen model, then a live test with
   `live_readout.py`. Quantised behaviour has not been measured for any GOAL B model of this line.
9. **GOAL A (real vs fake), if it comes back.** The ball-streak feature (0.91) says the cue is
   there; the SNN fails on position. Re-centre on the thrower, or use a head that pools over x
   (check riverops export support). Then retry with interleaved recordings from new sessions.

## 8. File map

| path | goal | content |
|---|---|---|
| `training/pretrain/` | SHARED / A | DVS Gesture pretraining, first throw fine-tune, input statistics |
| `experiments/robust_release/` | A | robust_cv (shortcut-aware throw/feint and throw/no-throw), results JSONs, `b_x_symwin_final.pt` |
| data-pipeline branch `dataset/`, `tools/` | A (+ B data) | own recordings, auto labels, synth/aug/splice/video generation, the repo's own event simulator, shortcut probe |
| noah branch `dataset/_dataset01_01020304` | A + B | current labelled clip set |
| throw-detector branch `throwdet/` | B | teammate's throw-motion detector, delivered models A–E, live readout, replay, site adaptation |
| `experiments/early_v1/` | B | stream trainer `early_cv.py`, `aggregate_early.py`, `report_ext.py`, `launch_ext.sh`, `jobs_ext.txt`, `models/`, `results_ext/` |
| `experiments/early_v1/extpos/` | B | external-positive pipeline (fetch, v2e, match, probe, bank) and its README |
| `data/extpos/` (git-ignored) | B | local sample: DailyDVS raw (446 recordings), HMDB videos (4,088), matched clips, banks |
| A100 `~/extpos`, `~/early/runs_ext*` | B | full builds, all runs, logs |
