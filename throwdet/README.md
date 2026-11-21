# Throw-motion detector (binary SNN, board format)

**Question the network answers, every 10 ms in the live feed: is a throw motion happening right now?**
A real throw and a feint both count as throw motion (the arm swings like a throw). Everything
else does not: standing, walking, waving, an empty scene. Deciding whether the ball is actually
released is a separate task (see `pretrain/` and `~/robust` on the Tim branch).

Status: work in progress (overnight 21/22 Nov). The results table below is updated as runs finish.

## Board format

Same as `pretrain/` on the Tim branch, so it runs on the KV260:

| | |
|---|---|
| Input | binary `[T, 2, 80, 80]`: 10 ms frames, a 4x4 block of the 320x320 GenX320 image is 1 if any event of that polarity fell into it (despot `downsampling: 4`, `frequency: 100`) |
| Network | spiking conv layers only (riverops `SpikingConvBN2d`, head `SpikingConv2d` -> `[T, 2, 1, 1]`) |
| Output | 2 neurons: 0 = no throw motion, 1 = throw motion |
| Live readout (host) | `score_t = (1 - alpha) * score_{t-1} + alpha * (spike1_t - spike0_t)`, alarm when `score_t > threshold`, then 1 s refractory. `alpha` and `threshold` are in each model's `readout.json` |

The network state is never reset in live use. Training uses random 0.8 s windows cut from 1.2 s clips, so the model never relies on starting from a reset state.

## Data (`prep.py` builds one table, 5,615 clips of 1.2 s)

| Domain | Source | throw motion (1) | no throw motion (0) |
|---|---|---|---|
| own | our GenX320 recordings, noah branch `dataset/_dataset01_01020304` | 77 throws + 104 feints | 288 idle clips (the 5:47 min idle recording `04/211821`) |
| own | noah branch `dataset/_dataset01_010203`, idle only | | 255 idle clips (pauses between the swings, person in view) |
| own_aug | `dataset/build/aug_train`, `splice_train` (derived from the 01 recordings) | 423 | 108 |
| sim | `dataset/build/synth_train` (synth v2) + `video_train` (phone videos) | 832 | 416 |
| dd | DailyDVS-200 (real DVS, 47 people, MIT licence), `tools/dailydvs_import.py` | 590: throwing ball, pitch, throw things, toss | 2,522: arm swings without release (badminton swing, punch, wave, point, ...) and everyday idle actions |

The motion clips have the decision moment (event-rate peak of the swing) at 0.7 s. Targets per 10 ms frame:
- **Motion clips:** 1 from 250 ms before the peak to 300 ms after it. Our labels put the wind-up start at a median of 250 ms before the peak. All other frames of a motion clip are ignored.
- **Other clips:** 0 on every frame after a 150 ms warm-up.

## Avoiding shortcuts

Plain features already tell the domains apart: a logistic regression on activity features identifies DailyDVS vs. our camera with 93 % accuracy (`baseline.py`). DailyDVS is mostly negatives and our data mostly positives, so a careless model would learn "DailyDVS look = no throw". Countermeasures in `train.py`:
- **Balanced sampling.** Each (domain, target) pair gets a fixed share of every batch, for example own:1 = own:0 = 20 % and dd:1 = dd:0 = 15 %, so the domain carries no information about the target.
- **Density matching.** DailyDVS, synth and aug clips are randomly thinned to the active-cell density of our camera (30 to 200 cells per frame). Untouched, DailyDVS has about 270 to 470 cells per frame, our camera about 25 to 85.
- **Background compositing.** In half of the external clips, one of our own idle clips is OR-ed in, so they carry our scene and sensor noise.
- **Augmentation.** Mirror, scale ±25 %, shift ±8 px, time shift ±150 ms, speed ±20 %, cell dropout and noise.

## Evaluation (never used to pick a checkpoint: the model after the last step is evaluated)

- **Own camera, grouped 4-fold CV by recording.** The idle recording is split into 4 contiguous time chunks. Aug and splice clips derived from held-out recordings are removed from training. Reported:
  - balanced accuracy at the clip level: a motion clip counts as detected if the score crosses the threshold within -250..+300 ms of the peak, an idle clip is a false alarm if the score crosses anywhere
  - AUC
  - detection latency relative to the peak
- **Live stream.** Each held-out chunk of the idle recording is played as one continuous stream, giving false alarms per minute.
- **DailyDVS held-out persons** (official val/test persons): generalisation to new people, scenes and a different sensor.

Reference: `baseline.py`, a logistic regression on the activity curve, reaches 77 % balanced accuracy on the own-camera CV. A model has to beat that clearly; otherwise it has only learned "much motion = throw".

## Files

| File | Purpose |
|---|---|
| `prep.py` | builds `~/throwdet/data/table.npz` from all sources |
| `train.py` | training plus grouped CV, live-stream false-alarm test, DailyDVS person holdout |
| `baseline.py` | non-spiking reference and domain probe |
| `export.py` | `model.json`, `weights.txt`, `config.yaml`, `readout.json` for the Kria, plus a float vs fixed-point check |
| `replay.py` | runs a recording (any length) through a model exactly like the live loop; writes alarm times and a plot |
| `worker.sh` | job queue on the A100 (`~/throwdet/queue.txt`) |
| `model.py`, `speedpatch.py` | copied from `pretrain/` on the Tim branch |

## Results

All numbers are out-of-fold on our own camera (724 clips, 181 with throw motion, grouped 4-fold CV by
recording) plus the held-out idle stream (5.8 min, people walking through the view). `summary.md` on the
A100 (`~/throwdet/summary.md`) is regenerated automatically after every finished run.

**Recall at a false-alarm budget** is the number that matters for the live demo: the share of throw
motions (throws and feints) detected between 250 ms before and 300 ms after the swing peak, using the
best readout (alpha, hold, threshold) that stays within the budget on the idle stream (alarms per
minute) and on the idle clips.

| run | what changes | AUC own | recall @1 FA/min | recall @2 FA/min | recall @4 FA/min |
|---|---|---|---|---|---|
| activity baseline | logistic regression on the activity curve (not an SNN) | bal. acc 0.77 | | | |
| e02_ownonly_v1 | own clips + own aug only | 0.845 | 0.39 | 0.62 | 0.73 |
| e01_all_v1 | + synth, video, DailyDVS (density-matched, balanced) | 0.903 | 0.43 | 0.72 | 0.88 |
| e05_all_v1_nodens | as e01, DailyDVS not thinned | 0.898 | | | |

Findings so far:
- **The external data helps.** DailyDVS, synth and video raise the out-of-fold AUC on new recordings from 0.845 to 0.90. The domain balancing keeps them from acting as a shortcut.
- **Thinning DailyDVS to our density does not help our camera** (AUC 0.898 vs. 0.903). It does make DailyDVS throws unlearnable: held-out DailyDVS AUC is 0.56 thinned vs. 0.63 unthinned.
- **Almost all false alarms come from people walking close to the camera** (legs and torso filling the view) and from a hand right in front of the lens. An empty scene and whole-image motion do not trigger alarms. Run e11 and later sample such hard negatives more often.
- **Detection time.** At the 2 FA/min operating point, the alarm comes about 150 to 180 ms before the swing peak, i.e. during the wind-up.
