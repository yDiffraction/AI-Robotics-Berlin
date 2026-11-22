# Early throw detector (throw-like wind-up vs. everything else)

Goal: one output neuron that fires as early as possible when someone is about to throw,
and stays silent for anything else. It runs indefinitely on the Kria (riverops SNN,
2×80×80 binary frames every 10 ms).

## Data

All sources are converted by `prep_sources.py` into one sparse format: sorted int32
codes `((frame*2+p)*80+y)*80+x`, binary per 10 ms frame. This is the same encoding as
`training/pretrain/prep_dvsgesture.py`.

| bank | camera | role | clips | groups for CV |
|---|---|---|---|---|
| `noah` | own GenX320 | **positive**: 77 throws + 104 feints, release at 700 ms of 1.2 s; **negative**: 288 idle clips from our own scene | 469 | recording session; the idle recording is split into time blocks |
| `mb` | GenX320 (Durance motion_buddy) | negative: sit / lay-down / hello / dance / no-action, 6 clips per class × session | 1254 | folder (session) |
| `dd` | DVXplorer (DailyDVS-200) | negative: 10 arm actions (punch, salute, racket, rotate arm, …) + 10 idle-type actions | 2522 | person |
| `dd` heldpos | DVXplorer | DailyDVS throw/toss: **never trained on** by default. This is the unseen-camera positive test | 590 | person |
| `dvsg` | DVS128 (DVS Gesture) | negative: 11 hand/arm gestures, clipped to 3 s | 1341 | user |

Provenance:
- `noah`: `dataset/_dataset01_01020304` (noah branch, c936f55).
- `mb`: `data/motion_buddy_dataset` (Durance zip, **not to be committed or published**).
- `dd`: `~/ext/dailydvs` on the A100, imported from DailyDVS-200.
- `dvsg`: `data/dvsg_{train,test}.npz`.

To look at the clips: `python experiments/early_v1/viewer/viewer.py [more.npz | clip folder ...]`, then open
http://127.0.0.1:8765. It plays every bank and every `match.py` clip folder (80×80, or native 320×320
for clip folders) and shows the release marker, the activity per frame and all metadata.

## Training samples are streams, not clips

The chip runs forever, so every training sample is a multi-second stream (default
3 s = 300 frames) stitched from about 1 s clips:
- **Positive streams:** with probability 0.5, one positive clip is placed so its release
  lands at a random position.
- **Negative segments:** everything else is negative segments from random sources, with
  per-source weights noah-idle 0.3, mb 0.25, dd 0.25, dvsg 0.2.
- **Quiet gaps:** 15 % of the slots are gaps of up to 1.5 s with nothing but sensor noise.

Per-frame target:
- **1** from `R - onset` to `R + post` (default 300 ms / 300 ms).
- **Ignored** for `dc` ms before the onset and after `R + post` until the positive clip ends.
- **0** everywhere else.

The readout is an EMA of the output spikes (α = 0.85 per frame, about 65 ms), computed on
the host. The loss is a weighted MSE on that readout; positive frames get weight 3, and
the onset gets an extra ×2, falling linearly to the end of the target.

### Keeping the net off the "which dataset is this" clues

Positives come from one camera and one scene, and the negatives mostly from others.
Each clue a source carries is handled as follows:

| clue | counter-measure |
|---|---|
| our scene / camera = throw | Noah idle clips enter the streams exactly like positives, with the same scene transitions. Half of the evaluation contexts end in our own idle scene. |
| event density (Noah ~230 active cells per frame, DailyDVS ~860, DVSG ~290) | density-normalising thinning: every segment is thinned to a random common level of 80–500 cells per frame (`--dnorm`) |
| sensor noise / background texture | a faint "bed" layer of foreign negative clips (3–25 % kept) under 70 % of the streams; random noise and hot pixels |
| polarity balance (ON fraction differs per sensor) | random polarity swap per segment |
| framing / subject size / position | flip, ±20 px x shift, ±6 px y shift, zoom ×/÷1.25 |
| clip boundaries | every segment starts and ends at random points, with 0–10 frame overlaps; negatives have boundaries just as often |
| speed | time stretch ×/÷1.25, dithered so no empty frames appear |

### Running indefinitely

- PLIF time constants are capped (`--tau-max 8` frames = 80 ms) after every step, so no
  neuron can turn into a slow integrator that ramps up over minutes.
- Training streams contain long quiet stretches.
- Evaluation checks this on 60 s streams; see below.

## Evaluation (`early_cv.py`, 4 group folds; `aggregate_early.py`)

Per held-out fold:
- **False alarms per minute:** one 60 s stream of held-out negatives per source
  (noah-idle, mb, dd, dvsg) plus a mixed one. Alarm onsets are counted with a 1 s
  refractory period at every threshold.
- **Drift:** a 60 s quiet stream (sensor noise only). Reports false alarms, readout and
  per-layer firing rate in the first vs the last 10 s.
- **Positives:** each held-out positive is placed after 3 s of held-out negatives.
  - *Detection*: the readout crosses θ within [−800 ms, +300 ms] of release.
  - *Latency*: first crossing − release.
  - *Early*: the readout had already crossed during the context.

The operating threshold θ* is the lowest θ whose mean false-alarm rate over the four
source streams is ≤ 1 per minute. All numbers are reported at θ*.

## Files

- `prep_sources.py`: raw sources → sparse banks.
- `early_cv.py`: stream generator, training, CV evaluation, `--final` export (`final.pt`, config with readout α and τs).
- `aggregate_early.py`: pooled table and threshold choice.
- `run_grid.sh`, `jobs_*.txt`: A100 grids (`~/early`, 6 processes in parallel, ~10 GB each).

## Results

(see below, filled per iteration)
