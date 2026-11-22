# External throw positives that look like our GenX320 (`extpos/`)

Goal: real throw-motion **positives** from other datasets for the early throw detector
(`../early_cv.py`). Before this, external data was used almost only as negatives, so a network
could learn "looks like our camera = throw". Every external clip here, positive or negative,
goes through the same sensor-matching pipeline. The source therefore carries no label, and the
external positives look like ours on the board grid (2 × 80 × 80 binary cells, 10 ms).

## Sources and labels (`labels.py`)

| source | sensor | pos (throw motion) | neg | amb (never used as negative) |
|---|---|---|---|---|
| DailyDVS-200 | DVXplorer Lite 320×240, real events | throwing ball, throw things, pitch (262 recordings, 47 people) | punch, wave, point, rotate arm, stretch, surrender, salute, bounce ball, 10 idle-type actions | badminton swing, hit people with things, throw hats in the air, throw rubbish, bounce volleyball |
| HMDB51 (HF mirror `divm/hmdb51`) | RGB video → **v2e** | throw | wave, punch, clap, walk, stand, turn, pick, shake hands, kick (ball), jump, run, brush hair, drink, talk, smoke, push, climb stairs | catch, shoot ball, swing baseball, golf, hit, fencing, sword classes |
| motion_buddy, DVS Gesture | GenX320 / DVS128 | none: no throw-like class exists | unchanged | – |

`amb` exists because feints count as positives: an overhand swing without release *is* a
feint by our definition, so a badminton smash cannot also be a negative.

## Pipeline

```
fetch_dailydvs.py   index + fetch raw aedat4 events (HTTP range requests into the 91 GB zip)
fetch_hmdb.py       per-clip mp4 download with HMDB tags (camera motion, view, body, quality)
v2e_convert.py      video -> events: camera-motion rejection, crop, SuperSloMo x10, v2e pixel model
profile_own.py      own_profile.json: our clips' statistics and our sensor's noise map
match.py            raw events -> 1.2 s GenX320-look clips (same steps for pos / neg / amb)
probe.py            shortcut probe (grouped-CV logistic regression on board-grid features)
to_bank.py          clips -> sparse bank for early_cv.py (amb in a separate *_amb.npz)
run_cloud.sh        everything on the A100
```

### Video → events (`v2e_convert.py`)

- **Simulator.** v2e 1.8.5 (`pip install v2e`, Python 3.14): SuperSloMo frame interpolation
  plus the v2e DVS pixel model. Thresholds, mismatch, photoreceptor bandwidth and refractory
  period are sampled per clip. v2e adds no noise; noise comes from our sensor later.
- **Interpolation.** ×10 (30 → 300 fps). Without it, every 10 ms board frame would see events
  from only every third video frame: a 33 ms comb that a network learns at once.
  `--interp dis` is an optical-flow CPU fallback (about 2.5 s per clip, against about 500 s
  for SuperSloMo on a CPU).
- **Camera motion.** Background motion is measured with LK feature tracking and a RANSAC
  similarity fit. Clips with p90 > 0.5 px/frame are rejected (calibrated on the HMDB tags:
  keeps 72 % of "static" clips and 10 % of "moving" ones). Our camera never moves; a panning
  background would be a source cue.
- **Crop.** Square crop of the full frame height around the motion, so there is no padding.

### Sensor matching (`match.py`), identical for every role

Targets are sampled per clip from **our** throws and feints (`own_profile.json`, 181 clips),
so the external clips follow our distribution rather than a single value.

1. **Foreign noise out.** Background-activity filter, then cells with fewer than 2 active
   neighbours are removed. Without this, the DVXplorer noise outlined the source field of
   view as a visible rectangle.
2. **Actor mask.** Only the main actor region is kept (activity-density map, largest
   component). Positives with less than 75 % of their activity on one actor are dropped.
3. **Anchor.** Event-rate peak becomes release = 700 ms, the same rule as our labels.
   Negatives are anchored the same way (their busiest moment), so "burst of motion at
   700 ms" is not a label cue.
4. **Geometry.** Motion height and centre are scaled and moved to values drawn from our
   clips. Source frame edges that end up inside the image are faded out, so there is no
   straight cut through the body.
5. **Polarity.** The ON share of cells is matched by raising the dominant polarity's
   threshold (DVXplorer is ON-heavy: 0.55 against our 0.48).
6. **Density.** Active cells per frame in the swing window are matched by a per-pixel
   threshold increase (keep every k-th event, bisection on k). This thins weak edges first,
   as a less sensitive sensor would. Random dropout would leave speckle.
7. **Noise.** Our sensor's per-cell noise map is added (92 empty-scene idle clips, including
   hot cells), with a random gain.

## Does it work? Shortcut probe (`probe.py`)

Logistic regression with grouped CV; balanced accuracy, where 0.5 means indistinguishable.
DailyDVS sample: 366 positive and 96 negative clips, against our 469 clips.

| question | features | naive conversion (crop + rescale, as the old importer) | matched |
|---|---|---|---|
| external positives vs our positives | sensor (noise, isolated cells, ON share, edges, temporal) | 0.99 | **0.63** |
| | all | 0.99 | **0.60** |
| any external vs any own clip | sensor | 0.94 | **0.50** |
| | all | 0.90 | **0.52** |
| | 8×8 position map | 0.56 | 0.47 |

Medians of our positives / naive DailyDVS positives / matched DailyDVS positives:

| | ours | naive | matched |
|---|---|---|---|
| active cells per frame (swing) | 90 | 320 | 89 |
| noise floor (10th percentile of cells per frame) | 36 | 200 | 27 |
| isolated cells | 0.035 | 0.124 | 0.030 |
| ON share | 0.48 | 0.54 | 0.47 |
| motion height (cells) | 38 | 60 | 37 |

What is still different is motion *content*: DailyDVS throws are wider (26 against 16 cells)
and partly lateral. That is real variation of throws, not a sensor artefact.

**Full DailyDVS build (A100).** From 2,063 recordings (Move = 0; 32 of 3,114 fetches lost to
rate limits): 366 positive, 2,906 negative and 356 amb clips. It gives the same probe result:
external vs own 0.53 (sensor) / 0.55 (all), external positives vs ours 0.60 / 0.58.

**HMDB51 does not pass.** 46 of 102 throw videos survive the camera-motion test, giving
172 positive and 82 negative clips (SuperSloMo on the GPU). The clips stay recognisable as
external: external vs own 0.73 (all), positives vs ours 0.94 (sensor). The reason is temporal:
events still bunch at the 30 fps video frames. Our 30 ms autocorrelation is 0.78, HMDB has 0.14,
and 0.39 after one frame period of timestamp jitter (`--time-jitter-us 33333`). Within HMDB,
sensor features give the label away (0.66). The bank (`hmdb.npz`) is built but **not used in
training**. It would teach "HMDB look = throw". Fixing it needs high-frame-rate source video
(≥ 120 fps, like the phone slow-motion takes in `tools/video_to_events.py`), not more
interpolation.

## Running

Local, a sample only. Full preprocessing is longer than 10 minutes here (DailyDVS is a
10 GB download; SuperSloMo takes about 500 s per clip on this CPU).

```bash
uv venv --python 3.14 v2eenv && uv pip install --python v2eenv v2e   # simulator env
python profile_own.py <own dataset dir> own_profile.json
python fetch_dailydvs.py index --out data/extpos/dailydvs
python fetch_dailydvs.py fetch --out data/extpos/dailydvs --limit 8      # + --actions 147,37,187
python match.py --src dd --raw data/extpos/dailydvs --profile own_profile.json --out data/extpos/clips_dd
python fetch_hmdb.py --out data/extpos/hmdb
python v2e_convert.py --manifest data/extpos/hmdb/manifest.csv --videos data/extpos/hmdb/videos \
    --out data/extpos/hmdb/raw --interp dis
python probe.py --own <own dataset dir> --ext data/extpos/clips_dd --ext data/extpos/clips_hmdb
```

Cloud (A100): `OUT=~/extpos OWN=~/extpos/own_clips bash run_cloud.sh`. Then link
`~/extpos/banks/{dd,hmdb}.npz` next to `noah/mb/dvsg.npz` in `~/early/banks_ext/`.

Notes:

- **Rate limits.** Hugging Face rate-limits bursts (HTTP 429). Both fetchers resume; re-run
  `fetch` with `--workers 4` to pick up failures.
- **Do not mix banks.** Never combine the new `dd.npz` with negatives from the old DailyDVS
  importer: a preprocessing difference between positives and negatives is itself a label.

## Training with it (`../early_cv.py`)

New options:

- `--pos-w-src noah:0.5,dd:0.3,hmdb:0.2`: positive shares per source.
- `--pos-srcs noah`: ablation, external positives are not trained on.
- `--w-hmdb`: negative share of HMDB clips.

Banks that are missing are skipped. In `../aggregate_early.py`, `--fa-src noah` picks the
threshold on our own idle stream only, which is the deployment criterion.
