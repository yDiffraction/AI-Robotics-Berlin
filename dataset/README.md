# Throw vs. feint dataset

## Raw recordings (GenX320 on the Kria)

Recorded on 2026-11-21 with `river-tools capture --host root@durance-kria.local`
(manual start/stop, 1.5–3 s per file). Our own recordings, not the Durance dataset.

| Folder | Files | Content |
|---|---|---|
| `01/samples/` | 25 | one person facing the camera, overhand throws and feints |
| `unsorted/samples/` | 1 | first test recording (8 s) |

Format (same as the Durance dataset): 4-byte big-endian length, then events
`x uint16, y uint16, polarity uint8, timestamp int64 (µs)`, all big-endian.

### Labels — `01/samples/labels.csv`

Auto labels from each recording's motion profile (event rate, 20 ms bins), written by
`tools/build_dataset.py`:

| Pattern | Label |
|---|---|
| one sharp swing (half-width ≤ 200 ms), scene quiet afterwards (post/pre < 0.55) | `throw` |
| activity continues after the swing (post/pre > 0.6): arm stops and comes back | `feint` |
| peak at the very end of the file | `invalid` (cut off) |

Uncertain labels (confidence ≤ 0.6) take the label of both neighbouring takes when they agree
(takes were recorded in blocks: 13:05–13:08 throws, 13:08–13:14 feints).
Result for `01`: 13 throw, 11 feint, 1 invalid.

**Review findings (2026-11-21):** an independent frame-by-frame review agreed with all 24
usable auto labels as *arm-motion* labels, but **the yarn ball is not visible to the sensor**
in any clip — no detached object after the swing. Whether the ball really left the hand is
therefore not verifiable from these files; the network can only learn arm kinematics
(throw: fast swing, then still; feint: stop and return). To confirm a label, watch
`build/videos/_review/<file>.mp4` and set `status=confirmed`; confirmed rows are never overwritten.

**For the next recordings:** a brighter / larger ball against a calm background (check first
that a ball thrown alone produces events), note per take whether it was a real throw, and film
a 240 fps phone video of the same takes. Target ≥ 50 throws and ≥ 50 feints from several people.

## Training set — `build/` (generated)

```bash
uv run tools/build_dataset.py --raw dataset/01/samples --out dataset/build             # real, aug, splice
uv run tools/generate_data.py synth --out dataset/build/synth_train --per-class 600     # procedural
uv run tools/generate_data.py video --takes <takes.csv> --out dataset/build/video_train # phone videos
# on a GPU box (A100: CPU workers render, the GPU simulates 32 clips at once)
uv run tools/generate_data.py synth --out dataset/build/synth_train --per-class 10000 --device cuda --batch 32
```

Everything is in git, including `synth_train` (synth v2, `--seed 11`, 400 clips per class, ~2 GB).
The build uses no Durance data.

Every sample is a 1.2 s window in the same `.bin` format named
`gesture_<ID>-<label>-<source>-<tag>.bin` (01 throw, 02 feint, 03 idle). Throw/feint
windows span 0.7 s before to 0.5 s after the decision moment (swing peak, `release_us`
in the manifests). Each folder is one "session" for the training scripts.

| Folder | Source | Content |
|---|---|---|
| `real_train/ real_val/ real_test/` | real | cut windows + `idle` from the quiet parts before the wind-up / after a throw |
| `aug_train/` | real → augmented | from the full recording with a 250 ms halo (no clipping): speed 0.9–1.1×, mirror, scale 0.9–1.1, ±15 px, ±5°, event thinning 85–100 %, 20 % camera shake, 15 % distractor person, 0–500 ev/s noise + ≤ 3 hot pixels, refractory 100–500 µs, ±100 ms time jitter; idle stretches are augmented too |
| `splice_train/` | real → synthetic feint | wind-up of a real throw cross-faded (30 ms) into the return phase of a real feint, body positions aligned (`event_aug.feint_splice`); ~10 % of training clips |
| `synth_train/` | procedural | figure rendered at 500 fps, events from `tools/event_sim.py`; 3/4 frontal view with 3-D arm keyframes and perspective (ready → cocked → release towards the camera → follow-through across the body), torso twist, sleeves; ball visible in 30 % only (the real yarn ball is invisible to the sensor); 1/4 profile. Decision moment = event-rate peak, the same rule as the real auto labels |
| `video_train/` | phone video → events | `tools/video_to_events.py`: real people, clothes and backgrounds; ball visible in RGB, so labels and release times are reliable |

Splits are by source recording; every variant of a recording (and every splice that uses it
as donor) stays with it. Validation and test use only unmodified real windows.

### Quality gates

`tools/shortcut_probe.py` trains a simple classifier to tell a generated source from real
data (grouped by recording). Balanced accuracy near 50 % = no obvious artefact.

| Generated vs real | Before fixes | After fixes |
|---|---|---|
| cut-and-reverse feints vs real feints | 100 % | replaced by splice |
| splice feints vs real feints | 95.5 % | 84.7 % (still detectable, keep the share small) |
| aug throws / feints vs real | — | 62 % / 70 % |
| synth throws / feints vs real | 100 % / 95.5 % (stick figure, wind-up as the main event, looming ball) | 83 % / 86 % (frontal 3-D arm model, `synth_train` v2) |

Fixes came from a multi-agent review: uniform noise was ~10× too high (the Kria camera runs
with a noise filter), time stretching clipped context, idle windows were never augmented,
geometry was too strong, and the splice deceleration contradicted real feints (their activity
peaks at the decision moment).

### Honest evaluation

`training/cv_snn.py` holds out every real recording once (grouped k-fold) and removes all of its
generated derivatives from training; no model selection on the test fold.

Videos: `build/videos/reel_<folder>.mp4` (12 clips per page, 4× slow motion, label and
decision marker), `build/videos/_review/` (every raw recording with its label),
`build/synth_train/videos/` (rendered source next to the events).
