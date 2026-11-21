# Throw vs. feint dataset

## Raw recordings (GenX320 on the Kria)

Recorded on 2026-11-21 with `river-tools capture --host root@durance-kria.local`
(manual start/stop, 1.5–3 s per file). Our own recordings, not the Durance dataset.

| Folder | Files | Content |
|---|---|---|
| `01/samples/` | 25 | thrower in profile view, throws and feints |
| `unsorted/samples/` | 1 | first test recording (8 s) |

Same format as the Durance dataset (4-byte length, then events `x, y, polarity, timestamp`):

```bash
uv run tools/events_tool.py info dataset/01/samples/*.bin
uv run tools/events_video.py clip dataset/01/samples/<file>.bin --out clip.mp4
```

### Labels — `01/samples/labels.csv`

Written by `tools/build_dataset.py` from each recording's motion profile
(event rate over time, 20 ms bins):

| Pattern | Label |
|---|---|
| one sharp swing (half-width ≤ 200 ms), scene quiet afterwards (post/pre < 0.55) | `throw` |
| activity continues after the swing (post/pre > 0.6): arm stops and comes back | `feint` |
| peak at the very end of the file | `invalid` (cut off) |

Uncertain labels (confidence ≤ 0.6) take the label of both neighbouring takes when they agree,
because takes were recorded in blocks (13:05–13:08 throws, 13:08–13:14 feints).
Result for `01`: 13 throw, 11 feint, 1 invalid; 1 throw below the confidence cut-off (0.6).

**To confirm a label:** watch `build/videos/_review/<file>.mp4`, then set `label` and
`status=confirmed` in `labels.csv`. Confirmed rows are never overwritten.

## Training set — `build/` (generated, not in git)

```bash
uv run tools/build_dataset.py --raw dataset/01/samples --out dataset/build
```

Every sample is a 1.0 s window in the same `.bin` format, named
`gesture_<ID>-<label>-<source>-<tag>.bin` (01 throw, 02 feint, 03 idle), so
`training/train_snn.py` reads it directly. Throw/feint windows span 0.7 s before to
0.3 s after the decision moment (swing peak); `manifest.csv` lists its offset as `release_us`.

| Folder | Source | Content |
|---|---|---|
| `real_train/ real_val/ real_test/` | real | cut windows + `idle` windows from the quiet parts before the wind-up / after a throw |
| `aug_train/` | real → augmented | mirror, shift, scale, ±10° rotation, speed 0.8–1.25×, event dropout + ON/OFF imbalance, background noise + hot pixels, ±100 ms time jitter, 30 % with a distractor person mixed in |
| `cutrev_train/` | real throw → synthetic feint | cut 40–220 ms before release, short hold, wind-up replayed backwards (time reversal swaps ON/OFF) |
| `synth_train/` | procedural | stick-figure thrower rendered at 500 fps + event simulator (`tools/synth_throw.py`); profile and frontal view, ball looming towards the camera in frontal throws |
| `durance_train/` | Durance gesture dataset | other human motion (sit, wave, dance …) as `idle`; **proprietary, never commit** |

Splits are by source recording: all variants of a recording stay in its split, and
`real_val` / `real_test` contain only unmodified real windows.

Videos: `build/videos/reel_<folder>.mp4` (12 clips per page, 4× slow motion, label and
release marker) and `build/videos/_review/` (every raw recording with its label).
