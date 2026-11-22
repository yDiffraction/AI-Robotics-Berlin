# _dataset01_01020304

Labelled version of our own recordings `dataset/01` to `04` (GenX320 on the Kria, 2026-11-21), cut into clips. Throw and
feint come from 01 to 03 and the throw recording `04/215305`; idle comes only from the idle recording `04/211821`.
Our own recordings, not the Durance dataset. Predecessor: `_dataset01_010203` (without 04, idle from the pauses between throws).

| Class | Folder | ID | Clips | Meaning |
|---|---|---|---|---|
| throw | `throw/` | 01 | 77 | real throw, the ball leaves the hand |
| feint | `feint/` | 02 | 104 | fake throw, the same motion without letting go |
| idle | `idle/` | 03 | 288 | no throw motion: the 5:47 min idle recording `04/211821`, cut into clips without gaps |

## Clips

- File name: `gesture_<ID>-<label>-d<dataset>_<time of recording>-<tag>.bin`.
  `t01234` = throw motion with its decision time at 12.34 s of the recording, `g01234` = idle clip starting at 12.34 s of the recording.
- Each clip is 1.2 s long, timestamps start at 0. For throw and feint the decision time (maximum of the event rate in
  the swing) is at 0.7 s, so `release_us` = 700000.
- Idle clips are thinned: per 10 ms, polarity and 4×4 pixel block only the first event is kept. That is the model's input
  grid (`training/prep_clips.py`: 120 frames of 10 ms, 80×80), so the model sees the same frames as from the full clip.
  This way the idle clips take 138 MB instead of 717 MB. Throw and feint clips contain all events.
- Same format as the recordings: 4-byte length, then events `x, y, polarity, timestamp`. Readable with
  `tools/events_tool.py` and `river-tools play`.
- `preview_<label>_<n>.png`: one row per clip, 12 frames of 100 ms each (ON green, OFF magenta).

## labels.csv

One row per clip: `file, folder, label, class_id, dataset, source, parent, start_s, decision_s, release_us, events, verified, note`.

- `source` is the recording the clip comes from; `start_s` and `decision_s` are seconds in that recording.
- `verified`: `1` = checked by Noah in the label tool, `claude` = labelled by Claude from frame sequences,
  `recording` = idle clip from a recording that is idle as a whole.
- Clips with the same `parent` come from the same motion.

## Provenance

| Dataset | Recording | throw | feint | idle |
|---|---|---|---|---|
| 01 | 25 short files, 10 motions kept | 7 | 3 | 0 |
| 02 | 170441 | 5 | 10 | 0 |
| 03 | 195808 | 6 | 4 | 0 |
| 03 | 200001 | 6 | 19 | 0 |
| 03 | 200828 | 14 | 32 | 0 |
| 03 | 201452 | 5 | 0 | 0 |
| 03 | 201622 | 4 | 0 | 0 |
| 03 | 201703 | 2 | 0 | 0 |
| 04 | 211821 (idle recording) | 0 | 0 | 288 |
| 04 | 215305 (with `tools/record_tool.py`) | 28 | 36 | 0 |

## Recording 04/215305

- Recorded with `tools/record_tool.py`: Space pressed for a throw, `k` for a feint (`dataset/04/samples/hints.csv`, 69 hints).
- Five Claude agents matched each hint to its swing and put the decision time on the event-rate maximum
  (`verified` = `claude`, the note gives the time of the hint). The label comes from the key; the ball is visible
  in only a few throws.
- Set invalid: 1 hint with no motion nearby, 3 duplicate hints on the same swing. The throw at 0.65 s is missing
  because it is too close to the start of the file.
- Uncertain: the throw at 198.12 s (looks more like a feint), the feint at 137.30 s (hardly any motion) and the
  matching at 103.01, 105.13, 186.28 and 289.72 s.

## Notes

- Split train and test by `source` or `dataset`, not by clip: clips from the same recording look very similar.
- A good 60 % of the clips are idle. Balance the classes in training (`training/train_throw_own.py` samples every class equally often).
- All idle clips come from a single recording, so a holdout by recording contains either all idle clips or none; a holdout of dataset `04` contains all idle clips.
- 127 idle clips contain almost no events (under 1000 after thinning, empty scene). In about 90 idle clips the whole
  image moves, mostly in the first 130 s and the last 40 s of the recording. The largest idle clip is 4.5 MB.
- The motions labelled `idle` and the pauses between throws from 01 to 03 are not included. Their labels are still in the recordings' `labels.csv`.
- Motions marked `invalid` (15 in 01, 120 in 03, 4 in 04) are left out.

## Rebuild

The labels are in `dataset/<NN>/samples/labels.csv` and can be edited with `python tools/label_tool.py`. Then:

```bash
python tools/build_dataset.py --idle-from dataset/04/samples/sample_20261121_211821_487061.bin
```

Without `--idle-from` the idle clips come from the labelled recordings again (motions labelled `idle` and pauses between throws).
