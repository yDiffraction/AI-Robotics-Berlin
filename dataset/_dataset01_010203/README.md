# _dataset01_010203

Labelled version of our own recordings `dataset/01`, `02` and `03` (GenX320 on the Kria, 2026-11-21), cut into clips.
Our own recordings, not the Durance dataset.

| Class | Folder | ID | Clips | Meaning |
|---|---|---|---|---|
| throw | `throw/` | 01 | 49 | real throw, the ball leaves the hand |
| feint | `feint/` | 02 | 68 | fake throw, the same motion without letting go |
| idle | `idle/` | 03 | 255 | no throw motion (standing, walking, picking up the ball, empty scene) |

## Clips

- File name: `gesture_<ID>-<label>-d<dataset>_<time of recording>-<tag>.bin`.
  `t01234` = throw motion with its decision time at 12.34 s of the recording, `g01234` = idle clip from a pause, starting at 12.34 s.
- Each clip is 1.2 s long, timestamps start at 0. For throw and feint the decision time (maximum of the event rate in
  the swing) is at 0.7 s, so `release_us` = 700000.
- Same format as the recordings: 4-byte length, then events `x, y, polarity, timestamp`. Readable with
  `tools/events_tool.py` and `river-tools play`.
- `preview_<label>_<n>.png`: one row per clip, 12 frames of 100 ms each (ON green, OFF magenta).

## labels.csv

One row per clip: `file, folder, label, class_id, dataset, source, parent, start_s, decision_s, release_us, events, verified, note`.

- `source` is the recording the clip comes from; `start_s` and `decision_s` are seconds in that recording.
- `verified`: `1` = checked by Noah in the label tool, `claude` = labelled by Claude from frame sequences,
  `auto` = idle clip from the pause between two motions.
- Clips with the same `parent` come from the same motion or the same pause.

## Provenance

| Dataset | Recording | throw | feint | idle |
|---|---|---|---|---|
| 01 | 25 short files, 10 motions kept | 7 | 3 | 5 |
| 02 | 170441 | 5 | 10 | 32 |
| 03 | 195808 | 6 | 4 | 21 |
| 03 | 200001 | 6 | 19 | 48 |
| 03 | 200828 | 14 | 32 | 119 |
| 03 | 201452 | 5 | 0 | 24 |
| 03 | 201622 | 4 | 0 | 4 |
| 03 | 201703 | 2 | 0 | 2 |

## Notes

- Split train and test by `source` or `dataset`, not by clip: clips from the same recording look very similar.
- Two thirds of the clips are idle. Balance the classes in training (`training/train_throw_own.py` samples every class equally often).
- 24 idle clips contain almost no events (empty scene).
- Motions marked `invalid` (15 in 01, 120 in 03) are left out, also as idle clips. Stretches where the camera was moved are cut out.

## Rebuild

The labels are in `dataset/<NN>/samples/labels.csv` and can be edited with `python tools/label_tool.py`. Then:

```bash
python tools/build_dataset.py
```
