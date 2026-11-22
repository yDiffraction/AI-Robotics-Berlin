# Our own recordings (GenX320 on the Kria)

Recorded on 2026-11-21 with `river-tools capture --host root@durance-kria.local` (manual start/stop, 1.5–3 s per file).
These are the team's own recordings, not the Durance dataset.

| Folder | Files | Content |
|---|---|---|
| `01/samples/` | 25 | one throw motion each, 1.5–3 s |
| `02/samples/` | 1 | long recording with 15 throw motions |
| `03/samples/` | 6 | long recordings with 92 usable throw motions |
| `04/samples/` | 2 | made with `tools/record_tool.py`: an idle recording without throw motions (5:47 min) and a throw recording with 69 hints (5:18 min). The recordings themselves are too large for GitHub; only `hints.csv` and `labels.csv` are checked in |
| `_dataset01_01020304/` | 469 clips | current version: labelled and cut into 1.2 s clips (throw and feint from 01 to 04, idle only from the idle recording in 04), see the README there |
| `_dataset01_010203/` | 372 clips | predecessor without 04, idle taken from the pauses between throws |
| `unsorted/samples/` | 1 | first test recording (8 s) |

Same format as the Durance dataset (4-byte length, then events `x, y, polarity, timestamp`), so it can be read directly with:

```bash
uv run tools/events_tool.py info dataset/01/samples/*.bin
river-tools play dataset/01/samples/<file>.bin
```

Splits used for training and evaluation are in [SPLITS.md](SPLITS.md).

## Recording with hints

```bash
python tools/record_tool.py                       # recordings go to dataset/04/samples, page at http://127.0.0.1:8766
python tools/record_tool.py --out dataset/05/samples --host root@durance-kria.local
```

Start and stop with the button; the whole recording ends up as one `sample_<date>_<time>.bin` in the folder. During the
recording, press a key at roughly the moment of the throw: Space for a real throw, `k` for a feint, Backspace deletes
the last hint. The hints are written to `hints.csv` in the same folder (file, `throw` or `feint`, seconds from the
file's first event, wall-clock time). No other camera app may run on the board meanwhile (stop
`despot_send_and_forget` first).

## Labelling

Each `samples/` folder has a `labels.csv` with one row per throw motion: file, label (`throw` real throw, `feint`
fake throw, `idle`, `invalid`, `end`), decision time in seconds, and `verified`.

```bash
python tools/find_swings.py dataset/04/samples/*.bin   # new recordings: find the motions, label "?"
python tools/label_tool.py                             # watch clips in a loop, label with hotkeys (http://127.0.0.1:8765)
python tools/build_dataset.py --idle-from dataset/04/samples/sample_20261121_211821_487061.bin   # cut clips into dataset/_dataset01_01020304, idle only from this recording
```

`training/propose_labels.py` fills open rows with a trained model's suggestions. They only count once they are confirmed
in the label tool.
