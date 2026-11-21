# Recording protocol — throw vs. feint

Goal: the Go2 sees a person wind up to throw a soft object (yarn ball, rolled sock).
It must dodge **only if the throw is completed** (object released) and ignore feints.

## Classes

| ID | Label | What happens |
|---|---|---|
| 01 | `throw` | Full wind-up, object leaves the hand towards the robot |
| 02 | `feint` | Same wind-up, arm stops or pulls back, object stays in the hand |
| 03 | `idle` | No throw: standing, walking past, waving, picking up the ball, other arm moves |

`idle` is the safe default and needs the most variety (it is what the robot sees most of the time).
Keep `throw` and `feint` balanced; they must look identical until the release.

## File naming

Same scheme as the Durance dataset, so the training loader works unchanged:

```
recording/sessions/<SS>/gesture_<ID>-<label>-<person>-<style>-<NN>.bin
e.g. recording/sessions/01/gesture_01-throw-laurenz-over-007.bin
```

- `<SS>` session = one person + one setup. **Splits are by session** (never mix one session across train/val/test).
- `<style>`: `over` (overhand), `under` (underhand), `side`, `lefthand`, …
- Log every take in `recording/takes.csv` (one row per file), including failed takes (`valid=0`).

## Setup per session

- Camera at its **final mounting position and height** on the robot (or the same height and tilt on a tripod). Robot standing still.
- Thrower 2–4 m in front of the camera, whole upper body in view; note the distance.
- Release roughly in the middle of the clip. If the recorder makes 1 s clips, start the wind-up right after it starts.
- Vary between sessions: person, clothing, hand, throw style, distance, projectile, background (someone walking behind), lighting.
- Safety: soft projectiles only, never aim at the Kria, the camera or the robot's head.

## Targets for the first 20-minute slot

| Class | Takes |
|---|---|
| throw | 40 |
| feint | 40 |
| idle | 40 |

≈ 2 people × 3 classes × 20 takes. Check 3–4 files after the first minutes:

```bash
uv run tools/events_tool.py info recording/sessions/01/*.bin
uv run tools/events_tool.py preview recording/sessions/01/gesture_01-*.bin --out recording/previews
```

Look for: person visible, arm motion clearly visible, ball visible after release, no clip that is pure noise.

If the recorder writes one long file instead of clips, cut it around each motion:

```bash
uv run tools/events_tool.py cut long_take.bin --out recording/sessions/01 --pre 0.3 --post 1.0
```

## Latency measurements (do these in the same slot)

The judges want the **sensor-to-command delay**, separately from **model-only inference time**.

1. **Score log with timestamps.** While the board runs the existing gesture model, run on the laptop:
   ```bash
   uv run tools/spleat_logger.py --host durance-kria.local --port 3333 --out recording/logs/slot1-spleat.csv
   ```
   Press Enter at each release to write a MARK row. The CSV contains the server's `latency` field (model/server side) and the frame rate.
2. **ROS timestamps.** On the Kria, record what the pipeline publishes:
   ```bash
   ros2 bag record -o /root/slot1 /durance/gestures /durance/clearance /api/sport/request /api/sport/response
   ```
3. **End-to-end with a phone.** Film at 240 fps (slow motion) with the thrower's hand **and** the robot (or a screen that flashes when the command is sent) in the same frame. Count frames from release to robot reaction: `delay = frames / 240 s`. Do ≥10 trials and note them in `takes.csv`.
4. **Baseline now:** do the same with the existing gesture demo (e.g. "hello") to get the current pipeline's delay before our model exists.

Write down for every measurement: distance, lighting, who threw, success/failure, false triggers.
