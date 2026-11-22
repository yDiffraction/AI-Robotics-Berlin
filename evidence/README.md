# Evidence

Every number carries a label saying how it was measured:

- **BOARD**: measured on the Kria (FPGA inference) or the real Go2.
- **FLOAT-OOF**: float model on the laptop/GPU, out of fold (the model never saw that recording).
- **IN-SAMPLE**: evaluated on data the model was trained or calibrated on (optimistic).
- **SIM**: simulator.
- **OPEN**: not measured yet. To be filled from the live trials (template: [`trials_template.csv`](trials_template.csv)).

## End-to-end (sensor → robot)

| Quantity | Value | Label | Conditions / source |
|---|---|---|---|
| Throw trials: dodges started before the ball arrives | OPEN | BOARD | fill in from `trials_template.csv`: thrower distance, light, auto-dodge on |
| False triggers per minute (people walking, feints, picking up the ball) | OPEN | BOARD | count `/snn/detection` with `acted=true` during non-throw minutes |
| Sensor → `/reflex/dash` delay | OPEN | BOARD | see "How to measure the delay" below |
| `/reflex/dash` → robot moving | OPEN | BOARD | 240 fps video with an LED on `/reflex/dash`, or `/dodge/event` vs odometry |

## Perception

| Quantity | Value | Label | Conditions / source |
|---|---|---|---|
| Model-only inference + transport on the Kria | 1.85 ms median round trip (laptop → SPLEAT server → laptop) | BOARD | `perception/hitl/README.md` "Checks done"; v3 model, recording 201703. Earlier, with Noah's ball model on the same firmware, the server reported 0.5–0.6 ms model latency and 1.9 ms round trip |
| Live camera → board throughput | 100 frames/s (10 ms frames), 1.7 ms round trip | BOARD | empty scene, `perception/hitl/README.md` |
| Board vs float agreement | 94.6 % of frames, 2/2 alarms identical | BOARD vs float | v3, recording 201703 |
| Bridge readout vs reference readout | identical alarms on 100k random frames | unit test | `perception/model_hitl_step1/README.md` |
| throwdet (family of the deployed model), grouped 4-fold CV | AUC 0.865–0.924 (mean 0.89), recall 0.54–0.79 at 2 false alarms/min (4 seeds) | FLOAT-OOF | `training/throwdet/README.md`, `results_summary.md` |
| `D_e10_fold3` on the unseen recordings `01` | 23/25 swings, 4 other alarms in 55 s, alarm ≈ 150 ms before release | FLOAT-OOF | `training/throwdet/README.md`, `replay_01_heldout.png` |
| `d_neg11` (start model of the RL step) on labelled person01 windows | 45/53 throws (85 %), 2/68 false alarms (3 %), Q2.6 fixed point | IN-SAMPLE | `perception/model_d_neg11/RESULTS.txt` |
| `hitl_step1` vs start model vs RL snapshots 80–470 | picked as best in a live A/B session | BOARD, not saved | the per-model hits / misses / FA per min were shown live and not written to a file; recompute from session `20261122_153235` |
| Early detection on whole recordings (earlier approach) | gate **failed**: no variant reached ≥ 60 % at ≤ 2–3 FA/min | FLOAT-OOF | `experiments/early_ctx_eval/REPORT.md` |

## Robot motion

| Quantity | Value | Label | Conditions / source |
|---|---|---|---|
| Deployed dodge (`dodge/go2_dodge.py`, sport API): back 1.1 m/s for 0.5 s | 0.27 m back, return within 2 cm of the start, max tilt 9° | SIM (Gazebo CHAMP) | `dodge/README.md` |
| Low-level RL/optimised dodge (not deployed) | 0/500 failures in harsh runs, 30 cm after 0.58 s median | SIM (MuJoCo) | `sim/README.md` |

## How to measure the delay (sensor → command)

`snn_bridge.py` already measures the FPGA round trip per frame (`/api/state` → `fpga.rtt_ms`, `fpga.latency_ms`).
The full chain is: event arrives → 10 ms frame closes → FPGA round trip → readout (needs `hold_frames` = 3
frames above θ) → `/reflex/dash`. So the pure processing delay is about one frame (≤ 10 ms) + round trip (~2 ms),
while the decision itself needs at least 3 frames (30 ms) above threshold. To measure it end to end, film an LED that
blinks in view of the event camera together with the robot at 240 fps, or log the camera event timestamp of the
frame that raised the alarm next to the `/reflex/dash` publish time.
