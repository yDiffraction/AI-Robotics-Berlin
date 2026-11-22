# Throw Reflex: a Go2 that dodges a throw, seen by a spiking network on RiverFPGA

Durance Vision Lab track, Berlin HackLab Robotics, 21–22 Nov 2026.

A Prophesee GenX320 event camera watches a person. A spiking CNN on the RiverFPGA (Kria KV260) decides every 10 ms
whether a throw motion is happening. When it is, a ROS 2 node makes the Unitree Go2 jump back out of the way and
then walk back to where it stood. We recorded and labelled our own data, trained our own SNN with riverops,
fine-tuned it on the board with human-in-the-loop reinforcement learning, and run the whole loop on the Kria.

**Track level: 2 (Build)**: own data, own SNN, deployed on RiverFPGA, ROS 2 behaviour on the Go2. For level 3,
see [Robustness](#robustness-and-safe-defaults) and the honest status in [Evidence](#evidence).

> Video walkthrough: _link to be added_

## Run the demo

The robot stack lives in `/root/dodge/` on the Kria that is mounted on the Go2. From a laptop on the same cable:

```bash
ssh root@<kria> "source /root/dodge/env.sh && ros2 launch /root/dodge/robot.launch.py"
#   <kria> is 192.168.77.1 on the wired link, or its IPv6 link-local address, e.g. fe80::...%<adapter index>
```

Then open `http://<kria>:8080`. The page shows the event camera, the SNN output spikes, the readout against its
threshold, a THROW DETECTED banner and the robot state.

- **Auto-dodge starts OFF.** Press **Y** on the Go2 remote (or the switch on the page) to let detections move the dog.
- **X** on the remote is a manual dodge, independent of the SNN.
- `dry_run:=true` runs everything live but the robot only logs; `snn:=mock camera:=replay:<file.bin>` runs on a desk
  without board or robot.

Install once on the Kria: see [dodge/README.md](dodge/README.md) (copy `dodge/` to `/root/dodge/`, `kria/*.sh` flat
into it) and load the model with `python perception/hitl/board_ctl.py load perception/model_hitl_step1 --ssh root@<kria>`.

## Pipeline

```
 GenX320 ──MIPI──▶ Kria KV260 ─────────────────────────────────────────────────────────────┐
 events            despot_debug_camera ─▶ snn_bridge.py                                     │ on the Kria
                   (Durance)              10 ms frames, 2×80×80 binary                      │ (ARM + FPGA)
                                          │            ▲                                    │
                                          ▼            │ spikes (11 outputs, we use 0 and 1)│
                                   SPLEAT server ─▶ RIVER IP on the FPGA                    │
                                   (Durance)        our SNN: weights.txt (Q2.6)             │
                                          │                                                 │
                         readout: y = 0.8·y + 0.2·(spike1 − spike0), alarm if y > 0.6 for   │
                         3 frames, 1 s refractory  ──▶ /snn/detection, web page :8080        │
                                          │ only while auto-dodge is ON (Y on the remote)   │
                                          ▼                                                 │
                                   /reflex/dash ─▶ go2_dodge.py ─▶ /api/sport/request ──────┘
                                                                      │ Ethernet, DDS domain 0
                                                                      ▼
                                                            Unitree Go2 sport service:
                                                            back 1.1 m/s for 0.5 s, hold 1 s,
                                                            walk back to the start by odometry
```

## Model lineage

| Step | Model | Where | Data | Who |
|---|---|---|---|---|
| 1. Pretraining | SNN backbone on DVS Gesture (11 classes) | `training/pretrain/` | IBM DVS Gesture (public) | Tim |
| 2. Supervised throw-motion detector | throwdet `D_e10_fold3` (2 outputs: none / throw motion), grouped 4-fold CV | `training/throwdet/` | own recordings 02–04 + augmented, synthetic, phone-video and DailyDVS clips | Laurenz |
| 3. Person fine-tune | `D_person01_neg`: head fine-tuned on 121 labelled windows, false alarms weighted ×2 | `training/throwdet/models/` | own person01 windows | Laurenz |
| 4. Board export | `d_neg11`: same net exported for the 11-output Q2.6 firmware already on the board (no new bitstream) | `perception/model_d_neg11/` | – | Laurenz |
| 5. **RL on the board** | **`hitl_step1`**: one policy-gradient (REINFORCE) step on the exact alarm rule, from live sessions where a person presses SPACE when the net should fire; picked in a live A/B test against the start model and RL snapshots 80–470 | `perception/model_hitl_step1/`, loop in `perception/hitl/` | live sessions on the board | Tim |

Board file: `perception/model_hitl_step1/weights.txt`, 78,501 values, md5 `d8fd7d025d666acb48ee31b4bc558fc7`.

## Evidence

Full table with conditions and labels: [evidence/README.md](evidence/README.md). Every number is labelled BOARD,
FLOAT-OOF (out of fold), IN-SAMPLE, SIM or OPEN.

| What | Number | Label |
|---|---|---|
| Model-only inference + transport, laptop ↔ Kria | 1.85 ms median round trip, 100 frames/s | BOARD |
| Board vs float model | 94.6 % frame agreement | BOARD |
| throwdet family, own camera, grouped 4-fold CV | AUC 0.89 mean, recall 0.54–0.79 at 2 false alarms/min | FLOAT-OOF |
| Unseen recordings `01` | 23/25 swings, 4 other alarms in 55 s, ≈ 150 ms before release | FLOAT-OOF |
| Start model d_neg11 on person01 | 45/53 throws, 2/68 false alarms | IN-SAMPLE |
| Dodge motion (deployed parameters) | 0.27 m back, returns within 2 cm | SIM |
| Dodges before ball arrival, false triggers/min, sensor → command delay on the real Go2 | to be filled in from the live trials | OPEN |

The delay chain is one 10 ms frame + ~2 ms FPGA round trip + at least 3 frames above threshold before
`/reflex/dash`; [evidence/README.md](evidence/README.md#how-to-measure-the-delay-sensor--command) says how to measure it end to end.

## What is new and what comes from the Durance starter

| From the Durance starter (not in this repo) | Ours (this repo) |
|---|---|
| RiverFPGA bitstream (11-output gesture firmware, used **unchanged**), SPLEAT server, `despot_debug_camera` | the SNN weights and readout that run on it (`perception/model_hitl_step1`) |
| riverops (training, quantisation, export) and river-tools (capture) | own dataset, labels and tools (`dataset/`, `recording/`, `tools/`), training (`training/`), RL loop (`perception/hitl/`) |
| go2-kria ROS 2 overlay (`unitree_api` / `unitree_go` messages, gesture_node, action_bridge) | `dodge/`: snn_bridge (camera → FPGA → readout → ROS 2), go2_dodge (dodge and return by odometry), web UI, launch file, 27 unit tests |

Durance's tools, SDK, documentation and gesture dataset are proprietary and are **not** included. The setup guide
in [docs/SETUP.md](docs/SETUP.md) says where to place them after downloading them from riversdk.dev.

## Robustness and safe defaults

- **Auto-dodge is off at start**; the robot only moves on Y (or X for a manual dodge).
- **Stale data never triggers**: no camera frame for 0.3 s, or the FPGA lost, resets the readout; a frame older than
  that never publishes `/reflex/dash`. The page shows camera / FPGA / ROS status.
- **Refractory periods**: 1 s after an alarm in the readout, and `go2_dodge` ignores triggers while a dodge runs and for 1 s after.
- **Clamped motion**: speed ≤ 1.5 m/s, duration ≤ 2 s, return speed ≤ 0.5 m/s; every phase has a time limit;
  without odometry the return walks only 80 % of the distance, so it never overshoots towards the thrower.
- **Stops**: `/dodge/stop` and Ctrl+C send StopMove (3 times on exit); `dry_run:=true` logs instead of moving.
- **Distractors in training**: walking, picking up the ball and camera handling from our own recordings, DailyDVS
  arm swings without release, unpressed live time in the RL sessions, and the board's own false alarms as hard
  negatives in the next round.
- **Known limits**: no obstacle check behind the dog (keep 1 m free); one thrower in the training data; feints count
  as throw motion (the arm swing is the same, the yarn ball is barely visible to the sensor).

## Reproduce

| Step | Command / doc |
|---|---|
| Environment | `uv sync` (riverops and river-tools wheels from riversdk.dev in `downloads/`), see [docs/SETUP.md](docs/SETUP.md) |
| Record and label | [dataset/README.md](dataset/README.md), [recording/PROTOCOL.md](recording/PROTOCOL.md) |
| Pretrain | [training/pretrain/README.md](training/pretrain/README.md) |
| Supervised detector | [training/throwdet/README.md](training/throwdet/README.md), data generators in `training/throwdet/datagen/` |
| RL on the board | [perception/hitl/README.md](perception/hitl/README.md) (`live_loop.py`, GPU host via `--gpu` or `HITL_GPU`) |
| Export and load | `perception/hitl/export_board.py`, `perception/hitl/board_ctl.py load <model dir>` |
| Robot | [dodge/README.md](dodge/README.md); tests: `python3 dodge/tests/test_snn_bridge.py` |

Dataset sources and splits: [dataset/SPLITS.md](dataset/SPLITS.md).

## Repository layout

| Path | What |
|---|---|
| `dodge/` | **robot stack that runs on the Kria**: snn_bridge, go2_dodge, web UI, launch file, tests |
| `perception/` | **deployed model** `model_hitl_step1`, its start model `model_d_neg11`, the RL loop `hitl/`, live viewer |
| `training/` | `pretrain/` (DVS Gesture), `throwdet/` (supervised detector + data generators), `early_ctx/` (labels used by the RL loop) |
| `dataset/`, `recording/`, `tools/` | own recordings, labels, splits, recording protocol, record / label / event tools |
| `evidence/` | measured numbers with conditions, trial template |
| `sim/` | Go2 simulation (MuJoCo, Gazebo) used to tune and check the dodge |
| `experiments/` | approaches we tried and did not deploy, with their results ([index](experiments/README.md)) |
| `docs/` | setup guide, the overnight report (`reports/`) |
| `scripts/` | ROS 2 / sim install helpers |

## Team and contributions

| Who | Main contributions |
|---|---|
| Laurenz | throwdet supervised detector and its data pipeline (augmentation, synthetic clips, phone video, DailyDVS import), d_neg11 export, recording protocol, first SNN export, Gazebo throw-vs-feint scene |
| Noah | `dodge/` robot stack (snn_bridge, go2_dodge, web UI, launch), own recordings and label tools, `_dataset01_*`, MuJoCo sport emulator, early-detection evaluation |
| Tim | DVS Gesture pretraining, early-detection models (v2b, v3), human-in-the-loop RL loop and `hitl_step1`, live viewer, low-level dodge controller in simulation |

## Trade-offs

- **One RL step, chosen live.** Pure RL with few presses drifts towards silence within ~20 steps, so we kept the
  snapshot that did best in a live A/B session instead of the last one.
- **Throw motion, not ball release.** The yarn ball is barely visible to the GenX320, so the net detects the arm swing
  (throws and feints alike). That makes the dodge early but means a feint also triggers it.
- **Reuse the gesture bitstream.** Exporting into the existing 11-output Q2.6 firmware avoided a new FPGA build
  during the event; outputs 2–10 are zero-weighted and never fire.
- **Sport API instead of the low-level controller.** The low-level RL dodge in `sim/` is faster in simulation, but the
  sport API is safer on the real dog and needs no gain tuning.
