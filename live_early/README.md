# Early throw-motion detector on the Kria (GOAL B)

Two exports of the early-detection SNN, both in the board firmware's format (arch v1, 11 outputs, 78501 values,
weights/bias Q2.6, potentials/threshold/leak Q4.12) and both with the same readout rule, only θ differs:

| folder | model | weights.txt md5 | θ (provisional) | held-out 02/03, θ-free comparison at ≤ 1 pause FA/min |
|---|---|---|---|---|
| `model/` | v2b: Noah's whfor_s1 + 8 min of v2 training | `a0a6f2a9afa1af61fa5d49f6fa33918e` | 0.70 | 56/107 swings, chance 20 (+36), lead −390 ms |
| `model_v3/` | v3_t8_s1: 40 min v2 training from DVS-Gesture init | `44f403275d8ea3c87d4e0bf93078cf08` | 0.85 | 48/107 swings, chance 18 (+30), lead −330 ms; 1.7 hard-neg FA/min |

The θ-free numbers pick θ on the test sessions (they compare models; they are not an operating point). v3 is the more
stable recipe across seeds (+28 to +34 vs +25/+36 for v2) and the first whose pre-registered θ rule works
(43/107 at 0.33 pause FA/min, lead −320 ms); v2b alarms a bit earlier. Load one, compare live, then calibrate θ on
the new session. Below, `model/` stands for whichever folder you load.

The rest of this section describes `model/` (v2b); `model_v3/` is built the same way (layer 1 scaled by 2.022,
float vs fixed point 95–97 % frame agreement, `model_v3/export_log.json`).

- **Output channel 0** = "throw motion is happening" (throws *and* feints count). Channels 1–10 are padding and never fire.
- **Host readout** (`model/readout.json`): `y = 0.85 * y + 0.15 * spike_ch0` per 10 ms frame; alarm when `y` rises past
  **θ = 0.7**, then 1 s refractory. **θ is provisional**: it was read off held-out test sessions and has to be
  recalibrated on the new recording session.
- Model: `runs/early_ctx` v2 pipeline (`ctx2.py` on the A100, `~/early_ctx/v2/runs/v2b/model_f0.pt`): trained on
  whole recordings of the 03 sessions + idle take + foreign negatives, tonic-drive penalty, clean BN statistics.
  It is a cross-validation fold model (not trained on 170441 / 201703 / 215305), i.e. a stopgap until an all-data model.
- Export (`export_early.py`): head padded to 11 outputs; layer 1 weights, bias and threshold divided by 2.288 so the
  weights fit Q2.6 (exact in float, the threshold is per layer in `weights.txt`). Float vs fixed-point emulation on two
  held-out recordings: 93–97 % of frames agree on the output spike, alarms 5/5 and 7/9 matched within 50 ms
  (`model/export_log.json`). **Not yet run on the real board.**

## Load it on the board

The laptop needs an address on the board subnet (board = 192.168.77.1), e.g. 192.168.77.198/24 on the Ethernet port
the board is plugged into. Loading replaces whatever model is on the board now (probably Noah's ball model), so back it up:

```bash
ssh root@192.168.77.1 'cp -n /root/weights.txt /root/weights_backup_$(md5sum /root/weights.txt | cut -c1-32).txt; ls /root/weights_backup_*'
scp live_early/model/weights.txt root@192.168.77.1:/root/weights.txt
ssh root@192.168.77.1 'md5sum /root/weights.txt && systemctl restart spleat-server && systemctl is-active spleat-server'
```

For v3 use `live_early/model_v3/weights.txt`. The md5 must match the table above. To go back: copy the backup over `/root/weights.txt` and restart `spleat-server`.

## Check and watch it

Needs Python with numpy (and torch + riverops for the float model, e.g. the repo's `.venv`).

```bash
# board vs float model on a recording (headless): prints spike agreement, alarm matches, round trip
python live_early/viewer.py --replay <frames.npy or recording.bin> --backend both --check --fast
# live camera through the FPGA, viewer at http://127.0.0.1:8770
python live_early/viewer.py
# FPGA and the float model side by side
python live_early/viewer.py --backend both
# with v3 on the board: same, plus --model live_early/model_v3 (float model and θ of v3)
python live_early/viewer.py --model live_early/model_v3 --backend both
```

`--model` must match what is on the board: it sets θ and the float model the FPGA is compared against.

The viewer starts `despot_debug_camera` on the board over SSH; the camera connects back to the laptop on port 3334
(allow Python through the firewall). Stop other camera users on the board first (`despot_send_and_forget`,
`record_tool`, `live_view`). `.npy` replays are bit-packed frames from `runs/early_ctx/prep.py`.

Files: `viewer.py` (camera/replay → FPGA and/or float model → readout → web page), `model/` (export + float
checkpoint `model.pt`), `snn_model.py` / `speedpatch.py` (Tim's model code, unchanged copies, for the float backend).
