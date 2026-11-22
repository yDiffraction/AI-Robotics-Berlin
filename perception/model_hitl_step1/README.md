# model_hitl_step1: the board model chosen in the final live evaluation (22 Nov 2026)

Snapshot step 1 of the human-in-the-loop run 20261122_151118 (`perception/hitl/live_loop.py`, RL on the A100,
`--loss pg`), started from `perception/model_d_neg11` (throwdet D_person01_neg_11out). Picked as the best model
in `live_loop.py --eval` against the start model and steps 80, 159, 237, 393, 470.

- Board: firmware with 11 outputs, weights Q2.6, potentials Q4.12; `weights.txt` has 78,501 values,
  md5 `d8fd7d025d666acb48ee31b4bc558fc7`. `config.yaml` / `model.json` are the same as v3's and d_neg11's.
- Readout (`readout.json`): score = 0.8 * score + 0.2 * (spike[1] - spike[0]) per 10 ms frame;
  alarm when score > 0.6 for 3 consecutive frames, then 1 s refractory. Output 0 = none, 1 = throw motion.
- `model.pt`: float checkpoint (EMA weights after one RL step from d_neg11).

## On the robot (installed 22 Nov 2026)

- Kria: `/root/weights.txt` = this model (also `/root/weights_hitl_step1.txt`). `spleat-server` is enabled at
  boot, so it survives reboots. FPGA firmware unchanged (11 outputs, Q2.6).
- Robot stack: Noah's `noah-web` branch (c9aea75) `dodge/` in `/root/dodge/` (`robot.launch.py`, `snn_bridge.py`,
  `web/`, `go2_dodge.py`, `kria/*.sh`). The previous `/root/dodge` is in `/root/dodge_backup_20250530_185247`
  (board clock).
- `/root/dodge/model/readout.json` = `readout_bridge.json` (this readout plus a name for the page).
- `snn_bridge.py` and `web/app.js` on the board have `snn_bridge_throwdet.patch` applied: a third readout mode `throwdet`
  (EMA of spike[1] - spike[0], hold frames), picked automatically from this readout.json. Without it the bridge
  would read the file as v3's EMA on channel 0. The page shows it as y in -1..1 with both neurons (ch 1 throw, ch 0 none) and θ -0.99..0.99. Checked: identical alarms to `perception/hitl/readout_spec.py`
  on 100k random frames; Noah's 27 tests pass. To be merged into noah-web.
- Start: `source /root/dodge/env.sh && ros2 launch /root/dodge/robot.launch.py`, page on port 8080.
