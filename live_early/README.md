# Early throw-motion detector on the Kria (GOAL B, v2b)

`model/weights.txt` is the early-detection SNN in the board firmware's format: arch v1, 11 outputs, 78501 values,
weights/bias Q2.6, potentials/threshold/leak Q4.12, md5 `a0a6f2a9afa1af61fa5d49f6fa33918e`.

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

The md5 must be `a0a6f2a9afa1af61fa5d49f6fa33918e`. To go back: copy the backup over `/root/weights.txt` and restart `spleat-server`.

## Check and watch it

Needs Python with numpy (and torch + riverops for the float model, e.g. the repo's `.venv`).

```bash
# board vs float model on a recording (headless): prints spike agreement, alarm matches, round trip
python live_early/viewer.py --replay <frames.npy or recording.bin> --backend both --check --fast
# live camera through the FPGA, viewer at http://127.0.0.1:8770
python live_early/viewer.py
# FPGA and the float model side by side
python live_early/viewer.py --backend both
```

The viewer starts `despot_debug_camera` on the board over SSH; the camera connects back to the laptop on port 3334
(allow Python through the firewall). Stop other camera users on the board first (`despot_send_and_forget`,
`record_tool`, `live_view`). `.npy` replays are bit-packed frames from `runs/early_ctx/prep.py`.

Files: `viewer.py` (camera/replay → FPGA and/or float model → readout → web page), `model/` (export + float
checkpoint `model.pt`), `snn_model.py` / `speedpatch.py` (Tim's model code, unchanged copies, for the float backend).
