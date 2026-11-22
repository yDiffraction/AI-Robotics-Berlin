# Human-in-the-loop training for the early throw detector

The loop: run the current model on the Kria, a person presses a button whenever the network *should* fire,
fine-tune on those presses (GPU), load the new weights onto the board, and repeat.

```
camera ─SSH tunnel─▶ laptop ─frames─▶ Kria (SPLEAT) ─spikes─▶ readout/alarm ─▶ label page (you press SPACE)
                                                                                    │ sessions/<stamp>/
board_ctl.py load ◀── <run>/board/weights.txt ◀── train.py (GPU, float replica + BPTT) ◀┘
```

| file | what |
|---|---|
| `collect.py` + `collect.html` | live camera → board (and/or float model) → web page at :8771 with a press button, live hit/miss/false-alarm score; records frames, board spikes, readout, presses |
| `train.py` | fine-tunes from v3 on sessions (plus optionally the early_ctx recordings), validates on held-out time, exports `<out>/board/` with θ* |
| `export_board.py` | checkpoint → board `weights.txt` (reproduces `model_v3/weights.txt` bit for bit, md5 `44f40327…`) |
| `board_ctl.py` | `status` (which model is on the board), `load <folder>` (backup, copy, restart, md5 check) |

## Connect to the board

The board is cabled straight to the laptop. Without an IPv4 address on that adapter (192.168.77.x), use the IPv6
link-local address; `%50` is the adapter index (`Get-NetAdapter`) and can change after replugging:

```powershell
$B = "fe80::20a:35ff:fe28:d34c%50"
python perception/hitl/board_ctl.py status --ssh "root@$B"
```

With an IPv6 `--host`, `collect.py` starts `despot_debug_camera` through an SSH reverse tunnel (the camera connects
to the board's own 127.0.0.1, and SSH forwards that to the laptop), so no firewall rule or IPv4 address is needed.

## One round

```powershell
# 1. label live (v3 is on the board). Press SPACE when the network should fire: at the throw's release
#    (the target starts 500 ms before your press). BACKSPACE undoes a press. Ctrl+C ends the session.
python perception/hitl/collect.py --host $B --model perception/model_v3
# 2. train (on the laptop with a CUDA GPU, or on the A100, see below)
python perception/hitl/train.py perception/hitl/sessions/* --out perception/hitl/runs/r1 --minutes 30
# 3. look at report.json: "init" (v3 float) and "board_recorded" (what really ran) vs "selected"; then load
python perception/hitl/board_ctl.py load perception/hitl/runs/r1/board --ssh "root@$B"
# 4. next round with the new model; its false alarms become hard negatives automatically
python perception/hitl/collect.py --host $B --model perception/hitl/runs/r1/board
```

Labelling notes:
- Mix throws, feints, walking and ball pick-ups. Unpressed stretches are negatives, and alarms the board makes
  there are oversampled as hard negatives in the next training round.
- A press usually comes 200–300 ms after the release. `--lag N` (frames) shifts every press earlier;
  `--label hold` uses the time the button was held instead of the press moment.
- `collect.py --replay rec.bin --speed 0.5` labels a recording at half speed instead of live.

## Live loop: collect here, RL on the A100 at the same time (`live_loop.py`)

```powershell
python perception/hitl/live_loop.py --host "fe80::20a:35ff:fe28:d34c%50"        # window: http://127.0.0.1:8771
```

- Start model: `--model perception/model_d_neg11` (default). That is throwdet's `D_person01_neg_11out` from the
  throw-detector branch (b538a6d), exported for the 11-output Q2.6 firmware already on the board, so no new
  bitstream is needed. Its readout is `score = 0.8·score + 0.2·(spike[1] − spike[0])`, with an alarm when
  score > θ (−0.04) for 3 frames. `--model perception/model_v3` goes back to v3 and its EMA readout.
  `readout_spec.py` reads the rule from the folder's readout.json; the trainer, the exports and the window all
  follow it.
- Syncs the trainer code to the A100 (`~/hitl`) and starts `online_train.py` there (`--loss pg` = REINFORCE on the
  alarm rule; `--loss pg,mil` or `sup,mil` for the gradient losses; `--train-args "--bs 16 --lr 1e-4"` for more).
- Uploads 10 s chunks, presses, θ and the watching flag every ~5 s. The trainer reloads them every 30 s and
  exports a snapshot every `--snapshot-s` (120 s).
- In the window:
  - **θ slider:** sets the board readout's θ immediately and the θ the RL reward uses.
  - **Just watching:** the frames are recorded but get weight 0 in training (presses there are ignored).
    Switch it OFF only while you are labelling, because unpressed labelled time means "should not fire".
  - **Update board now:** loads the newest snapshot. The board restarts (~3 s) and the trace shows no spikes meanwhile.
  - **Auto-update every N min:** 0 means only the button.
  - **Back to v3.**
  - Trainer step, labelled minutes, presses, RL return, the output firing rate, and the upload state.
- Snapshots that went onto the board are saved under `runs/live_<id>/`. The A100 keeps `~/hitl/live/<id>/`
  (sessions, snapshots, `train.log`). Ctrl+C stops the window and the trainer.
- Pure RL with few presses mostly learns to stay silent: the output rate drops to 0 within ~20 steps when the
  data is mostly unpressed. Watch the output rate before you update the board, and use "Back to v3" if needed.

### Final evaluation (`live_loop.py --eval`)

No trainer and no uploads. The window lists the start model and every snapshot under `runs/live_*/<step>/` (the
ones that went onto the board). "Load" puts one on the board. Each model keeps its own hits / misses / false
alarms / FA per min / alarm − press from the moment it is loaded, and switching back adds to its row. Press SPACE
whenever it should fire. The session is recorded as usual.

## Training (`train.py`)

The ctx2 (v3) recipe, continued from v3: same targets (500 ms before to 300 ms after the press), tonic-drive
penalty, τ ≤ 8 frames, weight EMA, flip/shift/thinning/noise augmentation, frozen BatchNorm statistics.
Hard negatives are the board's unclaimed alarms plus the idle/invalid labels of the early_ctx recordings.
`--ctx-frames ~/early_ctx/frames` replays the old whole recordings next to the new sessions so the model
does not forget them.

Losses (`--loss`, default `sup,mil`):
- `sup`: ctx2's frame-wise weighted MSE on the readout.
- `mil`: event level. Each press window must push max(y) above θ + 0.05, and negatives must stay below θ − 0.05.
  It targets v3's failure: precise, but its readout often peaks just under θ. On held-out sessions it crosses its
  pre-registered θ on 19 % of throws, but θ = 0.7 would already catch 56 %.
- `pg`: REINFORCE on the exact alarm rule (K = 4 sampled spike trains, leave-one-out baseline). Optional, use a small
  weight; see "Does RL help" below.

Validation streams the last 20 % of every session with carried state, like the board, and picks θ* = the lowest θ
with ≤ 1 false alarm per minute. The selected checkpoint is the one with the most hits at θ*.

## GPU

Measured on the A100, one step with a 300 + 60 warm-up frame window:

| batch | peak GPU memory |
|---|---|
| 4 | 2.0 GB |
| 8 | 4.0 GB |
| 16 | 7.9 GB |
| 32 | 15.8 GB |

**A laptop RTX 4070 (8 GB) is enough at `--bs 8` to `--bs 12`.** The step is launch-bound (a Python loop over
360 frames × 5 layers of tiny tensors), so the A100 mostly buys batch size, not speed. About 2.5 s per batch-16
step on a shared A100. ctx2's fused Triton loop (`training/early_ctx/lif_triton.py`) is ~5× faster but needs
Linux/WSL. A full hour of sessions is ~0.6 GB of packed frames and fits on the GPU as-is.

On the A100 (`~/hitl` has a copy of the trainer):

```bash
scp -i ~/.ssh/hackathon_a100 -r perception/hitl/sessions team-1-user@38.128.233.249:hitl/
ssh -i ~/.ssh/hackathon_a100 team-1-user@38.128.233.249 'cd ~/hitl && ~/snn_pretrain/.venv/bin/python -W ignore \
  perception/hitl/train.py sessions/* --ctx-frames ~/early_ctx/frames --bs 32 --minutes 30 --out runs/r1'
scp -i ~/.ssh/hackathon_a100 -r team-1-user@38.128.233.249:hitl/runs/r1 perception/hitl/runs/
```

## Checks done (2026-11-22)

- `export_board.py` on `model_v3/model.pt` → md5 `44f403275d8ea3c87d4e0bf93078cf08` (identical to the shipped file).
- The training replica (`train.run`) matches the riverops forward spike for spike (104/104 on 400 frames,
  also chunked with carried state). Gradients reach every layer.
- Board (v3) vs float on recording 201703: 276 vs 255 output spikes (276 = the fixed-point emulation in
  `export_log.json`), 94.6 % frame agreement, 2/2 alarms, round trip 1.85 ms median.
- Live camera → tunnel → board: 100 frames/s, 1.7 ms round trip (empty scene, no spikes).
- `board_ctl.py load perception/model_v3`: backup, copy, restart, md5 verified.
- `train.py` end to end with `sup,mil,pg` on two replayed sessions, on CPU and on the A100, through to export.
  **Not yet shown: that fine-tuning on real labelled sessions beats v3.** That needs the first real session.
