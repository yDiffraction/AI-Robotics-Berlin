# Running our own controller on the real Go2

## Real robot: SAFE dodge with sport mode as the default

`go2_modes.py` keeps Unitree's sport mode in charge and switches to our code only on request:

| Where | Button (remote) | Key (laptop terminal) | What happens |
|---|---|---|---|
| sport mode | hold **L1+R1** 1 s | `o` | sport mode lies the dog down, is released; `supervisor.py` starts `dodge_controller.py`, which stands the dog up (~3 s) |
| our mode | **D-pad ←** or **A** | `l` or `d` | SAFE dodge **left**: 0.5 s stand, ~2 s trotting in place, then ~30 cm (up to ~0.5 m) left, brake, stand. Refused unless the dog has stood still ≥ 1 s. The dodge is the one validated in sim (`../results/safe_dodge.json`), joint targets clamped 0.1 rad inside the range as validated |
| our mode | **D-pad →** | `r` | the same dodge to the **right**: its mirror image (all sideways commands negated), validated separately (`../results/safe_dodge_right.json`, below) |
| our mode | **Y** | `s` | dog lies down, our code exits, sport mode comes back (the dog stays lying: stand it up with the remote as usual) |
| our mode | **L2+B** | **Space** or Ctrl+C | **E-STOP**: controller killed, damping (supervisor). Repeated presses do nothing more; damping stays on until `s` / Y |
| after an E-STOP | **Y** | `s` | damping stops (after ≥ 2 s), sport mode comes back |
| sport mode | | `q` / Ctrl+C | quit |

Buttons count only alone (A, Y, D-pad) or exactly together (L1+R1), seen for 60 ms, and each needs a release first. B is not used alone: it is half of the L2+B e-stop. Before handing control to us, `go2_modes.py` checks that sport mode's StandDown has folded the legs; otherwise it aborts and sport mode stays in charge. No command is ever sent while sport mode is active (tested).

### Laptop setup (once)

This laptop has no Ethernet port: use a USB-Ethernet adapter.
1. Plug it in and connect it to the Go2.
2. In an **administrator** PowerShell:
   ```powershell
   Get-NetAdapter
   .\setup_laptop.ps1 -Adapter "<name of the USB adapter>"
   wsl --shutdown
   ```
   It sets 192.168.123.222/24 (no gateway; internet stays on WLAN). It also allows inbound UDP from 192.168.123.0/24 only, in WSL's Hyper-V firewall (default: Block) and the Windows firewall. `-Undo` reverts all of it.
3. WSL already runs with `networkingMode=mirrored` (`~/.wslconfig`), so the adapter appears inside WSL; the scripts find it by its 192.168.123.x address.
4. Disconnect Cisco AnyConnect while working with the robot.

### First session, in this order

1. `.\run_go2.ps1 check`: the interface is listed and 192.168.123.161 answers.
2. `.\run_go2.ps1 probe` (read-only, sport mode on):
   - lowstate at ~500 Hz;
   - pressing L1+R1, A, Y, D-pad ← / → and L2+B shows them under `bytes ... keys`. `go2_modes.py` reads the buttons there; if they don't show, use the keys on the laptop.
   - Also check that L1+R1 alone does nothing in sport mode on this firmware (it isn't in the manuals' button lists, but those differ between firmware versions).
   - Finally, switch the remote off for 5 s and note what the probe shows.
3. **Dog hanging in a harness**, `.\run_go2.ps1`:
   - L1+R1: the legs fold, then extend to the stand pose.
   - Press L2+B: the legs go soft. Y: sport mode back.
   - Again L1+R1, then **unplug the cable**, and note what the dog does (the open question from below).
   - **Do not press A or the D-pad while it hangs** (the policy would trot in the air).
4. On the floor, no dodge yet: L1+R1, the dog stands; Y, the dog lies down and sport mode is back.
5. **Dodge**:
   - Flat floor with grip, no payload, charged battery.
   - **≥ 1 m free on the dodge side and ≥ 0.5 m in front**: the 2 s of trotting in place drifts it forward by up to ~0.3 m in the fake robot. The dog may also end up turned by up to ~20° (left) / ~30° (right) in sim.
   - Someone holds the remote with a thumb near L2+B.
   - Start with `.\run_go2.ps1 gentle` and press D-pad ← (or A): all sideways speeds × 0.6.
     - In 300 harsh sim runs per side: 0 falls, 0 unsettled ends, peak tilt 11° left / 10° right (the full SAFE dodge reaches 17°).
     - It ends 24–30 cm (left) / 22–27 cm (right) away. It usually does not reach 30 cm, which is the only check it fails.
     - Then D-pad → for the gentle right dodge.
   - Then `.\run_go2.ps1` for the full SAFE dodge.

The controller prints the longest gap between two commands after each dodge. On the test machine it was 4 ms, while the supervisor trips at 50 ms.

### Right dodge: sim validation (`../results/safe_dodge_right.json`)

The right dodge is the SAFE dodge mirrored: kick, hold and brake commands negated, everything else the same. It was run through the same 500 harsh-physics runs as the left one: friction 0.35–1.1, payload −1…+3 kg, motors 80–100 %, 0–30 ms latency, sensor noise, random gait phase at the trigger. Both sides got the same 500 conditions (seed 2026).

| 500 harsh runs | Left (SAFE) | Right (mirrored) |
|---|---|---|
| failures (fall, unsettled end, load limit, < 30 cm) | 0 | 0 |
| 30 cm after the kick: median / p95 / worst | 0.58 / 0.64 / 0.88 s | 0.60 / 0.66 / 0.83 s |
| final distance: median (min) | 0.50 (0.31) m | 0.48 (0.34) m |
| peak tilt | 16.7° | 13.8° |
| foot force / trunk acceleration, worst (limits 1.81 BW / 1.93 g) | 1.62 BW / 1.65 g | 1.68 BW / 1.63 g |
| joint margin to the stop, worst | 0.15 rad | 0.22 rad |
| forward/backward drift, worst | 0.18 m | 0.14 m |
| heading change, worst | 19° | 29° |
| gentle (× 0.6, 300 runs): falls / end distance | 0 / 0.24–0.30 m | 0 / 0.22–0.27 m |

The policy is close to symmetric. The right dodge loads the joints a little less, but turns the dog more (up to 29°). Walk-these-ways has no heading control in this setup, so the dog ends wherever the dodge leaves it pointing.

### Tests of the mode switching ([test_modes.py](test_modes.py), `../results/modes_test.log`)

`fake_go2.py` now also provides fake `motion_switcher` and `sport` RPC services on the real Unitree RPC layer.
- While "sport mode" is active, the fake robot holds its own pose and records any command it receives.
- After a release, SelectMode answers "busy" for 10 s.

The test drives the unmodified `go2_modes.py` → `supervisor.py` → `dodge_controller.py`. **12/12 scenarios pass.**

| Scenario | Result |
|---|---|
| remote: L1+R1, A or D-pad ←, Y (robot holds / goes limp without commands) | dodge 0.52–0.53 m left, 30 cm after 0.57–0.58 s, tilt ≤ 0.21 rad, standing after; back in sport mode, dog lying |
| remote: D-pad → | dodge 0.52 m right, 30 cm after 0.58 s, tilt 0.19 rad |
| keyboard (`o`, `d`, `s`) in a pseudo-terminal | same as left |
| `--gentle`: A (remote) / `r` (keyboard) | 0.35 m left / 0.30 m right, tilt 0.19 rad |
| L2+B / Ctrl+C 0.25 s into the kick, then Y | damping after 10–30 ms, dog on its belly (roll < 0.1 rad), Y releases, sport mode back |
| A and Y during the stand-up | A refused, dog lies down, SelectMode "busy" retried until sport mode is back |
| L1+R1 for 0.6 s; L1+R1+A; A, Y, L2+B in sport mode | nothing happens, nothing sent |
| go2_modes.py killed while standing | controller lies down by itself (its input closed), supervisor ends cleanly, no processes left |
| all | **no command ever reached the robot while sport mode was active** |

Bugs found on the way:
- The first policy call after A took > 50 ms (TorchScript optimises on first use), so the supervisor tripped. The controller now warms the policy up at start.
- A normal Python exit takes > 50 ms, so the supervisor saw "controller silent" before the clean exit. The controller now exits immediately.
- Switching from A to Y within 20 ms was missed: the two button sources update at 500 Hz and 50 Hz, so the change looked like A+Y for one sample. A button now counts once it has been seen alone for 60 ms.

## E-stop and disconnect handling for the low-level Go2 controller

With sport mode released, Unitree's own protections are gone: the remote's L2+B damping, fall handling, and any defined behaviour when commands stop. This folder adds them back as a **separate supervisor process** that owns the controller:

```
supervisor.py  (small, reviewed, no policy code)
  │ starts ──> controller (child process: dodge policy, or stand_hold.py for tests)
  │            own session (terminal Ctrl+C reaches only the supervisor);
  │            killed automatically if the supervisor dies (PR_SET_PDEATHSIG)
  │
  ├─ listens: rt/lowstate (IMU, tick, wireless_remote, CRC)   rt/wirelesscontroller (keys)
  │           rt/lowcmd from the controller (CRC + sanity check of every command)
  └─ on any trip: SIGKILL the controller's process group, then publish DAMPING
                  (kp=0, kd=5 on all 12 joints) at 500 Hz until a second Ctrl+C
```

Only one process publishes `rt/lowcmd` at any time: the controller until the trip, then only the supervisor. Before the controller has sent its first valid command, the supervisor sends nothing, because sport mode may still be active then.

## Triggers

| Trigger | Rule | Why this value |
|---|---|---|
| **E-stop** | L2+B on the remote, read both from `rt/lowstate.wireless_remote` and from `rt/wirelesscontroller` | the combination Go2 operators already know; walk-these-ways' deploy uses it too |
| Operator | Ctrl+C or SIGTERM to the supervisor | a Ctrl+C must never just kill the controller and leave the last stiff target |
| Fall | roll or pitch > 0.8 rad | walk-these-ways' threshold; the SAFE dodge peaks at 0.26 rad (15°) under harsh physics |
| Controller dead or hanging | no valid `rt/lowcmd` for 50 ms (crash, hang, stopped process, wrong CRC, too slow) | the robot keeps stepping on stale targets meanwhile; the sim tolerates 30 ms latency |
| Controller output garbage | NaN/inf, target outside the joint range (where kp > 0), kp > 80, kd > 8, \|tau\| > 10 Nm, mode ≠ 1 | catches broken math and joint-order slips; 5450 commands of 20 harsh SAFE-dodge runs: 0 rejected |
| Robot link | no valid `rt/lowstate` for 100 ms | published at ~500 Hz |
| Robot frozen | `tick` stops for 100 ms while messages still arrive | only armed once the tick has been seen moving (no false trip if the firmware leaves it at 0) |
| Remote link | `rt/wirelesscontroller` silent for 0.5 s | only armed once the topic has been seen. **Unverified** whether the Go2 stops publishing when the remote is off: check with `remote_probe.py` |

Tripping is latched: the first reason is kept and the controller is never restarted. Getting the dog up again means a new start (or sport mode via the app).

## Tests

**End to end over real Unitree DDS** ([test_supervisor.py](test_supervisor.py), results in `../results/supervisor_test.log`).
- **Setup:** the unmodified `supervisor.py` and a test controller (`stand_hold.py`), against `fake_go2.py`. The fake robot is a MuJoCo Go2 that publishes `rt/lowstate` (with CRC, tick and remote bytes) and `rt/wirelesscontroller`, and executes `rt/lowcmd`. It runs on a private domain 77, loopback only.
- **Runs:** 18 scenarios × 3, alternating the two guesses for what the robot does without commands ("hold last command" and "go limp").
- **Result: 54/54 pass.**

"Time to damping" runs from the fault to the first damping command arriving at the robot.

| Fault | Trip reason | Time to damping |
|---|---|---|
| L2+B (lowstate / remote topic) | e-stop | 2–4 ms / 12–24 ms |
| controller crashes, SIGKILL | controller exited | 2–4 ms |
| controller hangs, SIGSTOP, sends every 80 ms, wrong CRC | controller silent | 50–54 ms |
| controller sends NaN / out-of-range target | bad command | 0–4 ms |
| `rt/lowstate` stops / tick frozen | robot link | 102–104 ms |
| remote topic stops | remote link | 500–504 ms |
| robot pushed over | tilt | ~100 ms after the push starts |
| Ctrl+C / SIGTERM to the supervisor | operator | 2–4 ms |
| supervisor SIGKILLed | (none) | controller dies with it; no more commands reach the robot |
| 10 s without fault, L2/B/L2+A/R2+B pressed | no trip | |

Every tripped run ended with the controller process dead, no controller command after the first damping command, and the robot lying down (trunk at 0.08–0.10 m).

**Bench bug found on the way:** the test harness created new DDS readers for every scenario. Python DDS readers are not freed reliably, so the readers piled up, each one processing every command. By the 9th scenario delivery lagged by 0.3–1.4 s, and old damping samples leaked into the next run. The fake robot now keeps one set of endpoints for the whole suite. For the real controller this means: **create DDS readers once, never per episode.**

**What the stop does to the robot mid-dodge** ([test_stop_sim.py](test_stop_sim.py), `../results/stop_sim.log`).
- **Setup:** 100 harsh-physics runs. The controller stops at a random moment, from 0.3 s before the trigger to the end of the stand-up. The last command is kept for a random 0–80 ms detection delay, then the stop mode applies.

| Stop mode | Tipped onto its side | Trunk peak g, p95 / max | Trunk hits floor at | Joint speed at a mechanical stop | Foot force max |
|---|---|---|---|---|---|
| **damping kd=5 (used)** | **0/100** | **2.3 / 5.5** | **≤ 0.45 m/s** | **≤ 2.7 rad/s** | **1.2 BW** |
| damping kd=2 | 0/100 | 3.7 / 5.0 | ≤ 0.75 m/s | ≤ 7.0 rad/s | 2.1 BW |
| damping kd=8 | 0/100 | 1.7 / 5.1 | ≤ 0.23 m/s | ≤ 1.6 rad/s | 1.3 BW |
| zero torque (power cut / firmware goes limp) | 0/100 | 9.3 / 10.4 | ≤ 1.6 m/s | ≤ 15.8 rad/s | 3.1 BW |
| last command forever (nobody sends, firmware holds) | **48/100** | 7.2 / 7.7 | ≤ 0.86 m/s | ≤ 10.9 rad/s | 3.6 BW |

Damping lowers the dog onto its belly; median trunk peak 1.0 g.
- 93 of 100 stops stay under the dodge's own load limit (1.93 g, 1.25 × the worst normal sideways walk).
- The 7 above it, up to 5.5 g, are all stops 0.3–0.45 s into the dodge, when the dog moves sideways at full speed and slides down onto its belly.
- The two alternatives are far worse:
  - Holding the last command tipped the dog over in half the runs.
  - Going limp hits the floor 3.6 times faster.

kd=8 is gentler in sim, but kd=5 is the value that has been used on real Go2s. A higher kd also amplifies velocity-sensor noise, which the sim does not model. So kd stays at 5 until kd=8 has been tried on the hanging robot.

**Rules and false alarms** ([test_safety.py](test_safety.py)): 33 checks, all pass.
- Remote bit layout as in Unitree's example.
- Command check: NaN, joint range, gains, mode.
- Tilt from the quaternion matches MuJoCo to 1e-15.
- Every timeout fires at its threshold, and never before the check is armed.
- 20 harsh SAFE-dodge runs: 5450 commands, **0 rejected**; peak tilt 0.26 rad vs the 0.8 rad trip.

## What this cannot catch

- **The cable between this computer and the Go2 is pulled, or this computer freezes or loses power.** Nothing reaches the robot. Whether the Go2 then holds the last command, goes limp or damps is not documented. Walk-these-ways sends a single damping command and then stops publishing, so it relies on "holds the last command", but that is not a test. **Test once with the dog hanging:** run `stand_hold.py` under the supervisor, unplug the cable, and watch.
- **The supervisor itself is killed** (SIGKILL, out of memory): the controller dies with it (tested), so the robot gets no commands, the same as an unplugged cable.
- **The remote is switched off or out of range**: detected only if the Go2 stops publishing `rt/wirelesscontroller` (see the probe). Otherwise only the other triggers remain; keep someone near the power button.
- **Sport-mode handover** is done by `go2_modes.py` (StandDown, legs folded, ReleaseMode; back with SelectMode, retried while the robot answers "busy", reported for ~10 s after a release). The supervisor runs only in between. Tested against a fake motion-switcher service, not yet against the real one.
- Lift detection, motor temperature and battery are still missing (see the risk list). The lidar and Collision Monitor watchdog in `../avoid` only zeroes the velocity; it does not damp.

## Run

```bash
# WSL / Linux with /root/go2_sim/venv (unitree_sdk2py + CycloneDDS):
python test_safety.py                      # rules + false-alarm check against the SAFE dodge (also on Windows)
python test_supervisor.py                  # 18 scenarios x 3 over real DDS against fake_go2 (~10 min)
python test_stop_sim.py --n 100            # what the stop does to the robot mid-dodge (also on Windows)
python test_modes.py                       # go2_modes -> supervisor -> dodge_controller vs fake robot + fake sport mode (~5 min)

# real robot, read-only first (safe with sport mode on):
python remote_probe.py --iface eth0
# then, dog HANGING in a harness:
python supervisor.py --iface eth0 -- python stand_hold.py --iface eth0 --release-sport-mode
#   press L2+B -> legs go soft; Ctrl+C -> same; unplug the cable -> note what the dog does
```

For the dodge, don't start these by hand: `go2_modes.py` (`.\run_go2.ps1` on Windows) runs `dodge_controller.py` as the supervisor's child.
