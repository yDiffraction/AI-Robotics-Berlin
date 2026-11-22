# Noah's throw reflex on the real Go2 (laptop wired to the dog)

Noah's ROS 2 nodes, unchanged, run natively on the laptop (WSL Ubuntu 22.04, ROS 2 Humble) together with the Durance starter's `gesture_node` and `obstacle_detection`. They talk to the dog directly over the cable.

This folder is **separate from `sim/deploy/`** (Tim's low-level RL dodge). Neither imports from the other. Try them one at a time on the dog.

## Quick start: the command we run on the dog

Laptop wired to the Go2, run from PowerShell in `deploy_noah`:

```powershell
.\run_noah.ps1 robot arm_time:=1.0 approach_speed:=0 retreat_speed:=1.0 retreat_dist:=0.5 return_speed:=0.3
```

This is the two-stage reflex with the **defaults of Noah's `two_stage_reflex` node** (his Gazebo values), except for one change: **no walking forward** (`approach_speed:=0`).
- **Wind-up** (the scores leave idle): the reflex arms and the dog stays where it is for up to 1.0 s.
- **Throw** while armed: the dog backs off **0.5 m at 1.0 m/s**, waits 1 s, then walks forward 0.5 m back to its start at 0.3 m/s.
- **Feint, idle again, or no throw within 1.0 s:** it disarms, and the dog doesn't move.
- Distances come from the dog's own odometry (`/utlidar/robot_odom`).

The keyboard gate starts **DISARMED**. Nothing reaches the dog until you press `a`.

| Key | |
|---|---|
| `a` | arm: the reflex drives the dog |
| **Space** | STOP (disarm + StopMove) |
| `x` | DAMP, the same as the remote's L2+B: the dog drops |
| `t` / `f` | fake throw / feint (when the Kria is not connected) |
| `u` / `d` | stand up / lie down (while disarmed) |
| `h` | status |
| `q`, Ctrl+C | StopMove and quit |

Before running:
- **Network:** once, as admin, run `sim/deploy/setup_laptop.ps1 -Adapter "<adapter>"`, then `wsl --shutdown`.
- **Build:** once, `.\run_noah.ps1 build`.
- **Dog:** sport mode on and standing, with **at least 1 m free behind it**. Backing off has no LiDAR check behind the dog.
- **Remote:** its buttons don't reach the laptop, so Space and `x` on the laptop are the stops.
- **Rehearse first:** `.\run_noah.ps1 dry ...` with the same arguments sends everything to `/dry/api/sport/request`, so the dog gets nothing.

```
Kria (spleat-server, Noah's 3-class SNN) --TCP 3333--> gesture_node --> /durance/gestures --> gesture_trigger
   (no Kria: real/control.py publishes fake scores, keys t / f)                     /reflex/arm|dash|disarm |
Go2 LiDAR: height map --> obstacle_detection --> /durance/clearance ----------+                              v
Go2 odometry /utlidar/robot_odom ---------------------------------------------+--> two_stage_reflex (Noah)
                                                                             /noah/api/sport/request |
Go2 sport service <-- /api/sport/request <-- real/control.py (keyboard gate: forwards only while ARMED) <-+
```

Two of Noah's reflexes, chosen with `reflex:=`:

**`two_stage` (default; Noah's newer one, from `2ed4aec`)**: `gesture_trigger` turns the scores into `/reflex/arm|dash|disarm`, and `two_stage_reflex` acts on them:
- **Wind-up** (non-idle share > 0.5): it arms and leans in 0.10 m at 0.2 m/s (only when the LiDAR says CLEAR), then waits up to 1.5 s.
- **Throw** while armed: it backs off 0.4 m at 0.6 m/s, waits 1 s, then walks back to the start at 0.25 m/s.
- **Feint, or idle again:** it walks back.
- Move is re-sent every 0.1 s. Distances come from the dog's `/utlidar/robot_odom` (144 Hz on the dog).
- The defaults are Noah's real-robot values from `tools/go2_reflex.py`. His Gazebo values are faster.
- Backing off has **no LiDAR check behind the dog**. Keep about 1 m free behind it.

**`sidestep` (the older `throw_reflex`)**:
- A throw counts when its share of the scores is > 0.6 for 2 frames in a row.
- On **CLEAR**: sidestep with `Move(vy=±0.8)` for 1 s, StopMove, walk back. Sides alternate. One action per 3 s.
- On **OBSTACLE or UNKNOWN**: duck (StandDown, then StandUp after 1.5 s).
- Feints are logged and ignored. Events go out as JSON on `/reflex/event`.

## Layout

| Path | What |
|---|---|
| `sim/ros_ws/src/go2_sim_tools/` | Noah's package (throw_reflex, fake_lidar, …), **unchanged**. |
| `sim/mujoco/`, `sim/tools/spleat_emulator.py`, `tools/spleat_logger.py` | Noah's MuJoCo sport emulator and SPLEAT tools, **unchanged**. |
| `real/launch/real_robot.launch.py` | The nodes of his Gazebo `throw_demo`, minus the simulation: gesture_node, obstacle_detection, throw_reflex. `send:=false` sends commands to `/dry/api/sport/request` instead, so the dog gets nothing. |
| `real/launch/desk.launch.py` | The same launch with stand-ins: Noah's SPLEAT emulator (3 scores), his `fake_lidar`, the starter's `robot_sim`. |
| `real/build_ws.sh` | Builds `/root/noah_ws` (see below). |
| `real/gesture_node_num_scores.patch` | The one change to starter code (see below). |
| `real/env.sh` | ROS + DDS settings: `robot` uses the wired 192.168.123.x interface on domain 0; `desk` uses loopback only on domain 73. |
| `real/probe_topics.py`, `real/check_link.sh`, `real/run.sh` | Helpers behind `run_noah.ps1`. |

The Noah paths mirror his branch (copied from `noah` at `2ed4aec`). To take his newer changes, run this from the repo root:

```bash
git archive origin/noah sim/ros_ws/src/go2_sim_tools sim/mujoco sim/tools/spleat_emulator.py sim/tools/measure_reflex.py tools/spleat_logger.py tools/go2_reflex.py dodge | tar -x -C deploy_noah
```

Then run `.\run_noah.ps1 build`.

## What the build needs (checked in WSL, 2026-11-22)

Already installed:
- ROS 2 Humble desktop, `rmw_cyclonedds_cpp`, colcon and rosdep;
- nlohmann-json;
- the Durance starter source at `/root/hackathon/go2-kria/src` (only cross-built for the Kria there).

**Installed now:** `ros-humble-rosidl-generator-dds-idl` (apt). The starter's `unitree_go` / `unitree_api` message packages need it.

`build_ws.sh` puts together `/root/noah_ws` on the Linux filesystem, outside the repo. The Durance code stays out of this public repo.
- **Linked from the starter:** `common`, `msgs`, `unitree_go`, `unitree_api`, `obstacle_detection`, `robot_sim`, `debug_node`.
- **`gesture_node`:** copied, then patched (see the next section).
- **Noah's `go2_sim_tools`:** linked from this folder.
- **Left out on purpose:** `action_bridge`, which would send the starter's own gesture actions to the dog. Also `bringup`, `qualia_sim` and `profiling_node`.

If the starter source lives somewhere else, set `DURANCE_SRC=/path/to/go2-kria/src`.

### The gesture_node patch

The starter `gesture_node` accepts only payloads of exactly 11 float32 scores (DVS-Gesture) and drops everything else. Noah's model sends 3 scores (throw, feint, idle), so every frame from the real Kria would be dropped. His Gazebo demo didn't hit this because his fake thrower pads its scores to 11.

The patch adds a parameter `num_scores`:
- The default is 11, so the node behaves as before.
- With 3, the 3 scores fill slots 0–2 of the unchanged 11-slot message, and the other slots stay 0. That is the layout `throw_reflex` and Noah's `throw_scene` already use.

The patch file contains a few lines of `gesture_node.cpp` as diff context. Check that this is acceptable before pushing to a public repo, since the track rules say not to publish Durance tools.

### DDS fix needed on this laptop

Under WSL's mirrored networking, **UDP datagrams above ~1470 bytes are dropped silently**, even on loopback. Measured: 1296 B arrive, 1444 B do not. CycloneDDS sends up to 14720 B per datagram by default, so every height map was lost: `obstacle_detection` never published, and `throw_reflex` stayed at UNKNOWN and only ducked.

`env.sh` sets `MaxMessageSize 1400B`, so DDS fragments the messages itself. With that, 80×80 maps (25.6 kB) arrive at 10 Hz. This only controls what the laptop **sends**. The dog sends its own large messages with its own settings: `.\run_noah.ps1 topics` shows whether its height map gets through.

## On the Kria (on the dog, 192.168.123.99)

- Noah's model (3 classes: throw, feint, idle) must be deployed, and `spleat-server` must be running.
- After a reboot, run `despot_test_camera`, then `despot_send_and_forget --to_display 0`.
- **The starter ROS pipeline (`ros2 launch bringup pipeline.launch.py`) must not run on the Kria.** Its `action_bridge` would send its own commands to the dog. Its `gesture_node` would also publish a second `/durance/gestures` stream on the same network.

## Keyboard control (`dry`, `robot`, `desk`)

`throw_reflex` publishes on `/noah/api/sport/request`. `real/control.py` runs in the foreground and is the only thing that talks to the dog. It forwards the reflex's commands only while ARMED and **starts DISARMED**. The launch runs in the background, with its log in `logs/`.

| Key | |
|---|---|
| `a` | ARM (refused while a reflex action is still running) |
| Space | STOP: disarm + StopMove |
| `x` | DAMP: disarm + Damp, the same as the remote's L2+B (the dog drops) |
| `u` / `d` | stand up (StandUp, then BalanceStand) / lie down; only while disarmed |
| `t` / `f` | fake throw / feint (only with no Kria, `scores:=keys`): a 0.6 s wind-up, then 0.4 s of throw / feint, the score shape of Noah's Gazebo thrower |
| `h` | status: armed, score rate, LiDAR clearance |
| `q`, Ctrl+C | StopMove, then quit |

Scores: if the Kria (`.99:3333`) is reachable, it uses the real model; otherwise it uses keyboard fake throws. Force it with `scores:=kria` or `scores:=keys`.

Noah's standalone `tools/go2_reflex.py` (the same two-stage moves, but with no ROS and no LiDAR, keys a/d/s/Space) runs with `.\run_noah.ps1 sdk`. Add `--dry-run` to print the requests only. It uses Tim's WSL venv `/root/go2_sim/venv` only for its Python and `unitree_sdk2py`; none of Tim's code is used. His `dodge/` (a minimal ROS dodge node) is copied for reference and not wired in.

## Run, in this order

From PowerShell, in `deploy_noah`:

```powershell
.\run_noah.ps1 build     # once, and after updating Noah's files
.\run_noah.ps1 desk      # no hardware: throw -> sidestep (fake LiDAR clear), feint ignored. Ctrl+C to stop
.\run_noah.ps1 check     # dog wired: interface, dog .161, Kria .99, port 3333
.\run_noah.ps1 topics    # does the dog's height map arrive? (sends nothing)
.\run_noah.ps1 dry       # real Kria + real LiDAR; the dog gets nothing
```

During `dry`:
- Throw and feint at the camera, and watch the `THROW` / `feint` lines. Count false triggers.
- `obstacle_detection` logs every switch between CLEAR and OBSTACLE. Walk around the dog to see it.
- Watch the commands it would send: `ros2 topic echo /dry/api/sport/request`, after `source real/env.sh robot`.

Real dog: sport mode on, dog standing. Keep about 1 m free behind it (two_stage backs off) or to both sides (sidestep). The remote's buttons don't reach the laptop, so the laptop keys Space (stop) and `x` (damp) are the stops:

```powershell
.\run_noah.ps1 robot retreat_speed:=0.4 approach_speed:=0.0   # gentle first: no lean-in, slower back-off
.\run_noah.ps1 robot                                           # Noah's real-robot values
.\run_noah.ps1 robot reflex:=sidestep dodge_speed:=0.3         # the older sidestep reflex
```

Launch arguments (`name:=value`):
- Scores: `kria_host`, `port`, `num_scores`, `window`, `threshold`, `confirm_frames`.
- Reflex: `reflex` (two_stage | sidestep).
- Two-stage: `idle_class`, `windup_threshold`, `arm_time`, `approach_speed`, `approach_dist`, `retreat_speed`, `retreat_dist`, `return_speed`, `hold_time`, `tolerance`, `odom_topic`.
- Sidestep: `dodge_speed`, `dodge_time`, `cooldown`, `alternate_sides`, `recenter`, `stale_timeout`.
- LiDAR: `radius`, `blind_radius`, `min_height`, `block_unknown`.
- Scores source: `scores` (kria | keys; default: the Kria if reachable).

## Watch for on the real dog (Noah's code, not changed here)

- **Move is sent once per sidestep.** If the dog only twitches instead of stepping ~1 s sideways, the Go2 needs Move repeated. That's a change for Noah's `throw_reflex`.
- **After a duck it sends StandUp, not BalanceStand.** If the dog stays locked standing and later sidesteps do nothing, that's why.
- **The clearance check covers only 0.35–0.5 m around the dog.** At 0.8 m/s the sidestep travels further than that.
- These four apply to `reflex:=sidestep`. The two-stage reflex re-sends Move and has no duck. Quitting the keyboard gate always sends StopMove.

## Tested (without the dog)

Desk launch, the same nodes as `real_robot.launch.py`:
- Emulator scores (3 per frame) → patched gesture_node → throw_reflex.
- **Throw while clear:** `1008 y=+0.8` → `1003` → `1008 y=-0.8` → `1003`.
- **Feint:** ignored.
- **Obstacle placed at 0.45 m:** clearance switched to OBSTACLE, and the next throws ducked (`1005`, then `1004`).
- **`send:=false`:** all commands went to `/dry/api/sport/request`, and `robot_sim` received 0 requests.
