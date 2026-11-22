# Go2 simulation (MuJoCo)

Physics sim of the Unitree Go2 via [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (Python variant).
Independent of the Gazebo/Docker sim in `sim/` (`sim.sh`, `ros_ws/`): runs directly on the host without ROS.

One-time setup (both clones are excluded from the repo by `.gitignore`):

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco          # tested with 1eb6642
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python  # tested with 814556d
conda create -n durance python=3.10 && conda run -n durance pip install mujoco pygame cyclonedds==0.10.2 -e third_party/unitree_sdk2_python
```

In `third_party/unitree_mujoco/simulate_python/config.py` set `USE_JOYSTICK = 0` so no gamepad is needed;
`DOMAIN_ID = 1` and `INTERFACE = "lo"` are already the defaults there.

Start (from the repo root):

```bash
# Terminal 1: simulator (MuJoCo viewer)
cd third_party/unitree_mujoco/simulate_python && conda run -n durance --no-capture-output python unitree_mujoco.py

# Terminal 2: sport emulator, the Go2 stands up and waits for requests
conda run -n durance --no-capture-output python sim/mujoco/sport_emulator.py

# Terminal 3: send a single action the way action_bridge does (name or api_id)
conda run -n durance --no-capture-output python sim/mujoco/send_action.py hello
```

`check_sim.py` is the low-level smoke test without the emulator (do not run it together with the emulator; both write `rt/lowcmd`).

- The sim runs on DDS domain **1**, interface `lo` (`simulate_python/config.py`). The real Go2 / Kria use domain 0, so a
  sim command can never reach the real robot.
- unitree_mujoco only knows low level: `rt/lowcmd` in, `rt/lowstate` + `rt/sportmodestate` out. `sport_emulator.py`
  replaces the real Go2's high-level sport service: it reads `rt/api/sport/request` (ROS topic `/api/sport/request`,
  which the starter's `action_bridge` publishes), answers on `rt/api/sport/response` (code 0, 3203 for an unknown
  api_id) and plays the matching keyframe motion from `sport_actions.py`.
- Emulated api_ids: 1001 damp, 1002 balance stand, 1004 stand up, 1005 stand down, 1006 recovery stand, 1009 sit,
  1010 rise from sit, 1016 hello, 1017 stretch, 1020 content, 1022 dance, 1029 and 1036 (both approximated as hello).
  A new request aborts the running motion.
- The motions are hand-tuned PD keyframes, not Unitree's controller: good for checking behaviour logic (which action
  when, order, abort), not for judging real dynamics or duration. No walking (`Move`), no LiDAR (`/utlidar/*`).
