# Go2 Reflex — event-vision SNN behavior on RiverFPGA

Durance AI track, HackLab Berlin (21–22 Nov 2026).
Pipeline: Prophesee GenX320 event camera → SNN on Kria KV260 (RIVER IP) → ROS 2 Humble → Unitree Go2.

> Durance's tools, SDK, docs and dataset are proprietary and must **not** be committed.
> They live in `vendor/` and `data/`, both git-ignored.

## Layout

| Path | Content | In git |
|---|---|---|
| `pyproject.toml` | Python 3.10 training env (riverops, torch, jupyter) | yes |
| `vendor/downloads/` | Files from riversdk.dev (wheel, SDK installers, dataset zip) | no |
| `vendor/docs/` | Offline copy of the riversdk.dev hackathon cheatsheets + track slides | no |
| `vendor/go2-kria/go2-kria/` | Durance ROS 2 workspace (gesture_node, action_bridge, …) + cross-build scripts | no |
| `data/motion_buddy_dataset/` | Gesture dataset, sessions `01`–`14`, 1 s event recordings at 320×320 | no |
| `documentation/` | Markdown notes on Kria, deployment, cross-compilation, riverops API | yes |
| `scripts/install_ros_humble.sh` | ROS 2 Humble + Gazebo Go2 sim setup for Ubuntu 22.04 / WSL | yes |

Files from riversdk.dev need a login. After downloading, unpack:

```bash
unzip vendor/downloads/go2-kria-generated.zip -d vendor/go2-kria
unzip vendor/downloads/durance-ai-handgesture-recognition.zip -d data -x '__MACOSX/*'
```

## Setup (macOS, Apple Silicon)

No Linux VM with ROS is needed on the laptop: ROS 2 Humble runs on the Kria (Yocto image),
the laptop only cross-compiles inside a Docker container.

### 1. Training environment

```bash
uv sync                      # Python 3.10 + riverops 0.2.0 + torch (MPS)
uv run jupyter lab
```

### 2. Docker for cross-compilation

The build image is `linux/amd64`. On Apple Silicon run it through Colima with Apple
Virtualization and Rosetta (much faster than QEMU):

```bash
colima start default --vm-type vz --vz-rosetta --cpu 8 --memory 16 --disk 60
docker context use colima
```

### 3. Build the ROS 2 overlay for the Kria

```bash
cd vendor/go2-kria/go2-kria
mkdir -p sdk && cp ../../downloads/x86_64_sdk_installer.sh sdk/installer.sh
./manage.sh image        # once; SHA-256 of installer: ae40d393…6f18
./manage.sh check
./manage.sh build        # or: ./manage.sh build --packages-up-to gesture_node
./manage.sh package      # -> kria-runtime.tar.gz
```

### 4. Deploy to the Kria

```bash
scp kria-runtime.tar.gz root@durance-kria.local:/tmp/
ssh root@durance-kria.local      # no password, user root
release=$(mktemp -d /opt/go2-XXXXXX) && tar -xzf /tmp/kria-runtime.tar.gz -C "$release"
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
source /opt/ros/humble/setup.bash && source "$release/local_setup.bash"
python3 "$release/smoke.py"
```

UART fallback: `minicom -D /dev/cu.usbserial-XXXX -b 115200` (hardware flow control off).

### 5. Deploy a new SNN

Files to update on the board (see `vendor/docs/deploy_a_new_model.txt`):

- `/usr/lib/firmware/xilinx/spleat/spleat.bit.bin` + `spleat.dtbo` — generated on riversdk.dev → Firmware jobs (board KV260, IP GenX320)
- `/root/.config/spleat-server/config.yaml` — input/output shapes, `scaling = 2**fraction_bits`
- `/root/metavision_camera.json` — frame frequency, downsampling, camera biases
- `/root/weights.txt` — integer weights, no empty lines (not even at the end)

### 6. Run with the Go2

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml
ros2 topic echo --once --qos-reliability best_effort /utlidar/robot_pose geometry_msgs/msg/PoseStamped
ros2 launch bringup pipeline.launch.py despot:=false port:=3333
```

## Submission checklist (from the track brief)

- Working demo: 2-minute walkthrough showing sensor input, on-board inference, robot response; state what is new vs. the starter demo.
- Reproducible handoff: this README, launch/training steps, model/config, ROS 2 code, dataset sources and splits, contribution summary.
- Evidence: successes/trials, false triggers, measured sensor-to-command latency (separate from model-only inference time), test conditions, failure cases.
