#!/usr/bin/env bash
# Build the ROS 2 workspace for Noah's throw reflex natively on the laptop (WSL Ubuntu 22.04, ROS 2 Humble).
#
#   bash real/build_ws.sh            (run_noah.ps1 build)
#
# Workspace: $NOAH_WS (default /root/noah_ws, on the Linux filesystem, outside the repo). It contains
#   - the Durance starter packages, linked from $DURANCE_SRC (the go2-kria source, NOT in this public repo):
#     common, msgs, unitree_go, unitree_api, obstacle_detection, robot_sim, debug_node
#   - gesture_node copied from there with real/gesture_node_num_scores.patch applied (adds parameter
#     num_scores, default 11 = unchanged; Noah's model sends 3)
#   - Noah's go2_sim_tools, linked from this folder (sim/ros_ws/src/go2_sim_tools, unchanged from his branch)
# Not built: action_bridge (it would send the starter's own gesture actions to the dog), bringup, qualia_sim,
# profiling_node.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
DURANCE_SRC=${DURANCE_SRC:-/root/hackathon/go2-kria/src}
NOAH_WS=${NOAH_WS:-/root/noah_ws}

[ -d "$DURANCE_SRC/msgs" ] || { echo "Durance starter source not found at $DURANCE_SRC (set DURANCE_SRC)"; exit 1; }
set +u; source /opt/ros/humble/setup.bash; set -u     # ROS setup scripts use unset variables

mkdir -p "$NOAH_WS/src/durance"
for pkg in common msgs unitree_go unitree_api obstacle_detection robot_sim debug_node; do
    ln -sfn "$DURANCE_SRC/$pkg" "$NOAH_WS/src/durance/$pkg"
done
rm -rf "$NOAH_WS/src/durance/gesture_node"
cp -r "$DURANCE_SRC/gesture_node" "$NOAH_WS/src/durance/gesture_node"
patch -p1 -d "$NOAH_WS/src/durance/gesture_node" < "$here/real/gesture_node_num_scores.patch"
ln -sfn "$here/sim/ros_ws/src/go2_sim_tools" "$NOAH_WS/src/go2_sim_tools"

cd "$NOAH_WS"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -Wno-dev "$@"
echo "built: source $NOAH_WS/install/setup.bash"
