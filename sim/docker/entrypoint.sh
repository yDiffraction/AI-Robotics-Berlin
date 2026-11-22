#!/bin/bash
# Source ROS, the optional Go2 Gazebo overlay and our workspace (if built).
set -e
source /opt/ros/humble/setup.bash
[ -f /opt/go2_sim/install/setup.bash ] && source /opt/go2_sim/install/setup.bash
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh
[ -f "$WS/out/install/setup.bash" ] && source "$WS/out/install/setup.bash"
exec "$@"
