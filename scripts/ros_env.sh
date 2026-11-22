# Source in WSL Ubuntu-22.04:  source /mnt/c/<path-to-repo>/scripts/ros_env.sh [robot|sim]
#   robot: same DDS settings as the Kria / Go2 (domain 0, CycloneDDS, network-wide)
#   sim:   Gazebo Go2 sim on a separate domain so it never mixes with the real robot
set +u
source /opt/ros/humble/setup.bash
[ -f /root/go2_ws/install/setup.bash ] && source /root/go2_ws/install/setup.bash
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
case "${1:-sim}" in
  robot)
    export ROS_DOMAIN_ID=0
    export ROS_LOCALHOST_ONLY=0
    ;;
  sim)
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    ;;
  *) echo "usage: source ros_env.sh [robot|sim]"; return 1 2>/dev/null || exit 1 ;;
esac
# With WSL mirrored networking, the CLI's "is a daemon running?" probe hangs when none is
# running (closed localhost ports time out instead of refusing). Starting it up front avoids that.
ros2 daemon start >/dev/null 2>&1
echo "ROS 2 $ROS_DISTRO | mode=${1:-sim} | ROS_DOMAIN_ID=$ROS_DOMAIN_ID | RMW=$RMW_IMPLEMENTATION"
