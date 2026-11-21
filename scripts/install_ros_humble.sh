#!/usr/bin/env bash
# ROS 2 Humble + Gazebo Classic + Unitree Go2 (CHAMP) sim on Ubuntu 22.04 (WSL)
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# Locale
apt-get update
apt-get install -y locales curl git software-properties-common ca-certificates
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
add-apt-repository -y universe

# ROS 2 apt source (official ros2-apt-source package)
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
  dpkg -i /tmp/ros2-apt-source.deb
fi

apt-get update
apt-get upgrade -y

# ROS 2 Humble desktop + dev tools + DDS used by the Kria/Go2
apt-get install -y ros-humble-desktop ros-dev-tools python3-rosdep \
  ros-humble-rmw-cyclonedds-cpp

# Gazebo Classic (gazebo11) + ROS integration + Go2 sim deps (from README)
apt-get install -y ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control ros-humble-xacro ros-humble-robot-localization \
  ros-humble-ros2-controllers ros-humble-ros2-control \
  ros-humble-velodyne ros-humble-velodyne-gazebo-plugins ros-humble-velodyne-description \
  ros-humble-teleop-twist-keyboard ros-humble-joint-state-publisher-gui

# rosdep
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init
rosdep update --rosdistro humble

# Go2 workspace
mkdir -p /root/go2_ws/src
cd /root/go2_ws/src
[ -d unitree-go2-ros2 ] || git clone --branch humble https://github.com/anujjain-dev/unitree-go2-ros2.git
# Load controllers with `spawner` instead of the daemon-based CLI (avoids robot collapsing on spawn)
python3 "$(dirname "$(readlink -f "$0")")/patch_go2_spawners.py" 2>/dev/null \
  || python3 /mnt/c/Users/diffr/code/AI-Robotics-Berlin/scripts/patch_go2_spawners.py
cd /root/go2_ws
set +u
source /opt/ros/humble/setup.bash
set -u
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --symlink-install

# Shell setup
grep -q "/opt/ros/humble/setup.bash" /root/.bashrc || cat >> /root/.bashrc <<'EOF'

# ROS 2 Humble + Go2 sim
source /opt/ros/humble/setup.bash
source /root/go2_ws/install/setup.bash
source /usr/share/gazebo/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# WSL mirrored networking: start the ros2 CLI daemon up front, otherwise the first
# `ros2 topic/node ...` hangs probing for a daemon that isn't there.
ros2 daemon start >/dev/null 2>&1
EOF

echo "INSTALL_DONE"
