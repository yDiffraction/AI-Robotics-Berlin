#!/usr/bin/env bash
# Go2 low-level simulation stack in WSL Ubuntu-22.04 (no NVIDIA GPU needed; MuJoCo on CPU).
#   wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/<path-to-repo>/scripts/install_go2_sim.sh
# Installs into /root/go2_sim:
#   venv/                  python3.10 venv: mujoco, torch (CPU), numpy, matplotlib, unitree_sdk2py
#   walk-these-ways-go2/   public pretrained Go2 locomotion policy (gait-conditioned)
#   unitree_mujoco/        Unitree's official MuJoCo Go2 model + DDS bridge (same LowCmd/LowState as the dog)
#   unitree_sdk2_python/   Unitree SDK2 Python bindings
#   unitree_rl_gym/        Unitree's RL repo (humanoid policies only; kept for reference)
#   cyclonedds/            CycloneDDS 0.10.x, built from source (needed by unitree_sdk2py)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get install -y python3-pip python3-venv git cmake build-essential libglfw3 libglfw3-dev >/dev/null

mkdir -p /root/go2_sim && cd /root/go2_sim
[ -d unitree_rl_gym ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git
[ -d unitree_mujoco ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git
[ -d unitree_sdk2_python ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git
[ -d walk-these-ways-go2 ] || git clone --depth 1 https://github.com/Teddy-Liao/walk-these-ways-go2.git

if [ ! -f cyclonedds/install/lib/libddsc.so ]; then
  [ -d cyclonedds ] || git clone --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds
  (cd cyclonedds && mkdir -p build install && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF >/dev/null \
    && cmake --build . --target install -j"$(nproc)" >/dev/null)
fi

[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q mujoco numpy pyyaml matplotlib pygame
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
CYCLONEDDS_HOME=/root/go2_sim/cyclonedds/install pip install -q -e unitree_sdk2_python

python -c "import mujoco, torch; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('mujoco', mujoco.__version__, '| torch', torch.__version__, '| unitree_sdk2py OK')"
echo GO2_SIM_READY
