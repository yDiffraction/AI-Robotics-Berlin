#!/usr/bin/env bash
# Desk simulation helper. Run from anywhere; see sim/README.md.
set -euo pipefail
cd "$(dirname "$0")"

# Pick the running container, otherwise $SIM (sil | gazebo), default sil.
service=${SIM:-}
if [ -z "$service" ]; then
    if docker ps --format '{{.Names}}' | grep -qx go2-sim-gazebo; then service=gazebo; else service=sil; fi
fi
container=go2-sim-$service
in_ws() { docker exec -it "$container" /entrypoint.sh bash -c "$*"; }

case "${1:-help}" in
    image)    docker compose build "$service" ;;
    up)       docker compose up -d "$service"
              [ "$service" = gazebo ] && echo "Desktop: http://localhost:6080/vnc.html" || true ;;
    down)     docker compose down ;;
    build)    shift
              in_ws "cd /ws && colcon --log-base /ws/out/log build --base-paths /ws/src \
                     --build-base /ws/out/build --install-base /ws/out/install \
                     --cmake-args -DCMAKE_BUILD_TYPE=Release $*" ;;
    run)      shift; in_ws "ros2 launch go2_sim_tools sil.launch.py $*" ;;
    gazebo)   shift; in_ws "ros2 launch go2_sim_tools gazebo.launch.py $*" ;;
    emulator) shift; docker exec -it "$container" python3 /ws/tools/spleat_emulator.py "$@" ;;
    smoke)    in_ws "python3 /ws/durance-scripts/smoke.py" ;;
    shell)    in_ws bash ;;
    *)  cat <<'USAGE'
Usage: ./sim.sh <command>          (SIM=gazebo ./sim.sh ... selects the Gazebo container)
  image      build the Docker image (once)
  up         start the container          down   stop it
  build      colcon-build Durance + sim packages inside the container
  smoke      run the Durance smoke test against the native build
  run        launch pipeline + robot_sim + fake_lidar   (args: host:=... robot:=false ...)
  emulator   start the SPLEAT emulator, e.g. ./sim.sh emulator --mode manual
  gazebo     launch the Gazebo Go2 + sport API bridge (gazebo container only)
  shell      open a sourced shell in the container
USAGE
    ;;
esac
