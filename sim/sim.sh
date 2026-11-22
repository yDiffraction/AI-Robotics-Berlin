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
    up)       if [ "${2:-}" = windows ]; then
                  # the X cookie with a wildcard host, so that it is valid inside the container
                  export SIM_XAUTH=${XDG_RUNTIME_DIR:-/tmp}/go2-sim.xauth
                  : > "$SIM_XAUTH"
                  xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$SIM_XAUTH" nmerge -
                  docker compose -f compose.yaml -f compose.windows.yaml up -d gazebo
                  echo "Gazebo and RViz open as windows on display $DISPLAY"
              else
                  docker compose up -d "$service"
                  [ "$service" = gazebo ] && echo "Desktop: http://localhost:6080/vnc.html" || true
              fi ;;
    down)     docker compose down ;;
    build)    shift
              in_ws "cd /ws && colcon --log-base /ws/out/log build --base-paths /ws/src \
                     --build-base /ws/out/build --install-base /ws/out/install \
                     --cmake-args -DCMAKE_BUILD_TYPE=Release $*" ;;
    run)      shift; in_ws "ros2 launch go2_sim_tools sil.launch.py $*" ;;
    gazebo)   shift; in_ws "export XDG_RUNTIME_DIR=/tmp/runtime-root DISPLAY=\${DISPLAY:-:1}; ros2 launch go2_sim_tools gazebo.launch.py $*" ;;
    demo)     shift; in_ws "export XDG_RUNTIME_DIR=/tmp/runtime-root DISPLAY=\${DISPLAY:-:1}; ros2 launch go2_sim_tools throw_demo.launch.py $*" ;;
    demo2)    shift; in_ws "export XDG_RUNTIME_DIR=/tmp/runtime-root DISPLAY=\${DISPLAY:-:1}; ros2 launch go2_sim_tools two_stage_demo.launch.py $*" ;;
    keys)     in_ws "ros2 run go2_sim_tools reflex_keyboard" ;;
    measure)  shift; in_ws "python3 /ws/tools/measure_reflex.py $*" ;;
    emulator) shift; docker exec -it "$container" python3 /ws/tools/spleat_emulator.py "$@" ;;
    smoke)    in_ws "python3 /ws/durance-scripts/smoke.py" ;;
    shell)    in_ws bash ;;
    *)  cat <<'USAGE'
Usage: ./sim.sh <command>          (SIM=gazebo ./sim.sh ... selects the Gazebo container)
  image      build the Docker image (once)
  up         start the container          down   stop it
  up windows Gazebo container with its windows on this desktop instead of noVNC (Linux, X11)
  build      colcon-build Durance + sim packages inside the container
  smoke      run the Durance smoke test against the native build
  run        launch pipeline + robot_sim + fake_lidar   (args: host:=... robot:=false ...)
  emulator   start the SPLEAT emulator, e.g. ./sim.sh emulator --mode manual
  gazebo     launch the Gazebo Go2 + sport API bridge (gazebo container only)
  demo       throw-vs-feint dodge demo in Gazebo, view at http://localhost:6080/vnc.html
  demo2      two-stage demo: lean in on the wind-up, back off on the throw (args: keyboard:=true gui:=true retreat_speed:=0.7 ...)
  keys       trigger the two-stage reflex by hand: a = arm, d = dash, s = disarm (with demo2 keyboard:=true)
  measure    time the two-stage reflex while demo2 runs, e.g. ./sim.sh measure --trials 12, or --delays 0.05,0.3 to trigger it itself
  shell      open a sourced shell in the container
USAGE
    ;;
esac
