#!/usr/bin/env bash
# Entry point used by run_noah.ps1 (WSL). Extra arguments go to ros2 launch as name:=value.
#   run.sh check            network: interface, dog, Kria, SPLEAT port
#   run.sh topics           real Go2 network: receive rates of LiDAR / scores / clearance (nothing is sent)
#   run.sh desk  [args]     everything on loopback with stand-ins for Kria, LiDAR and dog
#   run.sh dry   [args]     real LiDAR (+ Kria if up); the gate sends to /dry/api/sport/request, the dog gets nothing
#   run.sh robot [args]     real dog, through the keyboard gate (starts DISARMED)
#   run.sh sdk   [args]     Noah's standalone tools/go2_reflex.py (no ROS, no LiDAR, keys a/d/s/space); --dry-run
#                           Python: Tim's WSL venv /root/go2_sim/venv (has unitree_sdk2py), read-only use
here="$(cd "$(dirname "$0")/.." && pwd)"
mode=${1:-}; shift || true

# The launch (reflex, LiDAR clearance, scores) runs in the background with its log in logs/; the keyboard gate
# (real/control.py) runs in the foreground and is the only thing that publishes to the dog.
run_gated() {   # run_gated <launch file> <gate out topic> <label> <default scores> [launch args]
    local launch=$1 out=$2 label=$3 scores=$4; shift 4
    local kria=192.168.123.99 port=3333 a args=()
    for a in "$@"; do
        case "$a" in
            scores:=*) scores=${a#scores:=} ;;
            kria_host:=*) kria=${a#kria_host:=}; args+=("$a") ;;
            port:=*) port=${a#port:=}; args+=("$a") ;;
            *) args+=("$a") ;;
        esac
    done
    if [ "$scores" = auto ]; then
        if timeout 1 bash -c "</dev/tcp/$kria/$port" 2>/dev/null; then scores=kria; else scores=keys; fi
        echo "scores: $([ $scores = kria ] && echo "Kria $kria:$port reachable -> real model" \
                                           || echo "Kria $kria:$port NOT reachable -> keyboard fake throws (t / f)")"
    fi
    mkdir -p "$here/logs"
    log="$here/logs/$mode-$(date +%Y%m%d-%H%M%S).log"     # global: the EXIT trap needs log and lpid
    setsid ros2 launch "$launch" scores:="$scores" "${args[@]}" >"$log" 2>&1 < /dev/null &
    lpid=$!
    stop_launch() {
        kill -INT -- "-$lpid" 2>/dev/null
        for _ in $(seq 50); do kill -0 "$lpid" 2>/dev/null || break; sleep 0.1; done
        kill -KILL -- "-$lpid" 2>/dev/null
        echo "launch stopped. Log: $log"
    }
    trap stop_launch EXIT
    echo "launch running in the background, log: $log"
    python3 -u "$here/real/control.py" --out "$out" --scores "$scores" --label "$label"
}

case "$mode" in
    check)  exec bash "$here/real/check_link.sh" ;;
    topics) source "$here/real/env.sh" robot || exit 1; exec python3 "$here/real/probe_topics.py" "$@" ;;
    desk)   source "$here/real/env.sh" desk  || exit 1
            run_gated "$here/real/launch/desk.launch.py" /api/sport/request "DESK (robot_sim on loopback)" kria "$@" ;;
    dry)    source "$here/real/env.sh" robot || exit 1
            run_gated "$here/real/launch/real_robot.launch.py" /dry/api/sport/request "DRY (dog gets nothing)" auto "$@" ;;
    robot)  source "$here/real/env.sh" robot || exit 1
            run_gated "$here/real/launch/real_robot.launch.py" /api/sport/request "REAL DOG" auto "$@" ;;
    sdk)    iface=$(ip -4 -o addr | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')
            [ -n "$iface" ] || { echo "no 192.168.123.x interface (sim/deploy/setup_laptop.ps1)"; exit 1; }
            exec "${NOAH_SDK_PY:-/root/go2_sim/venv/bin/python}" -u "$here/tools/go2_reflex.py" --iface "$iface" "$@" ;;
    *) sed -n '2,9p' "$0"; exit 1 ;;
esac
