#!/usr/bin/env bash
# Source this: ROS 2 Humble + the noah workspace + the DDS settings for one of two networks.
#
#   source real/env.sh robot   # the real Go2 on the wired 192.168.123.x interface, DDS domain 0 (Go2 default)
#   source real/env.sh desk    # loopback only, private domain 73: nothing can reach the dog, even if it is wired
NOAH_WS=${NOAH_WS:-/root/noah_ws}
[ -f "$NOAH_WS/install/setup.bash" ] || { echo "workspace not built: run_noah.ps1 build"; return 1; }
set +u
# MaxMessageSize 1400B: under WSL mirrored networking, UDP datagrams above ~1470 bytes are dropped silently (measured:
# 1296 B arrive, 1444 B do not, even on loopback). CycloneDDS's default 14720 B lost every height map; with 1400 B
# it fragments in DDS instead and 25.6 kB maps arrive at full rate.
source /opt/ros/humble/setup.bash
source "$NOAH_WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0

case "${1:-}" in
    robot)
        iface=$(ip -4 -o addr | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')
        if [ -z "$iface" ]; then
            echo "no interface with 192.168.123.x: run sim/deploy/setup_laptop.ps1 (admin), then wsl --shutdown"
            return 1
        fi
        export ROS_DOMAIN_ID=0
        export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"$iface\"/></Interfaces><MaxMessageSize>1400B</MaxMessageSize></General></Domain></CycloneDDS>"
        echo "DDS: real Go2 network, interface $iface, domain 0"
        ;;
    desk)
        export ROS_DOMAIN_ID=73
        export CYCLONEDDS_URI='<CycloneDDS><Domain Id="73"><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast><MaxMessageSize>1400B</MaxMessageSize></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
        echo "DDS: desk (loopback only, domain 73)"
        ;;
    *)
        echo "usage: source real/env.sh robot|desk"; return 1 ;;
esac
