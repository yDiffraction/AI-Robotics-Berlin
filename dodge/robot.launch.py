#!/usr/bin/env python3
"""Robot launch: the dodge node and the SNN bridge with its web UI, one command, no package build.

  ros2 launch dodge/robot.launch.py                       # on the Kria: camera + FPGA local, web on :8080
  ros2 launch dodge/robot.launch.py dry_run:=true         # everything live, the robot only logs its moves
  ros2 launch dodge/robot.launch.py camera:=replay:/abs/path/rec.bin snn:=mock dry_run:=true   # desk test
  ros2 launch dodge/robot.launch.py --show-args           # all arguments

What runs (paths relative to this file):
  go2_dodge.py   X on the Go2 remote = manual dodge; /reflex/dash = dodge
                 (dry_run, speed, return_speed, odom, odom_topic)
  snn_bridge.py  event camera -> SNN on the FPGA -> throw detection; publishes /reflex/dash only while auto-dodge
                 is on (off at start; Y on the remote toggles it, so does the web page); web UI on web_port

Both are restarted after a crash (respawn:=false to turn that off). Ctrl+C stops both; go2_dodge sends StopMove.
Needs the ROS environment of the robot (on the Kria: source /root/dodge/env.sh) and, for the Go2 remote and the
color camera, the unitree_go / unitree_api messages of the go2-kria overlay.
"""
import os
import re
import subprocess
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

HERE = os.path.dirname(os.path.abspath(__file__))

ARGS = [  # name, default (Kria), description
    ("camera", "local", "local | ssh | listen | replay:<file.bin> (despot_debug_camera here, over SSH, by hand, file)"),
    ("snn", "fpga", "fpga (SPLEAT server) | mock (no board: airspace tripwire stand-in)"),
    ("spleat_host", "127.0.0.1", "SPLEAT server (the SNN on the FPGA)"),
    ("spleat_port", "3333", "SPLEAT server port"),
    ("theta", "", "detection threshold, '' = from model/readout.json (0.85, provisional)"),
    ("readout_mode", "", "ema | diff, '' = from model/readout.json"),
    ("auto", "false", "auto-dodge at start (SNN detections move the robot); Y on the remote toggles"),
    ("web_port", "8080", "web UI port"),
    ("color", "false", "show the Go2 front camera on the page at start (also a checkbox there)"),
    ("ssh_host", "root@192.168.77.1", "camera:=ssh: where despot_debug_camera runs"),
    ("listen_port", "3334", "port despot_debug_camera connects to"),
    ("remote", "both", "where the remote buttons are read: lowstate | wirelesscontroller | both"),
    ("dry_run", "false", "go2_dodge: log the moves, send nothing to the robot"),
    ("speed", "1.1", "go2_dodge: dodge speed m/s (max 1.5)"),
    ("return_speed", "0.25", "go2_dodge: speed back to the start m/s (max 0.5)"),
    ("odom", "sportmodestate", "go2_dodge: sportmodestate | odometry | none"),
    ("odom_topic", "", "go2_dodge: odometry topic, '' = default of the odom source"),
    ("respawn", "true", "restart a node that crashed (after 2 s)"),
    ("python", "", "Python interpreter for the nodes, '' = the one running ros2 launch"),
]


def _bool(name, v):
    low = v.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return "true"
    if low in ("false", "0", "no", "off"):
        return "false"
    raise ValueError("%s:=%s must be true or false" % (name, v))


def _float(name, v):
    try:
        return repr(float(v))      # always with a decimal point: ROS would read "1" as an integer
    except ValueError:
        raise ValueError("%s:=%s must be a number" % (name, v))


def _int(name, v):
    try:
        return str(int(v))
    except ValueError:
        raise ValueError("%s:=%s must be an integer" % (name, v))


def _urls(port):
    """Web addresses of this machine (ip -4 addr; BusyBox has it too)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
        addrs = [a for a in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", out) if not a.startswith("127.")]
    except (OSError, subprocess.SubprocessError):
        addrs = []
    return ["http://%s:%s/" % (a, port) for a in addrs] or ["http://<this machine>:%s/" % port]


def _setup(context):
    a = {name: LaunchConfiguration(name).perform(context) for name, _, _ in ARGS}
    py = a["python"] or sys.executable or "python3"

    def p(name, value):
        return ["-p", "%s:=%s" % (name, value)]

    dodge = [py, os.path.join(HERE, "go2_dodge.py"), "--ros-args",
             *p("dry_run", _bool("dry_run", a["dry_run"])), *p("speed", _float("speed", a["speed"])),
             *p("return_speed", _float("return_speed", a["return_speed"])), *p("odom", a["odom"]),
             *p("remote", a["remote"])]
    if a["odom_topic"]:
        dodge += p("odom_topic", a["odom_topic"])
    bridge = [py, os.path.join(HERE, "snn_bridge.py"), "--ros-args",
              *p("camera", a["camera"]), *p("snn", a["snn"]), *p("spleat_host", a["spleat_host"]),
              *p("spleat_port", _int("spleat_port", a["spleat_port"])), *p("auto", _bool("auto", a["auto"])),
              *p("web_port", _int("web_port", a["web_port"])), *p("color", _bool("color", a["color"])),
              *p("ssh_host", a["ssh_host"]), *p("listen_port", _int("listen_port", a["listen_port"])),
              *p("remote", a["remote"])]
    if a["theta"]:
        bridge += p("theta", _float("theta", a["theta"]))
    if a["readout_mode"]:
        bridge += p("readout_mode", a["readout_mode"])
    respawn = _bool("respawn", a["respawn"]) == "true"
    common = dict(output="screen", additional_env={"PYTHONUNBUFFERED": "1"}, respawn=respawn, respawn_delay=2.0,
                  sigkill_timeout="5")
    return [
        LogInfo(msg="Throw Reflex web UI: %s   (X = dodge, Y = auto-dodge on/off; auto-dodge %s at start%s)" % (
            "  ".join(_urls(a["web_port"])), "ON" if a["auto"].lower() in ("true", "1") else "off",
            ", DRY RUN" if a["dry_run"].lower() in ("true", "1") else "")),
        ExecuteProcess(cmd=dodge, name="go2_dodge", sigterm_timeout="5", **common),
        ExecuteProcess(cmd=bridge, name="snn_bridge", sigterm_timeout="8", **common),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(name, default_value=default, description=desc)
                              for name, default, desc in ARGS] + [OpaqueFunction(function=_setup)])
