"""Noah's throw-vs-feint reflex on the REAL Go2, run from the laptop wired to the dog.

  Kria (SPLEAT server, Noah's 3-class SNN)  --TCP-->  gesture_node  -> /durance/gestures --+
  Go2 LiDAR  /utlidar/robot_pose + /utlidar/height_map_array -> obstacle_detection          |
                                              -> /durance/clearance ------------------------+-> throw_reflex
                                                                         /api/sport/request <-+   (Noah's node)
  the Go2's sport service executes /api/sport/request (Move / StopMove / StandDown / StandUp)

The same nodes Noah's Gazebo demo (sim/ros_ws/src/go2_sim_tools/launch/throw_demo.launch.py) runs, minus the
simulation: no throw_scene, fake_lidar, sport_api_bridge or Gazebo. The starter's action_bridge is NOT started:
it would send its own gesture actions to the dog.

throw_reflex is remapped to /noah/api/sport/request. Nothing here talks to the dog: real/control.py (started next
to this launch by run.sh) forwards those requests to /api/sport/request only while ARMED, or to
/dry/api/sport/request in dry mode.

  reflex:=two_stage (default)  gesture_trigger + two_stage_reflex: arm on the wind-up (lean in 0.1 m), back off
                               0.4 m on the throw, walk back; Move re-sent every 0.1 s, distances from odometry
  reflex:=sidestep             the older throw_reflex (sidestep, or duck when the LiDAR is not clear)
  scores:=keys   no Kria: gesture_node is not started, control.py publishes /durance/gestures (keys t / f)

Run through run_noah.ps1 (it sources real/env.sh robot first), or:
  source real/env.sh robot && ros2 launch real/launch/real_robot.launch.py   (+ real/control.py)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals as Is
from launch.substitutions import LaunchConfiguration as Arg
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = [
    # scores from the Kria
    ("kria_host", "192.168.123.99", "SPLEAT server (Kria on the Go2 network; IPv4 only)"),
    ("port", "3333", "SPLEAT server port (config.yaml on the Kria)"),
    ("num_scores", "3", "scores per frame of the model on the Kria (Noah's: throw, feint, idle)"),
    ("window", "3", "gesture_node moving average in frames (Noah's demo: 3)"),
    # LiDAR clearance (starter defaults)
    ("radius", "0.5", "obstacle_detection: check radius around the robot (m)"),
    ("blind_radius", "0.35", "obstacle_detection: ignored inner radius (the robot itself)"),
    ("min_height", "0.2", "obstacle_detection: lowest obstacle height above the floor (m)"),
    ("block_unknown", "false", "obstacle_detection: unobserved cells block too"),
    # which of Noah's reflexes
    ("reflex", "two_stage", "two_stage: lean in on the wind-up, back off on the throw (gesture_trigger + "
                            "two_stage_reflex); sidestep: the older throw_reflex"),
    # scores -> decision (both reflexes)
    ("throw_class", "0", ""), ("feint_class", "1", ""), ("idle_class", "2", "two_stage only"),
    ("threshold", "0.6", "throw / feint share of the scores to count a frame"),
    ("windup_threshold", "0.5", "two_stage: non-idle share that arms (wind-up)"),
    ("confirm_frames", "2", "consecutive frames above threshold"),
    # two_stage_reflex; defaults = Noah's real-robot values from tools/go2_reflex.py
    ("arm_time", "1.5", "s the robot stays armed (leaned in) waiting for the throw"),
    ("approach_speed", "0.2", "m/s forward when armed (only with LiDAR CLEAR), 0: stay put"),
    ("approach_dist", "0.10", "m forward at most when armed"),
    ("retreat_speed", "0.6", "m/s backwards on the throw"),
    ("retreat_dist", "0.4", "m backwards on the throw"),
    ("return_speed", "0.25", "m/s back to the start"),
    ("hold_time", "1.0", "s to wait after backing off"),
    ("tolerance", "0.03", "m, closer than this counts as at the start"),
    ("odom_topic", "/utlidar/robot_odom", "nav_msgs/Odometry of the Go2; missing: dead reckoning"),
    # throw_reflex (sidestep, Noah's defaults)
    ("dodge_speed", "0.8", "lateral speed of the sidestep (m/s)"),
    ("dodge_time", "1.0", "sidestep duration (s)"),
    ("cooldown", "3.0", "time per action incl. walking back (s)"),
    ("alternate_sides", "true", ""), ("recenter", "true", "walk back after the sidestep"),
    ("stale_timeout", "0.5", "StopMove if no scores for this long while moving (s)"),
    # where the scores come from
    ("scores", "kria", "kria: gesture_node reads the SPLEAT server; keys: control.py publishes fake scores"),
]

TYPES = {"idle_class": int, "windup_threshold": float, "arm_time": float, "approach_speed": float,
         "approach_dist": float, "retreat_speed": float, "retreat_dist": float, "return_speed": float,
         "hold_time": float, "tolerance": float, "odom_topic": str, "port": int, "num_scores": int, "window": int, "radius": float, "blind_radius": float,
         "min_height": float, "block_unknown": bool, "throw_class": int, "feint_class": int,
         "threshold": float, "confirm_frames": int, "dodge_speed": float, "dodge_time": float,
         "cooldown": float, "alternate_sides": bool, "recenter": bool, "stale_timeout": float}


def p(name):
    return ParameterValue(Arg(name), value_type=TYPES[name])


def generate_launch_description():
    reflex_params = [{k: p(k) for k in ("throw_class", "feint_class", "threshold", "confirm_frames",
                                        "dodge_speed", "dodge_time", "cooldown", "alternate_sides",
                                        "recenter", "stale_timeout")}]
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default, description=desc) for name, default, desc in ARGS],

        Node(package="gesture_node", executable="gesture_node", output="screen",
             parameters=[{"host": Arg("kria_host"), "port": p("port"), "num_scores": p("num_scores"),
                          "window": p("window")}],
             condition=Is("scores", "kria")),
        Node(package="obstacle_detection", executable="obstacle_detection", output="screen",
             parameters=[{k: p(k) for k in ("radius", "blind_radius", "min_height", "block_unknown")}]),

        # two-stage (default): scores -> /reflex/arm|dash|disarm -> lean in / back off / walk back
        Node(package="go2_sim_tools", executable="gesture_trigger", output="screen",
             parameters=[{k: p(k) for k in ("throw_class", "feint_class", "idle_class", "windup_threshold",
                                            "threshold", "confirm_frames")}],
             condition=Is("reflex", "two_stage")),
        Node(package="go2_sim_tools", executable="two_stage_reflex", output="screen",
             parameters=[{k: p(k) for k in ("arm_time", "approach_speed", "approach_dist", "retreat_speed",
                                            "retreat_dist", "return_speed", "hold_time", "tolerance",
                                            "odom_topic")}],
             remappings=[("/api/sport/request", "/noah/api/sport/request")],
             condition=Is("reflex", "two_stage")),
        # sidestep (older): throw -> Move sideways, or duck when the LiDAR is not clear
        Node(package="go2_sim_tools", executable="throw_reflex", output="screen", parameters=reflex_params,
             remappings=[("/api/sport/request", "/noah/api/sport/request")],
             condition=Is("reflex", "sidestep")),
    ])
