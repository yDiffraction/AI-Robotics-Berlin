"""Two-stage dodge demo in Gazebo: lean in on the wind-up, back off on the throw.

  throw_scene (virtual thrower; SPLEAT stand-in on port 3335)
     -> gesture_node -> /durance/gestures -> gesture_trigger -> /reflex/arm, dash, disarm
     -> two_stage_reflex -> /api/sport/request -> sport_api_bridge -> CHAMP Go2 in Gazebo (physics)
  fake_lidar -> obstacle_detection -> /durance/clearance -> two_stage_reflex

keyboard:=true leaves out the thrower; trigger by hand with `ros2 run go2_sim_tools reflex_keyboard`.

The CHAMP gait of the Go2 model is capped at 0.3 m/s; `max_speed` sets that cap for
this demo (0.7 by default). The robot also gets a second, fast ground-truth topic (/odom/ground_truth_fast,
200 Hz) because the model's own one only updates at 10 Hz - too coarse for timing.
"""

import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, description); passed on to two_stage_reflex (reflex) or throw_scene (scene)
REFLEX = {
    "arm_time": ("1.0", "how long the robot stays armed after a wind-up, in s"),
    "approach_speed": ("0.3", "forward speed during the wind-up in m/s, 0.0 = wait standing"),
    "approach_dist": ("0.15", "how far the robot leans in at most, in m"),
    "retreat_speed": ("0.7", "backward speed after a throw in m/s (the CHAMP model flips over now and then at 1.0)"),
    "retreat_dist": ("0.5", "how far the robot backs off, in m"),
    "return_speed": ("0.3", "speed for walking back to the start in m/s"),
}
SCENE = {
    "windup": ("0.6", "length of the thrower's wind-up in s"),
    "gap": ("4.0", "pause between two acts in s"),
    "throw_ratio": ("0.6", "share of acts that end in a throw"),
    "perception_delay": ("0.06", "delay of the scripted scores behind the scene in s"),
}

FAST_ODOM = "/odom/ground_truth_fast"
ROBOT = """<?xml version="1.0"?>
<robot name="go2" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find go2_description)/xacro/robot.xacro"/>
  <gazebo>
    <plugin name="p3d_timing" filename="libgazebo_ros_p3d.so">
      <ros><remapping>odom:=odom/ground_truth_fast</remapping></ros>
      <body_name>base_link</body_name>
      <frame_name>world</frame_name>
      <update_rate>200.0</update_rate>
      <gaussian_noise>0.0</gaussian_noise>
    </plugin>
  </gazebo>
</robot>
"""


def simulation(context):
    """The CHAMP Go2 in Gazebo, as go2_config's gazebo.launch.py starts it, with its own gait limit and robot file."""
    config = get_package_share_directory("go2_config")
    work = tempfile.mkdtemp(prefix="two_stage_demo_")
    with open(os.path.join(config, "config/gait/gait.yaml")) as f:
        gait = yaml.safe_load(f)
    gait["/**"]["ros__parameters"]["gait"]["max_linear_velocity_x"] = float(LaunchConfiguration("max_speed").perform(context))
    gait_path, robot_path = os.path.join(work, "gait.yaml"), os.path.join(work, "robot.xacro")
    with open(gait_path, "w") as f:
        yaml.safe_dump(gait, f)
    with open(robot_path, "w") as f:
        f.write(ROBOT)

    def include(package, file, arguments):
        path = os.path.join(get_package_share_directory(package), "launch", file)
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=arguments.items())

    common = {"description_path": robot_path, "use_sim_time": "true", "robot_name": "go2"}
    headless = "False" if LaunchConfiguration("gui").perform(context).lower() == "true" else "True"
    # a group, so that the arguments of the included files (rviz, gui, ...) do not overwrite ours
    return [GroupAction([
        include("champ_bringup", "bringup.launch.py", {
            **common, "gait_config_path": gait_path,
            "joints_map_path": os.path.join(config, "config/joints/joints.yaml"),
            "links_map_path": os.path.join(config, "config/links/links.yaml"),
            "gazebo": "true", "rviz": "false", "hardware_connected": "false", "publish_foot_contacts": "false",
            "joint_controller_topic": "joint_group_effort_controller/joint_trajectory", "close_loop_odom": "true"}),
        include("champ_gazebo", "gazebo.launch.py", {
            **common, "world": os.path.join(config, "worlds/default.world"), "gui": "false", "headless": headless,
            "world_init_z": "0.275", "world_init_heading": "0.0", "close_loop_odom": "true"}),
    ])]


def generate_launch_description():
    rviz = os.path.join(get_package_share_directory("go2_sim_tools"), "rviz", "throw_demo.rviz")

    def values(names):
        return {name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in names}

    thrower = UnlessCondition(LaunchConfiguration("keyboard"))
    behaviour = [
        Node(package="go2_sim_tools", executable="throw_scene", output="screen", condition=thrower,
             parameters=[{"port": 3335, **values(SCENE)}]),
        Node(package="gesture_node", executable="gesture_node", output="screen", condition=thrower,
             parameters=[{"host": "127.0.0.1", "port": 3335, "window": LaunchConfiguration("window")}]),
        Node(package="go2_sim_tools", executable="gesture_trigger", output="screen", condition=thrower),
        Node(package="obstacle_detection", executable="obstacle_detection", output="screen"),
        Node(package="go2_sim_tools", executable="fake_lidar", output="screen"),
        Node(package="go2_sim_tools", executable="two_stage_reflex", output="screen",
             parameters=[{"odom_topic": FAST_ODOM, **values(REFLEX)}]),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("window", default_value="3",
                              description="gesture_node smoothing window in frames (10 ms each)"),
        DeclareLaunchArgument("rviz", default_value="true", description="show the scene in RViz"),
        DeclareLaunchArgument("gui", default_value="false", description="show the Gazebo window"),
        DeclareLaunchArgument("keyboard", default_value="false",
                              description="no virtual thrower, trigger with reflex_keyboard instead"),
        *[DeclareLaunchArgument(name, default_value=default, description=text)
          for name, (default, text) in {**REFLEX, **SCENE}.items()],
        DeclareLaunchArgument("max_speed", default_value="0.7",
                              description="forward/backward speed limit of the CHAMP gait in m/s (the model ships with 0.3)"),
        OpaqueFunction(function=simulation),
        # the reflex refreshes Move itself, a short hold stops the robot if it ever goes quiet
        Node(package="go2_sim_tools", executable="sport_api_bridge", output="log",
             parameters=[{"move_hold": 0.5}]),
        Node(package="go2_sim_tools", executable="world_tf", output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz], output="log",
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
        TimerAction(period=12.0, actions=behaviour),
    ])
