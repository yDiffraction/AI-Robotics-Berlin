"""Throw-vs-feint dodge demo in Gazebo, shown in RViz.

  throw_scene (virtual thrower; SPLEAT stand-in on port 3335)
     -> gesture_node -> /durance/gestures -> throw_reflex -> /api/sport/request
     -> sport_api_bridge -> CHAMP Go2 in Gazebo (physics)
  fake_lidar -> obstacle_detection -> /durance/clearance -> throw_reflex
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    go2 = os.path.join(get_package_share_directory("go2_config"), "launch", "gazebo.launch.py")
    rviz = os.path.join(get_package_share_directory("go2_sim_tools"), "rviz", "throw_demo.rviz")
    window = LaunchConfiguration("window")

    behaviour = [
        Node(package="go2_sim_tools", executable="throw_scene", output="screen",
             parameters=[{"port": 3335}]),
        Node(package="gesture_node", executable="gesture_node", output="screen",
             parameters=[{"host": "127.0.0.1", "port": 3335, "window": window}]),
        Node(package="obstacle_detection", executable="obstacle_detection", output="screen"),
        Node(package="go2_sim_tools", executable="fake_lidar", output="screen"),
        Node(package="go2_sim_tools", executable="throw_reflex", output="screen"),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("window", default_value="3",
                              description="gesture_node smoothing window in frames (10 ms each)"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(go2),
                                 launch_arguments={"gui": "false", "rviz": "false"}.items()),
        Node(package="go2_sim_tools", executable="sport_api_bridge", output="screen",
             parameters=[{"move_hold": 3.0}]),
        Node(package="go2_sim_tools", executable="world_tf", output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz], output="log",
             parameters=[{"use_sim_time": True}]),
        TimerAction(period=12.0, actions=behaviour),
    ])
