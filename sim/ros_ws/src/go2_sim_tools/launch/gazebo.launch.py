"""Gazebo Go2 + Durance pipeline: gestures from the SPLEAT emulator move a simulated Go2.

  SPLEAT emulator -> gesture_node -> action_bridge -> /api/sport/request
                                                      -> sport_api_bridge -> CHAMP Go2 in Gazebo
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
    sil = os.path.join(get_package_share_directory("go2_sim_tools"), "launch", "sil.launch.py")

    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="3333"),
        DeclareLaunchArgument("window", default_value="50"),
        DeclareLaunchArgument("rviz", default_value="false"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(go2),
            launch_arguments={"rviz": LaunchConfiguration("rviz"), "gui": "true"}.items(),
        ),
        Node(package="go2_sim_tools", executable="sport_api_bridge", output="screen"),
        # Give Gazebo and the controller time to come up before actions can arrive.
        TimerAction(period=15.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sil),
                launch_arguments={
                    "robot": "false",
                    "host": LaunchConfiguration("host"),
                    "port": LaunchConfiguration("port"),
                    "window": LaunchConfiguration("window"),
                }.items(),
            ),
        ]),
    ])
