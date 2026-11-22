"""Software-in-the-loop: the real Durance pipeline against simulated hardware.

  SPLEAT emulator (camera + SNN)  ->  gesture_node -> action_bridge -> robot_sim (Go2)
  fake_lidar (Go2 LiDAR)          ->  obstacle_detection ---^

The SPLEAT emulator runs separately (sim/tools/spleat_emulator.py) so gestures can be
typed interactively. Set robot:=false when another node (e.g. the Gazebo bridge)
answers /api/sport/request instead of robot_sim.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pipeline = os.path.join(get_package_share_directory("bringup"), "launch", "pipeline.launch.py")
    robot_sim = os.path.join(get_package_prefix("robot_sim"), "lib", "robot_sim", "robot_sim")

    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="127.0.0.1",
                              description="SPLEAT server address (IPv4 only)"),
        DeclareLaunchArgument("port", default_value="3333"),
        DeclareLaunchArgument("window", default_value="50"),
        DeclareLaunchArgument("robot", default_value="true",
                              description="start robot_sim as the Go2 stand-in"),
        DeclareLaunchArgument("lidar", default_value="true",
                              description="start fake_lidar"),
        DeclareLaunchArgument("block_unknown", default_value="false"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline),
            launch_arguments={
                "despot": "false",
                "host": LaunchConfiguration("host"),
                "port": LaunchConfiguration("port"),
                "window": LaunchConfiguration("window"),
                "block_unknown": LaunchConfiguration("block_unknown"),
            }.items(),
        ),
        ExecuteProcess(cmd=[robot_sim], output="screen", name="robot_sim",
                       condition=IfCondition(LaunchConfiguration("robot"))),
        Node(package="go2_sim_tools", executable="fake_lidar", output="screen",
             condition=IfCondition(LaunchConfiguration("lidar"))),
    ])
