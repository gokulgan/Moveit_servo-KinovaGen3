#!/usr/bin/env python3
"""
Servo launch file for Kinova Gen3 7DOF + Robotiq 2F-85.

Uses official ros2_kortex packages:
  - kortex_description        → URDF/xacro
  - kinova_gen3_7dof_robotiq_2f_85_moveit_config → SRDF, kinematics, joint_limits
  - kortex_bringup            → servo config YAML
"""

import os
import yaml
import subprocess

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}


def generate_launch_description() -> LaunchDescription:

    # ── Declare arguments ────────────────────────────────────────────────
    declared_arguments = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("planning_group", default_value="manipulator"),
    ]

    # ── Package share directories ────────────────────────────────────────
    kortex_description_dir = get_package_share_directory("kortex_description")
    moveit_config_dir = get_package_share_directory(
        "kinova_gen3_7dof_robotiq_2f_85_moveit_config"
    )
    kortex_bringup_dir = get_package_share_directory("kortex_bringup")

    # ── 1. URDF via xacro ────────────────────────────────────────────────
    xacro_path = os.path.join(
        kortex_description_dir, "robots", "kinova.urdf.xacro"
    )

    xacro_command = [
        "xacro",
        xacro_path,
        "name:=gen3",
        "prefix:=",
        "arm:=gen3",
        "dof:=7",
        "vision:=false",
        "gripper:=robotiq_2f_85",
        "use_fake_hardware:=true",
        "fake_sensor_commands:=false",
        "sim_gazebo:=true",
        "sim_isaac:=false",
    ]

    result = subprocess.run(xacro_command, capture_output=True, text=True, check=True)
    urdf_content = result.stdout

    # ── 2. SRDF ──────────────────────────────────────────────────────────
    srdf_path = os.path.join(moveit_config_dir, "config", "gen3.srdf")
    with open(srdf_path, "r") as f:
        srdf_content = f.read()

    # ── 3. Kinematics & joint limits ─────────────────────────────────────
    kinematics = load_yaml(
        os.path.join(moveit_config_dir, "config", "kinematics.yaml")
    )
    joint_limits = load_yaml(
        os.path.join(moveit_config_dir, "config", "joint_limits.yaml")
    )

    # ── 4. Servo parameters ──────────────────────────────────────────────
    servo_yaml = load_yaml(
        os.path.join(kortex_bringup_dir, "config", "Nservo.yaml")
    )
    # Wrap under the expected 'moveit_servo' namespace if not already
    if "moveit_servo" not in servo_yaml:
        servo_params = {"moveit_servo": servo_yaml}
    else:
        servo_params = servo_yaml

    # Override move_group_name with the launch argument
    servo_params["moveit_servo"]["move_group_name"] = LaunchConfiguration(
        "planning_group"
    )

    # ── 5. Assemble node parameters ──────────────────────────────────────
    node_parameters = [
        {"robot_description": urdf_content},
        {"robot_description_semantic": srdf_content},
        {"robot_description_kinematics": kinematics},
        {"robot_description_planning": joint_limits},
        servo_params,
        {"use_sim_time": LaunchConfiguration("use_sim_time")},
    ]

    # ── 6. Servo node ────────────────────────────────────────────────────
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        parameters=node_parameters,
        output="screen",
    )

    return LaunchDescription(declared_arguments + [servo_node])