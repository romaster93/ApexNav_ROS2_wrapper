"""
ROS2 Launch file for ApexNav exploration system with IsaacSim FFW-SG2 robot.
Based on exploration_traj.launch.py — real-world mode with IsaacSim-specific parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetParameter
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directories
    exploration_manager_share = get_package_share_directory('exploration_manager')
    trajectory_manager_share = get_package_share_directory('trajectory_manager')

    # FFW-SG2 specific planning parameters
    planning_param_file = os.path.join(
        trajectory_manager_share, 'config', 'planning_param_ffw.yaml'
    )

    # Control parameter YAML
    control_param_file = os.path.join(
        trajectory_manager_share, 'config', 'control_param.yaml'
    )

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument(
        'map_size_x', default_value='30.0',
        description='Map size in X direction'
    )
    map_size_y_arg = DeclareLaunchArgument(
        'map_size_y', default_value='30.0',
        description='Map size in Y direction'
    )
    is_real_world_arg = DeclareLaunchArgument(
        'is_real_world', default_value='true',
        description='Real-world mode flag (IsaacSim counts as real-world)'
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic', default_value='/habitat/odom',
        description='Odometry topic from isaacsim_apexnav_bridge'
    )
    sensor_pose_topic_arg = DeclareLaunchArgument(
        'sensor_pose_topic', default_value='/habitat/sensor_pose',
        description='Camera pose topic from isaacsim_apexnav_bridge (Habitat forward transform)'
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/habitat/camera_depth',
        description='Normalized depth image topic (32FC1, [0,1] range)'
    )
    # Zed Mini camera intrinsics for 640x480
    cx_arg = DeclareLaunchArgument('cx', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx', default_value='245.33',
        description='Zed Mini fx (from /zed_mini/camera_info K matrix)')
    fy_arg = DeclareLaunchArgument('fy', default_value='245.33',
        description='Zed Mini fy (from /zed_mini/camera_info K matrix)')

    # Include algorithm_traj launch file with IsaacSim parameters
    algorithm_traj_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            exploration_manager_share, '/launch/algorithm_traj.launch.py'
        ]),
        launch_arguments={
            'is_real_world_': LaunchConfiguration('is_real_world'),
            'map_size_x_': LaunchConfiguration('map_size_x'),
            'map_size_y_': LaunchConfiguration('map_size_y'),
            'odometry_topic_': LaunchConfiguration('odom_topic'),
            'sensor_pose_topic_': LaunchConfiguration('sensor_pose_topic'),
            'depth_topic_': LaunchConfiguration('depth_topic'),
            'cx_': LaunchConfiguration('cx'),
            'cy_': LaunchConfiguration('cy'),
            'fx_': LaunchConfiguration('fx'),
            'fy_': LaunchConfiguration('fy'),
        }.items()
    )

    # Include RViz visualization
    rviz_traj_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            exploration_manager_share, '/launch/rviz_traj.launch.py'
        ])
    )

    # Trajectory server for FFW-SG2 swerve controller
    # Publishes /cmd_vel consumed by swerve_controller.py
    traj_server_node = Node(
        package='trajectory_manager',
        executable='traj_server',
        name='traj_server_node',
        output='screen',
        parameters=[
            control_param_file,
            planning_param_file,
            {
                'need_init': False,
                'max_correction_vel': 1.0,
                'max_correction_omega': 1.57,
            }
        ],
        remappings=[
            ('odometry', LaunchConfiguration('odom_topic')),
            ('trajectory', '/planning/trajectory'),
            ('cmd_vel', '/cmd_vel'),
        ]
    )

    return LaunchDescription([
        # Enable sim time for IsaacSim ROS2 bridge clock
        SetParameter(name='use_sim_time', value=True),
        # Launch arguments
        map_size_x_arg,
        map_size_y_arg,
        is_real_world_arg,
        odom_topic_arg,
        sensor_pose_topic_arg,
        depth_topic_arg,
        cx_arg,
        cy_arg,
        fx_arg,
        fy_arg,
        # Include algorithm (exploration_node + tsp_solver)
        algorithm_traj_launch,
        # Include RViz
        rviz_traj_launch,
        # Trajectory server
        traj_server_node,
    ])
