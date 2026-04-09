# Factory/warehouse environment profile (large open spaces, sparse obstacles)
"""
ROS2 Launch file for ApexNav exploration system (trajectory mode with MPC control).
Factory/warehouse environment profile (large open spaces, 1000x1000m).
Based on exploration_traj.launch.py with factory-optimized parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directories
    exploration_manager_share = get_package_share_directory('exploration_manager')
    trajectory_manager_share = get_package_share_directory('trajectory_manager')

    # Control parameter YAML
    control_param_file = os.path.join(
        trajectory_manager_share, 'config', 'control_param.yaml'
    )

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument(
        'map_size_x', default_value='200.0',
        description='Map size in X direction'
    )
    map_size_y_arg = DeclareLaunchArgument(
        'map_size_y', default_value='200.0',
        description='Map size in Y direction'
    )
    is_real_world_arg = DeclareLaunchArgument(
        'is_real_world', default_value='true',
        description='Real-world mode flag'
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic', default_value='/habitat/odom',
        description='Topic of odometry (VIO or LIO)'
    )
    sensor_pose_topic_arg = DeclareLaunchArgument(
        'sensor_pose_topic', default_value='/habitat/camera_pose',
        description='Camera pose from TF-based bridge (actual camera position)'
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/habitat/camera_depth',
        description='Normalized depth image topic (32FC1, [0,1] range)'
    )
    cx_arg = DeclareLaunchArgument('cx', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx', default_value='245.33',
        description='Zed Mini fx (from /zed_mini/camera_info K matrix)')
    fy_arg = DeclareLaunchArgument('fy', default_value='245.33',
        description='Zed Mini fy (from /zed_mini/camera_info K matrix)')

    # Include algorithm_traj_factory launch file
    algorithm_traj_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            exploration_manager_share, '/launch/algorithm_traj_factory.launch.py'
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

    # === 2026-04-07 traj_server disabled ===
    # Replaced by scripts/swerve_path_follower.py (holonomic pure-pursuit).
    # Re-enable by uncommenting the block below and stopping the follower.
    # traj_server_node = Node(
    #     package='trajectory_manager',
    #     executable='traj_server',
    #     name='traj_server_node',
    #     output='screen',
    #     parameters=[
    #         control_param_file,
    #         {
    #             'need_init': False,
    #             'max_correction_vel': 0.3,
    #             'max_correction_omega': 1.57,
    #         }
    #     ],
    #     remappings=[
    #         ('odometry', LaunchConfiguration('odom_topic')),
    #         ('trajectory', '/planning/trajectory'),
    #         ('cmd_vel', '/cmd_vel'),
    #     ]
    # )

    return LaunchDescription([
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
        # Include algorithm
        algorithm_traj_launch,
        # traj_server disabled — see swerve_path_follower.py
        # traj_server_node,
    ])
