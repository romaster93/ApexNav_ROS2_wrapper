"""
ROS2 Launch file for ApexNav algorithm nodes (trajectory mode).
Converted from ROS1 algorithm_traj.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import math
import yaml


def generate_launch_description():
    # Read shared max_depth from bridge config (single source of truth)
    _bridge_yaml = os.path.expanduser('~/ms_AIworker/config/apexnav_bridge.yaml')
    try:
        with open(_bridge_yaml) as _f:
            _cfg = yaml.safe_load(_f)
        MAX_DEPTH = float(_cfg['isaacsim_apexnav_bridge']['ros__parameters']['max_depth'])
    except (FileNotFoundError, KeyError, TypeError):
        MAX_DEPTH = 5.0  # fallback

    # Get package directories
    exploration_manager_share = get_package_share_directory('exploration_manager')
    trajectory_manager_share = get_package_share_directory('trajectory_manager')
    lkh_mtsp_solver_share = get_package_share_directory('lkh_mtsp_solver')

    # Declare launch arguments
    map_size_x_arg = DeclareLaunchArgument('map_size_x_', default_value='80.0')
    map_size_y_arg = DeclareLaunchArgument('map_size_y_', default_value='80.0')
    odometry_topic_arg = DeclareLaunchArgument('odometry_topic_', default_value='/habitat/odom')
    sensor_pose_topic_arg = DeclareLaunchArgument('sensor_pose_topic_', default_value='/habitat/camera_pose')
    depth_topic_arg = DeclareLaunchArgument('depth_topic_', default_value='/habitat/camera_depth')
    cx_arg = DeclareLaunchArgument('cx_', default_value='320.0')
    cy_arg = DeclareLaunchArgument('cy_', default_value='240.0')
    fx_arg = DeclareLaunchArgument('fx_', default_value='245.33')
    fy_arg = DeclareLaunchArgument('fy_', default_value='245.33')
    is_real_world_arg = DeclareLaunchArgument('is_real_world_', default_value='true')

    # Load trajectory planning parameters
    planning_param_file = os.path.join(
        trajectory_manager_share, 'config', 'planning_param_ffw.yaml'
    )

    # Calculate perception angles
    left_angle = 30 * math.pi / 180.0
    right_angle = 30 * math.pi / 180.0

    # Main exploration node
    exploration_node = Node(
        package='exploration_manager',
        executable='exploration_node',
        name='exploration_node',
        output='screen',
        parameters=[
            planning_param_file,
            {
                # FSM mode selection
                'is_real_world': LaunchConfiguration('is_real_world_'),

                # Mapping - SDF Map
                'sdf_map.ray_mode': 0,
                'sdf_map.resolution': 0.05,
                'sdf_map.map_size_x': LaunchConfiguration('map_size_x_'),
                'sdf_map.map_size_y': LaunchConfiguration('map_size_y_'),
                'sdf_map.obstacles_inflation': 0.45,  # 2026-04-07 복원: footprint inflation (실측 0.56x0.51 기준). 이전 0.1은 회귀였음.
                'sdf_map.local_bound': 10.0,  # frontier 검색 범위 (depth 범위와 독립, raycasting은 max_ray_length로 제한됨)
                'sdf_map.p_hit': 0.99,
                'sdf_map.p_miss': 0.48,
                'sdf_map.p_min': 0.10,
                'sdf_map.p_max': 0.98,
                'sdf_map.p_occ': 0.80,
                'sdf_map.max_ray_length': MAX_DEPTH - 0.01,
                'sdf_map.optimistic': True,
                'sdf_map.signed_dist': False,

                # Map ROS
                'map_ros/cx': LaunchConfiguration('cx_'),
                'map_ros/cy': LaunchConfiguration('cy_'),
                'map_ros/fx': LaunchConfiguration('fx_'),
                'map_ros/fy': LaunchConfiguration('fy_'),
                'map_ros/depth_filter_maxdist': MAX_DEPTH - 0.01,
                'map_ros/depth_filter_mindist': 0.0,
                'map_ros/depth_filter_margin': 2,
                'map_ros/filter_min_height': 0.5,  # 2026-04-07 복원: 가짜 바닥 occupancy 방지 (World2 개방 환경). InteriorAgent 실내는 후속 PR에서 재평가.
                'map_ros/filter_max_height': 1.5,
                'map_ros/free_ray_extrapolation': 0.95,  # max-range 픽셀 FREE 연장 (maxdist × 0.95)
                'map_ros/k_depth_scaling_factor': 65535.0,
                'map_ros/skip_pixel': 1,
                'map_ros/frame_id': 'World',
                'map_ros/virtual_ground_height': -0.34,

                # FSM timing
                'fsm/replan_time': 1.0,             # 2.0→1.0: 더 빠른 replan 주기
                'fsm/replan_traj_end_threshold': 1.5,  # 1.0→1.5: trajectory 끝 1.5초 전에 replan 시작
                'fsm/replan_frontier_change_delay': 999.0,
                'fsm/replan_timeout': 1.0,          # 10.0→1.0: REPLAN state에서 빠르게 PLAN_TRAJ 재시도 (멈춤 단축)

                # Exploration manager
                'exploration/policy': 2,
                'exploration/sigma_threshold': 0.015,
                'exploration/max_to_mean_threshold': 1.10,
                'exploration/max_to_mean_percentage': 0.90,
                'exploration/tsp_dir': os.path.join(lkh_mtsp_solver_share, 'resource'),

                # Frontier
                'frontier.cluster_min': 5,
                'frontier.cluster_size_xy': 0.65,
                'frontier.min_view_finish_fraction': 0.2,
                'frontier.min_contain_unknown': 30,

                # Object
                'object.min_observation_num': 2,
                'object.fusion_type': 1,
                'object.use_observation': True,
                'object.vis_cloud': True,

                # Perception utils
                'perception_utils.left_angle': left_angle,
                'perception_utils.right_angle': right_angle,
                'perception_utils.max_dist': 4.0,  # frontier 감지 거리 (맵 기반, depth 범위와 독립)
                'perception_utils.vis_dist': 1.0,

                # A* path searching
                'astar.lambda_heu': 1.0,
                'astar.resolution_astar': 0.1,
            }
        ],
        remappings=[
            ('/odom_world', LaunchConfiguration('odometry_topic_')),
            ('/map_ros/pose', LaunchConfiguration('sensor_pose_topic_')),
            ('/map_ros/depth', LaunchConfiguration('depth_topic_')),
        ]
    )

    # TSP solver node
    tsp_solver_node = Node(
        package='lkh_mtsp_solver',
        executable='tsp_node',
        name='tsp_solver',
        output='log',
        parameters=[{
            'exploration/tsp_dir': os.path.join(lkh_mtsp_solver_share, 'resource'),
        }]
    )

    return LaunchDescription([
        # Arguments
        map_size_x_arg,
        map_size_y_arg,
        odometry_topic_arg,
        sensor_pose_topic_arg,
        depth_topic_arg,
        cx_arg,
        cy_arg,
        fx_arg,
        fy_arg,
        is_real_world_arg,
        # Nodes
        exploration_node,
        tsp_solver_node,
    ])
