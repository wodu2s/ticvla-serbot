"""serbot_camera camera_node 실행용 launch 파일."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """camera_node 를 config/camera.yaml 파라미터와 함께 실행한다.

    Returns:
        camera_node 실행 액션을 담은 LaunchDescription.
    """
    # 설치된 share 경로의 YAML 을 기본 파라미터 파일로 사용한다.
    default_params = os.path.join(
        get_package_share_directory('serbot_camera'),
        'config',
        'camera.yaml',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='camera_node 파라미터 YAML 경로',
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='ROS 2 로그 레벨 (debug|info|warn|error|fatal)',
    )

    camera_node = Node(
        package='serbot_camera',
        executable='camera_node',
        # YAML 최상위 키가 camera_node 이므로 노드 이름을 바꾸면 안 된다.
        name='camera_node',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
        arguments=[
            '--ros-args', '--log-level',
            LaunchConfiguration('log_level'),
        ],
    )

    return LaunchDescription([
        params_file_arg,
        log_level_arg,
        camera_node,
    ])
