"""web_dashboard_node 실행용 launch 파일."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: FastAPI/uvicorn 이 설치된 가상환경 인터프리터.
#: 시스템 python3 로 실행하면 fastapi 를 찾지 못하므로 prefix 로 강제한다.
DEFAULT_VENV_PYTHON = '/home/soda/TIC-VLA/venvs/serbot-bridge/bin/python3'


def generate_launch_description() -> LaunchDescription:
    """web_dashboard_node 를 venv 인터프리터와 YAML 파라미터로 실행한다.

    Returns:
        web_dashboard_node 실행 액션을 담은 LaunchDescription.
    """
    default_params = os.path.join(
        get_package_share_directory('web_dashboard'),
        'config',
        'web_dashboard.yaml',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='web_dashboard_node 파라미터 YAML 경로',
    )

    python_executable_arg = DeclareLaunchArgument(
        'python_executable',
        default_value=DEFAULT_VENV_PYTHON,
        description='노드를 실행할 python 인터프리터 (fastapi/uvicorn 이 설치된 venv)',
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='ROS 2 로그 레벨 (debug|info|warn|error|fatal)',
    )

    dashboard_node = Node(
        package='web_dashboard',
        executable='web_dashboard_node',
        # YAML 최상위 키가 web_dashboard_node 이므로 노드 이름을 바꾸면 안 된다.
        name='web_dashboard_node',
        output='screen',
        emulate_tty=True,
        # prefix 로 지정한 인터프리터가 콘솔 스크립트의 shebang 을 덮어쓴다.
        prefix=LaunchConfiguration('python_executable'),
        parameters=[LaunchConfiguration('params_file')],
        arguments=[
            '--ros-args', '--log-level',
            LaunchConfiguration('log_level'),
        ],
    )

    return LaunchDescription([
        params_file_arg,
        python_executable_arg,
        log_level_arg,
        dashboard_node,
    ])
