"""serbot_camera 패키지 설치 스크립트."""

import os
from glob import glob

from setuptools import setup

package_name = 'serbot_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        # launch/ 와 config/ 를 share 경로에 설치해야 launch 파일에서
        # get_package_share_directory() 로 YAML 을 찾을 수 있다.
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soda',
    maintainer_email='soda@serbot.local',
    description='SerBot II IMX219 CSI 카메라 단일 진입점 ROS 2 노드',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = serbot_camera.camera_node:main',
        ],
    },
)
