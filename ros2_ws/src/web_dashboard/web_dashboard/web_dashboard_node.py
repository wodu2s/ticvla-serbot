#!/usr/bin/env python3
"""SerBot II FastAPI 웹 대시보드 노드.

설계 원칙:

* 카메라를 직접 열지 않는다. serbot_camera 노드가 발행하는 토픽만 구독한다.
* 모터를 직접 제어하지 않는다. pop 라이브러리를 import 하지도 않는다.
* Start/Stop/E-Stop 은 토픽 발행까지만 하고, 실제 구동/차단은 별도 safety 노드가 맡는다.

실행 구조는 rclpy 스핀을 데몬 스레드에서 돌리고 uvicorn 을 메인 스레드에서
돌리는 형태다. 두 이벤트 루프가 공유하는 상태는 모두 락으로 보호한다.
"""

from __future__ import annotations

import asyncio
import json
import math
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

import cv2
import numpy as np
import rclpy
import uvicorn
from collect_interfaces.srv import SetConfig, SubmitResult
from cv_bridge import CvBridge
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from nav_msgs.msg import Odometry
from pydantic import BaseModel, Field
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

#: TIC-VLA 프로젝트 루트. 추론 스크립트/venv 기본 경로를 여기서 유도한다.
TICVLA_ROOT: Path = Path.home() / 'TIC-VLA'

#: 추론 종료(SIGINT) 후 정상 종료를 기다릴 시간(초). ticvla_bridge.py 는
#: predict() 도중이면 --join-timeout(기본 3s) 만큼 더 걸릴 수 있어 여유를 둔다.
INFERENCE_STOP_WAIT_SEC: float = 8.0
#: 그래도 안 죽으면 terminate() 후 이만큼 더 기다리고 kill() 한다.
INFERENCE_KILL_WAIT_SEC: float = 3.0

#: FPS 이동평균을 계산할 시간 창(초).
FPS_WINDOW_SEC: float = 2.0

#: MJPEG multipart 경계 문자열.
MJPEG_BOUNDARY: str = 'frame'

#: 노드 목록 갱신 주기(초). 매 요청마다 rclpy 를 호출하지 않기 위한 캐시 주기.
NODE_COUNT_PERIOD_SEC: float = 2.0

#: 추론 subprocess 생사 확인 주기(초).
INFERENCE_POLL_PERIOD_SEC: float = 2.0

#: /ticvla/status 발행자가 아직 없을 때 표시할 문자열.
STATUS_NOT_AVAILABLE: str = 'N/A'

#: joystick_colormarker.py 와 공유하는 토픽/서비스 이름.
COLLECT_STATUS_TOPIC: str = '/collect/status'
COLLECT_SET_CONFIG_SRV: str = '/collect/set_config'
COLLECT_SUBMIT_RESULT_SRV: str = '/collect/submit_result'

#: /collect/status 가 이보다 오래 안 왔으면 "수집기 미실행"으로 본다.
COLLECT_STATUS_STALE_SEC: float = 3.0
#: 서비스가 아직 떠 있지 않을 때 기다리는 시간(초). 대시보드 요청을 오래
#: 붙잡지 않도록 짧게 잡는다 — 이 시간 안에 없으면 "수집기 미실행"으로 답한다.
COLLECT_SERVICE_DISCOVER_SEC: float = 1.0
#: 서비스 호출 자체(수집기가 응답하기까지)의 타임아웃(초).
COLLECT_SERVICE_CALL_TIMEOUT_SEC: float = 5.0


# ======================================================================
# 요청 본문 모델
# ======================================================================
class InstructionRequest(BaseModel):
    """POST /api/instruction 의 요청 본문."""

    text: str = Field(default='', description='TIC-VLA 에 전달할 자연어 지시문')


class CollectSetConfigRequest(BaseModel):
    """POST /api/collect/set_config 의 요청 본문."""

    layout: int = Field(description='마커 배치 1|2|3')
    target_color: str = Field(description='목표색 red|green|blue')


class CollectSubmitResultRequest(BaseModel):
    """POST /api/collect/submit_result 의 요청 본문."""

    success: bool = Field(description='마커 앞에서 성공적으로 정지했는가')
    final_distance: float = Field(default=0.0, description='최종 거리(m)')
    notes: str = Field(default='', description='비고(선택)')


# ======================================================================
# ROS 2 노드
# ======================================================================
class WebDashboardNode(Node):
    """대시보드용 토픽 구독/발행과 공유 상태 관리를 담당하는 노드.

    카메라 프레임은 구독 콜백에서 곧바로 리사이즈 + JPEG 인코딩까지 끝내고
    바이트만 슬롯에 보관한다. HTTP 요청마다 재인코딩하지 않기 위해서다.
    """

    def __init__(self) -> None:
        """파라미터를 검증하고 퍼블리셔·서브스크라이버·타이머를 구성한다."""
        super().__init__('web_dashboard_node')

        # ── 파라미터 선언 (기본값은 config/web_dashboard.yaml 과 동일) ──
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8000)
        self.declare_parameter('image_topic', '/camera/image_raw_fast')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        # 기본값은 데이터 수집이 저장하는 카메라 네이티브 해상도와 맞춘다
        # (config/web_dashboard.yaml 참고). 화면에 보이는 프레임이 실제 저장되는
        # 프레임과 같아야 격자 오버레이로 위치를 맞추는 게 의미가 있다.
        self.declare_parameter('stream_width', 1280)
        self.declare_parameter('stream_height', 720)
        self.declare_parameter('stream_fps', 10)
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('camera_timeout_sec', 1.0)
        self.declare_parameter('odom_timeout_sec', 1.0)

        # ── 추론(ticvla_bridge.py) 시작/정지 — 대시보드가 subprocess 로 띄운다 ──
        # camera_topic 은 이 대시보드가 구독하는 image_topic 과 같아야 한다.
        # (동시에 카메라를 열면 nvarguscamerasrc 가 실패하므로, 추론도 반드시
        # camera_node 토픽을 구독하게 --camera-topic 으로 실행한다.)
        self.declare_parameter(
            'inference_python', str(TICVLA_ROOT / 'venvs/ticvla/bin/python3'))
        self.declare_parameter(
            'inference_script', str(TICVLA_ROOT / 'scripts/ticvla_bridge.py'))
        self.declare_parameter('inference_gpu_lock', 'shared')
        self.declare_parameter('inference_vlm_period', 30.0)
        self.declare_parameter('inference_log_dir', str(TICVLA_ROOT / 'logs'))
        # 추가로 넘길 인자 (공백 구분, 예: "--speed-scale 0.5 --max-speed 0.15").
        self.declare_parameter('inference_extra_args', '')

        self.host: str = str(self.get_parameter('host').value)
        self.port: int = int(self.get_parameter('port').value)
        self._image_topic: str = str(self.get_parameter('image_topic').value)
        self._odom_topic: str = str(self.get_parameter('odom_topic').value)
        self._stream_width: int = int(self.get_parameter('stream_width').value)
        self._stream_height: int = int(self.get_parameter('stream_height').value)
        self.stream_fps: int = int(self.get_parameter('stream_fps').value)
        self._jpeg_quality: int = int(self.get_parameter('jpeg_quality').value)
        self._camera_timeout_sec: float = float(
            self.get_parameter('camera_timeout_sec').value)
        self._odom_timeout_sec: float = float(
            self.get_parameter('odom_timeout_sec').value)

        self._inference_python: str = str(self.get_parameter('inference_python').value)
        self._inference_script: str = str(self.get_parameter('inference_script').value)
        self._inference_gpu_lock: str = str(self.get_parameter('inference_gpu_lock').value)
        self._inference_vlm_period: float = float(
            self.get_parameter('inference_vlm_period').value)
        self._inference_log_dir: str = str(self.get_parameter('inference_log_dir').value)
        self._inference_extra_args: str = str(
            self.get_parameter('inference_extra_args').value)

        self._validate_parameters()

        # ── 공유 상태 (모두 _lock 으로 보호) ──
        self._lock: threading.Lock = threading.Lock()
        self._bridge: CvBridge = CvBridge()

        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_time: Optional[float] = None
        self._frame_times: Deque[float] = deque()
        self._source_width: int = 0
        self._source_height: int = 0

        self._odom_received: bool = False
        self._last_odom_time: Optional[float] = None
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_yaw_deg: float = 0.0
        self._odom_vx: float = 0.0
        self._odom_vy: float = 0.0
        self._odom_wz: float = 0.0

        self._ticvla_status: str = STATUS_NOT_AVAILABLE
        self._instruction: str = ''
        self._enabled: bool = False
        self._estop: bool = False

        # /collect/status 스냅샷 (joystick_colormarker.py 가 2Hz 로 발행하는
        # JSON). 파싱 실패나 미수신은 collect_snapshot() 에서 처리한다.
        self._collect_status: Optional[dict[str, Any]] = None
        self._collect_status_at: Optional[float] = None

        self._node_count: int = 0
        self._start_time: float = time.monotonic()

        # ── 추론 프로세스 (ticvla_bridge.py) 생명주기 ──
        # collect_lock 과 같은 이유로 별도 락을 쓴다: subprocess 시작/정지는
        # HTTP 요청 스레드에서 오고, 상태 조회는 폴링 타이머(rclpy 스레드)와
        # /api/status(HTTP 스레드)에서 동시에 온다.
        self._inference_lock: threading.Lock = threading.Lock()
        self._inference_process: Optional[subprocess.Popen] = None
        self._inference_last_notice: str = ''

        # 조이스틱/데이터 수집과 같은 컴퓨터를 나눠 쓸 때 이 노드가 CPU 를
        # 불필요하게 잡아먹지 않도록 두 가지 안전장치를 둔다:
        #   1) stream_fps 를 넘는 프레임은 리사이즈/인코딩 자체를 하지 않는다
        #      (카메라가 그보다 빨리 보내도 여분은 그냥 버린다).
        #   2) /api/stream/pause 로 인코딩을 통째로 멈출 수 있다 — 페이지는
        #      열어 둔 채 (odom/상태는 계속 보임) 영상만 끈다.
        self._min_frame_interval: float = 1.0 / max(1, self.stream_fps)
        self._stream_paused: bool = False

        # 프레임이 없을 때/일시정지 중일 때 내보낼 플레이스홀더는 한 번만 만들어 재사용한다.
        self._placeholder_jpeg: bytes = self._build_placeholder_jpeg('NO SIGNAL')
        self._paused_jpeg: bytes = self._build_placeholder_jpeg('STREAM PAUSED')

        # ── QoS 프로파일 ──
        # 센서/고빈도 데이터: 최신 값만 중요하므로 BEST_EFFORT + depth 1.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 명령류: 유실되면 안 되므로 RELIABLE.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 비상정지: 나중에 뜨는 safety 노드도 마지막 값을 받아야 하므로 latched.
        estop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── 구독 ──
        self.create_subscription(
            Image, self._image_topic, self._on_image, sensor_qos)
        # BEST_EFFORT 구독자는 RELIABLE 발행자와도 매칭되므로 어느 쪽이든 받는다.
        self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(
            String, '/ticvla/status', self._on_ticvla_status, command_qos)
        self.create_subscription(
            String, COLLECT_STATUS_TOPIC, self._on_collect_status, command_qos)

        # ── 데이터 수집 서비스 클라이언트 (joystick_colormarker.py 가 서버) ──
        # 이 노드는 지금까지 명령을 토픽 발행(fire-and-forget)으로만 다뤘다.
        # 배치/목표색 설정과 결과 제출은 수락/거부 사유가 바로 필요해 서비스로
        # 부른다 — rclpy 가 이미 별도 스레드에서 spin 중이라 동기 Client.call()
        # 은 못 쓰고, call_async()+콜백+Event 로 블로킹 대기한다
        # (_call_collect_service 참고).
        self._collect_set_config_cli = self.create_client(
            SetConfig, COLLECT_SET_CONFIG_SRV)
        self._collect_submit_result_cli = self.create_client(
            SubmitResult, COLLECT_SUBMIT_RESULT_SRV)

        # ── 발행 ──
        self._enable_pub = self.create_publisher(
            Bool, '/ticvla/enable', command_qos)
        self._estop_pub = self.create_publisher(
            Bool, '/ticvla/emergency_stop', estop_qos)
        self._instruction_pub = self.create_publisher(
            String, '/ticvla/instruction', command_qos)
        self._heartbeat_pub = self.create_publisher(
            Bool, '/ticvla/heartbeat', command_qos)
        # 브라우저가 아니라 이 노드의 생존을 알린다. 노드가 죽으면 끊긴다.
        self._heartbeat_timer = self.create_timer(0.5, self._publish_heartbeat)

        # 노드 개수는 주기적으로 캐시한다 (HTTP 스레드에서 rclpy 호출을 피하려고).
        self._node_count_timer = self.create_timer(
            NODE_COUNT_PERIOD_SEC, self._update_node_count)
        self._update_node_count()

        # 추론 subprocess 가 (조이스틱 collect 처럼) 죽었는지 주기적으로 확인한다.
        self._inference_poll_timer = self.create_timer(
            INFERENCE_POLL_PERIOD_SEC, self._poll_inference)

        self.get_logger().info(
            f'web_dashboard_node 시작: http://{self.host}:{self.port} '
            f'(image={self._image_topic}, odom={self._odom_topic}, '
            f'stream={self._stream_width}x{self._stream_height}@'
            f'{self.stream_fps}fps q={self._jpeg_quality})'
        )

    # ------------------------------------------------------------------
    # 파라미터 검증
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        """파라미터 값이 유효 범위인지 확인한다.

        Raises:
            ValueError: 포트/해상도/FPS/품질/타임아웃 중 하나라도 범위를 벗어난 경우.
        """
        if not self.host:
            raise ValueError('host 는 비어 있을 수 없다')
        if not 1 <= self.port <= 65535:
            raise ValueError(f'port 는 1~65535 범위여야 한다: {self.port}')
        if self._stream_width <= 0 or self._stream_height <= 0:
            raise ValueError(
                f'스트림 해상도는 0 보다 커야 한다: '
                f'{self._stream_width}x{self._stream_height}'
            )
        if not 1 <= self.stream_fps <= 60:
            raise ValueError(f'stream_fps 는 1~60 범위여야 한다: {self.stream_fps}')
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError(
                f'jpeg_quality 는 1~100 범위여야 한다: {self._jpeg_quality}')
        if self._camera_timeout_sec <= 0.0 or self._odom_timeout_sec <= 0.0:
            raise ValueError('타임아웃 값은 0 보다 커야 한다')
        if not self._image_topic or not self._odom_topic:
            raise ValueError('image_topic 과 odom_topic 은 비어 있을 수 없다')

    # ------------------------------------------------------------------
    # 구독 콜백
    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        """카메라 프레임을 받아 리사이즈 + JPEG 인코딩 후 슬롯에 보관한다.

        camera_node 는 최대 framerate(기본 15Hz)로 보내지만, 이 콜백은
        stream_fps 를 넘는 프레임은 처리하지 않고 버린다 — 조이스틱/데이터
        수집과 CPU 를 나눠 쓰는 상황에서 필요 이상으로 리사이즈/JPEG 인코딩을
        하지 않기 위해서다. 일시정지 중(--stream_paused)이면 그마저도 건너뛴다.

        Args:
            msg: 구독한 sensor_msgs/Image 메시지.
        """
        now = time.monotonic()
        with self._lock:
            if self._stream_paused:
                return
            last = self._last_frame_time
        if last is not None and now - last < self._min_frame_interval:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 소스가 이미 목표 해상도와 같으면(기본값은 카메라 네이티브와 동일)
            # 리사이즈를 건너뛴다 — 불필요한 보간 연산과 화질 손실을 피한다.
            if (frame.shape[1], frame.shape[0]) == (self._stream_width, self._stream_height):
                resized = frame
            else:
                resized = cv2.resize(
                    frame,
                    (self._stream_width, self._stream_height),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(
                '.jpg',
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if not ok:
                self.get_logger().warn('JPEG 인코딩 실패, 프레임을 건너뛴다')
                return
        except Exception as exc:
            # 프레임 하나의 실패로 노드를 죽이지 않되, 조용히 넘기지도 않는다.
            self.get_logger().error(f'프레임 처리 실패: {exc}')
            return

        with self._lock:
            self._latest_jpeg = buffer.tobytes()
            self._last_frame_time = now
            self._source_width = int(msg.width)
            self._source_height = int(msg.height)
            # 최근 FPS_WINDOW_SEC 구간의 수신 시각만 남겨 이동평균을 만든다.
            self._frame_times.append(now)
            while self._frame_times and now - self._frame_times[0] > FPS_WINDOW_SEC:
                self._frame_times.popleft()

    def _on_odom(self, msg: Odometry) -> None:
        """오도메트리에서 위치/자세/속도를 뽑아 저장한다.

        Args:
            msg: 구독한 nav_msgs/Odometry 메시지.
        """
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            twist = msg.twist.twist
            yaw_deg = math.degrees(self._quaternion_to_yaw(
                orientation.x, orientation.y, orientation.z, orientation.w))
        except Exception as exc:
            self.get_logger().error(f'오도메트리 처리 실패: {exc}')
            return

        with self._lock:
            self._odom_received = True
            self._last_odom_time = time.monotonic()
            self._odom_x = float(position.x)
            self._odom_y = float(position.y)
            self._odom_yaw_deg = float(yaw_deg)
            self._odom_vx = float(twist.linear.x)
            self._odom_vy = float(twist.linear.y)
            self._odom_wz = float(twist.angular.z)

    def _on_ticvla_status(self, msg: String) -> None:
        """TIC-VLA 상태 문자열을 저장한다.

        Args:
            msg: 구독한 std_msgs/String 메시지.
        """
        with self._lock:
            self._ticvla_status = msg.data

    def _on_collect_status(self, msg: String) -> None:
        """joystick_colormarker.py 가 보내는 /collect/status(JSON)를 저장한다.

        Args:
            msg: 구독한 std_msgs/String (JSON 페이로드).
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'/collect/status JSON 파싱 실패: {exc}')
            return
        with self._lock:
            self._collect_status = payload
            self._collect_status_at = time.monotonic()

    def _publish_heartbeat(self) -> None:
        """0.5초마다 생존 신호를 발행한다.

        브리지는 이 신호가 --heartbeat-timeout 동안 끊기면 0 을 발행한다.
        """
        self._heartbeat_pub.publish(Bool(data=True))

    def _update_node_count(self) -> None:
        """현재 그래프에 보이는 노드 개수를 캐시에 갱신한다."""
        try:
            count = len(self.get_node_names())
        except Exception as exc:
            self.get_logger().warn(f'노드 목록 조회 실패: {exc}')
            return
        with self._lock:
            self._node_count = count

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """쿼터니언에서 yaw(rad)만 추출한다.

        Args:
            x: 쿼터니언 x 성분.
            y: 쿼터니언 y 성분.
            z: 쿼터니언 z 성분.
            w: 쿼터니언 w 성분.

        Returns:
            Z 축 회전각(라디안).
        """
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # 명령 발행 (실제 제어는 하지 않고 토픽만 쏜다)
    # ------------------------------------------------------------------
    def publish_enable(self, value: bool) -> None:
        """/ticvla/enable 에 Start(True)/Stop(False)를 발행한다.

        Args:
            value: True 면 Start, False 면 Stop.
        """
        self._enable_pub.publish(Bool(data=value))
        with self._lock:
            self._enabled = value
        self.get_logger().info(f'/ticvla/enable ← {value}')

    def publish_estop(self) -> None:
        """/ticvla/emergency_stop 에 True 를 발행하고 enable 을 내린다.

        E-Stop 은 latched(TRANSIENT_LOCAL)로 발행되므로 나중에 실행되는
        safety 노드도 마지막 상태를 즉시 받는다.
        """
        self._estop_pub.publish(Bool(data=True))
        # 비상정지 시 enable 도 함께 내려 두 신호가 어긋나지 않게 한다.
        self._enable_pub.publish(Bool(data=False))
        with self._lock:
            self._estop = True
            self._enabled = False
        self.get_logger().warn('비상정지 발행: /ticvla/emergency_stop ← True')

    def publish_instruction(self, text: str) -> None:
        """/ticvla/instruction 에 지시문을 발행한다.

        Args:
            text: 사용자가 입력한 자연어 지시문.
        """
        self._instruction_pub.publish(String(data=text))
        with self._lock:
            self._instruction = text
        self.get_logger().info(f'/ticvla/instruction ← "{text}"')

    def pause_stream(self) -> None:
        """영상 인코딩을 멈춘다. 페이지는 열어 둔 채(odom/상태는 계속 갱신) 영상만 끈다.

        조이스틱으로 직접 수동 조종하며 데이터를 수집할 때, 이 노드가 매 프레임
        리사이즈+JPEG 인코딩에 쓰는 CPU 를 아예 0 으로 만들고 싶을 때 쓴다.
        """
        with self._lock:
            self._stream_paused = True
        self.get_logger().info('영상 스트리밍 일시정지 (조이스틱/수집 중 CPU 절약용)')

    def resume_stream(self) -> None:
        """`pause_stream()` 으로 멈춘 영상 인코딩을 재개한다."""
        with self._lock:
            self._stream_paused = False
        self.get_logger().info('영상 스트리밍 재개')

    # ------------------------------------------------------------------
    # 추론(ticvla_bridge.py) 생명주기
    # ------------------------------------------------------------------
    def start_inference(self, instruction: str) -> tuple[bool, str]:
        """ticvla_bridge.py 를 subprocess 로 띄운다.

        camera_topic 을 이 대시보드가 구독하는 image_topic 과 동일하게 넘긴다 —
        추론이 카메라를 직접 열면 camera_node(대시보드 영상의 출처)와
        nvarguscamerasrc 자원을 다퉈 둘 다 실패한다. --require-enable 로 띄워서
        기동 직후에는 실제 주행 명령을 내지 않고(START 버튼을 눌러야 움직임)
        VLM 로딩/카메라 연결만 먼저 끝내 둔다.

        Args:
            instruction: 최초 지시문. 비어 있으면 ticvla_bridge.py 기본값을 쓴다.

        Returns:
            (성공 여부, 사람이 읽을 메시지).
        """
        with self._inference_lock:
            if (self._inference_process is not None
                    and self._inference_process.poll() is None):
                return False, f'이미 실행 중이다 (pid={self._inference_process.pid})'

        script = Path(self._inference_script)
        if not script.is_file():
            return False, f'추론 스크립트를 찾을 수 없다: {script}'
        python = Path(self._inference_python)
        if not python.is_file():
            return False, f'추론용 python 인터프리터를 찾을 수 없다: {python}'

        # camera_node 가 실제로 이 토픽에 발행 중인지 미리 확인한다 — 안 떠 있으면
        # ticvla_bridge.py 는 첫 프레임 타임아웃(수 초) 뒤에야 실패를 알린다.
        if self.count_publishers(self._image_topic) <= 0:
            return False, (
                f'{self._image_topic} 에 발행자가 없다 — camera_node(serbot_camera)'
                '가 떠 있는지 먼저 확인할 것')

        log_dir = Path(self._inference_log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f'로그 폴더를 만들 수 없다: {exc}'
        log_path = log_dir / f'inference_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'

        cmd = [
            str(python), str(script),
            '--enable', '--require-enable',
            '--camera-topic', self._image_topic,
            '--require-camera',
            '--gpu-lock', self._inference_gpu_lock,
            '--vlm-period', str(self._inference_vlm_period),
            '--odom-topic', self._odom_topic,
            '--log-json', str(log_path),
        ]
        if instruction.strip():
            cmd += ['--instruction', instruction.strip()]
        if self._inference_extra_args.strip():
            cmd += shlex.split(self._inference_extra_args)

        try:
            proc = subprocess.Popen(cmd, cwd=str(TICVLA_ROOT))
        except Exception as exc:  # noqa: BLE001
            return False, f'실행 실패: {exc!r}'

        with self._inference_lock:
            self._inference_process = proc
            self._inference_last_notice = f'시작됨 pid={proc.pid}'
        self.get_logger().info(
            f'추론 시작 pid={proc.pid} | camera_topic={self._image_topic} | '
            f'instruction={instruction!r} | log={log_path}')
        return True, f'시작됨 (pid={proc.pid}). START 버튼을 눌러야 실제로 움직인다.'

    def stop_inference(self) -> tuple[bool, str]:
        """실행 중인 추론 subprocess 를 SIGINT 로 안전하게 종료한다.

        ticvla_bridge.py 의 SIGINT 핸들러는 정지 명령을 여러 번 발행한 뒤
        끝난다(collect_gotoobject.py 와 같은 패턴). predict() 도중이면
        --join-timeout 만큼 더 걸릴 수 있어 넉넉히 기다린 뒤에만
        terminate()/kill() 로 올린다.

        Returns:
            (성공 여부, 사람이 읽을 메시지).
        """
        with self._inference_lock:
            proc = self._inference_process
        if proc is None or proc.poll() is not None:
            with self._inference_lock:
                self._inference_process = None
            return False, '실행 중인 추론 프로세스가 없다'

        self.get_logger().info(f'추론 정지 요청: SIGINT → pid={proc.pid}')
        try:
            proc.send_signal(signal.SIGINT)
        except Exception as exc:  # noqa: BLE001
            return False, f'SIGINT 전송 실패: {exc!r}'

        def _cleanup() -> None:
            """정상 종료를 기다리다 안 되면 terminate/kill 까지 올린다.

            wait() 가 최대 수 초 블록되므로 HTTP 요청 스레드를 잡아두지 않게
            별도 스레드에서 돈다.
            """
            try:
                proc.wait(timeout=INFERENCE_STOP_WAIT_SEC)
            except subprocess.TimeoutExpired:
                self.get_logger().warn(
                    f'{INFERENCE_STOP_WAIT_SEC:.0f}초 내 종료 안 됨 → terminate()')
                try:
                    proc.terminate()
                    proc.wait(timeout=INFERENCE_KILL_WAIT_SEC)
                except subprocess.TimeoutExpired:
                    self.get_logger().error('terminate 실패 → kill()')
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            with self._inference_lock:
                if self._inference_process is proc:
                    self._inference_process = None
                    self._inference_last_notice = (
                        f'정지됨 (returncode={proc.returncode})')
            self.get_logger().info(f'추론 종료 확인 returncode={proc.returncode}')

        threading.Thread(target=_cleanup, name='inference_stop', daemon=True).start()
        return True, f'정지 요청 보냄 (pid={proc.pid})'

    def _poll_inference(self) -> None:
        """추론 subprocess 가 (수동 종료 없이) 스스로 끝났는지 확인한다."""
        with self._inference_lock:
            proc = self._inference_process
            if proc is None or proc.poll() is None:
                return
            self._inference_process = None
            returncode = proc.returncode
            self._inference_last_notice = f'종료됨 (returncode={returncode})'
        self.get_logger().warn(f'추론 프로세스가 스스로 종료됨 returncode={returncode}')

    def inference_snapshot(self) -> dict[str, Any]:
        """/api/status 에 실어 보낼 추론 프로세스 상태.

        Returns:
            running/pid/last_notice 를 담은 딕셔너리.
        """
        with self._inference_lock:
            proc = self._inference_process
            running = proc is not None and proc.poll() is None
            pid = proc.pid if running else None
            notice = self._inference_last_notice
        return {'running': running, 'pid': pid, 'last_notice': notice}

    # ------------------------------------------------------------------
    # 데이터 수집 (joystick_colormarker.py) 연동
    # ------------------------------------------------------------------
    def collect_snapshot(self) -> dict[str, Any]:
        """/api/status 에 실어 보낼 /collect/status 스냅샷.

        Returns:
            수집기가 안 떠 있거나(한 번도 수신 안 함) 상태가 오래됐으면
            `{"available": False}` 만 담은 딕셔너리, 아니면 마지막으로 받은
            필드 전체에 `available: True` 를 더한 딕셔너리.
        """
        with self._lock:
            payload = self._collect_status
            at = self._collect_status_at
        age = (time.monotonic() - at) if at is not None else float('inf')
        if payload is None or age > COLLECT_STATUS_STALE_SEC:
            return {'available': False}
        return {**payload, 'available': True}

    def _call_collect_service(self, client: Any, request: Any) -> tuple[bool, str, Any]:
        """수집기 서비스를 블로킹으로 호출한다 (없으면 즉시 실패로 답한다).

        rclpy 실행기는 `_spin_ros` 데몬 스레드에서 이미 spin 중이라, 이 HTTP
        요청 스레드에서 동기 `Client.call()` 을 쓰면 같은 노드를 두 스레드가
        spin 하려는 셈이 돼 안전하지 않다. 대신 `call_async()` 로 얻은
        future 에 완료 콜백을 걸어 `threading.Event` 를 set 하게 하고, 이
        스레드는 그 Event 를 기다린다 — 콜백은 executor 스레드가 실행하므로
        안전하다.

        Args:
            client: 호출할 서비스 클라이언트.
            request: 채워진 요청 메시지.

        Returns:
            (수락 여부, 사람이 읽을 메시지, 원본 응답 또는 None).
        """
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=COLLECT_SERVICE_DISCOVER_SEC):
                return False, '수집기 미실행 — joystick_colormarker.py 가 떠 있는지 확인할 것', None

        done = threading.Event()
        result_holder: list[Any] = []

        def _on_done(fut: Any) -> None:
            result_holder.append(fut)
            done.set()

        future = client.call_async(request)
        future.add_done_callback(_on_done)
        if not done.wait(timeout=COLLECT_SERVICE_CALL_TIMEOUT_SEC):
            return False, '수집기 응답 시간 초과', None

        fut = result_holder[0]
        exc = fut.exception()
        if exc is not None:
            return False, f'서비스 호출 실패: {exc!r}', None
        response = fut.result()
        return bool(response.accepted), str(response.reason), response

    def call_set_config(self, layout: int, target_color: str) -> dict[str, Any]:
        """/collect/set_config 를 호출한다.

        Args:
            layout: 마커 배치 1|2|3.
            target_color: 목표색 red|green|blue.

        Returns:
            {ok, message, target_side}.
        """
        request = SetConfig.Request(layout=int(layout), target_color=str(target_color))
        ok, reason, response = self._call_collect_service(
            self._collect_set_config_cli, request)
        return {
            'ok': ok,
            'message': reason or ('수락됨' if ok else '거부됨'),
            'target_side': (response.target_side if response is not None else ''),
        }

    def call_submit_result(self, success: bool, final_distance: float,
                           notes: str) -> dict[str, Any]:
        """/collect/submit_result 를 호출한다.

        Args:
            success: 성공 여부.
            final_distance: 최종 거리(m).
            notes: 비고.

        Returns:
            {ok, message}.
        """
        request = SubmitResult.Request(
            success=bool(success), final_distance=float(final_distance),
            notes=str(notes))
        ok, reason, _response = self._call_collect_service(
            self._collect_submit_result_cli, request)
        return {'ok': ok, 'message': reason or ('수락됨' if ok else '거부됨')}

    # ------------------------------------------------------------------
    # HTTP 계층에서 읽어가는 상태
    # ------------------------------------------------------------------
    def get_stream_jpeg(self) -> bytes:
        """스트리밍에 사용할 JPEG 바이트를 돌려준다.

        Returns:
            최신 프레임의 JPEG 바이트. 일시정지 중이면 STREAM PAUSED 이미지,
            프레임이 없거나 오래됐으면 NO SIGNAL 이미지.
        """
        now = time.monotonic()
        with self._lock:
            paused = self._stream_paused
            jpeg = self._latest_jpeg
            last = self._last_frame_time

        if paused:
            return self._paused_jpeg
        if jpeg is None or last is None or (now - last) > self._camera_timeout_sec:
            return self._placeholder_jpeg
        return jpeg

    def build_status(self) -> dict[str, Any]:
        """/api/status 응답용 상태 딕셔너리를 만든다.

        Returns:
            카메라/ROS/오도메트리/TIC-VLA/시스템 상태를 담은 딕셔너리.
        """
        now = time.monotonic()
        with self._lock:
            last_frame = self._last_frame_time
            frame_count = len(self._frame_times)
            src_w, src_h = self._source_width, self._source_height
            odom_received = self._odom_received
            last_odom = self._last_odom_time
            odom_x, odom_y, odom_yaw = self._odom_x, self._odom_y, self._odom_yaw_deg
            odom_vx, odom_vy, odom_wz = self._odom_vx, self._odom_vy, self._odom_wz
            ticvla_status = self._ticvla_status
            instruction = self._instruction
            enabled = self._enabled
            estop = self._estop
            node_count = self._node_count
            stream_paused = self._stream_paused

        frame_age = (now - last_frame) if last_frame is not None else float('inf')
        camera_connected = frame_age <= self._camera_timeout_sec

        odom_age = (now - last_odom) if last_odom is not None else float('inf')
        odom_available = odom_received and odom_age <= self._odom_timeout_sec

        return {
            'camera': {
                'connected': camera_connected,
                'fps': round(frame_count / FPS_WINDOW_SEC, 2),
                # 무한대는 JSON 표준이 아니므로 -1.0 으로 바꿔 보낸다.
                'last_frame_age_sec': (
                    round(frame_age, 3) if math.isfinite(frame_age) else -1.0),
                'resolution': (
                    f'{src_w}x{src_h}' if src_w and src_h else STATUS_NOT_AVAILABLE),
                'stream_paused': stream_paused,
            },
            'ros': {
                'ok': rclpy.ok(),
                'node_count': node_count,
            },
            'odom': {
                'available': odom_available,
                'x': round(odom_x, 4),
                'y': round(odom_y, 4),
                'yaw_deg': round(odom_yaw, 2),
                'vx': round(odom_vx, 4),
                'vy': round(odom_vy, 4),
                'wz': round(odom_wz, 4),
                'last_age_sec': (
                    round(odom_age, 3) if math.isfinite(odom_age) else -1.0),
            },
            'ticvla': {
                'status': ticvla_status,
                'instruction': instruction,
                'enabled': enabled,
                'estop': estop,
            },
            'inference': self.inference_snapshot(),
            'collect': self.collect_snapshot(),
            'system': {
                'uptime_sec': round(now - self._start_time, 1),
            },
        }

    def _build_placeholder_jpeg(self, text: str) -> bytes:
        """주어진 문구가 그려진 검은 JPEG 를 만든다 (NO SIGNAL / STREAM PAUSED 등).

        Args:
            text: 화면 중앙에 표시할 문구.

        Returns:
            플레이스홀더 이미지의 JPEG 바이트.

        Raises:
            RuntimeError: 플레이스홀더 인코딩에 실패한 경우.
        """
        canvas = np.zeros((self._stream_height, self._stream_width, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.8, self._stream_width / 640.0 * 1.4)
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        origin = (
            max(0, (self._stream_width - text_w) // 2),
            (self._stream_height + text_h) // 2,
        )
        cv2.putText(canvas, text, origin, font, scale,
                    (60, 60, 220), thickness, cv2.LINE_AA)

        ok, buffer = cv2.imencode(
            '.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            raise RuntimeError(f'{text!r} 플레이스홀더 인코딩에 실패했다')
        return buffer.tobytes()


# ======================================================================
# FastAPI 애플리케이션
# ======================================================================
def create_app(node: WebDashboardNode) -> FastAPI:
    """대시보드 HTTP 엔드포인트를 가진 FastAPI 앱을 만든다.

    Args:
        node: 상태를 읽고 명령을 발행할 ROS 2 노드 인스턴스.

    Returns:
        구성이 끝난 FastAPI 애플리케이션.
    """
    app = FastAPI(title='SerBot II Dashboard', docs_url=None, redoc_url=None)

    @app.get('/', response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """대시보드 HTML 페이지를 반환한다."""
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get('/video_feed')
    async def video_feed() -> StreamingResponse:
        """MJPEG(multipart/x-mixed-replace) 스트림을 반환한다."""
        return StreamingResponse(
            _mjpeg_generator(node),
            media_type=f'multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}',
            headers={'Cache-Control': 'no-store, no-cache, must-revalidate'},
        )

    @app.get('/api/status')
    async def api_status() -> JSONResponse:
        """현재 카메라/ROS/오도메트리/TIC-VLA 상태를 JSON 으로 반환한다."""
        return JSONResponse(content=node.build_status())

    @app.post('/api/start')
    async def api_start() -> JSONResponse:
        """/ticvla/enable 에 True 를 발행한다."""
        node.publish_enable(True)
        return JSONResponse(content={'ok': True})

    @app.post('/api/stop')
    async def api_stop() -> JSONResponse:
        """/ticvla/enable 에 False 를 발행한다."""
        node.publish_enable(False)
        return JSONResponse(content={'ok': True})

    @app.post('/api/estop')
    async def api_estop() -> JSONResponse:
        """/ticvla/emergency_stop 에 True 를 발행한다."""
        node.publish_estop()
        return JSONResponse(content={'ok': True})

    @app.post('/api/instruction')
    async def api_instruction(payload: InstructionRequest) -> JSONResponse:
        """입력된 지시문을 /ticvla/instruction 으로 발행한다."""
        node.publish_instruction(payload.text)
        return JSONResponse(content={'ok': True})

    @app.post('/api/stream/pause')
    async def api_stream_pause() -> JSONResponse:
        """영상 인코딩을 멈춘다 (조이스틱/데이터 수집 중 CPU 절약용)."""
        node.pause_stream()
        return JSONResponse(content={'ok': True})

    @app.post('/api/stream/resume')
    async def api_stream_resume() -> JSONResponse:
        """영상 인코딩을 재개한다."""
        node.resume_stream()
        return JSONResponse(content={'ok': True})

    @app.post('/api/inference/start')
    async def api_inference_start(payload: InstructionRequest) -> JSONResponse:
        """ticvla_bridge.py 를 subprocess 로 띄운다. body.text 를 최초 지시문으로 쓴다."""
        ok, message = node.start_inference(payload.text)
        return JSONResponse(content={'ok': ok, 'message': message},
                            status_code=200 if ok else 409)

    @app.post('/api/inference/stop')
    async def api_inference_stop() -> JSONResponse:
        """실행 중인 추론 subprocess 를 정지한다."""
        ok, message = node.stop_inference()
        return JSONResponse(content={'ok': ok, 'message': message},
                            status_code=200 if ok else 409)

    @app.get('/collect', response_class=HTMLResponse)
    async def collect_page() -> HTMLResponse:
        """데이터 수집 페이지를 반환한다."""
        return HTMLResponse(content=COLLECT_HTML)

    # 아래 두 엔드포인트만 async def 가 아니라 def 다 — 서비스 응답을
    # 기다리는 동안(최대 COLLECT_SERVICE_CALL_TIMEOUT_SEC) 블로킹하므로,
    # FastAPI 가 스레드풀에서 돌려 uvicorn 의 이벤트 루프를 막지 않게 한다.
    @app.post('/api/collect/set_config')
    def api_collect_set_config(payload: CollectSetConfigRequest) -> JSONResponse:
        """배치/목표색을 수집기에 설정 요청한다."""
        result = node.call_set_config(payload.layout, payload.target_color)
        return JSONResponse(content=result, status_code=200 if result['ok'] else 409)

    @app.post('/api/collect/submit_result')
    def api_collect_submit_result(payload: CollectSubmitResultRequest) -> JSONResponse:
        """에피소드 결과(성공/실패/거리/비고)를 수집기에 제출한다."""
        result = node.call_submit_result(
            payload.success, payload.final_distance, payload.notes)
        return JSONResponse(content=result, status_code=200 if result['ok'] else 409)

    return app


async def _mjpeg_generator(node: WebDashboardNode):
    """stream_fps 주기로 최신 JPEG 을 multipart 청크로 흘려보낸다.

    Args:
        node: JPEG 슬롯을 보유한 ROS 2 노드.

    Yields:
        multipart/x-mixed-replace 규격의 바이트 청크.
    """
    period = 1.0 / float(node.stream_fps)
    boundary = f'--{MJPEG_BOUNDARY}\r\n'.encode('ascii')
    try:
        while True:
            jpeg = node.get_stream_jpeg()
            yield (
                boundary
                + b'Content-Type: image/jpeg\r\n'
                + f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii')
                + jpeg
                + b'\r\n'
            )
            # 브라우저가 끊으면 여기서 CancelledError 가 올라온다.
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        # 클라이언트 연결 종료는 정상 흐름이므로 조용히 빠져나간다.
        raise
    except Exception as exc:
        node.get_logger().error(f'MJPEG 스트림 오류: {exc}')
        raise


# ======================================================================
# 대시보드 HTML (외부 CDN/폰트/라이브러리 없이 완전 오프라인 동작)
# ======================================================================
DASHBOARD_HTML: str = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SerBot II Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    background: #12141a; color: #e8eaf0;
    font-family: "DejaVu Sans", "Noto Sans CJK KR", sans-serif;
  }
  body.estop { border: 8px solid #e02020; }
  h1 { font-size: 20px; margin: 0 0 12px 0; letter-spacing: 0.5px; }
  .nav-link {
    font-size: 13px; font-weight: 400; margin-left: 14px;
    color: #60a5fa; text-decoration: none;
  }
  .nav-link:hover { text-decoration: underline; }
  .layout { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  .video-box { flex: 1 1 640px; min-width: 320px; }
  .video-frame { position: relative; }
  .video-box img {
    width: 100%; background: #000; border: 1px solid #2a2f3a; border-radius: 6px;
    display: block;
  }
  /* 데이터 수집 정렬용 격자. 8x8 등분 + 중앙 십자선. img 와 정확히 같은 박스에
     겹치도록 video-frame(position:relative) 안에서 절대배치한다. */
  .grid-overlay {
    position: absolute; inset: 0; border-radius: 6px; pointer-events: none;
    display: none;
    background-image:
      repeating-linear-gradient(to right, rgba(255,255,255,0.30),
        rgba(255,255,255,0.30) 1px, transparent 1px, transparent 12.5%),
      repeating-linear-gradient(to bottom, rgba(255,255,255,0.30),
        rgba(255,255,255,0.30) 1px, transparent 1px, transparent 12.5%),
      linear-gradient(to right, rgba(255,70,70,0.75), rgba(255,70,70,0.75)),
      linear-gradient(to bottom, rgba(255,70,70,0.75), rgba(255,70,70,0.75));
    background-repeat: repeat, repeat, no-repeat, no-repeat;
    background-position: 0 0, 0 0, 50% 0, 0 50%;
    background-size: 100% 100%, 100% 100%, 2px 100%, 100% 2px;
  }
  .grid-overlay.on { display: block; }
  .video-toolbar {
    display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
  }
  .video-toolbar button {
    background: #2a2f3a; font-size: 13px; padding: 7px 12px;
  }
  .video-toolbar button.active { background: #2563eb; }
  .video-toolbar button.paused { background: #b45309; }
  .panel {
    flex: 0 1 380px; min-width: 300px;
    background: #1a1d26; border: 1px solid #2a2f3a; border-radius: 6px; padding: 14px;
  }
  .panel h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
    color: #8b93a7; margin: 0 0 8px 0;
  }
  .section { margin-bottom: 16px; }
  .row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; font-size: 14px;
  }
  .row .label { color: #9aa3b8; }
  .row .value { font-family: "DejaVu Sans Mono", monospace; }
  .hint { font-size: 12px; color: #6b7280; line-height: 1.5; margin-top: 8px; }
  .dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; background: #555;
  }
  .dot.on { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
  .dot.off { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
  .controls { margin-top: 16px; }
  .btn-row { display: flex; gap: 8px; margin-bottom: 10px; }
  button {
    font-family: inherit; font-size: 14px; font-weight: 600;
    border: none; border-radius: 5px; padding: 10px 16px; cursor: pointer;
    color: #fff;
  }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-start { background: #16a34a; flex: 1; }
  .btn-stop { background: #4b5563; flex: 1; }
  .btn-estop {
    background: #dc2626; width: 100%; font-size: 20px; padding: 18px;
    letter-spacing: 2px;
  }
  .btn-send { background: #2563eb; }
  .instruction { display: flex; gap: 8px; margin-top: 12px; }
  .instruction input {
    flex: 1; padding: 10px; font-size: 14px; font-family: inherit;
    background: #0e1015; color: #e8eaf0;
    border: 1px solid #2a2f3a; border-radius: 5px;
  }
  .estop-banner {
    display: none; background: #dc2626; color: #fff; font-weight: 700;
    padding: 10px; border-radius: 5px; margin-bottom: 12px; text-align: center;
    letter-spacing: 1px;
  }
  body.estop .estop-banner { display: block; }
</style>
</head>
<body>
<h1>SerBot II &mdash; TIC-VLA Dashboard
  <a class="nav-link" href="/collect">데이터 수집 →</a>
</h1>
<div class="estop-banner">EMERGENCY STOP ENGAGED</div>

<div class="layout">
  <div class="video-box">
    <div class="video-toolbar">
      <button id="btn-grid" onclick="toggleGrid()">격자 표시</button>
      <button id="btn-stream" onclick="toggleStream()">스트리밍 일시정지</button>
    </div>
    <div class="video-frame">
      <img id="video" src="/video_feed" alt="camera stream">
      <div class="grid-overlay" id="grid-overlay"></div>
    </div>
  </div>

  <div class="panel">
    <div class="section">
      <h2>Camera</h2>
      <div class="row">
        <span class="label">연결</span>
        <span class="value">
          <span id="cam-dot" class="dot"></span><span id="cam-state">-</span>
        </span>
      </div>
      <div class="row">
        <span class="label">FPS</span><span class="value" id="cam-fps">-</span>
      </div>
      <div class="row">
        <span class="label">마지막 프레임</span><span class="value" id="cam-age">-</span>
      </div>
      <div class="row">
        <span class="label">해상도</span><span class="value" id="cam-res">-</span>
      </div>
    </div>

    <div class="section">
      <h2>ROS 2</h2>
      <div class="row">
        <span class="label">상태</span>
        <span class="value">
          <span id="ros-dot" class="dot"></span><span id="ros-state">-</span>
        </span>
      </div>
      <div class="row">
        <span class="label">노드 수</span><span class="value" id="ros-nodes">-</span>
      </div>
      <div class="row">
        <span class="label">업타임</span><span class="value" id="uptime">-</span>
      </div>
    </div>

    <div class="section">
      <h2>Odometry</h2>
      <div class="row">
        <span class="label">수신</span>
        <span class="value">
          <span id="odom-dot" class="dot"></span><span id="odom-state">-</span>
        </span>
      </div>
      <div class="row">
        <span class="label">x / y</span><span class="value" id="odom-xy">-</span>
      </div>
      <div class="row">
        <span class="label">yaw</span><span class="value" id="odom-yaw">-</span>
      </div>
      <div class="row">
        <span class="label">vx / vy</span><span class="value" id="odom-v">-</span>
      </div>
      <div class="row">
        <span class="label">wz</span><span class="value" id="odom-wz">-</span>
      </div>
    </div>

    <div class="section">
      <h2>TIC-VLA</h2>
      <div class="row">
        <span class="label">상태</span><span class="value" id="tv-status">-</span>
      </div>
      <div class="row">
        <span class="label">enabled</span><span class="value" id="tv-enabled">-</span>
      </div>
      <div class="row">
        <span class="label">instruction</span><span class="value" id="tv-instr">-</span>
      </div>
    </div>

    <div class="section">
      <h2>추론 (ticvla_bridge.py)</h2>
      <div class="row">
        <span class="label">상태</span>
        <span class="value">
          <span id="inf-dot" class="dot"></span><span id="inf-state">-</span>
        </span>
      </div>
      <div class="row">
        <span class="label">알림</span><span class="value" id="inf-notice">-</span>
      </div>
      <div class="btn-row">
        <button class="btn-start" onclick="startInference()">추론 시작</button>
        <button class="btn-stop" onclick="stopInference()">추론 정지</button>
      </div>
      <div class="hint">
        아래 instruction 입력창의 값으로 시작한다. 시작 직후에는 아직 실제로
        움직이지 않는다 — 이어서 START 버튼을 눌러야 주행이 허용된다.
      </div>
    </div>

    <div class="controls">
      <div class="btn-row">
        <button class="btn-start" id="btn-start" onclick="post('/api/start')">START</button>
        <button class="btn-stop" onclick="post('/api/stop')">STOP</button>
      </div>
      <button class="btn-estop" onclick="post('/api/estop')">EMERGENCY STOP</button>
      <div class="instruction">
        <input id="instr-input" type="text" placeholder="instruction 입력 후 Enter">
        <button class="btn-send" onclick="sendInstruction()">전송</button>
      </div>
    </div>
  </div>
</div>

<script>
function setDot(id, ok) {
  document.getElementById(id).className = 'dot ' + (ok ? 'on' : 'off');
}

function fmtAge(sec) {
  return (sec < 0) ? '수신 없음' : sec.toFixed(2) + ' s';
}

// ── 격자 오버레이 (순수 클라이언트 측 — 서버 프레임에는 손대지 않는다) ──
function applyGridState(on) {
  document.getElementById('grid-overlay').classList.toggle('on', on);
  document.getElementById('btn-grid').classList.toggle('active', on);
}

function toggleGrid() {
  const on = !document.getElementById('grid-overlay').classList.contains('on');
  applyGridState(on);
  // 이 브라우저에서만 기억한다 (서버/다른 탭에는 영향 없음).
  try { localStorage.setItem('ticvla_grid_on', on ? '1' : '0'); } catch (e) {}
}

try {
  applyGridState(localStorage.getItem('ticvla_grid_on') === '1');
} catch (e) {}

// ── 스트리밍 일시정지 (조이스틱/데이터 수집 중 CPU 절약용) ──
let streamPaused = false;
function applyStreamButton(paused) {
  streamPaused = paused;
  const btn = document.getElementById('btn-stream');
  btn.textContent = paused ? '스트리밍 재개' : '스트리밍 일시정지';
  btn.classList.toggle('paused', paused);
}

async function toggleStream() {
  await post(streamPaused ? '/api/stream/resume' : '/api/stream/pause');
}

async function post(url, body) {
  try {
    await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})
    });
  } catch (e) {
    console.error('요청 실패', url, e);
  }
  refresh();
}

function sendInstruction() {
  const input = document.getElementById('instr-input');
  const text = input.value.trim();
  if (!text) { return; }
  post('/api/instruction', {text: text});
  input.value = '';
}

// ── 추론(ticvla_bridge.py) 시작/정지 — 실패 사유를 알림창으로 바로 보여준다 ──
async function postJson(url, body) {
  let data = {};
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})
    });
    data = await res.json().catch(() => ({}));
    if (!data.ok) {
      alert(data.message || ('요청 실패: HTTP ' + res.status));
    }
  } catch (e) {
    alert('요청 실패: ' + e);
  }
  refresh();
  return data;
}

function startInference() {
  const text = document.getElementById('instr-input').value.trim();
  postJson('/api/inference/start', {text: text});
}

function stopInference() {
  postJson('/api/inference/stop', {});
}

document.getElementById('instr-input').addEventListener('keydown', function (e) {
  if (e.key === 'Enter') { sendInstruction(); }
});

async function refresh() {
  let s;
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    s = await res.json();
  } catch (e) {
    // 서버가 내려간 경우: 모든 표시등을 꺼서 즉시 알 수 있게 한다.
    setDot('cam-dot', false);
    setDot('ros-dot', false);
    document.getElementById('ros-state').textContent = 'DISCONNECTED';
    return;
  }

  setDot('cam-dot', s.camera.connected);
  document.getElementById('cam-state').textContent =
      s.camera.connected ? 'CONNECTED' : 'NO SIGNAL';
  document.getElementById('cam-fps').textContent = s.camera.fps.toFixed(1);
  document.getElementById('cam-age').textContent = fmtAge(s.camera.last_frame_age_sec);
  document.getElementById('cam-res').textContent = s.camera.resolution;
  applyStreamButton(!!s.camera.stream_paused);

  setDot('ros-dot', s.ros.ok);
  document.getElementById('ros-state').textContent = s.ros.ok ? 'OK' : 'DOWN';
  document.getElementById('ros-nodes').textContent = s.ros.node_count;
  document.getElementById('uptime').textContent = s.system.uptime_sec.toFixed(0) + ' s';

  setDot('odom-dot', s.odom.available);
  document.getElementById('odom-state').textContent =
      s.odom.available ? 'ACTIVE' : 'NO DATA';
  document.getElementById('odom-xy').textContent =
      s.odom.x.toFixed(3) + ' / ' + s.odom.y.toFixed(3);
  document.getElementById('odom-yaw').textContent = s.odom.yaw_deg.toFixed(1) + '\\u00b0';
  document.getElementById('odom-v').textContent =
      s.odom.vx.toFixed(3) + ' / ' + s.odom.vy.toFixed(3);
  document.getElementById('odom-wz').textContent = s.odom.wz.toFixed(3);

  document.getElementById('tv-status').textContent = s.ticvla.status;
  document.getElementById('tv-enabled').textContent = s.ticvla.enabled ? 'true' : 'false';
  document.getElementById('tv-instr').textContent = s.ticvla.instruction || '-';

  setDot('inf-dot', s.inference.running);
  document.getElementById('inf-state').textContent = s.inference.running
      ? ('실행중 (pid=' + s.inference.pid + ')') : '정지됨';
  document.getElementById('inf-notice').textContent = s.inference.last_notice || '-';

  // E-Stop 상태면 화면 전체 적색 테두리 + START 비활성화
  document.body.classList.toggle('estop', s.ticvla.estop);
  document.getElementById('btn-start').disabled = s.ticvla.estop;
}

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


# ======================================================================
# 데이터 수집 페이지 (joystick_colormarker.py 연동)
# ======================================================================
COLLECT_HTML: str = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SerBot II Collect</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    background: #12141a; color: #e8eaf0;
    font-family: "DejaVu Sans", "Noto Sans CJK KR", sans-serif;
  }
  h1 { font-size: 20px; margin: 0 0 12px 0; letter-spacing: 0.5px; }
  .nav-link {
    font-size: 13px; font-weight: 400; margin-left: 14px;
    color: #60a5fa; text-decoration: none;
  }
  .nav-link:hover { text-decoration: underline; }
  .down-banner {
    display: none; background: #dc2626; color: #fff; font-weight: 700;
    padding: 10px; border-radius: 5px; margin-bottom: 12px; text-align: center;
    letter-spacing: 1px;
  }
  body.down .down-banner { display: block; }
  .layout { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  .video-box { flex: 2 1 640px; min-width: 320px; }
  .video-box img {
    width: 100%; background: #000; border: 1px solid #2a2f3a; border-radius: 6px;
    display: block;
  }
  .hint { font-size: 12px; color: #6b7280; line-height: 1.5; margin-top: 8px; }
  .panel {
    flex: 1 1 380px; min-width: 320px;
    background: #1a1d26; border: 1px solid #2a2f3a; border-radius: 6px; padding: 14px;
  }
  .panel h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
    color: #8b93a7; margin: 0 0 8px 0;
  }
  .section { margin-bottom: 16px; }
  .row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; font-size: 14px;
  }
  .row .label { color: #9aa3b8; }
  .row .value { font-family: "DejaVu Sans Mono", monospace; }
  .dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; background: #555;
  }
  .dot.on { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
  .dot.off { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
  .dot.big { width: 16px; height: 16px; }
  .btn-row { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  button {
    font-family: inherit; font-size: 14px; font-weight: 600;
    border: none; border-radius: 5px; padding: 10px 14px; cursor: pointer;
    color: #fff; background: #2a2f3a; flex: 1; min-width: 90px;
  }
  button:disabled { opacity: 0.35; cursor: not-allowed; }
  button.active { background: #2563eb; }
  button.color-red.active { background: #b91c1c; }
  button.color-green.active { background: #15803d; }
  button.color-blue.active { background: #1d4ed8; }
  .btn-success { background: #16a34a; }
  .btn-fail { background: #b91c1c; }
  .color-hint { font-size: 11px; color: #8b93a7; display: block; margin-top: 2px; }
  table.counts { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.counts th, table.counts td {
    border: 1px solid #2a2f3a; padding: 5px 8px; text-align: center;
  }
  table.counts th { color: #8b93a7; font-weight: 600; }
  table.counts td.current { background: #1e3a8a; font-weight: 700; }
  .total-row { margin-top: 6px; font-size: 13px; color: #9aa3b8; }
  .result-form { display: none; }
  .result-form.enabled { display: block; }
  .result-form input[type=text], .result-form input[type=number] {
    width: 100%; padding: 8px; margin-top: 4px; margin-bottom: 10px;
    background: #0e1015; color: #e8eaf0; border: 1px solid #2a2f3a;
    border-radius: 5px; font-family: inherit; font-size: 14px;
  }
  .result-form label { font-size: 13px; color: #9aa3b8; }
  .state-pill {
    display: inline-block; padding: 3px 10px; border-radius: 10px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
  }
  .state-pill.idle { background: #374151; }
  .state-pill.recording { background: #b45309; }
  .state-pill.awaiting_result { background: #7c3aed; }
</style>
</head>
<body>
<h1>SerBot II &mdash; 데이터 수집
  <a class="nav-link" href="/">← 실시간 대시보드</a>
</h1>
<div class="down-banner">수집기 미실행 — joystick_colormarker.py 가 떠 있는지 확인할 것
  (colrec 로 실행했는지, /collect/status 가 오는지)</div>

<div class="layout">
  <!-- 1) 카메라 라이브 뷰 — 3색이 다 보이는지 시작 전에 반드시 확인 -->
  <div class="video-box">
    <img src="/video_feed" alt="camera stream">
    <div class="hint">
      시작 전에 화면에 3색 마커가 전부 보이는지 확인할 것 — 한 색만 보이는
      장면에서 찍으면 색을 무시해도 정답이 나와 그 에피소드가 무의미해진다.
    </div>
  </div>

  <div class="panel">
    <!-- 5) 상태 표시 -->
    <div class="section">
      <h2>상태</h2>
      <div class="row">
        <span class="label">state</span>
        <span class="value"><span id="state-pill" class="state-pill idle">-</span></span>
      </div>
      <div class="row">
        <span class="label">odom</span>
        <span class="value">
          <span id="odom-dot" class="dot big"></span><span id="odom-text">-</span>
        </span>
      </div>
      <div class="row">
        <span class="label">frame_hz</span><span class="value" id="frame-hz">-</span>
      </div>
      <div class="row">
        <span class="label">직전 폴더</span><span class="value" id="last-episode">-</span>
      </div>
    </div>

    <!-- 2) 배치 선택 -->
    <div class="section">
      <h2>배치 선택</h2>
      <div class="btn-row">
        <button id="layout-btn-1" onclick="selectLayout(1)">배치 1</button>
        <button id="layout-btn-2" onclick="selectLayout(2)">배치 2</button>
        <button id="layout-btn-3" onclick="selectLayout(3)">배치 3</button>
      </div>
      <div class="hint" id="layout-current">현재 배치: -</div>
    </div>

    <!-- 3) 목표색 선택 -->
    <div class="section">
      <h2>목표색 선택</h2>
      <div class="btn-row">
        <button id="color-btn-red" class="color-red" onclick="selectColor('red')">
          red<span class="color-hint" id="slot-red">-</span>
        </button>
        <button id="color-btn-green" class="color-green" onclick="selectColor('green')">
          green<span class="color-hint" id="slot-green">-</span>
        </button>
        <button id="color-btn-blue" class="color-blue" onclick="selectColor('blue')">
          blue<span class="color-hint" id="slot-blue">-</span>
        </button>
      </div>
    </div>

    <!-- 4) 누적 카운터 -->
    <div class="section">
      <h2>누적 카운터</h2>
      <table class="counts" id="counts-table">
        <thead>
          <tr><th>배치</th><th>red</th><th>green</th><th>blue</th></tr>
        </thead>
        <tbody id="counts-body"></tbody>
      </table>
      <div class="total-row" id="counts-total">합계 -/72</div>
    </div>

    <!-- 6) 결과 입력 폼 (awaiting_result 일 때만 활성) -->
    <div class="section">
      <h2>결과 입력</h2>
      <div id="result-form" class="result-form">
        <label>최종 거리(m)</label>
        <input type="number" step="0.01" id="final-distance" value="0.80">
        <label>비고(선택)</label>
        <input type="text" id="notes" placeholder="">
        <div class="btn-row">
          <button class="btn-success" onclick="submitResult(true)">성공</button>
          <button class="btn-fail" onclick="submitResult(false)">실패</button>
        </div>
      </div>
      <div class="hint" id="result-hint">
        결과 대기 상태(awaiting_result)가 아니면 비활성화된다. 제출해야 다음
        에피소드로 넘어간다.
      </div>
    </div>
  </div>
</div>

<script>
const LAYOUT_SIDE_COLOR = {
  1: {left: 'blue', center: 'red', right: 'green'},
  2: {left: 'red', center: 'green', right: 'blue'},
  3: {left: 'green', center: 'blue', right: 'red'},
};
const COLOR_KO = {red: '빨강', green: '초록', blue: '파랑'};
const SIDE_KO = {left: '좌', center: '중', right: '우'};

let uiLayout = null;
let uiColor = null;
let lastState = 'idle';

function sideOf(layout, color) {
  const sides = LAYOUT_SIDE_COLOR[layout];
  for (const side in sides) { if (sides[side] === color) { return side; } }
  return null;
}

function layoutDesc(n) {
  const s = LAYOUT_SIDE_COLOR[n];
  return `좌=${COLOR_KO[s.left]} 중=${COLOR_KO[s.center]} 우=${COLOR_KO[s.right]}`;
}

async function collectPostJson(url, body) {
  let data = {ok: false, message: '요청 실패'};
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {}),
    });
    data = await res.json().catch(() => ({ok: false, message: 'HTTP ' + res.status}));
    if (!data.ok) { alert(data.message || ('요청 실패: HTTP ' + res.status)); }
  } catch (e) {
    alert('요청 실패: ' + e);
  }
  refreshCollect();
  return data;
}

async function selectLayout(n) {
  if (lastState !== 'idle') {
    alert('녹화/결과대기 중에는 배치를 바꿀 수 없다 (state=' + lastState + ')');
    return;
  }
  if (!confirm('마커를 배치 ' + n + ' 대로 놓았습니까?\\n' + layoutDesc(n))) { return; }
  const color = uiColor || 'red';
  const res = await collectPostJson('/api/collect/set_config', {layout: n, target_color: color});
  if (res.ok) { uiLayout = n; uiColor = color; }
}

async function selectColor(c) {
  if (lastState !== 'idle') {
    alert('녹화/결과대기 중에는 목표색을 바꿀 수 없다 (state=' + lastState + ')');
    return;
  }
  if (uiLayout == null) { alert('배치를 먼저 선택하세요'); return; }
  const res = await collectPostJson('/api/collect/set_config', {layout: uiLayout, target_color: c});
  if (res.ok) { uiColor = c; }
}

async function submitResult(success) {
  const distEl = document.getElementById('final-distance');
  const notesEl = document.getElementById('notes');
  const dist = parseFloat(distEl.value);
  await collectPostJson('/api/collect/submit_result', {
    success: success,
    final_distance: isNaN(dist) ? 0.0 : dist,
    notes: notesEl.value,
  });
  notesEl.value = '';
}

function renderCounts(counts, layout) {
  const body = document.getElementById('counts-body');
  body.innerHTML = '';
  let total = 0;
  for (const l of [1, 2, 3]) {
    const row = (counts && counts[String(l)]) || {red: 0, green: 0, blue: 0};
    const tr = document.createElement('tr');
    const cur = (layout === l);
    tr.innerHTML =
      '<td>' + l + (cur ? ' ★' : '') + '</td>' +
      ['red', 'green', 'blue'].map(c =>
        '<td' + (cur ? ' class="current"' : '') + '>' + (row[c] || 0) + '/8</td>'
      ).join('');
    body.appendChild(tr);
    total += (row.red || 0) + (row.green || 0) + (row.blue || 0);
  }
  document.getElementById('counts-total').textContent = '합계 ' + total + '/72';
}

async function refreshCollect() {
  let s;
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    s = (await res.json()).collect;
  } catch (e) {
    document.body.classList.add('down');
    return;
  }
  if (!s || !s.available) {
    document.body.classList.add('down');
    return;
  }
  document.body.classList.remove('down');

  lastState = s.state || 'idle';
  const pill = document.getElementById('state-pill');
  pill.textContent = lastState;
  pill.className = 'state-pill ' + lastState;

  setDotBig('odom-dot', !!s.odom_ok);
  document.getElementById('odom-text').textContent = s.odom_ok ? 'OK' : '미수신';
  document.getElementById('frame-hz').textContent =
    (typeof s.frame_hz === 'number' ? s.frame_hz.toFixed(1) : '-');
  document.getElementById('last-episode').textContent = s.last_episode_dir || '-';

  uiLayout = s.layout;
  uiColor = s.target_color;
  document.getElementById('layout-current').textContent =
    'current 배치: ' + (s.layout != null ? s.layout + ' (' + layoutDesc(s.layout) + ')' : '미설정');

  for (const n of [1, 2, 3]) {
    const btn = document.getElementById('layout-btn-' + n);
    btn.classList.toggle('active', s.layout === n);
    btn.disabled = (lastState !== 'idle');
  }
  for (const c of ['red', 'green', 'blue']) {
    const btn = document.getElementById('color-btn-' + c);
    btn.classList.toggle('active', s.target_color === c);
    btn.disabled = (lastState !== 'idle') || (s.layout == null);
    const slot = document.getElementById('slot-' + c);
    const side = (s.layout != null) ? sideOf(s.layout, c) : null;
    slot.textContent = side ? SIDE_KO[side] : '-';
  }

  renderCounts(s.counts, s.layout);

  const formEnabled = (lastState === 'awaiting_result');
  document.getElementById('result-form').classList.toggle('enabled', formEnabled);
  document.getElementById('result-hint').style.display = formEnabled ? 'none' : 'block';
}

function setDotBig(id, ok) {
  document.getElementById(id).className = 'dot big ' + (ok ? 'on' : 'off');
}

refreshCollect();
setInterval(refreshCollect, 1000);
</script>
</body>
</html>
"""


# ======================================================================
# 엔트리포인트
# ======================================================================
def _spin_ros(executor: SingleThreadedExecutor, node: WebDashboardNode) -> None:
    """rclpy 스핀 전용 스레드 본체.

    Args:
        executor: 노드가 등록된 실행기.
        node: 로깅에 사용할 노드.
    """
    try:
        executor.spin()
    except (ExternalShutdownException, KeyboardInterrupt):
        # 메인 스레드에서 종료를 지시한 정상 경로.
        pass
    except Exception as exc:
        node.get_logger().error(f'rclpy 스핀 스레드 종료: {exc}')


def main(args: Optional[list[str]] = None) -> None:
    """노드를 띄우고 uvicorn 을 메인 스레드에서 실행한다.

    Args:
        args: ROS 2 인자 목록. None 이면 sys.argv 를 사용한다.
    """
    rclpy.init(args=args)

    node: Optional[WebDashboardNode] = None
    executor: Optional[SingleThreadedExecutor] = None
    spin_thread: Optional[threading.Thread] = None

    try:
        node = WebDashboardNode()

        # rclpy 는 데몬 스레드에서, uvicorn 은 메인 스레드에서 돌린다.
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(
            target=_spin_ros, args=(executor, node), name='rclpy_spin', daemon=True)
        spin_thread.start()

        # uvicorn 이 SIGINT 를 받아 graceful shutdown 후 run() 이 반환된다.
        config = uvicorn.Config(
            create_app(node),
            host=node.host,
            port=node.port,
            log_level='info',
            access_log=False,
            timeout_graceful_shutdown=3,
        )
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('KeyboardInterrupt 수신 → 종료 절차 시작')
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f'예외로 종료: {exc}')
        else:
            # 노드 생성 전 실패(파라미터 검증 등)는 로거가 없으므로 그대로 올린다.
            rclpy.shutdown()
            raise
    finally:
        # 스핀 스레드 정지 → 노드 파괴 → 컨텍스트 종료 순서를 지킨다.
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
            if spin_thread.is_alive() and node is not None:
                node.get_logger().warn('rclpy 스핀 스레드가 2초 내에 종료되지 않았다')
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
