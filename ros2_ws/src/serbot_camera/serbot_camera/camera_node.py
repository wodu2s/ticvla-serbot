#!/usr/bin/env python3
"""SerBot II IMX219 CSI 카메라를 단독으로 열어 ROS 2 토픽으로 발행하는 노드.

시스템 전체에서 이 노드 하나만 카메라 디바이스를 연다. nvarguscamerasrc 는
동시에 두 프로세스가 같은 센서를 열면 Argus 에러로 실패하므로, 웹 대시보드 /
TIC-VLA 추론 / 데이터 수집은 모두 이 노드가 발행하는 토픽을 구독해야 한다.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

#: 연속 읽기 실패가 이 횟수에 도달하면 캡처를 닫고 재연결을 시도한다.
MAX_CONSECUTIVE_READ_FAILURES: int = 30

#: 프레임레이트 허용 범위 (nvarguscamerasrc IMX219 모드 기준).
MIN_FRAMERATE: int = 1
MAX_FRAMERATE: int = 60


def build_gst_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    framerate: int,
    flip_method: int,
) -> str:
    """nvarguscamerasrc 기반 GStreamer 파이프라인 문자열을 만든다.

    Jetson Orin NX + JetPack 6.0 + OpenCV 4.10.0(GStreamer 1.20.3) 조합에서
    15.09FPS / dropped 0 으로 실측 검증된 파이프라인이다.

    Args:
        sensor_id: nvarguscamerasrc 센서 인덱스.
        width: 캡처 가로 해상도(px).
        height: 캡처 세로 해상도(px).
        framerate: 초당 프레임 수.
        flip_method: nvvidconv 회전/반전 방식(0 이면 변환 없음).

    Returns:
        cv2.VideoCapture 에 그대로 전달할 수 있는 파이프라인 문자열.
    """
    # NVMM 메모리에서 받아 nvvidconv 로 BGRx 변환 후 videoconvert 로 BGR 로 낮춘다.
    # appsink 는 drop=true / max-buffers=1 로 두어 항상 최신 프레임만 남긴다.
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} ! '
        f'video/x-raw(memory:NVMM),width={width},height={height},'
        f'format=NV12,framerate={framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! video/x-raw,format=BGRx ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    )


class CameraNode(Node):
    """IMX219 CSI 카메라 프레임을 Image/CameraInfo 토픽으로 발행하는 노드.

    캡처와 발행을 분리한 구조를 사용한다.

    * 캡처 전용 데몬 스레드가 계속 ``cap.read()`` 를 돌면서 최신 프레임 1장만
      슬롯에 덮어쓴다. 느린 구독자나 발행 처리 때문에 캡처가 밀리지 않는다.
    * 발행 타이머(1/framerate 주기)가 슬롯에서 프레임을 꺼내 발행한다.
    * 읽기 실패가 연속으로 누적되면 캡처를 닫고 일정 시간 후 재연결한다.
    """

    def __init__(self) -> None:
        """파라미터를 읽어 검증하고, 퍼블리셔·캡처 스레드·타이머를 준비한다."""
        super().__init__('camera_node')

        # ── 파라미터 선언 및 로드 (기본값은 config/camera.yaml 과 동일) ──
        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('framerate', 15)
        # SerBot II 카메라는 180도 뒤집혀 장착. yaml 없이 ros2 run 해도 2 가 되게.
        # (config/camera.yaml 의 flip_method: 2 와 동일. TIC-VLA 브리지 기본과도 일치.)
        self.declare_parameter('flip_method', 2)
        self.declare_parameter('image_topic', '/camera/image_raw_fast')
        self.declare_parameter('info_topic', '/camera/camera_info')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('stats_period_sec', 5.0)
        self.declare_parameter('reopen_delay_sec', 2.0)

        self._sensor_id: int = int(self.get_parameter('sensor_id').value)
        self._width: int = int(self.get_parameter('width').value)
        self._height: int = int(self.get_parameter('height').value)
        self._framerate: int = int(self.get_parameter('framerate').value)
        self._flip_method: int = int(self.get_parameter('flip_method').value)
        self._image_topic: str = str(self.get_parameter('image_topic').value)
        self._info_topic: str = str(self.get_parameter('info_topic').value)
        self._frame_id: str = str(self.get_parameter('frame_id').value)
        self._stats_period_sec: float = float(
            self.get_parameter('stats_period_sec').value)
        self._reopen_delay_sec: float = float(
            self.get_parameter('reopen_delay_sec').value)

        self._validate_parameters()

        # ── 상태 변수 ──
        self._bridge: CvBridge = CvBridge()
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running: bool = True

        # 통계 카운터 (_lock 으로 보호)
        self._captured: int = 0
        self._published: int = 0
        self._read_fail: int = 0
        self._consecutive_failures: int = 0

        # ── 퍼블리셔 (센서 데이터용 QoS: 최신 프레임 우선, 재전송 없음) ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._image_pub = self.create_publisher(
            Image, self._image_topic, sensor_qos)
        self._info_pub = self.create_publisher(
            CameraInfo, self._info_topic, sensor_qos)

        # ── 카메라 오픈 (실패해도 캡처 스레드가 재연결을 계속 시도한다) ──
        self._pipeline: str = build_gst_pipeline(
            self._sensor_id,
            self._width,
            self._height,
            self._framerate,
            self._flip_method,
        )
        self.get_logger().info(f'GStreamer 파이프라인: {self._pipeline}')
        self._open_capture()

        # ── 캡처 전용 데몬 스레드 시작 ──
        self._capture_thread: threading.Thread = threading.Thread(
            target=self._capture_loop,
            name='csi_capture',
            daemon=True,
        )
        self._capture_thread.start()

        # ── 타이머: 발행(1/framerate)과 통계(stats_period_sec) ──
        self._publish_timer = self.create_timer(
            1.0 / float(self._framerate), self._publish_frame)
        self._stats_timer = self.create_timer(
            self._stats_period_sec, self._log_stats)

        self.get_logger().info(
            f'camera_node 시작: sensor_id={self._sensor_id}, '
            f'{self._width}x{self._height}@{self._framerate}fps, '
            f'image={self._image_topic}, info={self._info_topic}'
        )

    # ------------------------------------------------------------------
    # 파라미터 검증
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        """로드한 파라미터가 유효한 범위인지 확인한다.

        Raises:
            ValueError: 해상도가 0 이하이거나, 프레임레이트가 허용 범위를
                벗어나거나, 그 밖의 설정값이 물리적으로 불가능한 경우.
        """
        if self._sensor_id < 0:
            raise ValueError(f'sensor_id 는 0 이상이어야 한다: {self._sensor_id}')
        if self._width <= 0 or self._height <= 0:
            raise ValueError(
                f'해상도는 0 보다 커야 한다: {self._width}x{self._height}')
        if not MIN_FRAMERATE <= self._framerate <= MAX_FRAMERATE:
            raise ValueError(
                f'framerate 는 {MIN_FRAMERATE}~{MAX_FRAMERATE} 범위여야 한다: '
                f'{self._framerate}'
            )
        if not 0 <= self._flip_method <= 7:
            raise ValueError(
                f'flip_method 는 0~7 범위여야 한다: {self._flip_method}')
        if not self._image_topic or not self._info_topic:
            raise ValueError('image_topic 과 info_topic 은 비어 있을 수 없다')
        if not self._frame_id:
            raise ValueError('frame_id 는 비어 있을 수 없다')
        if self._stats_period_sec <= 0.0:
            raise ValueError(
                f'stats_period_sec 는 0 보다 커야 한다: {self._stats_period_sec}')
        if self._reopen_delay_sec < 0.0:
            raise ValueError(
                f'reopen_delay_sec 는 0 이상이어야 한다: {self._reopen_delay_sec}')

    # ------------------------------------------------------------------
    # 캡처 관리
    # ------------------------------------------------------------------
    def _open_capture(self) -> bool:
        """GStreamer 파이프라인으로 카메라를 연다.

        Returns:
            열기에 성공하면 True, 실패하면 False.
        """
        cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # 열기 실패한 핸들은 Argus 자원을 잡고 있을 수 있으므로 즉시 해제한다.
            cap.release()
            self.get_logger().error(
                'CSI 카메라 열기 실패. 다른 프로세스가 카메라를 점유 중인지 확인할 것.')
            return False

        self._cap = cap
        self._consecutive_failures = 0
        self.get_logger().info('CSI 카메라 열기 성공')
        return True

    def _release_capture(self) -> None:
        """열려 있는 캡처 핸들을 해제한다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        """캡처 전용 스레드 본체. 최신 프레임 1장만 슬롯에 유지한다."""
        while self._running:
            # 카메라가 닫혀 있으면 재연결을 시도한다.
            if self._cap is None:
                if not self._open_capture():
                    time.sleep(self._reopen_delay_sec)
                continue

            ok, frame = self._cap.read()

            if not ok or frame is None:
                with self._lock:
                    self._read_fail += 1
                self._consecutive_failures += 1

                # 연속 실패가 임계치를 넘으면 파이프라인이 죽은 것으로 보고 재연결한다.
                if self._consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    self.get_logger().warn(
                        f'읽기 연속 실패 {self._consecutive_failures}회 → '
                        f'{self._reopen_delay_sec}초 후 카메라 재연결'
                    )
                    self._release_capture()
                    time.sleep(self._reopen_delay_sec)
                else:
                    # 폭주 방지를 위해 짧게 쉬어 준다.
                    time.sleep(0.01)
                continue

            # 정상 프레임: 이전 프레임을 덮어써 항상 최신 1장만 남긴다.
            self._consecutive_failures = 0
            with self._lock:
                self._latest_frame = frame
                self._captured += 1

    # ------------------------------------------------------------------
    # 발행
    # ------------------------------------------------------------------
    def _publish_frame(self) -> None:
        """슬롯의 최신 프레임을 Image / CameraInfo 로 발행한다."""
        with self._lock:
            frame = self._latest_frame
            # 같은 프레임을 중복 발행하지 않도록 슬롯을 비운다.
            self._latest_frame = None

        if frame is None:
            return

        # Image 와 CameraInfo 의 stamp 는 반드시 동일한 값을 사용한다.
        stamp = self.get_clock().now().to_msg()

        try:
            image_msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        except Exception as exc:  # cv_bridge 변환 실패는 프레임 단위로 흘려보낸다.
            self.get_logger().error(f'cv_bridge 변환 실패: {exc}')
            return

        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self._frame_id

        info_msg = self._build_camera_info(stamp)

        self._image_pub.publish(image_msg)
        self._info_pub.publish(info_msg)

        with self._lock:
            self._published += 1

    def _build_camera_info(self, stamp) -> CameraInfo:  # noqa: ANN001
        """현재 해상도에 맞는 CameraInfo 메시지를 만든다.

        Args:
            stamp: Image 메시지와 공유할 builtin_interfaces/Time 값.

        Returns:
            채워진 CameraInfo 메시지.
        """
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.width = self._width
        info.height = self._height
        info.distortion_model = 'plumb_bob'

        # TODO: 카메라 캘리브레이션 미수행 상태다. camera_calibration 패키지로
        # 체커보드 캘리브레이션을 수행한 뒤 d/k/r/p 를 실제 값으로 교체할 것.
        # 현재는 왜곡 계수를 0 으로 두어 "왜곡 없음"으로 취급한다.
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [0.0] * 9
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [0.0] * 12
        return info

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------
    def _log_stats(self) -> None:
        """캡처/발행/읽기실패 카운터를 주기적으로 로그로 남긴다."""
        with self._lock:
            captured = self._captured
            published = self._published
            read_fail = self._read_fail

        # 주기 대비 실제 발행 FPS 를 함께 보여 파이프라인 지연을 바로 확인한다.
        fps = published / self._stats_period_sec if self._stats_period_sec else 0.0
        self.get_logger().info(
            f'[stats/{self._stats_period_sec:.0f}s] captured={captured} '
            f'published={published} read_fail={read_fail} '
            f'(publish {fps:.2f} FPS)'
        )

        with self._lock:
            self._captured = 0
            self._published = 0
            self._read_fail = 0

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """캡처 스레드를 정지시키고 카메라 자원을 해제한다.

        Argus 는 프로세스가 죽어도 자원을 곧바로 회수하지 못하는 경우가 있으므로
        반드시 스레드 join 후 cap.release() 까지 수행해야 다음 실행이 정상 동작한다.
        """
        self._running = False

        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                self.get_logger().warn('캡처 스레드가 2초 내에 종료되지 않았다')

        self._release_capture()
        self.get_logger().info('카메라 자원 해제 완료')


def main(args: Optional[list[str]] = None) -> None:
    """노드를 생성하고 스핀하며, 어떤 경로로 끝나든 자원을 정리한다.

    Args:
        args: ROS 2 인자 목록. None 이면 sys.argv 를 사용한다.
    """
    rclpy.init(args=args)

    node: Optional[CameraNode] = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('KeyboardInterrupt 수신 → 종료 절차 시작')
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f'예외로 종료: {exc}')
        else:
            # 노드 생성 전(파라미터 검증 실패 등)에는 로거가 없으므로 다시 올린다.
            rclpy.shutdown()
            raise
    finally:
        # 정상/비정상 종료 모두에서 스레드 정지 → 카메라 해제 → 노드 파괴 순서를 지킨다.
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
