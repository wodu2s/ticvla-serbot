#!/usr/bin/env python3
"""ReAct 단독 주행 어댑터 — TIC-VLA 없이 Gemini 비전 + 단순 cmd_vel 로 실제 주행.

``LiveTicVlaAdapter`` 가 TIC-VLA ``<think>`` 를 관측으로 쓰고 로봇은 움직이지
않는 것과 달리, 이 어댑터는:

- 관측: 실시간 카메라 프레임 1장을 Gemini 에 보내 장면 서술을 받는다.
- ``goto[X]``: Gemini 에 X 의 bbox 를 묻고 → bbox 중심으로 방위각을 환산해
  제자리 회전 → 추정 거리만큼(최대 ``max_step_m``) 전진. 이를 도착 판정
  (``stop_distance_m``/``arrive_bbox_h``)까지 ``max_servo_iters`` 번 반복한다.
- ``rotate[left|right]``: odom yaw 폐루프로 ``rotate_deg`` 만큼 제자리 회전.

비교 기준선(baseline) 용도다 — 저수준 정책이 없는 "LLM 만으로의 주행"이
얼마나 되는지를 본다.

안전장치:
- 출력은 ``/ticvla/cmd_vel`` → cmd_vel_mux 를 거친다(속도 클램프, 0.5s 무수신
  정지, 조이스틱 우선, E-Stop). mux 가 안 떠 있으면 시작하지 않는다.
- mux 상태가 ``joystick``/``estop`` 이 되면 즉시 0 을 내고 ``OperatorAbort``.
- odom/카메라가 ``sensor_stale_sec`` 이상 끊기면 정지하고 동작 실패 처리.
- 동작 하나가 ``motion_timeout_sec`` 를 넘으면 정지.
- 종료(정상/예외/Ctrl+C) 시 ``close()`` 가 0 속도를 여러 번 발행한다.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.config import DirectDriveConfig
from agent.llm_gemini import GeminiClient, LLMUnavailable
from agent.tic_vla_adapter import TicVlaAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 제어 루프 주기(Hz). mux cmd_timeout(0.5s)보다 충분히 촘촘해야 한다.
CONTROL_HZ = 20.0
#: 회전 완료 허용 오차(도).
YAW_TOL_DEG = 2.0
#: 회전 끝무렵 최소 각속도(rad/s) — 너무 작으면 마찰로 멈춘다.
MIN_WZ = 0.15

_DESCRIBE_PROMPT = """\
You are the eyes of a small indoor mobile robot. Describe what the robot's \
front camera sees in 2-4 short sentences. Name every distinct object or \
landmark you can see with a short, specific noun phrase (e.g. "red marker", \
"blue cooler", "office chair", "open door") and say whether it is on the \
left, center or right and roughly near or far. If the scene is dark or \
blurry, say so. Do not speculate about things that are not visible."""

_LOCATE_PROMPT = """\
Find "{target}" in this image from a robot's front camera.
Answer with ONLY one JSON object, no markdown:
{{"visible": true|false, "box_2d": [ymin, xmin, ymax, xmax], "distance_m": <float>}}
box_2d uses 0-1000 normalized coordinates. distance_m is your rough estimate \
of the distance from the camera to the target on the floor. If the target is \
not visible, answer {{"visible": false}}."""

_RE_JSON = re.compile(r'\{.*\}', re.DOTALL)


class OperatorAbort(RuntimeError):
    """운영자가 조이스틱/E-Stop 으로 개입해 주행을 끊었을 때."""


class MotionError(RuntimeError):
    """센서 끊김/시간 초과 등으로 동작을 끝까지 못 한 경우."""


def _yaw_from_quat(q: Any) -> float:
    """geometry_msgs/Quaternion → yaw(rad)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(angle: float) -> float:
    """각도를 (-pi, pi] 로 감싼다."""
    return math.atan2(math.sin(angle), math.cos(angle))


class DirectDriveAdapter(TicVlaAdapter):
    """Gemini 비전 + odom 폐루프 cmd_vel 로 실제 로봇을 움직이는 어댑터."""

    def __init__(self, cfg: DirectDriveConfig, llm: GeminiClient,
                 logger: Any, run_label: str) -> None:
        """ROS 노드를 띄우고 카메라/odom/mux 수신을 확인한다.

        Args:
            cfg: direct 모드 설정.
            llm: 비전 호출에 쓸 Gemini 클라이언트(루프와 같은 인스턴스).
            logger: ``.info``/``.warning``/``.error`` 를 쓰는 로거.
            run_label: 프레임 저장 하위 폴더명(react_loop 의 JSONL 라벨과 같게).

        Raises:
            RuntimeError: ROS 환경이 없거나 카메라/odom/mux 가 응답하지 않는 경우.
        """
        self._cfg = cfg
        self._llm = llm
        self._log = logger

        save_root = Path(cfg.frame_save_dir)
        if not save_root.is_absolute():
            save_root = PROJECT_ROOT / save_root
        self._frame_dir = save_root / run_label
        self._frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_seq = 0

        try:
            import rclpy
            from cv_bridge import CvBridge
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except ImportError as exc:
            raise RuntimeError(
                f'ROS 환경을 불러오지 못했다({exc}) — /opt/ros/humble 과 '
                'ros2_ws/install 을 source 한 셸에서 실행할 것.') from exc

        import cv2

        self._cv2 = cv2
        self._Twist = Twist
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame: Optional[Any] = None
        self._frame_at = 0.0
        self._odom: Optional[tuple[float, float, float]] = None
        self._odom_at = 0.0
        self._mux_reason: Optional[str] = None
        self._mux_at = 0.0

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = rclpy.create_node('react_direct_drive')
        self._cmd_pub = self._node.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self._node.create_subscription(
            Image, cfg.camera_topic, self._on_image, qos_profile_sensor_data)
        self._node.create_subscription(
            Odometry, cfg.odom_topic, self._on_odom, qos_profile_sensor_data)
        self._node.create_subscription(
            String, '/ticvla/mux_status', self._on_mux, 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._wait_ready(timeout=5.0)
        self._log.warning(
            '★ ReAct 단독 주행 모드 — 로봇이 실제로 움직인다. 조이스틱을 건드리거나 '
            '대시보드 E-Stop 을 누르면 즉시 멈추고 에피소드를 끝낸다.')

    # ── ROS 콜백 ──────────────────────────────────────────────────────

    def _on_image(self, msg: Any) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001 — 콜백에서 죽으면 안 된다
            self._log.warning(f'카메라 프레임 변환 실패: {exc}')
            return
        with self._lock:
            self._frame = frame
            self._frame_at = time.monotonic()

    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose
        with self._lock:
            self._odom = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))
            self._odom_at = time.monotonic()

    def _on_mux(self, msg: Any) -> None:
        with self._lock:
            self._mux_reason = msg.data
            self._mux_at = time.monotonic()

    # ── 상태 확인 ─────────────────────────────────────────────────────

    def _wait_ready(self, timeout: float) -> None:
        """카메라/odom/mux 가 모두 들어올 때까지 기다린다."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                ok = (self._frame is not None and self._odom is not None
                      and self._mux_reason is not None)
            if ok:
                break
            time.sleep(0.1)
        else:
            self._fail_not_ready(timeout)
        # enable 게이트는 운영자(대시보드) 몫이라 여기서 켜거나 끄지 않는다.
        time.sleep(0.3)
        with self._lock:
            reason = self._mux_reason
        if reason in ('disabled', 'estop'):
            self.close()
            raise RuntimeError(
                f'cmd_vel_mux 상태={reason} — 대시보드에서 E-Stop 해제/주행 허용을 '
                '켠 뒤 다시 실행할 것.')

    def _fail_not_ready(self, timeout: float) -> None:
        with self._lock:
            missing = [name for name, v in (
                (self._cfg.camera_topic, self._frame),
                (self._cfg.odom_topic, self._odom),
                ('/ticvla/mux_status (cmd_vel_mux)', self._mux_reason)) if v is None]
        self.close()
        raise RuntimeError(
            f'{timeout:.0f}초 안에 수신되지 않음: {", ".join(missing)} — '
            "'bot' 으로 센서를, 'joystick' 또는 cmd_vel_mux 를 먼저 띄울 것.")

    def _check_safety(self) -> None:
        """운영자 개입/센서 끊김을 확인한다. 문제가 있으면 예외."""
        now = time.monotonic()
        with self._lock:
            reason = self._mux_reason
            odom_age = now - self._odom_at
            frame_age = now - self._frame_at
        if reason in ('joystick', 'estop'):
            raise OperatorAbort(f'cmd_vel_mux 상태={reason} — 운영자 개입')
        if odom_age > self._cfg.sensor_stale_sec:
            raise MotionError(f'odom {odom_age:.1f}s 끊김')
        if frame_age > self._cfg.sensor_stale_sec:
            raise MotionError(f'카메라 {frame_age:.1f}s 끊김')

    # ── 저수준 동작 ───────────────────────────────────────────────────

    def _publish(self, vx: float, wz: float) -> None:
        msg = self._Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def _halt(self) -> None:
        for _ in range(5):
            self._publish(0.0, 0.0)
            time.sleep(1.0 / CONTROL_HZ)

    def _rotate_by(self, angle_rad: float) -> float:
        """odom yaw 폐루프로 제자리 회전한다. 실제 회전량(rad)을 돌려준다."""
        with self._lock:
            start_yaw = self._odom[2]
        target = _wrap(start_yaw + angle_rad)
        tol = math.radians(YAW_TOL_DEG)
        started = time.monotonic()
        try:
            while True:
                self._check_safety()
                if time.monotonic() - started > self._cfg.motion_timeout_sec:
                    raise MotionError('회전 시간 초과')
                with self._lock:
                    yaw = self._odom[2]
                err = _wrap(target - yaw)
                if abs(err) < tol:
                    break
                # 끝무렵 감속(비례), 하한/상한 클램프
                wz = max(MIN_WZ, min(self._cfg.angular_speed, 1.5 * abs(err)))
                self._publish(0.0, math.copysign(wz, err))
                time.sleep(1.0 / CONTROL_HZ)
        finally:
            self._halt()
        with self._lock:
            return _wrap(self._odom[2] - start_yaw)

    def _forward_by(self, dist_m: float) -> float:
        """odom 변위 폐루프로 직진한다. 실제 이동 거리(m)를 돌려준다."""
        with self._lock:
            x0, y0, _ = self._odom
        started = time.monotonic()
        moved = 0.0
        try:
            while moved < dist_m:
                self._check_safety()
                if time.monotonic() - started > self._cfg.motion_timeout_sec:
                    raise MotionError('전진 시간 초과')
                self._publish(self._cfg.linear_speed, 0.0)
                time.sleep(1.0 / CONTROL_HZ)
                with self._lock:
                    x, y, _ = self._odom
                moved = math.hypot(x - x0, y - y0)
        finally:
            self._halt()
        return moved

    # ── 비전 ──────────────────────────────────────────────────────────

    def _grab_jpeg(self, tag: str) -> bytes:
        """최신 프레임을 JPEG 로 인코딩하고 디스크에도 남긴다."""
        time.sleep(0.3)  # 정지 직후 모션 블러 방지
        self._check_safety()
        with self._lock:
            frame = self._frame.copy()
        ok, buf = self._cv2.imencode('.jpg', frame, [self._cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise MotionError('JPEG 인코딩 실패')
        self._frame_seq += 1
        safe_tag = re.sub(r'[^0-9A-Za-z_-]+', '-', tag)[:40]
        path = self._frame_dir / f'frame_{self._frame_seq:03d}_{safe_tag}.jpg'
        path.write_bytes(buf.tobytes())
        return buf.tobytes()

    def _describe(self, tag: str) -> str:
        jpeg = self._grab_jpeg(tag)
        return self._llm.generate_with_image(_DESCRIBE_PROMPT, jpeg).strip()

    def _locate(self, target: str, tag: str) -> Optional[dict]:
        """타깃 bbox/거리를 묻는다. 안 보이거나 파싱 실패면 None."""
        jpeg = self._grab_jpeg(tag)
        raw = self._llm.generate_with_image(_LOCATE_PROMPT.format(target=target), jpeg)
        match = _RE_JSON.search(raw)
        if match is None:
            self._log.warning(f'locate 응답에 JSON 없음: {raw!r}')
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            self._log.warning(f'locate JSON 파싱 실패: {raw!r}')
            return None
        if not data.get('visible') or len(data.get('box_2d') or []) != 4:
            return None
        return data

    # ── TicVlaAdapter 인터페이스 ──────────────────────────────────────

    def _safe(self, fn, *args) -> str:
        """비전/모션 오류를 관측 문자열로 바꾼다. OperatorAbort 는 그대로 올린다."""
        try:
            return fn(*args)
        except LLMUnavailable as exc:
            self._halt()
            return f'[ERROR] vision call failed ({exc.reason}). The robot stopped.'
        except MotionError as exc:
            self._halt()
            self._log.error(f'동작 실패: {exc}')
            return f'[ERROR] motion aborted ({exc}). The robot stopped.'

    def initial_observation(self) -> str:
        return self._safe(self._describe, 'init')

    def goto(self, target: str) -> str:
        return self._safe(self._goto, target)

    def _goto(self, target: str) -> str:
        cfg = self._cfg
        total_moved = 0.0
        for it in range(1, cfg.max_servo_iters + 1):
            loc = self._locate(target, f'goto-{target}-{it}')
            if loc is None:
                scene = self._describe(f'goto-{target}-lost')
                prefix = ('The target is not visible from here.' if it == 1 else
                          f'Moved {total_moved:.1f} m toward the {target}, then lost sight of it.')
                return f'{prefix} {scene}'
            ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in loc['box_2d'])
            cx = (xmin + xmax) / 2.0
            box_h = max(0.0, ymax - ymin)
            dist = float(loc.get('distance_m') or 0.0)
            bearing_deg = (0.5 - cx) * cfg.hfov_deg  # + = 왼쪽(반시계)
            self._log.info(
                f'[goto {target} #{it}] cx={cx:.2f} box_h={box_h:.2f} '
                f'dist≈{dist:.2f}m bearing={bearing_deg:+.1f}°')

            if box_h >= cfg.arrive_bbox_h or (0.0 < dist <= cfg.stop_distance_m):
                if abs(bearing_deg) > cfg.heading_tolerance_deg:
                    self._rotate_by(math.radians(bearing_deg))
                scene = self._describe(f'goto-{target}-arrived')
                return f'The robot is now directly in front of the {target}. {scene}'

            if abs(bearing_deg) > cfg.heading_tolerance_deg:
                self._rotate_by(math.radians(bearing_deg))
            step = cfg.max_step_m if dist <= 0.0 else min(
                cfg.max_step_m, max(0.1, dist - cfg.stop_distance_m))
            total_moved += self._forward_by(step)

        scene = self._describe(f'goto-{target}-limit')
        return (f'Moved {total_moved:.1f} m toward the {target} but did not reach '
                f'it within {cfg.max_servo_iters} attempts. {scene}')

    def rotate(self, direction: str) -> str:
        return self._safe(self._rotate_obs, direction)

    def _rotate_obs(self, direction: str) -> str:
        sign = 1.0 if direction == 'left' else -1.0
        turned = self._rotate_by(sign * math.radians(self._cfg.rotate_deg))
        scene = self._describe(f'rotate-{direction}')
        return f'Turned {direction} {abs(math.degrees(turned)):.0f} degrees. {scene}'

    def stop(self) -> None:
        try:
            self._halt()
        except Exception:  # noqa: BLE001 — stop() 은 예외를 던지면 안 된다
            pass

    def close(self) -> None:
        """정지 후 ROS 노드를 내린다. 여러 번 불려도 안전하다."""
        self.stop()
        try:
            self._executor.shutdown()
            self._node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    @property
    def frame_dir(self) -> Path:
        return self._frame_dir
