#!/usr/bin/env python3
"""TIC-VLA GoToObject 파인튜닝용 주행 데이터 수집 노드.

SerBot II 카메라·휠 오도메트리·조이스틱(또는 cmd_vel)을 10Hz 로 저장한다.
이 노드는 cmd_vel 을 **발행하지 않는다**. 구동은 별도 조이스틱 스크립트가 담당한다.

출력은 s01_batch_json_generation.py 가 바로 읽을 수 있는 형태다.

    <data-root>/GoToObject_v1/raw/epNNNN_<object>_<task>_<distance>/
    ├── frames/000000.jpg …
    ├── rgb -> frames          (s01 호환 심볼릭 링크)
    ├── trajectory.csv         ★ 휠 오도메트리 실측 (s01 입력)
    ├── cmd_vel.csv            조종 명령값 (원본)
    ├── trajectory_cmd.csv     명령 적분 추정 경로 (분석용, 학습 미사용)
    └── meta.json              instruction + files 설명

폴더명 예: ep0017_graybin_L_2m / ep0031_redbox_R_3m / ep0042_graybin_C_2p5m
파싱: seq, obj, task, dist = name.split("_", 3)  (seq 는 "ep" 접두사 제거)

참고: OmniVLA collect 의 조이스틱 리더만 참고한다. goal_img / actions /
joystick / instruction.txt 는 OmniVLA 잔재라 저장하지 않는다.

사용 예:

    # 기본 (정면 회색 통, cmd_vel 구독)
    python3 collect/collect_gotoobject.py --task C

    # 좌측 배치, 시작 거리 메모
    python3 collect/collect_gotoobject.py --task L --start-distance 2.0

    # odom 없이 강행 (학습에 쓰면 안 됨)
    python3 collect/collect_gotoobject.py --task C --no-odom
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


# ──────────────────────────────────────────────────────────────────────
# CSV 헤더
# ──────────────────────────────────────────────────────────────────────

#: s01 이 요구하는 필수 컬럼 + image(추적용 여분). 헤더·순서를 바꾸지 말 것.
TRAJECTORY_HEADER = ["time", "image", "x", "y", "z", "qx", "qy", "qz", "qw"]

#: 그 시점 조종 명령 (cmd_vel). joystick 이면 데드존 적용 후 값.
CMD_VEL_HEADER = ["time", "image", "linear_x", "linear_y", "angular_z"]

#: 명령을 적분한 추정 경로 (분석용, 학습에 쓰지 않음).
TRAJECTORY_CMD_HEADER = ["time", "image", "x", "y", "yaw"]

#: meta.json files — 폴더만 열어도 각 파일이 뭔지 알 수 있게.
EPISODE_FILES_DESC: dict[str, str] = {
    "trajectory.csv": "wheel odometry 실측. 학습 라벨 원본",
    "cmd_vel.csv": "조종 명령값 (linear_x, linear_y, angular_z)",
    "trajectory_cmd.csv": "명령을 적분한 추정 경로. 학습에 쓰지 않음",
    "frames/": "10Hz 카메라 프레임",
    "rgb": "frames 심볼릭 링크 (s01 요구)",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

#: odom 대기 상한(초). 이 안에 한 번도 안 오면 거부.
ODOM_WAIT_SEC: float = 10.0

#: 정지 판정에 쓰는 이동 거리 창(초).
STOP_WINDOW_SEC: float = 1.0

DEFAULT_INSTRUCTION = "Go to the gray bin."
VALID_TASKS = ("L", "C", "R")


# ──────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────

def default_data_root() -> Path:
    """데이터 루트 기본값을 돌려준다.

    Returns:
        TICVLA_DATA_ROOT 환경변수, 없으면 ~/TIC-VLA/data.
    """
    env = os.environ.get("TICVLA_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "TIC-VLA" / "data").resolve()


def apply_deadzone(value: float, deadzone: float) -> float:
    """데드존을 적용한 뒤 -1~+1 로 재정규화한다.

    Args:
        value: 원본 축 값.
        deadzone: 무시할 절댓값 하한.

    Returns:
        데드존 밖이면 재스케일된 값, 안이면 0.
    """
    value = float(value)
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - deadzone) / max(1e-9, 1.0 - deadzone)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """쿼터니언에서 yaw(rad) 를 뽑는다.

    Args:
        qx, qy, qz, qw: orientation.

    Returns:
        yaw (rad).
    """
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def format_distance_tag(meters: float) -> str:
    """--start-distance 를 폴더명용 태그로 만든다.

    Args:
        meters: 시작 거리(m).

    Returns:
        정수면 "2m", 소수면 점을 p 로 바꾼 "2p5m".
    """
    if abs(meters - round(meters)) < 1e-9:
        return f"{int(round(meters))}m"
    text = f"{meters:.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "m"


def next_episode_seq(raw_root: Path) -> int:
    """raw/ 전체 ep* 폴더를 스캔해 다음 일련번호를 돌려준다.

    task 별이 아니라 raw 전체를 통틀어 연속으로 매긴다.

    Args:
        raw_root: GoToObject_v1/raw 경로.

    Returns:
        다음 번호 (없으면 1).
    """
    max_seq = 0
    if not raw_root.is_dir():
        return 1
    for path in raw_root.iterdir():
        if not path.is_dir() or not path.name.startswith("ep"):
            continue
        # ep0017_graybin_L_2m → head=ep0017
        head = path.name.split("_", 1)[0]
        digits = head[2:]
        if digits.isdigit():
            max_seq = max(max_seq, int(digits))
    return max_seq + 1


def create_episode_dir(
    data_root: Path,
    object_name: str,
    task: str,
    start_distance: float,
) -> Path:
    """에피소드 디렉토리를 만든다. task 하위 폴더는 만들지 않는다.

    폴더명: epNNNN_<object>_<task>_<distance>
    파싱: seq, obj, task, dist = name.split("_", 3)
          (seq 는 "ep" 접두사를 제거한 숫자)

    Args:
        data_root: --data-root.
        object_name: 객체 이름 (예: graybin). '_' 불가.
        task: L / C / R.
        start_distance: --start-distance (m).

    Returns:
        생성된 에피소드 Path.

    Raises:
        ValueError: object 에 '_' 가 포함된 경우.
    """
    if "_" in object_name:
        raise ValueError(
            f"--object 에 '_' 를 넣을 수 없다 (폴더 파싱이 깨진다): "
            f"{object_name!r}")

    raw_root = data_root / "GoToObject_v1" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    seq = next_episode_seq(raw_root)
    dist_tag = format_distance_tag(start_distance)
    base_name = f"ep{seq:04d}_{object_name}_{task}_{dist_tag}"
    episode_dir = raw_root / base_name
    # 극히 드물게 동시 생성 충돌 시 번호만 올린다.
    while episode_dir.exists():
        seq += 1
        base_name = f"ep{seq:04d}_{object_name}_{task}_{dist_tag}"
        episode_dir = raw_root / base_name

    episode_dir.mkdir(parents=False, exist_ok=False)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir()
    # s01 은 'rgb' 디렉토리를 rglob 으로 찾는다. frames 와 1:1 대응 심볼릭 링크.
    rgb_link = episode_dir / "rgb"
    if not rgb_link.exists():
        rgb_link.symlink_to("frames")
    return episode_dir


# ──────────────────────────────────────────────────────────────────────
# 조이스틱 리더 (OmniVLA v2 참고, 독립 구현)
# ──────────────────────────────────────────────────────────────────────

class JoystickReader:
    """pygame USB 조이스틱 리더. cmd_vel 은 발행하지 않고 값만 읽는다."""

    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        """조이스틱을 연다.

        Args:
            args: CLI 인자.
            logger: rclpy 로거.

        Raises:
            RuntimeError: pygame/조이스틱 초기화 실패.
        """
        self.args = args
        self.logger = logger
        self.pygame = None
        self.joystick = None

        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        try:
            import pygame
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"pygame import 실패: {exc!r}") from exc

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count <= 0:
            raise RuntimeError(
                "조이스틱이 감지되지 않았다. USB 연결을 확인하거나 "
                "--action-source cmd_vel 을 쓰라.")

        self.joystick = pygame.joystick.Joystick(args.joystick_index)
        self.joystick.init()
        self.logger.info(
            f"조이스틱: {self.joystick.get_name()} "
            f"axes={self.joystick.get_numaxes()} "
            f"buttons={self.joystick.get_numbuttons()}")

    def read_raw(self) -> Tuple[float, float, float]:
        """데드존 적용 전 원본 축 값.

        Returns:
            (raw_left_x, raw_left_y, raw_right_x).
        """
        self.pygame.event.pump()
        n = self.joystick.get_numaxes()

        def axis(idx: int) -> float:
            if idx < 0 or idx >= n:
                return 0.0
            return float(self.joystick.get_axis(idx))

        return (
            axis(self.args.left_x_axis),
            axis(self.args.left_y_axis),
            axis(self.args.right_x_axis),
        )

    def read_cmd(self) -> Tuple[float, float, float]:
        """데드존·스케일 적용 후 (linear_x, linear_y, angular_z).

        Returns:
            로봇 명령 단위의 속도 튜플.
        """
        raw_lx, raw_ly, raw_rx = self.read_raw()
        left_x = apply_deadzone(raw_lx, self.args.deadzone)
        left_y = apply_deadzone(raw_ly, self.args.deadzone)
        right_x = apply_deadzone(raw_rx, self.args.deadzone)

        # pygame: 스틱 위 = y 음수 → 전진 +linear_x
        linear_x = -left_y * self.args.max_linear_x
        linear_y = -left_x * self.args.max_linear_y
        angular_z = right_x * self.args.max_angular_z

        if self.args.invert_linear_x:
            linear_x *= -1.0
        if self.args.invert_linear_y:
            linear_y *= -1.0
        if self.args.invert_angular_z:
            angular_z *= -1.0
        return float(linear_x), float(linear_y), float(angular_z)

    def close(self) -> None:
        """pygame 자원을 해제한다."""
        try:
            if self.pygame is not None:
                self.pygame.joystick.quit()
                self.pygame.quit()
        except Exception:  # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────
# 수집 노드
# ──────────────────────────────────────────────────────────────────────

class GoToObjectCollector(Node):
    """GoToObject 에피소드 수집 노드."""

    def __init__(self, args: argparse.Namespace) -> None:
        """구독·CSV·에피소드 폴더를 준비한다. 기록은 odom 확인 후 시작한다.

        Args:
            args: CLI 인자.
        """
        super().__init__("ticvla_gotoobject_collector")
        self.args = args
        self.log = self.get_logger()

        if args.task not in VALID_TASKS:
            raise ValueError(
                f"--task 는 {VALID_TASKS} 중 하나여야 한다: {args.task!r}")
        if "_" in args.object:
            raise ValueError(
                f"--object 에 '_' 를 넣을 수 없다: {args.object!r}")

        self.instruction = (
            args.instruction.strip() if args.instruction
            else DEFAULT_INSTRUCTION)

        # 폴더명에서 뺀 타임스탬프는 meta.created_at 으로 남긴다.
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.episode_dir = create_episode_dir(
            Path(args.data_root).expanduser().resolve(),
            args.object, args.task, args.start_distance)
        self.episode_id = self.episode_dir.name
        self.frames_dir = self.episode_dir / "frames"
        self.trajectory_path = self.episode_dir / "trajectory.csv"
        self.cmd_vel_path = self.episode_dir / "cmd_vel.csv"
        self.trajectory_cmd_path = self.episode_dir / "trajectory_cmd.csv"
        self.meta_path = self.episode_dir / "meta.json"
        # instruction 은 meta.json 에만 둔다 (별도 instruction.txt 없음).

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        # ── 최신 센서 ──
        self.latest_image_msg: Optional[Image] = None
        self._image_ok = False
        self._camera_fatal: Optional[str] = None
        self._last_image_at = 0.0

        self._odom_ok = False
        self._odom_count = 0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_z = 0.0
        self._odom_qx = 0.0
        self._odom_qy = 0.0
        self._odom_qz = 0.0
        self._odom_qw = 1.0
        #: 정지 판정용 (monotonic, x, y)
        self._odom_hist: Deque[Tuple[float, float, float]] = deque(
            maxlen=500)

        self.latest_linear_x = 0.0
        self.latest_linear_y = 0.0
        self.latest_angular_z = 0.0

        # ── 명령 적분 궤적 (분석용, trajectory_cmd.csv) ──
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_yaw = 0.0
        self._prev_timer_mono: Optional[float] = None

        # ── 기록 상태 ──
        self.start_time: Optional[float] = None
        self.timer = None
        self.frame_index = 0  # 다음 저장 인덱스 (000000 부터)
        self.rows_written = 0
        self.finished = False
        self._finish_done = False
        self._recording = False
        self._pending_start = False
        self._frame_intervals: list[float] = []
        self._last_save_mono: Optional[float] = None
        self._pose_samples: list[Tuple[float, float, float]] = []

        # ── 정지 유지 카운트다운 ──
        self._hold_started_at: Optional[float] = None
        self._hold_completed = False
        self._last_hold_log_at = 0.0

        self.joystick_reader: Optional[JoystickReader] = None
        if args.action_source == "joystick":
            self.joystick_reader = JoystickReader(args, self.log)

        # ── CSV (trajectory / cmd_vel / trajectory_cmd) ──
        self.csv_trajectory = self.trajectory_path.open(
            "w", encoding="utf-8", newline="")
        self.csv_cmd_vel = self.cmd_vel_path.open(
            "w", encoding="utf-8", newline="")
        self.csv_trajectory_cmd = self.trajectory_cmd_path.open(
            "w", encoding="utf-8", newline="")

        self.w_traj = csv.writer(self.csv_trajectory)
        self.w_cmd_vel = csv.writer(self.csv_cmd_vel)
        self.w_traj_cmd = csv.writer(self.csv_trajectory_cmd)

        self.w_traj.writerow(TRAJECTORY_HEADER)
        self.w_cmd_vel.writerow(CMD_VEL_HEADER)
        self.w_traj_cmd.writerow(TRAJECTORY_CMD_HEADER)
        for f in (self.csv_trajectory, self.csv_cmd_vel,
                  self.csv_trajectory_cmd):
            f.flush()

        # ── 구독 (카메라는 BEST_EFFORT 센서 QoS) ──
        self.create_subscription(
            Image, args.camera_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, args.odom_topic, self._on_odom, 10)
        if args.action_source == "cmd_vel":
            self.create_subscription(
                Twist, args.cmd_vel_topic, self._on_cmd_vel, 10)

        self.write_meta(status="waiting")
        self.log.info("═" * 60)
        self.log.info(f"에피소드 폴더 : {self.episode_dir}")
        self.log.info(f"object={args.object}  task={args.task}")
        self.log.info(f"instruction : {self.instruction}")
        self.log.info(
            f"camera={args.camera_topic}  odom={args.odom_topic}  "
            f"action={args.action_source}/{args.cmd_vel_topic}")
        self.log.info(
            f"hz={args.hz}  hold={args.hold_seconds}s  "
            f"stop_threshold={args.stop_threshold}m")
        self.log.info("이 노드는 cmd_vel 을 발행하지 않는다 (저장만).")
        self.log.info("═" * 60)

        if args.no_odom:
            self.log.warn(
                "⚠ --no-odom: 오도메트리 없이 강행한다. "
                "trajectory.csv 가 비거나 가짜가 되어 **학습에 쓸 수 없는 데이터**다.")
            self._arm_recording_when_ready(require_odom=False)
        else:
            self.log.info(
                f"odom 대기 중… ({ODOM_WAIT_SEC:.0f}초 안에 "
                f"{args.odom_topic} 이 와야 한다)")
            self._odom_wait_timer = self.create_timer(
                0.2, self._poll_odom_ready)

    # ── 시작 게이트 ─────────────────────────────────────────────────────

    def _poll_odom_ready(self) -> None:
        """odom 수신을 폴링한다. 타임아웃이면 치명 종료한다."""
        if self._recording or self.finished:
            return
        if self._odom_ok:
            try:
                self._odom_wait_timer.cancel()
            except Exception:  # noqa: BLE001
                pass
            self._arm_recording_when_ready(require_odom=True)
            return

        # 노드 생성 후 경과
        if not hasattr(self, "_wait_started"):
            self._wait_started = time.monotonic()
        elapsed = time.monotonic() - self._wait_started
        if elapsed < ODOM_WAIT_SEC:
            return

        self.log.error(
            f"odom 수신 실패 — {ODOM_WAIT_SEC:.0f}초 동안 "
            f"{self.args.odom_topic} 메시지가 없다.\n"
            "  확인:\n"
            "    1) odometry_publisher (또는 휠 오돔 노드) 가 떠 있는가\n"
            "    2) ros2 topic list | grep -i odom\n"
            "    3) ros2 topic echo /wheel/odom --once\n"
            "  강행이 필요하면 --no-odom (학습용 데이터로는 쓰지 말 것).")
        try:
            self._odom_wait_timer.cancel()
        except Exception:  # noqa: BLE001
            pass
        self.finish(abort_reason="odom_timeout")

    def _arm_recording_when_ready(self, *, require_odom: bool) -> None:
        """카메라(+odom)가 준비되면 기록 타이머를 무장한다.

        Args:
            require_odom: True 면 odom 필수.
        """
        if self._recording:
            return
        if require_odom and not self._odom_ok:
            return
        if self.latest_image_msg is None:
            self.log.info("첫 카메라 프레임을 기다리는 중…")
            # on_image 가 오면 시작
            self._pending_start = True
            return
        self._start_recording()

    def _start_recording(self) -> None:
        """고정 주기 기록 타이머를 켠다."""
        if self._recording or self.finished:
            return
        self._recording = True
        self.start_time = time.monotonic()
        self._prev_timer_mono = self.start_time
        self.timer = self.create_timer(1.0 / self.args.hz, self._on_timer)
        self.write_meta(status="recording")
        self.log.info(
            f"기록 시작 — {self.args.hz:.1f}Hz | "
            f"odom={'OK' if self._odom_ok else '없음(--no-odom)'}")

    # ── 콜백 ──────────────────────────────────────────────────────────

    def _on_image(self, msg: Image) -> None:
        """최신 카메라 프레임을 보관한다.

        Args:
            msg: sensor_msgs/Image.
        """
        self.latest_image_msg = msg
        self._image_ok = True
        self._last_image_at = time.monotonic()
        if (getattr(self, "_pending_start", False)
                and not self._recording
                and not self.finished):
            if self.args.no_odom or self._odom_ok:
                self._pending_start = False
                self._start_recording()

    def _on_odom(self, msg: Odometry) -> None:
        """휠 오도메트리를 저장한다. 변환 없이 그대로 쓴다.

        Args:
            msg: nav_msgs/Odometry (body→world FLU, REP-103).
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        now = time.monotonic()
        with self._lock:
            self._odom_ok = True
            self._odom_count += 1
            self._odom_x = float(p.x)
            self._odom_y = float(p.y)
            self._odom_z = float(p.z)
            self._odom_qx = float(q.x)
            self._odom_qy = float(q.y)
            self._odom_qz = float(q.z)
            self._odom_qw = float(q.w)
            self._odom_hist.append((now, self._odom_x, self._odom_y))

    def _on_cmd_vel(self, msg: Twist) -> None:
        """외부 cmd_vel(보통 조이스틱 발행)을 기록용으로 받는다.

        Args:
            msg: geometry_msgs/Twist.
        """
        self.latest_linear_x = float(msg.linear.x)
        self.latest_linear_y = float(msg.linear.y)
        self.latest_angular_z = float(msg.angular.z)

    # ── 타이머 ────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        """에피소드 시작 기준 경과 초.

        Returns:
            경과 시간(초). 시작 전이면 0.
        """
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def _snapshot_odom(self) -> Tuple[float, float, float, float, float, float, float]:
        """최신 odom pose 를 스냅샷한다.

        Returns:
            (x, y, z, qx, qy, qz, qw).
        """
        with self._lock:
            return (
                self._odom_x, self._odom_y, self._odom_z,
                self._odom_qx, self._odom_qy, self._odom_qz, self._odom_qw,
            )

    def _update_action_from_source(self) -> None:
        """action source 에서 최신 명령값을 읽는다.

        joystick 모드면 데드존 적용 후 값을 쓴다. 원본 축 값은 저장하지 않는다.
        """
        if self.args.action_source != "joystick":
            return
        assert self.joystick_reader is not None
        lx, ly, az = self.joystick_reader.read_cmd()
        self.latest_linear_x = lx
        self.latest_linear_y = ly
        self.latest_angular_z = az

    def _save_frame(self) -> Optional[str]:
        """최신 이미지를 frames/NNNNNN.jpg 로 저장한다.

        실패하면 None. 실패 시 CSV 행도 쓰지 않는다.

        Returns:
            저장한 파일명. 실패 시 None.
        """
        if self.latest_image_msg is None:
            self.log.error("카메라 프레임이 없다 — 수집을 중단한다 (폴백 없음).")
            self._camera_fatal = "no_image"
            return None

        # 카메라 끊김 감지 (2초 이상 무수신)
        if (self._last_image_at > 0.0
                and time.monotonic() - self._last_image_at > 2.0):
            self.log.error(
                "카메라 프레임이 2초 이상 끊겼다 — 수집을 중단한다 "
                "(폴백 이미지로 채우지 않는다).")
            self._camera_fatal = "image_stale"
            return None

        try:
            image = self.bridge.imgmsg_to_cv2(
                self.latest_image_msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"이미지 변환 실패 — 중단: {exc}")
            self._camera_fatal = "image_convert"
            return None

        name = f"{self.frame_index:06d}.jpg"
        path = self.frames_dir / name
        ok = cv2.imwrite(
            str(path), image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)])
        if not ok:
            self.log.error(f"이미지 저장 실패 — 이 스텝 CSV 행을 건너뛴다: {path}")
            return None

        now = time.monotonic()
        if self._last_save_mono is not None:
            self._frame_intervals.append(now - self._last_save_mono)
        self._last_save_mono = now
        self.frame_index += 1
        return name

    def _integrate_cmd(self, dt: float) -> None:
        """속도 명령을 적분해 분석용 궤적을 갱신한다.

        Args:
            dt: 직전 타이머와의 간격(초).
        """
        if dt <= 0.0:
            return
        lx = self.latest_linear_x
        ly = self.latest_linear_y
        az = self.latest_angular_z
        self.cmd_yaw += az * dt
        self.cmd_x += (lx * math.cos(self.cmd_yaw)
                       - ly * math.sin(self.cmd_yaw)) * dt
        self.cmd_y += (lx * math.sin(self.cmd_yaw)
                       + ly * math.cos(self.cmd_yaw)) * dt

    def _stop_window_travel(self) -> float:
        """최근 1초 창에서 odom 이동 거리를 돌려준다.

        Returns:
            창 내 이동 거리(m). 샘플 부족이면 inf.
        """
        with self._lock:
            hist = list(self._odom_hist)
        if len(hist) < 2:
            return float("inf")
        now = hist[-1][0]
        # 창의 시작에 가장 가까운 샘플
        start = hist[-1]
        for sample in reversed(hist):
            if now - sample[0] >= STOP_WINDOW_SEC:
                start = sample
                break
            start = sample
        if now - start[0] < STOP_WINDOW_SEC * 0.5:
            return float("inf")
        return math.hypot(hist[-1][1] - start[1], hist[-1][2] - start[2])

    def _update_hold_countdown(self) -> None:
        """정지 유지 카운트다운을 갱신하고 로그한다."""
        if not self._odom_ok:
            return
        travel = self._stop_window_travel()
        now = time.monotonic()

        if travel < self.args.stop_threshold:
            if self._hold_started_at is None:
                self._hold_started_at = now
                self.log.info(
                    f"정지 감지 (1초 이동 {travel:.4f}m < "
                    f"{self.args.stop_threshold:.3f}m) — "
                    f"{self.args.hold_seconds:.0f}초 유지를 시작한다")
            held = now - self._hold_started_at
            remain = max(0.0, self.args.hold_seconds - held)
            if (not self._hold_completed
                    and now - self._last_hold_log_at >= 0.5):
                self._last_hold_log_at = now
                self.log.info(
                    f"정지 유지 중… {held:4.1f}s / "
                    f"{self.args.hold_seconds:.0f}s  (남은 {remain:4.1f}s)")
            if (not self._hold_completed
                    and held >= self.args.hold_seconds):
                self._hold_completed = True
                self.log.info(
                    "✔ 정지 유지 완료 — 수집 정지(Ctrl+C)를 눌러도 된다")
        else:
            if self._hold_started_at is not None and not self._hold_completed:
                self.log.warn(
                    f"움직임 재개 (1초 이동 {travel:.4f}m) — "
                    "정지 유지 카운트다운을 리셋한다")
            self._hold_started_at = None
            # 완료 후에는 리셋하지 않는다 (이미 조건을 충족).
            if not self._hold_completed:
                pass

    def _on_timer(self) -> None:
        """고정 주기 저장. 이미지 실패 시 해당 스텝 CSV 를 쓰지 않는다."""
        if self.finished:
            return
        if self._camera_fatal is not None:
            self.finish(abort_reason=self._camera_fatal)
            return

        t = self.elapsed()
        self._update_action_from_source()

        image_name = self._save_frame()
        if image_name is None:
            if self._camera_fatal is not None:
                self.finish(abort_reason=self._camera_fatal)
            return

        # odom 스냅샷 (없으면 0 — --no-odom 경로)
        ox, oy, oz, qx, qy, qz, qw = self._snapshot_odom()
        self._pose_samples.append((t, ox, oy))

        # ★ 학습 라벨: 휠 오도메트리 실측 (헤더 변경 금지)
        self.w_traj.writerow([
            f"{t:.6f}", image_name,
            f"{ox:.6f}", f"{oy:.6f}", f"{oz:.6f}",
            f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
        ])

        # 조종 명령 원본 (joystick 이면 데드존 적용 후)
        self.w_cmd_vel.writerow([
            f"{t:.6f}", image_name,
            f"{self.latest_linear_x:.6f}",
            f"{self.latest_linear_y:.6f}",
            f"{self.latest_angular_z:.6f}",
        ])

        # 분석용: 명령 적분 추정 경로
        now_mono = time.monotonic()
        dt = (now_mono - self._prev_timer_mono
              if self._prev_timer_mono is not None else 0.0)
        self._prev_timer_mono = now_mono
        self._integrate_cmd(dt)
        self.w_traj_cmd.writerow([
            f"{t:.6f}", image_name,
            f"{self.cmd_x:.6f}", f"{self.cmd_y:.6f}", f"{self.cmd_yaw:.6f}",
        ])

        for f in (self.csv_trajectory, self.csv_cmd_vel,
                  self.csv_trajectory_cmd):
            f.flush()
        self.rows_written += 1

        self._update_hold_countdown()

        if (self.args.duration_sec is not None
                and self.args.duration_sec > 0
                and t >= self.args.duration_sec):
            self.log.info(
                f"--duration-sec {self.args.duration_sec} 도달 — 종료한다")
            self.finish()

    # ── 종료·검수 ─────────────────────────────────────────────────────

    def _path_length(self, samples: list[Tuple[float, float, float]]) -> float:
        """(t,x,y) 샘플의 누적 이동거리를 계산한다.

        Args:
            samples: 시간순 pose 샘플.

        Returns:
            누적 거리(m).
        """
        if len(samples) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(samples)):
            total += math.hypot(
                samples[i][1] - samples[i - 1][1],
                samples[i][2] - samples[i - 1][2])
        return total

    def _last_seconds_travel(self, seconds: float) -> float:
        """마지막 N초 구간의 이동거리.

        Args:
            seconds: 창 길이(초).

        Returns:
            이동거리(m).
        """
        if not self._pose_samples:
            return 0.0
        t_end = self._pose_samples[-1][0]
        window = [s for s in self._pose_samples if s[0] >= t_end - seconds]
        return self._path_length(window)

    def _build_summary(self) -> dict[str, Any]:
        """종료 검수 요약을 만든다.

        Returns:
            meta.json summary 및 터미널 출력용 딕셔너리.
        """
        n_frames = len([
            p for p in self.frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]) if self.frames_dir.exists() else 0

        intervals_ms = [x * 1000.0 for x in self._frame_intervals]
        target_ms = 1000.0 / max(self.args.hz, 1e-6)
        median_ms = float(np.median(intervals_ms)) if intervals_ms else 0.0
        max_ms = float(np.max(intervals_ms)) if intervals_ms else 0.0

        total_dist = self._path_length(self._pose_samples)
        last3 = self._last_seconds_travel(3.0)
        duration = self.elapsed()

        warnings: list[str] = []
        if n_frames != self.rows_written:
            warnings.append(
                f"프레임 수({n_frames})와 CSV 행({self.rows_written}) 불일치")

        # 세 CSV 데이터 행 수가 frames 와 같은지 확인 (헤더 제외)
        for label, path in (
            ("trajectory.csv", self.trajectory_path),
            ("cmd_vel.csv", self.cmd_vel_path),
            ("trajectory_cmd.csv", self.trajectory_cmd_path),
        ):
            if not path.exists():
                warnings.append(f"{label} 없음")
                continue
            with path.open("r", encoding="utf-8") as f:
                n_rows = max(sum(1 for _ in f) - 1, 0)
            if n_rows != n_frames:
                warnings.append(
                    f"{label} 행({n_rows}) ≠ 프레임({n_frames})")

        if self._odom_count == 0 and not self.args.no_odom:
            warnings.append("odom 수신 0건")
        if self.args.no_odom:
            warnings.append("--no-odom 강행: 학습에 쓸 수 없는 데이터")
        stop_limit = self.args.stop_threshold * 3.0
        if last3 > stop_limit:
            warnings.append(
                f"마지막 3초 이동거리 {last3:.4f}m > "
                f"stop_threshold*3 ({stop_limit:.4f}m) — 정지 유지 실패")
        if total_dist < 0.3:
            warnings.append(f"총 이동거리 {total_dist:.3f}m < 0.3m")
        if intervals_ms and max_ms > target_ms * 3.0:
            warnings.append(
                f"프레임 간격 최대 {max_ms:.0f}ms > 목표×3 ({target_ms * 3:.0f}ms)")

        ok = len(warnings) == 0
        return {
            "episode": self.episode_dir.name,
            "num_frames": n_frames,
            "num_csv_rows": self.rows_written,
            "duration_sec": round(duration, 3),
            "frame_interval_median_ms": round(median_ms, 1),
            "frame_interval_max_ms": round(max_ms, 1),
            "frame_interval_target_ms": round(target_ms, 1),
            "odom_messages": self._odom_count,
            "total_distance_m": round(total_dist, 4),
            "last_3sec_distance_m": round(last3, 4),
            "hold_completed": self._hold_completed,
            "ok": ok,
            "warnings": warnings,
        }

    def _print_summary(self, summary: dict[str, Any]) -> None:
        """검수 요약을 터미널에 크게 출력한다.

        Args:
            summary: _build_summary() 결과.
        """
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            f"■ {summary['episode']}",
            f"  {summary['num_frames']} 프레임 / {summary['duration_sec']:.1f}s"
            f"   (CSV {summary['num_csv_rows']} 행)",
            f"  프레임 간격  중앙값 {summary['frame_interval_median_ms']:.0f}ms"
            f"  최대 {summary['frame_interval_max_ms']:.0f}ms"
            f"   (목표 {summary['frame_interval_target_ms']:.0f}ms)",
            f"  odom 수신 {summary['odom_messages']}건",
            f"  총 이동거리 {summary['total_distance_m']:.3f}m"
            f"   마지막 3초 {summary['last_3sec_distance_m']:.4f}m",
            f"  정지 유지 완료: {summary['hold_completed']}",
        ]
        if summary["ok"]:
            lines.append("  ✔ 이상 없음")
        else:
            lines.append("  ✖ 경고 — 이 에피소드를 버릴지 판단할 것")
            for w in summary["warnings"]:
                lines.append(f"    · {w}")
        lines.append(
            "╚══════════════════════════════════════════════════════════╝")
        text = "\n".join(lines)
        # 로거 + stderr 동시 (검수는 눈에 띄게).
        self.log.info(text)
        sys.stderr.write(text + "\n")

    def write_meta(
        self,
        status: str,
        summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """meta.json 을 기록한다.

        Args:
            status: waiting / recording / finished / aborted_*.
            summary: 검수 요약 (종료 시).
        """
        meta: dict[str, Any] = {
            "dataset": "GoToObject-v1",
            "robot": "SerBot II",
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "object": self.args.object,
            "task": self.args.task,
            "instruction": self.instruction,
            "start_distance_m": self.args.start_distance,
            "hz": self.args.hz,
            "camera_topic": self.args.camera_topic,
            "odom_topic": self.args.odom_topic,
            "cmd_vel_topic": self.args.cmd_vel_topic,
            "action_source": self.args.action_source,
            "trajectory_source": (
                "none (--no-odom)" if self.args.no_odom
                else "wheel_odometry"),
            "coordinate_frame": (
                "REP-103 FLU (body->world). "
                "s01 에서 USE_CAMERA_OPTICAL_TO_FLU=False"),
            "hold_seconds": self.args.hold_seconds,
            "stop_threshold": self.args.stop_threshold,
            "note": self.args.note or "",
            "status": status,
            "safety": "subscribe/save only; no cmd_vel publish",
            "files": dict(EPISODE_FILES_DESC),
        }
        if summary is not None:
            meta["summary"] = summary
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def finish(self, abort_reason: Optional[str] = None) -> None:
        """기록을 끝내고 검수·meta 를 남긴다.

        Args:
            abort_reason: 카메라/치명 오류 사유. None 이면 정상 종료.
        """
        if getattr(self, "_finish_done", False):
            return
        self._finish_done = True
        self.finished = True

        if self.timer is not None:
            try:
                self.timer.cancel()
            except Exception:  # noqa: BLE001
                pass
            self.timer = None

        for f in (self.csv_trajectory, self.csv_cmd_vel,
                  self.csv_trajectory_cmd):
            if not f.closed:
                f.flush()
                f.close()

        # odom 미수신 등으로 프레임 0개면 빈 폴더를 남기지 않는다.
        n_frames = 0
        if self.frames_dir.exists():
            n_frames = len([
                p for p in self.frames_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ])
        if n_frames == 0:
            self.log.warn(
                f"프레임 0개 — 빈 에피소드 폴더를 삭제한다: {self.episode_dir}")
            try:
                shutil.rmtree(self.episode_dir)
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"빈 폴더 삭제 실패: {exc}")
            sys.stderr.write(
                f"[수집] 빈 에피소드 삭제됨 (사유={abort_reason or 'empty'}): "
                f"{self.episode_dir.name}\n")
            return

        summary = self._build_summary()
        status = (
            f"aborted_{abort_reason}" if abort_reason
            else ("finished_warn" if summary["warnings"] else "finished_ok"))
        self.write_meta(status=status, summary=summary)
        self._print_summary(summary)
        self.log.info(f"에피소드 종료 status={status} → {self.episode_dir}")

    def destroy_node(self) -> None:
        """자원 해제."""
        if self.joystick_reader is not None:
            self.joystick_reader.close()
        for f in (getattr(self, "csv_trajectory", None),
                  getattr(self, "csv_cmd_vel", None),
                  getattr(self, "csv_trajectory_cmd", None)):
            if f is not None and not f.closed:
                try:
                    f.flush()
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: sys.argv[1:] 형태.

    Returns:
        Namespace.
    """
    parser = argparse.ArgumentParser(
        description="TIC-VLA GoToObject 주행 데이터 수집 (저장만, cmd_vel 미발행)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", default=str(default_data_root()),
        help="데이터 루트. 기본은 $TICVLA_DATA_ROOT 또는 ~/TIC-VLA/data")
    parser.add_argument("--object", default="graybin",
                        help="목표 객체 이름 (폴더명에 들어감)")
    parser.add_argument(
        "--task", choices=list(VALID_TASKS), default="C",
        help="객체 배치 위치: L=좌 / C=정면 / R=우")
    parser.add_argument(
        "--instruction", default=None,
        help=f'자연어 지시문. 기본 "{DEFAULT_INSTRUCTION}"')
    parser.add_argument(
        "--start-distance", type=float, default=2.0,
        help="시작 시 객체까지 대략 거리(m). meta 기록용")
    parser.add_argument("--note", default="", help="meta.json note 필드")

    parser.add_argument(
        "--camera-topic", default="/camera/image_raw_fast")
    parser.add_argument("--odom-topic", default="/wheel/odom")
    parser.add_argument(
        "--cmd-vel-topic", default="/joystick_cmd_vel",
        help="action-source=cmd_vel 일 때 구독할 Twist")
    parser.add_argument(
        "--action-source", choices=("joystick", "cmd_vel"),
        default="cmd_vel")

    parser.add_argument("--hz", type=float, default=10.0,
                        help="저장 레이트. s01 DST_HZ=10 과 맞출 것")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--duration-sec", type=float, default=None,
        help="자동 종료 시간(초). 생략 시 Ctrl+C 까지")
    parser.add_argument(
        "--hold-seconds", type=float, default=8.0,
        help="목표 앞 정지 유지 시간(초)")
    parser.add_argument(
        "--stop-threshold", type=float, default=0.01,
        help="1초 창 이동거리가 이 값 미만이면 정지로 본다(m)")
    parser.add_argument(
        "--no-odom", action="store_true",
        help="odom 없이 강행 (학습에 쓸 수 없는 데이터)")

    # 조이스틱 (action-source=joystick)
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--left-x-axis", type=int, default=0)
    parser.add_argument("--left-y-axis", type=int, default=1)
    parser.add_argument("--right-x-axis", type=int, default=2)
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--max-linear-x", type=float, default=0.10)
    parser.add_argument("--max-linear-y", type=float, default=0.10)
    parser.add_argument("--max-angular-z", type=float, default=0.30)
    parser.add_argument("--invert-linear-x", action="store_true")
    parser.add_argument("--invert-linear-y", action="store_true")
    parser.add_argument("--invert-angular-z", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    """진입점. SIGINT/SIGTERM 시 정상 저장 후 종료.

    Returns:
        프로세스 종료 코드. odom 타임아웃·카메라 실패면 비0.
    """
    rclpy.init()
    cfg = parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    collector: Optional[GoToObjectCollector] = None
    try:
        collector = GoToObjectCollector(cfg)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[치명] 초기화 실패: {exc}\n")
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    requested = {"stop": False}

    def _on_signal(signum: int, frame: object) -> None:
        """종료 플래그만 세운다. 저장은 메인 루프가 한다."""
        requested["stop"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    exit_code = 0
    try:
        while rclpy.ok() and not requested["stop"] and not collector.finished:
            rclpy.spin_once(collector, timeout_sec=0.05)

        if requested["stop"] and not getattr(collector, "_finish_done", False):
            collector.log.info("Ctrl+C — 정상 저장 후 종료한다")
            collector.finish()

        fatal = getattr(collector, "_camera_fatal", None)
        if getattr(collector, "_finish_done", False):
            # finish 안에서 abort_reason 으로 끝난 경우
            meta = {}
            try:
                meta = json.loads(collector.meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
            status = str(meta.get("status", ""))
            if status.startswith("aborted_odom"):
                exit_code = 2
            elif status.startswith("aborted_"):
                exit_code = 3
        elif fatal:
            collector.finish(abort_reason=fatal)
            exit_code = 2 if fatal == "odom_timeout" else 3
    finally:
        try:
            if collector is not None and not getattr(
                    collector, "_finish_done", False):
                collector.finish()
            if collector is not None:
                collector.destroy_node()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"종료 처리 오류: {exc}\n")
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
