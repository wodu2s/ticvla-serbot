#!/usr/bin/env python3
"""joystick_errand.py — 심부름(Errand) 시나리오 다구간 수집기.

joystick_gotoobject.py 기반. 원본은 수정하지 않고 import 해서 재사용한다
(SerBot 백엔드, 데드존, 스틱→방향 변환, cmd_vel 스케일이 원본과 100% 동일).

기존 수집기와 다른 점
--------------------
- 기존: 목표 객체 1개 = 에피소드 1개. collect_gotoobject.py 를 subprocess 로 기동.
- 여기: 한 세션에서 구간(segment)이 A→B→C 로 이어지고, 구간마다 목표 객체와
  instruction 이 바뀐다. 구간 전환이 즉각적이어야 하고 세션 메타를 공유하므로
  subprocess 대신 이 프로세스가 직접 기록한다.

출력 구조 (구간마다 별도 에피소드 폴더. s01 이 에피소드 단위로 읽으므로 합치지 않는다)
--------------------------------------------------------------------------
    <data-root>/Errand_v0/raw/
      session_YYMMDD_HHMMSS_<scenario>/
        session.json          시나리오 메타 + 구간 목록 + 이벤트 로그
        seg00_<object>/
          rgb/rgb_00000.jpg …
          trajectory.csv
          episode.json
        seg01_<object>/…

이미지 전처리 (확정 사항)
------------------------
1280x720 → 448x448 squeeze resize. 종횡비 무시, 크롭 없음.
근거: InternVL dynamic_preprocess 는 max_num=1 이면 후보 격자가 [(1,1)] 뿐이라
      image.resize((448,448)) 한 줄만 실효하고 crop 박스는 (0,0,448,448) 전체다.
      학습(policy_data.py)·추론(브리지) 모두 max_num=1 로 동일.
cv2 가 아니라 PIL 을 쓴다. 브리지·학습 경로가 PIL resize 이므로 리샘플링 커널을 맞춘다.
카메라 flip-method=2(180도)는 camera_node 파라미터 그대로 두고 여기서 손대지 않는다.

trajectory.csv (기존 스펙 그대로. 컬럼을 바꾸지 말 것)
-----------------------------------------------------
    time,x,y,z,qx,qy,qz,qw
/wheel/odom 의 pose.pose 를 변환 없이 기록. time 은 **구간 시작 기준 초**.
이미지 1장 = CSV 1행. 저장 10Hz.

s01 이 시간축을 행 인덱스로 다시 만든다는 점(rel_s = tick / SRC_HZ)에 주의.
time 컬럼은 참고용이고 정렬에는 쓰이지 않는다. 그래서 프레임 누락은 조용히
시간축을 왜곡시킨다 — 아래 안전장치가 누락을 허용하지 않고 즉시 중단하는 이유다.

파일명은 rgb_00000.jpg 형식이어야 한다. policy_data.py 의 get_numeric_key 가
stem 에 '_' 가 없으면 ValueError 를 던진다.

안전장치
--------
- 시작 시 /wheel/odom 미수신 → 에러 종료(10). robot_state=0 환각 데이터 방지.
- 시작 시 카메라 미수신/변환 실패 → 에러 종료(11).
- 기록 중 카메라가 새 프레임을 못 주면 정지이미지를 재사용하지 않고 중단(11).
  camera_node 는 발행 후 슬롯을 비우므로 "새 메시지 없음" = 캡처 정지다.
- 기록 중 odom 이 끊기면 마지막 pose 를 반복 기록하지 않고 중단(10).
- 종료 시 구간별 rgb 파일 수 == csv 행 수 검증. 불일치면 종료 코드 12.

종료 코드: 0 정상 / 10 odom / 11 카메라 / 12 검증 불일치 / 2 인자 오류(argparse)

실행
----
    python3 collect/joystick_errand.py --scenario fetch_and_dump \\
        --segments "redbin,bluebin" --start-distance 2.0

주의: 처음 실제 주행 테스트는 반드시 바퀴를 띄운 상태에서 하세요.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

# pygame 은 SDL 드라이버를 결정한 뒤에 import 해야 한다 (원본과 같은 처리).
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────

#: 데이터셋 이름. <data-root>/<이 이름>/raw/ 아래에 세션이 쌓인다.
DATASET_NAME_DEFAULT = "Errand_v0"

#: 저장 레이트. s01 의 SRC_HZ=10.0 과 맞춰야 한다.
SAVE_HZ = 10.0

#: 학습·추론이 쓰는 입력 한 변 (InternVL image_size).
IMAGE_SIZE = 448

#: s01 이 요구하는 컬럼. 순서·이름을 바꾸지 말 것.
TRAJECTORY_HEADER = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]

#: policy_data.get_numeric_key 가 stem.split('_', 1)[1] 을 float 로 읽는다.
IMAGE_PATTERN = "rgb_%05d.jpg"

#: 수기 이벤트 마킹 키. 나중에 "Agent 가 여기서 판단했어야 한다"의 근거가 된다.
MARK_KEYS: dict[str, str] = {
    "o": "obstacle_detected",
    "r": "replan",
    "n": "new_order",
}

#: 토픽 기본값 (collect_gotoobject.py 와 동일).
ODOM_TOPIC_DEFAULT = "/wheel/odom"
CAMERA_TOPIC_DEFAULT = "/camera/image_raw_fast"

#: 종료 코드.
EXIT_OK = 0
EXIT_ODOM = 10
EXIT_CAMERA = 11
EXIT_VERIFY = 12

#: 객체 이름 규칙 — 단일 ASCII 단어, 언더스코어 금지 (폴더명 파싱을 단순하게).
OBJECT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
#: 시나리오 이름은 폴더명 뒤쪽 전체를 차지하므로 언더스코어를 허용한다.
SCENARIO_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

#: joystick_gotoobject 모듈 핸들. import 시 SerBot 백엔드가 로드되므로
#: 인자 검증이 끝난 뒤에만 채운다.
JG: Any = None


def load_joystick_module() -> Any:
    """joystick_gotoobject.py 를 import 해서 그대로 재사용한다.

    원본을 수정하지 않는다. import 시점에 load_serbot_backend() 가 실행되어
    실제 하드웨어 백엔드를 잡으므로 --help/인자 오류 경로에서는 부르지 않는다.

    Returns:
        joystick_gotoobject 모듈.
    """
    global JG
    if JG is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import joystick_gotoobject as _jg  # noqa: PLC0415

        JG = _jg
    return JG


# ──────────────────────────────────────────────────────────────────────
# 터미널 키 입력 (수기 이벤트 마킹)
# ──────────────────────────────────────────────────────────────────────

class KeyReader:
    """블로킹 없이 stdin 한 글자를 읽는다.

    조이스틱은 SDL dummy 드라이버로 돌기 때문에 pygame KEYDOWN 이 오지 않는다.
    SSH 세션에서도 동작하도록 termios cbreak 모드를 직접 쓴다.
    setraw 가 아니라 setcbreak 를 쓰므로 Ctrl+C 는 그대로 동작한다.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._fd: Optional[int] = None
        self._saved: Any = None

    def __enter__(self) -> "KeyReader":
        if not sys.stdin.isatty():
            print("[키입력] stdin 이 TTY 가 아니다 → 수기 이벤트 마킹 비활성화")
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            print(f"[키입력] cbreak 설정 실패 → 마킹 비활성화: {exc!r}")
            self.enabled = False
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._fd is not None and self._saved is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001
                pass

    def poll(self) -> list[str]:
        """지금 대기 중인 키를 모두 읽어 소문자 리스트로 준다.

        Returns:
            눌린 키 문자 목록. 비활성화 상태면 빈 리스트.
        """
        if not self.enabled or self._fd is None:
            return []
        keys: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch.lower())
        return keys


# ──────────────────────────────────────────────────────────────────────
# 설정 / 구간 상태
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """실행 설정."""

    scenario: str
    segments: list[str]
    instructions: list[str]
    start_distance: float
    data_root: Path
    dataset_name: str
    hz: float
    jpeg_quality: int
    odom_topic: str
    camera_topic: str
    cmd_vel_topic: str
    task_default: str
    next_button: int
    sensor_timeout: float
    odom_stall_sec: float
    camera_stall_sec: float


@dataclass
class Segment:
    """구간 하나(= 에피소드 하나)의 기록 상태."""

    index: int
    object_name: str
    instruction: str
    task: str
    dir_path: Path
    t_start_session: float
    t0_mono: float
    frames: int = 0
    stall_ticks: int = 0
    status: str = "recording"
    t_end_session: Optional[float] = None
    events: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _fh: Any = None
    _writer: Any = None

    @property
    def rgb_dir(self) -> Path:
        return self.dir_path / "rgb"

    @property
    def csv_path(self) -> Path:
        return self.dir_path / "trajectory.csv"

    @property
    def json_path(self) -> Path:
        return self.dir_path / "episode.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    """json 을 임시 파일에 쓴 뒤 replace 한다 (Ctrl+C 중 손상 방지).

    Args:
        path: 최종 경로.
        payload: 직렬화할 dict.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def count_csv_rows(path: Path) -> int:
    """헤더를 제외한 CSV 행 수.

    Args:
        path: CSV 경로.

    Returns:
        데이터 행 수. 파일이 없으면 0.
    """
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


# ──────────────────────────────────────────────────────────────────────
# 기록 노드
# ──────────────────────────────────────────────────────────────────────

class ErrandRecorder(Node):
    """센서를 구독하고 10Hz 로 구간별 프레임·궤적을 기록하는 노드.

    콜백과 타이머가 하나의 SingleThreadedExecutor 에서 직렬로 돌기 때문에
    파일 쓰기는 항상 이 executor 스레드 하나에서만 일어난다. 조이스틱 루프
    (메인 스레드)는 요청만 큐에 넣고 절대 파일을 만지지 않는다.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__("errand_recorder")
        self.cfg = cfg
        self.log = self.get_logger()
        self.bridge = CvBridge()

        # ── 센서 스냅샷 ──
        self._sensor_lock = threading.Lock()
        self._image_msg: Optional[ImageMsg] = None
        self._image_seq = 0
        self._image_at = 0.0
        self._saved_image_seq = 0
        self._image_shape: Optional[tuple[int, int]] = None

        self._odom: tuple[float, ...] = (0.0,) * 6 + (1.0,)
        self._odom_seq = 0
        self._odom_at = 0.0

        # ── 세션 상태 (state_lock 으로 보호) ──
        self._state_lock = threading.RLock()
        self._commands: list[tuple[float, str]] = []
        self.session_dir: Optional[Path] = None
        self.session_id: str = ""
        self.session_t0: Optional[float] = None
        self.events: list[dict] = []
        self.segments: list[Segment] = []
        self.current: Optional[Segment] = None
        self.task: str = cfg.task_default
        self.status: str = "waiting"
        self.warnings: list[str] = []
        self.fatal: Optional[str] = None
        self.finished = threading.Event()

        self._cmd_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.create_subscription(
            ImageMsg, cfg.camera_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, cfg.odom_topic, self._on_odom, 10)
        self.create_timer(1.0 / cfg.hz, self._on_tick)

    # ── 구독 콜백 ────────────────────────────────────────────────
    def _on_image(self, msg: ImageMsg) -> None:
        """최신 카메라 프레임을 보관한다.

        camera_node 는 발행 후 슬롯을 비우므로 같은 프레임이 두 번 오지 않는다.
        따라서 seq 가 늘지 않으면 캡처가 멈춘 것이다.

        Args:
            msg: sensor_msgs/Image (bgr8).
        """
        with self._sensor_lock:
            self._image_msg = msg
            self._image_seq += 1
            self._image_at = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        """휠 오도메트리를 변환 없이 그대로 보관한다.

        Args:
            msg: nav_msgs/Odometry.
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._sensor_lock:
            self._odom = (
                float(p.x), float(p.y), float(p.z),
                float(q.x), float(q.y), float(q.z), float(q.w),
            )
            self._odom_seq += 1
            self._odom_at = time.monotonic()

    # ── 메인 스레드에서 호출하는 요청 API ─────────────────────────
    def publish_cmd_vel(self, lx: float, ly: float, az: float) -> None:
        """조이스틱 입력을 Twist 로 발행한다 (기존 스택과 동일한 토픽).

        Args:
            lx: linear.x, ly: linear.y, az: angular.z.
        """
        msg = Twist()
        msg.linear.x = float(lx)
        msg.linear.y = float(ly)
        msg.angular.z = float(az)
        try:
            self._cmd_pub.publish(msg)
        except Exception:  # noqa: BLE001
            pass

    def request(self, command: str) -> None:
        """구간 전환류 명령을 큐에 넣는다. 실제 처리는 다음 tick.

        Args:
            command: "start" | "next" | "stop".
        """
        with self._state_lock:
            self._commands.append((time.monotonic(), command))

    def set_task(self, task: str) -> None:
        """현재(또는 다음) 구간의 task 라벨을 바꾸고 이벤트로 남긴다.

        Args:
            task: "L" | "C" | "R".
        """
        with self._state_lock:
            if self.task == task:
                return
            self.task = task
            if self.current is not None:
                self.current.task = task
            self.add_event("task_set", task=task)

    def mark(self, kind: str) -> dict:
        """수기 이벤트를 지금 시각으로 기록한다.

        Args:
            kind: obstacle_detected | replan | new_order | emergency_stop.

        Returns:
            기록된 이벤트 dict.
        """
        with self._state_lock:
            return self.add_event(kind)

    # ── 이벤트 / 세션 파일 ───────────────────────────────────────
    def session_time(self, mono: Optional[float] = None) -> float:
        """세션 시작 기준 경과 초.

        Args:
            mono: time.monotonic() 값. None 이면 지금.

        Returns:
            경과 초. 세션 시작 전이면 0.0.
        """
        if self.session_t0 is None:
            return 0.0
        return (time.monotonic() if mono is None else mono) - self.session_t0

    def add_event(self, kind: str, mono: Optional[float] = None,
                  **extra: Any) -> dict:
        """events 에 항목을 추가한다. state_lock 을 잡은 상태로 부를 것.

        segment / object 는 요청 시점의 현재 구간에서 채운다. 수기 이벤트는
        어느 구간 어느 프레임에서 눌렸는지가 핵심이므로 seg_t 와 frame 도 남긴다.

        Args:
            kind: 이벤트 종류.
            mono: 이벤트 발생 시각(monotonic). None 이면 지금.
            **extra: 추가 필드.

        Returns:
            추가된 이벤트 dict.
        """
        now = time.monotonic() if mono is None else mono
        event: dict = {"t": round(self.session_time(now), 3), "type": kind}
        seg = self.current
        if seg is not None:
            event["segment"] = seg.index
            event["object"] = seg.object_name
            event["seg_t"] = round(max(now - seg.t0_mono, 0.0), 3)
            event["frame"] = seg.frames
        event.update(extra)
        self.events.append(event)
        if seg is not None:
            seg.events.append(event)
        self.write_session_json()
        return event

    def session_payload(self) -> dict:
        """session.json 내용을 만든다.

        Returns:
            직렬화 가능한 dict.
        """
        cfg = self.cfg
        with self._sensor_lock:
            shape = self._image_shape
        return {
            "dataset": cfg.dataset_name,
            "session_id": self.session_id,
            "scenario": cfg.scenario,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": self.status,
            "hz": cfg.hz,
            "start_distance_m": cfg.start_distance,
            "segment_plan": [
                {"index": i, "object": obj, "instruction": ins}
                for i, (obj, ins) in enumerate(zip(cfg.segments, cfg.instructions))
            ],
            "image": {
                "dir": "rgb",
                "pattern": IMAGE_PATTERN,
                "size": [IMAGE_SIZE, IMAGE_SIZE],
                "source_size": list(shape) if shape else None,
                "preprocess": (
                    "PIL Image.resize((448,448)) — squeeze resize, "
                    "종횡비 무시·크롭 없음. InternVL max_num=1 경로와 동일"
                ),
                "flip_method": 2,
                "jpeg_quality": cfg.jpeg_quality,
            },
            "trajectory": {
                "file": "trajectory.csv",
                "columns": TRAJECTORY_HEADER,
                "source": f"{cfg.odom_topic} pose.pose (변환 없음)",
                "time_origin": "segment_start",
            },
            "topics": {
                "camera": cfg.camera_topic,
                "odom": cfg.odom_topic,
                "cmd_vel_out": cfg.cmd_vel_topic,
            },
            "segments": [self.segment_summary(s) for s in self.segments],
            "events": list(self.events),
            "warnings": list(self.warnings),
            "pipeline_note": (
                "s01 은 에피소드 폴더에서 instruction 을 instruction.txt 또는 "
                "meta.json 에서 찾는다. 여기 instruction 은 episode.json 에 있다."
            ),
        }

    def segment_summary(self, seg: Segment) -> dict:
        """구간 요약(session.json 의 segments 항목).

        Args:
            seg: 대상 구간.

        Returns:
            요약 dict.
        """
        rows = count_csv_rows(seg.csv_path)
        duration = None
        if seg.t_end_session is not None:
            duration = round(seg.t_end_session - seg.t_start_session, 3)
        return {
            "index": seg.index,
            "object": seg.object_name,
            "dir": seg.dir_path.name,
            "instruction": seg.instruction,
            "task": seg.task,
            "status": seg.status,
            "t_start": round(seg.t_start_session, 3),
            "t_end": None if seg.t_end_session is None else round(seg.t_end_session, 3),
            "duration_s": duration,
            "frames": seg.frames,
            "csv_rows": rows,
            "stall_ticks": seg.stall_ticks,
            "warnings": list(seg.warnings),
        }

    def write_session_json(self) -> None:
        """session.json 을 갱신한다. 세션 폴더가 없으면 아무것도 하지 않는다."""
        if self.session_dir is None:
            return
        try:
            write_json_atomic(
                self.session_dir / "session.json", self.session_payload())
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"session.json 저장 실패: {exc!r}")

    def write_episode_json(self, seg: Segment) -> None:
        """구간 폴더의 episode.json 을 갱신한다.

        Args:
            seg: 대상 구간.
        """
        cfg = self.cfg
        with self._sensor_lock:
            shape = self._image_shape
        payload = {
            "dataset": cfg.dataset_name,
            "session_id": self.session_id,
            "scenario": cfg.scenario,
            "segment_index": seg.index,
            "segment_dir": seg.dir_path.name,
            "segment_count": len(cfg.segments),
            "object": seg.object_name,
            "instruction": seg.instruction,
            "task": seg.task,
            "start_distance_m": cfg.start_distance if seg.index == 0 else None,
            "hz": cfg.hz,
            "status": seg.status,
            "num_frames": seg.frames,
            "num_csv_rows": count_csv_rows(seg.csv_path),
            "stall_ticks": seg.stall_ticks,
            "session_t_start": round(seg.t_start_session, 3),
            "session_t_end": (
                None if seg.t_end_session is None else round(seg.t_end_session, 3)),
            "duration_s": (
                None if seg.t_end_session is None
                else round(seg.t_end_session - seg.t_start_session, 3)),
            "image": {
                "dir": "rgb",
                "pattern": IMAGE_PATTERN,
                "size": [IMAGE_SIZE, IMAGE_SIZE],
                "source_size": list(shape) if shape else None,
                "preprocess": "PIL Image.resize((448,448)) squeeze, 크롭 없음",
                "flip_method": 2,
                "jpeg_quality": cfg.jpeg_quality,
            },
            "trajectory": {
                "file": "trajectory.csv",
                "columns": TRAJECTORY_HEADER,
                "source": f"{cfg.odom_topic} pose.pose (변환 없음)",
                "time_origin": "segment_start",
            },
            "events": list(seg.events),
            "warnings": list(seg.warnings),
        }
        try:
            write_json_atomic(seg.json_path, payload)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"episode.json 저장 실패: {exc!r}")

    # ── 세션 / 구간 생명주기 (executor 스레드에서만 실행) ──────────
    def _ensure_session_dir(self, mono: float) -> None:
        """첫 구간 시작 시점에 세션 폴더를 만든다.

        버튼을 한 번도 안 누르고 끝내면 빈 폴더가 남지 않도록 지연 생성한다.

        Args:
            mono: 세션 시작 시각(monotonic).
        """
        if self.session_dir is not None:
            return
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.session_id = f"session_{stamp}_{self.cfg.scenario}"
        raw_root = self.cfg.data_root / self.cfg.dataset_name / "raw"
        self.session_dir = raw_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.session_t0 = mono
        self.status = "recording"
        self.log.info(f"세션 폴더 생성: {self.session_dir}")

    def _open_segment(self, index: int, mono: float) -> None:
        """구간 폴더를 만들고 CSV 를 열어 기록을 시작한다.

        Args:
            index: 구간 번호(0-based).
            mono: 구간 시작 시각(monotonic).
        """
        cfg = self.cfg
        obj = cfg.segments[index]
        assert self.session_dir is not None
        seg_dir = self.session_dir / f"seg{index:02d}_{obj}"
        seg_dir.mkdir(parents=True, exist_ok=False)
        (seg_dir / "rgb").mkdir()

        seg = Segment(
            index=index,
            object_name=obj,
            instruction=cfg.instructions[index],
            task=self.task,
            dir_path=seg_dir,
            t_start_session=self.session_time(mono),
            t0_mono=mono,
        )
        seg._fh = seg.csv_path.open("w", encoding="utf-8", newline="")
        seg._writer = csv.writer(seg._fh)
        seg._writer.writerow(TRAJECTORY_HEADER)
        seg._fh.flush()

        self.segments.append(seg)
        self.current = seg
        self.write_episode_json(seg)
        self.add_event("segment_start", mono=mono,
                       segment=index, object=obj)
        self.log.info(
            f"구간 시작 seg{index:02d} object={obj} task={seg.task} "
            f"instruction={seg.instruction!r}")

    def _close_segment(self, mono: float, status: str) -> None:
        """현재 구간을 닫고 검증·메타를 남긴다.

        Args:
            mono: 종료 시각(monotonic).
            status: "done" 또는 "aborted_*".
        """
        seg = self.current
        if seg is None:
            return
        seg.t_end_session = self.session_time(mono)
        seg.status = status
        if seg._fh is not None:
            try:
                seg._fh.flush()
                os.fsync(seg._fh.fileno())
                seg._fh.close()
            except Exception:  # noqa: BLE001
                pass
            seg._fh = None
            seg._writer = None

        for warn in self.verify_segment(seg):
            seg.warnings.append(warn)
        self.add_event("segment_end", mono=mono,
                       segment=seg.index, object=seg.object_name)
        self.current = None
        self.write_episode_json(seg)
        self.log.info(
            f"구간 종료 seg{seg.index:02d} frames={seg.frames} "
            f"rows={count_csv_rows(seg.csv_path)} status={status}")

    def verify_segment(self, seg: Segment) -> list[str]:
        """rgb 파일 수 == csv 행 수 등을 검증한다.

        Args:
            seg: 대상 구간.

        Returns:
            경고 문자열 목록. 비어 있으면 정상.
        """
        warns: list[str] = []
        files = sorted(seg.rgb_dir.glob("rgb_*.jpg"))
        rows = count_csv_rows(seg.csv_path)
        if len(files) != rows:
            warns.append(f"rgb 파일 {len(files)}장 ≠ csv 행 {rows}개")
        if len(files) != seg.frames:
            warns.append(f"rgb 파일 {len(files)}장 ≠ 기록 카운터 {seg.frames}")
        if rows == 0:
            warns.append("행이 0개 — 학습에 쓸 수 없다")
        if files:
            expected = [seg.rgb_dir / (IMAGE_PATTERN % i) for i in range(len(files))]
            if files != expected:
                warns.append("rgb 파일 번호가 0..N-1 연속이 아니다")
        if seg.stall_ticks:
            warns.append(
                f"카메라 미수신 tick {seg.stall_ticks}회 — s01 은 시간축을 "
                "행 인덱스로 재생성하므로 그만큼 시간이 압축된다")
        return warns

    def _drain_commands(self, now: float) -> None:
        """큐에 쌓인 구간 명령을 처리한다.

        Args:
            now: 현재 tick 시각(monotonic).
        """
        with self._state_lock:
            pending = self._commands
            self._commands = []
            for mono, command in pending:
                if command == "start":
                    self._cmd_start(mono)
                elif command == "next":
                    self._cmd_next(mono)
                elif command == "stop":
                    self._cmd_stop(mono, "done")

    def _cmd_start(self, mono: float) -> None:
        """세션 시작 = 0번 구간 기록 시작.

        Args:
            mono: 요청 시각(monotonic).
        """
        if self.current is not None:
            self.log.warn("이미 기록 중이다. 구간 전환은 전환 버튼을 쓸 것.")
            return
        if self.segments:
            self.log.warn("이미 종료된 세션이다. 새로 시작하려면 스크립트를 다시 실행할 것.")
            return
        try:
            self._ensure_session_dir(mono)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"세션 폴더 생성 실패: {exc!r}")
            return
        self.add_event("session_start", mono=mono, scenario=self.cfg.scenario)
        self._open_segment(0, mono)

    def _cmd_next(self, mono: float) -> None:
        """현재 구간 종료 + 다음 구간 시작.

        Args:
            mono: 요청 시각(monotonic).
        """
        seg = self.current
        if seg is None:
            self.log.warn("기록 중이 아니다. 먼저 시작 버튼을 누를 것.")
            return
        nxt = seg.index + 1
        if nxt >= len(self.cfg.segments):
            self.log.warn(
                f"마지막 구간(seg{seg.index:02d})이다. 세션을 끝내려면 정지 버튼을 쓸 것.")
            return
        self._close_segment(mono, "done")
        self._open_segment(nxt, mono)

    def _cmd_stop(self, mono: float, status: str) -> None:
        """현재 구간 종료 + 세션 마감.

        Args:
            mono: 요청 시각(monotonic).
            status: 구간 종료 상태.
        """
        if self.current is None and self.status in ("completed", "waiting"):
            self.log.warn("기록 중이 아니다.")
            return
        self._close_segment(mono, status)
        done = len(self.segments)
        planned = len(self.cfg.segments)
        if status != "done":
            self.status = status
        elif done < planned:
            self.status = "completed_partial"
            self.warnings.append(f"계획 {planned}구간 중 {done}구간만 기록됨")
        else:
            self.status = "completed"
        self.add_event("session_end", mono=mono, status=self.status)
        self.write_session_json()
        self.finished.set()

    def stop_from_main(self, status: str = "done") -> None:
        """메인 스레드(Ctrl+C 등)에서 세션을 마감한다.

        executor 가 아직 살아 있으면 큐를 통해, 이미 멈췄으면 직접 처리한다.

        Args:
            status: 구간 종료 상태.
        """
        with self._state_lock:
            if self.current is None and self.status != "recording":
                return
        if status == "done":
            self.request("stop")
            if self.finished.wait(timeout=3.0):
                return
        with self._state_lock:
            self._cmd_stop(time.monotonic(), status)

    def _fatal(self, reason: str, message: str) -> None:
        """치명 오류 — 정지이미지/정지 pose 를 기록하지 않고 세션을 중단한다.

        Args:
            reason: "camera" 또는 "odom".
            message: 사람이 읽을 사유.
        """
        if self.fatal is not None:
            return
        self.fatal = reason
        self.log.error(f"치명 오류({reason}) — 기록 중단: {message}")
        with self._state_lock:
            self.warnings.append(f"{reason}: {message}")
            self._cmd_stop(time.monotonic(), f"aborted_{reason}")

    # ── 10Hz tick ────────────────────────────────────────────────
    def _on_tick(self) -> None:
        """구간 명령 처리 + 프레임 1장 / CSV 1행 기록."""
        now = time.monotonic()
        self._drain_commands(now)

        with self._state_lock:
            seg = self.current
            if seg is None or self.fatal is not None:
                return

        with self._sensor_lock:
            odom = self._odom
            odom_at = self._odom_at
            image_seq = self._image_seq
            image_msg = self._image_msg
            image_at = self._image_at

        # odom 이 끊기면 마지막 pose 를 반복 기록하지 않는다.
        if odom_at <= 0.0:
            self._fatal("odom", f"{self.cfg.odom_topic} 수신 이력 없음")
            return
        if (now - odom_at) > self.cfg.odom_stall_sec:
            self._fatal(
                "odom",
                f"{self.cfg.odom_topic} 수신 끊김 "
                f"({now - odom_at:.2f}s > {self.cfg.odom_stall_sec:.2f}s)")
            return

        # 새 프레임이 없으면 직전 프레임을 다시 쓰지 않고 tick 을 버린다.
        if image_msg is None or image_seq == self._saved_image_seq:
            seg.stall_ticks += 1
            if image_at <= 0.0:
                self._fatal("camera", f"{self.cfg.camera_topic} 수신 이력 없음")
            elif (now - image_at) > self.cfg.camera_stall_sec:
                self._fatal(
                    "camera",
                    f"{self.cfg.camera_topic} 새 프레임 없음 "
                    f"({now - image_at:.2f}s > {self.cfg.camera_stall_sec:.2f}s)")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self._fatal("camera", f"cv_bridge 변환 실패: {exc!r}")
            return

        name = IMAGE_PATTERN % seg.frames
        path = seg.rgb_dir / name
        try:
            # 1280x720 → 448x448 squeeze resize. 종횡비 무시, 크롭 없음.
            # resample 인자를 넘기지 않는 것이 의도다 — 브리지·학습이 쓰는
            # image.resize((w, h)) 호출과 정확히 같은 커널을 타게 한다.
            img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_448 = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_448.save(path, format="JPEG", quality=self.cfg.jpeg_quality)
        except Exception as exc:  # noqa: BLE001
            self._fatal("camera", f"이미지 저장 실패 {path}: {exc!r}")
            return

        with self._sensor_lock:
            self._saved_image_seq = image_seq
            self._image_shape = (int(frame.shape[1]), int(frame.shape[0]))

        seg._writer.writerow(
            [f"{now - seg.t0_mono:.6f}"] + [f"{v:.6f}" for v in odom])
        seg._fh.flush()
        seg.frames += 1

    # ── 사전 점검 ────────────────────────────────────────────────
    def wait_for_odom(self, timeout: float) -> tuple[bool, str]:
        """odom 발행을 기다린다.

        Args:
            timeout: 최대 대기 초.

        Returns:
            (성공 여부, 사유 문자열).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._sensor_lock:
                if self._odom_seq >= 2:
                    return True, f"{self._odom_seq}건 수신"
            time.sleep(0.05)
        with self._sensor_lock:
            got = self._odom_seq
        return False, f"{timeout:.1f}초 안에 2건 미달 (수신 {got}건)"

    def wait_for_camera(self, timeout: float) -> tuple[bool, str]:
        """카메라 발행과 bgr8 변환을 확인한다.

        연속 2건을 요구해 정지이미지(발행 자체가 멈춘 상태)를 걸러낸다.

        Args:
            timeout: 최대 대기 초.

        Returns:
            (성공 여부, 사유 문자열).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._sensor_lock:
                seq = self._image_seq
                msg = self._image_msg
            if seq >= 2 and msg is not None:
                try:
                    frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception as exc:  # noqa: BLE001
                    return False, f"cv_bridge bgr8 변환 실패: {exc!r}"
                h, w = int(frame.shape[0]), int(frame.shape[1])
                if w <= 0 or h <= 0:
                    return False, f"프레임 크기 이상: {w}x{h}"
                with self._sensor_lock:
                    self._image_shape = (w, h)
                return True, f"{seq}건 수신, {w}x{h} bgr8"
            time.sleep(0.05)
        with self._sensor_lock:
            got = self._image_seq
        return False, f"{timeout:.1f}초 안에 2건 미달 (수신 {got}건)"

# ──────────────────────────────────────────────────────────────────────
# 버튼 매핑
# ──────────────────────────────────────────────────────────────────────

def build_button_map(cfg: Config) -> tuple[dict[int, str], dict[int, str]]:
    """원본 매핑을 유지하고 구간 전환 버튼만 추가한다.

    Args:
        cfg: 실행 설정 (next_button 사용).

    Returns:
        ({버튼: 액션 키}, {버튼: 설명}).
    """
    jg = load_joystick_module()
    action = {
        jg.STOP_BUTTON: "estop",
        jg.TASK1_BUTTON: "task_L",
        jg.TASK2_BUTTON: "task_C",
        jg.TASK3_BUTTON: "task_R",
        jg.HORN_BUTTON: "horn",
        jg.COLLECT_START_BUTTON: "start",
        jg.COLLECT_STOP_BUTTON: "stop",
        cfg.next_button: "next",
    }
    desc = {
        jg.STOP_BUTTON: "긴급 정지 (모터만 정지, 기록은 계속 + 이벤트 기록)",
        jg.TASK1_BUTTON: "task L — 객체 좌측 (현재 구간 라벨)",
        jg.TASK2_BUTTON: "task C — 객체 정면 (현재 구간 라벨)",
        jg.TASK3_BUTTON: "task R — 객체 우측 (현재 구간 라벨)",
        jg.HORN_BUTTON: "경적 (placeholder, 원본 그대로)",
        jg.COLLECT_START_BUTTON: "세션 시작 — seg00 기록 시작",
        jg.COLLECT_STOP_BUTTON: "세션 종료 — 현재 구간 종료 + 세션 마감",
        cfg.next_button: "★ 구간 전환 — 현재 구간 종료 + 다음 구간 시작 (신규)",
    }
    return action, desc


def reserved_buttons() -> dict[int, str]:
    """원본이 이미 쓰는 버튼 목록.

    Returns:
        {버튼 번호: 원본 용도}.
    """
    jg = load_joystick_module()
    return {
        jg.STOP_BUTTON: "긴급정지",
        jg.AUTO_TOGGLE_BUTTON: "자율주행토글(원본에서도 미연결)",
        jg.TASK1_BUTTON: "task L",
        jg.HORN_BUTTON: "경적",
        jg.TASK2_BUTTON: "task C",
        jg.TASK3_BUTTON: "task R",
        jg.COLLECT_START_BUTTON: "수집시작",
        jg.COLLECT_STOP_BUTTON: "수집멈춤",
    }


def print_button_table(cfg: Config, desc: dict[int, str]) -> None:
    """조작 표를 터미널에 출력한다.

    Args:
        cfg: 실행 설정.
        desc: 버튼 설명 맵.
    """
    print("")
    print("=" * 74)
    print("  조작 (원본 joystick_gotoobject.py 매핑 유지 + 구간 전환 추가)")
    print("=" * 74)
    print(f"  {'버튼':<6} {'기능':<62}")
    print("  " + "-" * 70)
    for btn in sorted(desc):
        mark = "★" if btn == cfg.next_button else " "
        print(f"  {mark}{btn:<5} {desc[btn]}")
    used = set(desc)
    free = [b for b in range(10) if b not in used]
    if free:
        print("  " + "-" * 70)
        print(f"  {'남음':<6} {', '.join(str(b) for b in free)} (미할당)")
    print("  " + "-" * 70)
    print("  키보드   " + "  ".join(f"'{k}'={v}" for k, v in MARK_KEYS.items()))
    print("  Ctrl+C   현재 구간 종료 + 세션 마감 후 종료")
    print("=" * 74)
    print("")


# ──────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────

def print_status(cfg: Config, rec: ErrandRecorder, joystick: Any,
                 xyz: tuple[float, float, float], notice: str) -> None:
    """수집 상태 패널.

    Args:
        cfg: 실행 설정.
        rec: 기록 노드.
        joystick: pygame Joystick.
        xyz: 데드존 적용된 (x, y, z) 스틱 값.
        notice: 최근 알림 한 줄.
    """
    jg = load_joystick_module()
    sep = "-" * 74
    os.system("cls" if os.name == "nt" else "clear")

    x_pos, y_pos, z_pos = xyz
    lx, ly, az = jg.joystick_to_cmd_vel(x_pos, y_pos, z_pos)

    print(sep)
    print(f"  TIC-VLA Errand 다구간 수집 | {joystick.get_name()}")
    print(f"  SerBOT: {'실제' if jg.SERBOT_AVAILABLE else '더미(Dummy)'} "
          f"| 백엔드 {jg.bot.name} | 시나리오 {cfg.scenario}")
    print(sep)

    with rec._state_lock:
        seg = rec.current
        segments = list(rec.segments)
        status = rec.status
        task = rec.task
        events = list(rec.events)[-5:]
        session_dir = rec.session_dir

    print("  [조종]")
    print(f"  스틱 x={x_pos:+.3f} y={y_pos:+.3f} z={z_pos:+.3f} "
          f"| 방향각 {jg.degree_now:6.1f}deg | 속도 {jg.speed:4.1f}/{jg.MAX_SPEED}")
    print(f"  cmd_vel vx={lx:+.3f} vy={ly:+.3f} wz={az:+.3f} → {cfg.cmd_vel_topic}")

    print()
    print("  [세션]")
    print(f"  상태     : {status}"
          + (f"  (치명: {rec.fatal})" if rec.fatal else ""))
    print(f"  폴더     : {session_dir if session_dir else '(첫 구간 시작 시 생성)'}")
    print("  구간계획 : " + " → ".join(
        f"seg{i:02d}:{o}" for i, o in enumerate(cfg.segments)))
    print(f"  task     : {task} ({jg.TASK_LABELS.get(task, '')})")

    print()
    print("  [구간]")
    print(f"  {'seg':<6}{'object':<12}{'상태':<12}{'프레임':>8}{'행':>8}"
          f"{'초':>8}{'stall':>7}")
    for s in segments:
        rows = count_csv_rows(s.csv_path)
        dur = ((s.t_end_session if s.t_end_session is not None
                else rec.session_time()) - s.t_start_session)
        flag = "*" if s is seg else " "
        print(f"  {flag}{s.index:<5}{s.object_name:<12}{s.status:<12}"
              f"{s.frames:>8}{rows:>8}{dur:>8.1f}{s.stall_ticks:>7}")
    if not segments:
        print("  (대기 중 — 시작 버튼을 누르세요)")

    if events:
        print()
        print("  [최근 이벤트]")
        for e in events:
            extra = f" seg{e['segment']:02d}" if "segment" in e else ""
            frame = f" frame={e['frame']}" if "frame" in e else ""
            print(f"  t={e['t']:>8.2f}s  {e['type']:<20}{extra}{frame}")

    print(sep)
    keys = " ".join(f"{k}={v}" for k, v in MARK_KEYS.items())
    print(f"  {cfg.next_button}=구간전환  8=시작  9=종료  0=긴급정지  "
          f"3/6/7=task L/C/R  | 키보드 {keys}")
    if notice:
        print(f"  알림: {notice}")


def print_final_report(cfg: Config, rec: ErrandRecorder) -> int:
    """종료 요약을 출력하고 종료 코드를 정한다.

    Args:
        cfg: 실행 설정.
        rec: 기록 노드.

    Returns:
        프로세스 종료 코드.
    """
    print("")
    print("=" * 74)
    print("  수집 요약")
    print("=" * 74)
    if rec.session_dir is None:
        print("  기록된 세션이 없다 (시작 버튼을 누르지 않았다).")
        print("=" * 74)
        return EXIT_OK

    print(f"  세션 : {rec.session_dir}")
    print(f"  상태 : {rec.status}")
    print(f"  {'seg':<5}{'object':<12}{'프레임':>8}{'csv행':>8}{'초':>8}"
          f"{'stall':>7}  판정")
    bad = False
    for seg in rec.segments:
        rows = count_csv_rows(seg.csv_path)
        files = len(list(seg.rgb_dir.glob("rgb_*.jpg")))
        dur = 0.0 if seg.t_end_session is None else (
            seg.t_end_session - seg.t_start_session)
        verdict = "OK" if not seg.warnings else "경고"
        if seg.warnings:
            bad = True
        print(f"  {seg.index:<5}{seg.object_name:<12}{files:>8}{rows:>8}"
              f"{dur:>8.1f}{seg.stall_ticks:>7}  {verdict}")
        for warn in seg.warnings:
            print(f"        ! {warn}")

    marks = [e for e in rec.events if e["type"] in MARK_KEYS.values()]
    if marks:
        print("  " + "-" * 70)
        print(f"  수기 이벤트 {len(marks)}건")
        for e in marks:
            print(f"    t={e['t']:>8.2f}s  {e['type']:<20}"
                  f"seg{e.get('segment', -1):02d} frame={e.get('frame')}")

    if rec.warnings:
        print("  " + "-" * 70)
        for warn in rec.warnings:
            print(f"  ! {warn}")
    print("=" * 74)
    print("  이 데이터를 s01 에 넣기 전에: instruction 은 episode.json 에 있다.")
    print("=" * 74)

    if rec.fatal == "odom":
        return EXIT_ODOM
    if rec.fatal == "camera":
        return EXIT_CAMERA
    if bad:
        return EXIT_VERIFY
    return EXIT_OK

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def default_data_root() -> str:
    """--data-root 기본값.

    joystick_gotoobject.default_data_root() 와 같은 규칙이지만, 인자 검증
    단계에서 SerBot 백엔드를 잡지 않기 위해 여기 다시 적는다.

    Returns:
        TICVLA_DATA_ROOT 또는 ~/TIC-VLA/data.
    """
    env = os.environ.get("TICVLA_DATA_ROOT", "").strip()
    if env:
        return str(Path(env).expanduser())
    return str(Path.home() / "TIC-VLA" / "data")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """인자를 파싱한다.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        파싱 결과.
    """
    parser = argparse.ArgumentParser(
        description="TIC-VLA 심부름 시나리오 다구간 수집기 (구간마다 에피소드 1개)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", required=True,
                        help="시나리오 이름. 세션 폴더명에 들어간다")
    parser.add_argument("--segments", required=True,
                        help='순서대로 방문할 목표 객체. 예: "redbin,bluebin"')
    parser.add_argument("--start-distance", type=float, default=2.0,
                        help="첫 구간 시작 시 객체까지 대략 거리(m), 메타 기록용")
    parser.add_argument("--instruction-template",
                        default="Go to the {object}.",
                        help="구간별 instruction 템플릿. {object} 를 포함할 것")
    parser.add_argument("--data-root", default=default_data_root(),
                        help="데이터 루트. <root>/<dataset>/raw 아래에 세션이 쌓인다")
    parser.add_argument("--dataset-name", default=DATASET_NAME_DEFAULT,
                        help="데이터셋 폴더 이름")
    parser.add_argument("--hz", type=float, default=SAVE_HZ,
                        help="저장 레이트. s01 의 SRC_HZ 와 맞출 것")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="저장 JPEG 품질")
    parser.add_argument("--odom-topic", default=ODOM_TOPIC_DEFAULT,
                        help="궤적 원본 odom 토픽")
    parser.add_argument("--camera-topic", default=CAMERA_TOPIC_DEFAULT,
                        help="카메라 토픽 (flip-method 는 camera_node 쪽 설정)")
    parser.add_argument(
        "--cmd-vel-topic",
        default=os.environ.get("JOYSTICK_CMD_VEL_TOPIC", "/joystick_cmd_vel"),
        help="조이스틱 입력을 내보낼 Twist 토픽 (기존 스택 호환용)")
    parser.add_argument("--task-default", choices=["L", "C", "R"], default="C",
                        help="시작 시 선택돼 있는 task 라벨")
    parser.add_argument("--next-button", type=int, default=5,
                        help="구간 전환 버튼. 원본이 쓰지 않는 번호여야 한다")
    parser.add_argument("--sensor-timeout", type=float, default=5.0,
                        help="시작 시 odom/카메라 수신 대기 최대 초")
    parser.add_argument("--odom-stall-sec", type=float, default=0.5,
                        help="이 시간 동안 odom 이 안 오면 중단")
    parser.add_argument("--camera-stall-sec", type=float, default=0.5,
                        help="이 시간 동안 새 프레임이 안 오면 중단")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """인자를 검증해 Config 로 만든다.

    Args:
        args: parse_args 결과.

    Raises:
        SystemExit: 인자가 규칙을 위반하면 즉시 종료한다.

    Returns:
        검증된 Config.
    """
    scenario = args.scenario.strip()
    if not SCENARIO_RE.match(scenario):
        raise SystemExit(
            f"ERROR: --scenario 는 영문/숫자/-/_ 만 쓸 수 있다: {scenario!r}")

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    if not segments:
        raise SystemExit('ERROR: --segments 가 비었다. 예: --segments "redbin,bluebin"')
    for obj in segments:
        if "_" in obj:
            raise SystemExit(
                f"ERROR: 목표 객체에 언더스코어를 쓸 수 없다 (폴더명 파싱): {obj!r}")
        if not OBJECT_RE.match(obj):
            raise SystemExit(
                f"ERROR: 목표 객체는 영문으로 시작하는 단일 ASCII 단어여야 한다: {obj!r}")

    template = args.instruction_template
    if "{object}" not in template:
        raise SystemExit("ERROR: --instruction-template 에 {object} 가 없다")
    instructions = [template.format(object=obj) for obj in segments]

    if args.hz <= 0.0:
        raise SystemExit("ERROR: --hz 는 0보다 커야 한다")
    if args.hz != SAVE_HZ:
        print(f"[경고] --hz {args.hz} != s01 SRC_HZ {SAVE_HZ}. "
              "s01 상수도 같이 바꾸지 않으면 시간축이 틀어진다.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("ERROR: --jpeg-quality 는 1~100")

    # 여기서부터 SerBot 백엔드가 로드된다 (버튼 상수 확인 필요).
    reserved = reserved_buttons()
    if args.next_button in reserved:
        raise SystemExit(
            f"ERROR: --next-button {args.next_button} 은 원본이 이미 쓴다 "
            f"({reserved[args.next_button]}). 남는 번호: "
            f"{[b for b in range(10) if b not in reserved]}")

    return Config(
        scenario=scenario,
        segments=segments,
        instructions=instructions,
        start_distance=float(args.start_distance),
        data_root=Path(args.data_root).expanduser().resolve(),
        dataset_name=str(args.dataset_name),
        hz=float(args.hz),
        jpeg_quality=int(args.jpeg_quality),
        odom_topic=str(args.odom_topic),
        camera_topic=str(args.camera_topic),
        cmd_vel_topic=str(args.cmd_vel_topic),
        task_default=str(args.task_default),
        next_button=int(args.next_button),
        sensor_timeout=float(args.sensor_timeout),
        odom_stall_sec=float(args.odom_stall_sec),
        camera_stall_sec=float(args.camera_stall_sec),
    )


def preflight(rec: ErrandRecorder, cfg: Config) -> int:
    """odom·카메라 수신을 확인한다. 실패하면 종료 코드를 준다.

    경고만 하고 진행하지 않는다 — odom 없이 기록하면 robot_state 가 전부
    0 인 환각 데이터가 만들어진다.

    Args:
        rec: 기록 노드 (executor 가 이미 돌고 있어야 한다).
        cfg: 실행 설정.

    Returns:
        0 이면 통과, 그 외는 종료 코드.
    """
    print("")
    print("[점검] 센서 수신 확인 중…")

    ok, why = rec.wait_for_odom(cfg.sensor_timeout)
    print(f"[점검] odom   {cfg.odom_topic}: {'OK' if ok else '실패'} — {why}")
    if not ok:
        print("!" * 74)
        print("  odom 을 받지 못했다 — 수집을 시작하지 않는다.")
        print("  odometry_publisher 가 떠 있는지 확인할 것:")
        print("    ros2 topic hz " + cfg.odom_topic)
        print("  경고만 하고 진행하면 trajectory.csv 가 통째로 가짜가 된다.")
        print("!" * 74)
        return EXIT_ODOM

    ok, why = rec.wait_for_camera(cfg.sensor_timeout)
    print(f"[점검] camera {cfg.camera_topic}: {'OK' if ok else '실패'} — {why}")
    if not ok:
        print("!" * 74)
        print("  카메라 프레임을 받지 못했거나 변환에 실패했다 — 시작하지 않는다.")
        print("  serbot_camera(camera_node) 가 떠 있는지 확인할 것:")
        print("    ros2 topic hz " + cfg.camera_topic)
        print("!" * 74)
        return EXIT_CAMERA

    with rec._sensor_lock:
        shape = rec._image_shape
    if shape and shape != (1280, 720):
        print(f"[점검] 주의: 소스 해상도가 {shape[0]}x{shape[1]} 다. "
              f"448x448 squeeze resize 는 그대로 적용된다.")
    return EXIT_OK


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    """엔트리포인트.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        종료 코드.
    """
    args = parse_args(argv)
    cfg = build_config(args)
    jg = load_joystick_module()

    action_map, desc_map = build_button_map(cfg)

    print("")
    print(f"[설정] 시나리오 : {cfg.scenario}")
    print("[설정] 구간     : " + " → ".join(
        f"seg{i:02d}_{o}  ({ins!r})"
        for i, (o, ins) in enumerate(zip(cfg.segments, cfg.instructions))))
    print(f"[설정] 저장 루트: {cfg.data_root / cfg.dataset_name / 'raw'}")
    print(f"[설정] 저장     : {cfg.hz}Hz, {IMAGE_SIZE}x{IMAGE_SIZE} "
          f"squeeze resize (PIL), JPEG q={cfg.jpeg_quality}")
    print(f"[설정] 시작거리 : {cfg.start_distance:.1f} m")

    rclpy.init(args=None)
    rec = ErrandRecorder(cfg)
    executor = SingleThreadedExecutor()
    executor.add_node(rec)
    spin_thread = threading.Thread(
        target=executor.spin, name="errand-executor", daemon=True)
    spin_thread.start()

    exit_code = EXIT_OK
    try:
        exit_code = preflight(rec, cfg)
        if exit_code != EXIT_OK:
            return exit_code

        print_button_table(cfg, desc_map)
        joystick = jg.init_pygame_and_joystick()

        num_axes = joystick.get_numaxes()
        clock = pygame.time.Clock()
        debug_interval = max(1, int(jg.UPDATE_HZ / jg.DEBUG_HZ))
        frame_count = 0
        notice = "대기 중 — 시작 버튼을 누르세요"
        xyz = (0.0, 0.0, 0.0)

        with KeyReader() as keys:
            if not keys.enabled:
                rec.warnings.append("수기 이벤트 마킹 비활성화 (stdin 이 TTY 아님)")
            while True:
                try:
                    pygame.event.pump()
                except Exception as exc:  # noqa: BLE001
                    print("[pygame] event.pump 실패:", repr(exc))
                    raise

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type != pygame.JOYBUTTONDOWN:
                        continue
                    act = action_map.get(event.button)
                    if act is None:
                        continue
                    notice = handle_button(rec, act, event.button, desc_map)

                for ch in keys.poll():
                    kind = MARK_KEYS.get(ch)
                    if kind is None:
                        continue
                    ev = rec.mark(kind)
                    notice = (f"이벤트 {kind} @ t={ev['t']:.2f}s "
                              f"seg{ev.get('segment', -1):02d} "
                              f"frame={ev.get('frame')}")

                axes = [joystick.get_axis(i) for i in range(num_axes)]
                raw_x = axes[jg.LEFT_X_AXIS] if jg.LEFT_X_AXIS < num_axes else 0.0
                raw_y = axes[jg.LEFT_Y_AXIS] if jg.LEFT_Y_AXIS < num_axes else 0.0
                raw_z = axes[jg.RIGHT_X_AXIS] if jg.RIGHT_X_AXIS < num_axes else 0.0
                xyz = (jg.apply_deadzone(raw_x),
                       jg.apply_deadzone(raw_y),
                       jg.apply_deadzone(raw_z))

                try:
                    jg.move_serbot(*xyz)
                except Exception as exc:  # noqa: BLE001
                    print("[오류] move_serbot 실패:", repr(exc))
                    jg.stop_serbot()
                rec.publish_cmd_vel(*jg.joystick_to_cmd_vel(*xyz))

                if rec.fatal is not None:
                    jg.stop_serbot()
                    print(f"\n[중단] 센서 오류({rec.fatal})로 세션을 마감했다.")
                    break
                if rec.finished.is_set():
                    jg.stop_serbot()
                    print("\n[종료] 세션을 마감했다.")
                    break

                if frame_count % debug_interval == 0:
                    print_status(cfg, rec, joystick, xyz, notice)
                frame_count += 1
                clock.tick(jg.UPDATE_HZ)

    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C — 현재 구간을 마감하고 종료한다.")
    except Exception as exc:  # noqa: BLE001
        print("\n[오류] 예외 발생:", repr(exc))
        rec.warnings.append(f"예외: {exc!r}")
    finally:
        try:
            jg.stop_serbot()
        except Exception:  # noqa: BLE001
            pass
        try:
            rec.stop_from_main()
        except Exception as exc:  # noqa: BLE001
            print("[종료] 세션 마감 실패:", repr(exc))
        try:
            rec.publish_cmd_vel(0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            executor.shutdown()
            rec.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:  # noqa: BLE001
            pass

    report_code = print_final_report(cfg, rec)
    return report_code if exit_code == EXIT_OK else exit_code


def handle_button(rec: ErrandRecorder, act: str, button: int,
                  desc_map: dict[int, str]) -> str:
    """버튼 하나를 처리한다.

    Args:
        rec: 기록 노드.
        act: build_button_map 이 준 액션 키.
        button: 버튼 번호.
        desc_map: 설명 맵 (알림 문구용).

    Returns:
        화면에 띄울 알림 문구.
    """
    jg = load_joystick_module()
    if act == "estop":
        jg.stop_serbot()
        rec.mark("emergency_stop")
        return f"버튼 {button}: 긴급 정지 (이벤트 기록)"
    if act == "start":
        rec.request("start")
        return f"버튼 {button}: 세션 시작 요청"
    if act == "next":
        rec.request("next")
        return f"버튼 {button}: 구간 전환 요청"
    if act == "stop":
        rec.request("stop")
        return f"버튼 {button}: 세션 종료 요청"
    if act.startswith("task_"):
        task = act.split("_", 1)[1]
        rec.set_task(task)
        return f"버튼 {button}: task {task}"
    if act == "horn":
        return f"버튼 {button}: 경적 (placeholder)"
    return desc_map.get(button, f"버튼 {button}")


if __name__ == "__main__":
    sys.exit(main())
