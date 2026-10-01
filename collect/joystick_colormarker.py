#!/usr/bin/env python3
"""joystick_colormarker.py — 3색 마커 단일 방문 에피소드 수집기.

joystick_gotoobject.py 기반 (SerBot 백엔드·데드존·스틱→cmd_vel 변환을 원본
그대로 재사용). joystick_errand.py 에서 가져온 것 — rgb_00000.jpg 파일명
규칙, /wheel/odom 미수신 시 에러 종료, 프레임 누락 시 중단, 10Hz 유지.
버린 것 — 다구간(seg) 구조. 이 태스크는 한 에피소드 = 마커 1개 방문이다.

배경
----
회색 통 1종(GoToObject_v1) 대신 빨강·초록·파랑 고채도 마커 3개로 목표를
바꾼다. 사전 검증에서 VLM 인식은 3색을 정확히 구분하지만(3/3) 행동 계획이
목표 위치와 거의 무관하게 나온다는 문제가 확인됐다 — Action Expert
파인튜닝이 고쳐야 할 부분이고, 이 스크립트가 그 학습 데이터를 모은다.

수집 매트릭스 (총 72 에피소드)
------------------------------
마커 3개를 벽 앞에 일직선으로 두고 배치를 세 벌 돌아가며 쓴다 — 색을 한 칸씩
미는 순환이라, 3벌을 다 돌면 모든 색이 좌·중·우를 각각 한 번씩 거친다
(로봇 위치가 아니라 마커 배치 자체를 돌리는 방식으로 좌우 편향을 색과
분리한다):

    배치 1   좌=파랑   중=빨강   우=초록
    배치 2   좌=빨강   중=초록   우=파랑
    배치 3   좌=초록   중=파랑   우=빨강

각 배치 = 목표색 3 × 반복 8 = 24. 배치 3개 = 72.
로봇 시작 위치는 통제 축이 아니다 — 매 에피소드 대략 같은 자리에서 30~50cm
아무렇게나 옮긴다(어디 뒀는지 기록하지 않는다. 통제가 아니라 같은 그림이
반복되는 것을 막기 위함). 거리도 통제 축이 아니다 — 거리 선택 UI 없음,
종료 시 대략값만 수기 기록한다.

에피소드 진행 (사람이 조이스틱으로 직접 조종)
--------------------------------------------
접근(0.10 m/s 직행) → 마커 1.5m 부터 서서히 감속(★ 급정지 금지) →
마커 앞 0.8m 에서 정지 → 즉시 종료(9번). 정지 유지 구간 없음 — 도착 판정은
향후 라이다로 옮길 예정이라 지금은 운전자가 직접 판단해 9번을 누른다.
감속 램프는 소프트웨어가 강제하지 않는다(조이스틱 수동 조종이므로) —
운전자가 지켜야 할 규칙이다.

이 프로세스는 한 번 실행해서 여러 에피소드를 연달아 모은다 — 8=시작,
9=종료 가 매 에피소드마다 반복된다. joystick_errand.py 처럼 세션 하나
끝나면 프로세스가 종료되는 구조가 아니다.

조건 선택(배치/목표색)과 결과 입력(성공/실패/최종거리/비고)은 이 스크립트가
아니라 웹 대시보드(web_dashboard_node, /collect 페이지)가 ROS 서비스
(/collect/set_config, /collect/submit_result)로 넣는다 — 조이스틱은 주행과
8/9(시작/종료)만 맡는다. 대시보드가 죽어도 8/9 와 녹화는 항상 동작해야
하므로, 9번으로 종료해 결과 대기(awaiting_result) 상태가 된 뒤 대시보드가
없으면 비상 제출 버튼(EMERGENCY_SUBMIT_BUTTON)으로 "성공, 최종거리
기본값"으로 즉시 확정할 수 있다.

저장 구조
---------
    <root>/ep{NNNN}_L{layout}_{target_color}/
      rgb/rgb_00000.jpg, rgb_00001.jpg, …
      trajectory.csv     (time,x,y,z,qx,qy,qz,qw — /wheel/odom pose.pose 그대로)
      episode.json

폴더명 예: ep0001_L1_blue

이미지명 규칙은 GoToObject_v1 재발 방지용이다 — 그 데이터셋은
frames/000000.jpg + rgb 심볼릭 링크로 저장했다가 policy_data.py 의
get_numeric_key(stem 에 '_' 없으면 예외)에 걸려 학습 첫 배치에서 터졌다.
언더스코어가 더 들어가도 죽으므로 IMAGE_PATTERN 을 바꾸지 말 것.

이미지는 원본 해상도 그대로 저장한다 — 리사이즈를 하지 않는다.
cv2.resize 로 미리 줄이면 학습·추론(PIL resize) 경로와 커널이 달라진다.
리사이즈는 전처리 단계(policy_data.py)에 맡긴다.

지시문
------
    red   → "Go to the red cooler."
    green → "Go to the green cooler."
    blue  → "Go to the blue cooler."

명사는 반드시 cooler 다 — 사전 검증에서 cooler 는 12/12 로 목표가
critical object 로 지목됐고, marker 는 파랑에서 2회 모두 장애물로
분류돼 실패했다. 모델이 자발적으로 쓰는 명사라 내부 표현과 맞는 것으로
보인다. marker/point/box 로 바꾸지 말 것.

안전장치 (joystick_errand.py 와 동일한 철학)
--------------------------------------------
- 시작 시 /wheel/odom·카메라 미수신 → 에러 종료(10/11), 프로세스 전체 중단.
- 기록 중 odom 끊김/카메라 새 프레임 없음 → 정지 pose·이미지를 반복 기록
  하지 않고 그 에피소드만 중단(aborted_*) 하고 프로세스도 함께 끝낸다.
  s01 은 trajectory.csv 의 time 컬럼을 안 쓰고 행 인덱스로 시간을
  재구성하므로, 프레임 누락을 조용히 넘기면 시간축이 압축된다.
- 종료 시 rgb 파일 수 == csv 행 수 검증은 collect/verify_errand_session.py
  가 한다 (에피소드마다 자동 호출 — 세션/구간 하나짜리로 감싸서 그대로 부른다).

종료 코드: 0 정상 / 10 odom / 11 카메라 / 2 인자 오류(argparse)

금지 (지킨 것)
--------------
- third_party/TIC-VLA 는 읽기만 한다(수정 없음).
- data/GoToObject_v1 은 이 스크립트 어디에서도 경로로 등장하지 않는다.
  --root 기본값은 GoToColorMarker_v1/raw 로 완전히 분리돼 있다.

실행
----
    python3 collect/joystick_colormarker.py

    배치/목표색은 대시보드(/collect)에서 설정한다 — 여기엔 --layout 인자가
    없다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import rclpy
from collect_interfaces.srv import SetConfig, SubmitResult
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import String

# pygame 은 SDL 드라이버를 결정한 뒤에 import 해야 한다 (원본과 같은 처리).
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────

DATASET_NAME = "GoToColorMarker_v1"

#: 저장 레이트. s01 의 SRC_HZ=10.0 과 맞춰야 한다.
SAVE_HZ = 10.0

#: s01 이 요구하는 컬럼. 순서·이름을 바꾸지 말 것.
TRAJECTORY_HEADER = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]

#: policy_data.get_numeric_key 가 stem.split('_', 1)[1] 을 float 로 읽는다.
#: 접두 뒤에 언더스코어가 더 있으면 예외가 난다 — 형식을 바꾸지 말 것.
IMAGE_PATTERN = "rgb_%05d.jpg"

ODOM_TOPIC_DEFAULT = "/wheel/odom"
CAMERA_TOPIC_DEFAULT = "/camera/image_raw_fast"

EXIT_OK = 0
EXIT_ODOM = 10
EXIT_CAMERA = 11

TARGET_COLORS = ("red", "green", "blue")
LAYOUTS = (1, 2, 3)

#: 색 → 지시문. 명사는 반드시 cooler (사전 검증 근거는 모듈 docstring 참고).
INSTRUCTIONS: dict[str, str] = {
    "red": "Go to the red cooler.",
    "green": "Go to the green cooler.",
    "blue": "Go to the blue cooler.",
}

#: 화면 표시용 한글 색 이름 (내부 값은 계속 영문 "red"/"green"/"blue").
COLOR_KO: dict[str, str] = {"red": "빨강", "green": "초록", "blue": "파랑"}

#: 배치 → {슬롯: 색}. 색을 한 칸씩 미는 순환 — 배치 3벌을 돌면 모든 색이
#: 좌·중·우를 한 번씩 거친다.
LAYOUT_SIDE_COLOR: dict[int, dict[str, str]] = {
    1: {"left": "blue", "center": "red", "right": "green"},
    2: {"left": "red", "center": "green", "right": "blue"},
    3: {"left": "green", "center": "blue", "right": "red"},
}

VERIFY_SCRIPT = Path(__file__).resolve().parent / "verify_errand_session.py"

#: 상태 머신. 대시보드가 조회/변경하는 값은 전부 이 세 값 중 하나다.
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_AWAITING_RESULT = "awaiting_result"

#: 대시보드가 죽었을 때 결과 입력 비상구로 쓰는 버튼. 색 선택 버튼이 없어지며
#: 비게 된 구 red_button(1) 슬롯을 재사용한다 — 예약 버튼(0,2,3,4,6,7,8,9)과
#: 겹치지 않는다.
EMERGENCY_SUBMIT_BUTTON = 1

#: 비상 제출 시 쓰는 최종거리 기본값. "0.8m 에서 정지"라는 수집 절차의 정지
#: 기준값과 맞춘다 — 실제로 그 근처에서 멈췄을 것이라는 가정.
EMERGENCY_DEFAULT_DISTANCE = 0.8

#: /collect/status 발행 주기(초). 2Hz.
STATUS_PERIOD_SEC = 0.5

#: joystick_gotoobject 모듈 핸들. import 시 SerBot 백엔드가 로드되므로
#: 인자 검증이 끝난 뒤에만 채운다.
JG: Any = None


def load_joystick_module() -> Any:
    """joystick_gotoobject.py 를 import 해서 그대로 재사용한다 (원본 수정 없음).

    Returns:
        joystick_gotoobject 모듈.
    """
    global JG
    if JG is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import joystick_gotoobject as _jg  # noqa: PLC0415

        JG = _jg
    return JG


def reserved_buttons() -> dict[int, str]:
    """원본(joystick_gotoobject.py)이 이미 쓰는 버튼 목록.

    이 스크립트는 로봇 위치 선택 기능이 없어 TASK1/2/3_BUTTON 에 아무
    동작도 바인딩하지 않지만, 그 번호들은 여전히 원본 스크립트의 예약
    번호이므로 목표색 버튼이 같은 번호를 쓰지 않도록 계속 걸러낸다.

    Returns:
        {버튼 번호: 원본 용도}.
    """
    jg = load_joystick_module()
    return {
        jg.STOP_BUTTON: "긴급정지",
        jg.AUTO_TOGGLE_BUTTON: "자율주행토글(미연결, 예약됨)",
        jg.TASK1_BUTTON: "원본 TASK1(L) — 이 스크립트는 미사용",
        jg.HORN_BUTTON: "경적(미사용, 예약됨)",
        jg.TASK2_BUTTON: "원본 TASK2(C) — 이 스크립트는 미사용",
        jg.TASK3_BUTTON: "원본 TASK3(R) — 이 스크립트는 미사용",
        jg.COLLECT_START_BUTTON: "수집시작",
        jg.COLLECT_STOP_BUTTON: "수집종료",
    }


def layout_desc(layout: int, ko: bool = False) -> str:
    """배치의 좌/중/우 색 대응을 한 줄로 요약한다.

    Args:
        layout: 1 | 2 | 3.
        ko: True 면 한글 색 이름으로 표시한다.

    Returns:
        "좌: blue  중: red  우: green" 형식의 문자열.
    """
    sides = LAYOUT_SIDE_COLOR[layout]
    fmt = COLOR_KO.__getitem__ if ko else (lambda c: c)
    return (f"좌: {fmt(sides['left'])}   중: {fmt(sides['center'])}   "
            f"우: {fmt(sides['right'])}")


def derive_target_side(layout: int, target_color: str) -> str:
    """배치와 목표색만으로 target_side 를 조회한다.

    로봇 위치는 통제 축이 아니므로(에피소드마다 대략 옮기고 기록하지 않음),
    target_side 는 순수하게 "이 배치에서 이 색이 어느 슬롯에 있는가"다.
    평가에서 좌우 편향을 뽑는 축이라 반드시 정확해야 한다 — 같은 색인데
    좌/우로 갈리는 쌍의 성적 차이가 좌우 편향, 같은 target_side 인데 색만
    다른 쌍의 차이가 색 판별력이다.

    Args:
        layout: 1 | 2 | 3.
        target_color: "red" | "green" | "blue".

    Returns:
        "left" | "center" | "right".
    """
    sides = LAYOUT_SIDE_COLOR[layout]
    for side, color in sides.items():
        if color == target_color:
            return side
    raise ValueError(f"알 수 없는 target_color: {target_color!r}")


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


def next_episode_seq(root: Path) -> int:
    """root 전체 ep* 폴더를 스캔해 다음 일련번호를 돌려준다 (전역 연속).

    collect_gotoobject.py 의 next_episode_seq() 와 같은 규칙(배치별이 아니라
    전체 통틀어 연속 번호).

    Args:
        root: --root 경로.

    Returns:
        다음 번호 (없으면 1).
    """
    max_seq = 0
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir() or not path.name.startswith("ep"):
                continue
            head = path.name.split("_", 1)[0]
            digits = head[2:]
            if digits.isdigit():
                max_seq = max(max_seq, int(digits))
    return max_seq + 1


def default_root() -> str:
    """--root 기본값. TICVLA_DATA_ROOT 또는 ~/TIC-VLA/data 아래.

    Returns:
        <data-root>/GoToColorMarker_v1/raw.
    """
    env = os.environ.get("TICVLA_DATA_ROOT", "").strip()
    base = Path(env).expanduser() if env else (Path.home() / "TIC-VLA" / "data")
    return str(base / DATASET_NAME / "raw")


# ──────────────────────────────────────────────────────────────────────
# 설정 / 에피소드 상태
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """실행 설정. 배치/목표색은 여기 없다 — 대시보드가 서비스로 넣는
    런타임 상태(ColorMarkerRecorder.layout/target_color)다."""

    root: Path
    target_per_color: int
    hz: float
    jpeg_quality: int
    odom_topic: str
    camera_topic: str
    cmd_vel_topic: str
    sensor_timeout: float
    odom_stall_sec: float
    camera_stall_sec: float


@dataclass
class Episode:
    """에피소드 하나(마커 1개 방문)의 기록 상태."""

    seq: int
    layout: int
    target_color: str
    dir_path: Path
    t0_mono: float
    frames: int = 0
    stall_ticks: int = 0
    status: str = "recording"
    events: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _fh: Any = None
    _writer: Any = None

    @property
    def name(self) -> str:
        return self.dir_path.name

    @property
    def rgb_dir(self) -> Path:
        return self.dir_path / "rgb"

    @property
    def csv_path(self) -> Path:
        return self.dir_path / "trajectory.csv"

    @property
    def json_path(self) -> Path:
        return self.dir_path / "episode.json"


# ──────────────────────────────────────────────────────────────────────
# 기록 노드
# ──────────────────────────────────────────────────────────────────────

class ColorMarkerRecorder(Node):
    """센서를 구독하고 10Hz 로 에피소드 하나를 기록하는 노드.

    한 프로세스가 여러 에피소드를 연달아 기록한다 — 8(시작)/9(종료) 버튼이
    에피소드마다 반복된다. joystick_errand.py 의 세션(다구간) 개념은 없다.
    콜백과 타이머가 하나의 SingleThreadedExecutor 에서 직렬로 돌기 때문에
    파일 쓰기는 항상 이 executor 스레드 하나에서만 일어난다.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__("colormarker_recorder")
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
        #: 최근 2초간 프레임 수신 시각 — /collect/status 의 frame_hz 계산용.
        self._image_times: deque[float] = deque()

        self._odom: tuple[float, ...] = (0.0,) * 6 + (1.0,)
        self._odom_seq = 0
        self._odom_at = 0.0

        # ── 조건 선택 / 에피소드 상태 (state_lock 으로 보호) ──
        self._state_lock = threading.RLock()
        self._commands: list[tuple[float, str, str]] = []
        #: 대시보드(/collect/set_config)가 설정한다. 에피소드가 끝나도
        #: 유지된다 — 같은 배치/색을 여러 번(색당 8회) 반복 촬영하는 실제
        #: 작업 패턴에 맞춘 것이다.
        self.layout: Optional[int] = None
        self.target_color: Optional[str] = None
        #: idle / recording / awaiting_result. 대시보드와 콘솔이 그대로
        #: 읽어가는 값이라 문자열 상수(STATE_*)로 둔다.
        self.state: str = STATE_IDLE
        self.current: Optional[Episode] = None
        self.last_closed: Optional[Episode] = None
        self.fatal: Optional[str] = None
        self.episode_closed = threading.Event()
        #: 8번(시작)이 거부됐을 때 사유. (시각, 사유) — /collect/status 와
        #: 콘솔에 노출한다. 대시보드엔 시작 버튼이 없어 디버그 용도에 가깝다.
        self._last_start_rejection: Optional[tuple[float, str]] = None

        self._cmd_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.create_subscription(
            ImageMsg, cfg.camera_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, cfg.odom_topic, self._on_odom, 10)
        self.create_timer(1.0 / cfg.hz, self._on_tick)

        # ── 대시보드 연동: 서비스 서버 + 상태 퍼블리셔 ──
        self.create_service(SetConfig, "/collect/set_config", self._on_set_config)
        self.create_service(
            SubmitResult, "/collect/submit_result", self._on_submit_result)
        self._status_pub = self.create_publisher(String, "/collect/status", 10)
        self.create_timer(STATUS_PERIOD_SEC, self._publish_status)

    # ── 구독 콜백 ────────────────────────────────────────────────
    def _on_image(self, msg: ImageMsg) -> None:
        """최신 카메라 프레임을 보관한다.

        Args:
            msg: sensor_msgs/Image (bgr8).
        """
        now = time.monotonic()
        with self._sensor_lock:
            self._image_msg = msg
            self._image_seq += 1
            self._image_at = now
            self._image_times.append(now)
            while self._image_times and now - self._image_times[0] > 2.0:
                self._image_times.popleft()

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

    def odom_fresh(self, max_age: float = 1.0) -> bool:
        """odom 이 최근에 수신됐는지.

        Args:
            max_age: 이 시간(초) 안에 수신이 있어야 신선하다고 본다.

        Returns:
            신선하면 True.
        """
        with self._sensor_lock:
            at = self._odom_at
        return at > 0.0 and (time.monotonic() - at) < max_age

    def camera_fresh(self, max_age: float = 1.0) -> bool:
        """카메라 프레임이 최근에 수신됐는지.

        Args:
            max_age: 이 시간(초) 안에 수신이 있어야 신선하다고 본다.

        Returns:
            신선하면 True.
        """
        with self._sensor_lock:
            at = self._image_at
        return at > 0.0 and (time.monotonic() - at) < max_age

    def frame_hz(self) -> float:
        """최근 2초간 카메라 프레임 수신 Hz. /collect/status 표시용.

        Returns:
            Hz. 표본이 2개 미만이면 0.0.
        """
        with self._sensor_lock:
            times = list(self._image_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

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

    def request_start(self) -> None:
        """새 에피소드 시작을 큐에 넣는다. 실제 처리는 다음 tick."""
        with self._state_lock:
            self._commands.append((time.monotonic(), "start", ""))

    def request_stop(self, status: str = "done") -> None:
        """현재 에피소드 종료를 큐에 넣는다. 실제 처리는 다음 tick.

        Args:
            status: "done"(정상) 또는 "aborted_keyboard" 등.
        """
        with self._state_lock:
            self._commands.append((time.monotonic(), "stop", status))

    def mark_emergency(self) -> None:
        """긴급 정지 이벤트를 현재 에피소드에 기록한다 (기록은 계속됨)."""
        with self._state_lock:
            self._add_event(self.current, "emergency_stop")

    # ── 이벤트 ───────────────────────────────────────────────────
    def _add_event(self, ep: Optional[Episode], kind: str,
                   mono: Optional[float] = None) -> None:
        """이벤트를 에피소드 로그에 남긴다. state_lock 을 잡은 상태로 부를 것.

        Args:
            ep: 대상 에피소드. None 이면 아무 것도 하지 않는다.
            kind: 이벤트 종류.
            mono: 이벤트 시각(monotonic). None 이면 지금.
        """
        if ep is None:
            return
        now = time.monotonic() if mono is None else mono
        t = round(max(now - ep.t0_mono, 0.0), 3)
        ep.events.append({
            "t": t, "type": kind, "segment": 0, "seg_t": t, "frame": ep.frames,
        })

    # ── 에피소드 생명주기 (executor 스레드에서만 실행) ─────────────
    def _reject_start(self, reason: str) -> None:
        """8번(시작) 요청을 거부하고 사유를 남긴다.

        Args:
            reason: 사람이 읽을 거부 사유.
        """
        self.log.warn(f"시작 거부: {reason}")
        with self._state_lock:
            self._last_start_rejection = (time.monotonic(), reason)

    def _cmd_start(self, mono: float) -> None:
        """조건이 갖춰져 있으면 새 에피소드 폴더를 만들고 기록을 시작한다.

        배치/목표색은 더 이상 조이스틱이 아니라 /collect/set_config 로
        들어온다 — 여기서는 이미 설정돼 있는지와 state/센서 상태만 본다.

        Args:
            mono: 요청 시각(monotonic).
        """
        with self._state_lock:
            if self.state != STATE_IDLE:
                self._reject_start(f"state={self.state} (idle 이 아니다)")
                return
            if self.layout is None or self.target_color is None:
                self._reject_start(
                    "배치/목표색이 설정돼 있지 않다 — 대시보드(/collect)에서 설정할 것")
                return
            layout = self.layout
            target_color = self.target_color

        if not self.odom_fresh():
            self._reject_start(f"{self.cfg.odom_topic} 미수신")
            return
        if not self.camera_fresh():
            self._reject_start(f"{self.cfg.camera_topic} 미수신")
            return

        cfg = self.cfg
        seq = next_episode_seq(cfg.root)
        name = f"ep{seq:04d}_L{layout}_{target_color}"
        ep_dir = cfg.root / name
        while ep_dir.exists():
            seq += 1
            name = f"ep{seq:04d}_L{layout}_{target_color}"
            ep_dir = cfg.root / name
        ep_dir.mkdir(parents=True, exist_ok=False)
        (ep_dir / "rgb").mkdir()

        ep = Episode(
            seq=seq, layout=layout,
            target_color=target_color, dir_path=ep_dir, t0_mono=mono,
        )
        ep._fh = ep.csv_path.open("w", encoding="utf-8", newline="")
        ep._writer = csv.writer(ep._fh)
        ep._writer.writerow(TRAJECTORY_HEADER)
        ep._fh.flush()

        with self._state_lock:
            self.current = ep
            self.state = STATE_RECORDING
        self._add_event(ep, "episode_start")
        self._write_provisional_episode_json(ep)
        self.log.info(
            f"에피소드 시작 {name} target_side="
            f"{derive_target_side(layout, target_color)}")

    def _write_provisional_episode_json(self, ep: Episode) -> None:
        """기록 중/중단 시점의 잠정 episode.json (수기 필드는 아직 비움).

        Args:
            ep: 대상 에피소드.
        """
        payload = {
            "dataset": DATASET_NAME,
            "layout": ep.layout,
            "target_color": ep.target_color,
            "target_side": derive_target_side(ep.layout, ep.target_color),
            "instruction": INSTRUCTIONS[ep.target_color],
            "start_distance_approx": None,
            "success": None,
            "final_distance": None,
            "notes": "",
            "status": ep.status,
            "hz": self.cfg.hz,
            "num_frames": ep.frames,
            "num_csv_rows": count_csv_rows(ep.csv_path),
            "stall_ticks": ep.stall_ticks,
        }
        try:
            write_json_atomic(ep.json_path, payload)
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"episode.json 저장 실패: {exc!r}")

    def _cmd_stop(self, mono: float, status: str) -> None:
        """현재 에피소드를 닫는다. 수기 입력(성공/거리)은 메인 스레드가 채운다.

        Args:
            mono: 종료 시각(monotonic).
            status: "done" 또는 "aborted_*".
        """
        ep = self.current
        if ep is None:
            self.log.warn("기록 중이 아니다.")
            return
        ep.status = status
        if ep._fh is not None:
            try:
                ep._fh.flush()
                os.fsync(ep._fh.fileno())
                ep._fh.close()
            except Exception:  # noqa: BLE001
                pass
            ep._fh = None
            ep._writer = None

        for warn in self.verify_counts(ep):
            ep.warnings.append(warn)
        self._add_event(ep, "episode_end", mono=mono)
        self._write_provisional_episode_json(ep)
        self.log.info(
            f"에피소드 종료 {ep.name} frames={ep.frames} "
            f"rows={count_csv_rows(ep.csv_path)} status={status}")

        with self._state_lock:
            self.current = None
            self.last_closed = ep
            # 정상 종료(9번)만 결과 대기로 넘어간다. 치명 오류로 끊긴
            # 에피소드(aborted_*)는 애초에 결과를 물을 수 없으므로(운전자가
            # 판단할 상황이 아니었다) main() 이 즉시 success=False 로 확정한다
            # — 그쪽은 대시보드/서비스를 거치지 않는다.
            self.state = STATE_AWAITING_RESULT if status == "done" else STATE_IDLE
        self.episode_closed.set()

    def verify_counts(self, ep: Episode) -> list[str]:
        """rgb 파일 수 == csv 행 수 등 즉석 카운트 검증(경고만).

        본검수는 verify_errand_session.py 가 한다 — 여기는 화면/episode.json
        경고 필드를 바로 채우기 위한 빠른 확인이다.

        Args:
            ep: 대상 에피소드.

        Returns:
            경고 문자열 목록.
        """
        warns: list[str] = []
        files = sorted(ep.rgb_dir.glob("rgb_*.jpg"))
        rows = count_csv_rows(ep.csv_path)
        if len(files) != rows:
            warns.append(f"rgb 파일 {len(files)}장 ≠ csv 행 {rows}개")
        if rows == 0:
            warns.append("행이 0개 — 학습에 쓸 수 없다")
        if ep.stall_ticks:
            warns.append(f"카메라 미수신 tick {ep.stall_ticks}회")
        return warns

    def _fatal(self, reason: str, message: str) -> None:
        """치명 오류 — 정지이미지/정지 pose 를 기록하지 않고 에피소드를 중단한다.

        Args:
            reason: "camera" 또는 "odom".
            message: 사람이 읽을 사유.
        """
        if self.fatal is not None:
            return
        self.fatal = reason
        self.log.error(f"치명 오류({reason}) — 기록 중단: {message}")
        with self._state_lock:
            if self.current is not None:
                self.current.warnings.append(f"{reason}: {message}")
                self._cmd_stop(time.monotonic(), f"aborted_{reason}")

    def _drain_commands(self, now: float) -> None:
        """큐에 쌓인 시작/종료 명령을 처리한다.

        Args:
            now: 현재 tick 시각(monotonic).
        """
        with self._state_lock:
            pending = self._commands
            self._commands = []
            for mono, command, status in pending:
                if command == "start":
                    self._cmd_start(mono)
                elif command == "stop":
                    self._cmd_stop(mono, status or "done")

    # ── 10Hz tick ────────────────────────────────────────────────
    def _on_tick(self) -> None:
        """명령 처리 + (기록 중이면) 프레임 1장 / CSV 1행 기록."""
        now = time.monotonic()
        self._drain_commands(now)

        with self._state_lock:
            ep = self.current
            if ep is None or self.fatal is not None:
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
            ep.stall_ticks += 1
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

        name = IMAGE_PATTERN % ep.frames
        path = ep.rgb_dir / name
        try:
            # 원본 해상도 그대로 저장 — 리사이즈 없음(cv2.resize 도 PIL resize
            # 도 쓰지 않는다). 학습·추론 경로의 리사이즈 커널과 어긋나지
            # 않도록, 그리고 나중에 다른 해상도로 재처리할 여지를 남기도록.
            img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.save(path, format="JPEG", quality=self.cfg.jpeg_quality)
        except Exception as exc:  # noqa: BLE001
            self._fatal("camera", f"이미지 저장 실패 {path}: {exc!r}")
            return

        with self._sensor_lock:
            self._saved_image_seq = image_seq
            self._image_shape = (int(frame.shape[1]), int(frame.shape[0]))

        ep._writer.writerow(
            [f"{now - ep.t0_mono:.6f}"] + [f"{v:.6f}" for v in odom])
        ep._fh.flush()
        ep.frames += 1

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

    def stop_from_main(self, status: str) -> Optional[Episode]:
        """메인 스레드(Ctrl+C 등)에서 현재 에피소드를 강제 종료한다.

        executor 가 아직 살아 있으면 큐를 통해 처리되길 기다리고, 시간 안에
        안 되면 직접 처리한다(폴백).

        Args:
            status: 종료 상태 (예: "aborted_keyboard").

        Returns:
            닫힌 에피소드. 애초에 기록 중이 아니었으면 None.
        """
        with self._state_lock:
            if self.current is None:
                return None
        self.episode_closed.clear()
        self.request_stop(status)
        if self.episode_closed.wait(timeout=3.0):
            return self.last_closed
        with self._state_lock:
            self._cmd_stop(time.monotonic(), status)
        return self.last_closed

    # ── 대시보드 연동: 서비스 콜백 ─────────────────────────────────
    def _on_set_config(self, request: SetConfig.Request,
                       response: SetConfig.Response) -> SetConfig.Response:
        """/collect/set_config — idle 일 때만 배치/목표색을 받아들인다.

        Args:
            request: layout(1|2|3), target_color(red|green|blue).
            response: accepted/reason/target_side 를 채워 돌려준다.

        Returns:
            채워진 response.
        """
        with self._state_lock:
            if self.state != STATE_IDLE:
                response.accepted = False
                response.reason = (
                    f"state={self.state} — 녹화/결과대기 중에는 배치/목표색을 "
                    "바꿀 수 없다")
                response.target_side = ""
                return response
            if int(request.layout) not in LAYOUTS:
                response.accepted = False
                response.reason = f"layout 은 1|2|3 이어야 한다 (받은 값 {request.layout})"
                response.target_side = ""
                return response
            if request.target_color not in TARGET_COLORS:
                response.accepted = False
                response.reason = (
                    "target_color 는 red|green|blue 여야 한다 "
                    f"(받은 값 {request.target_color!r})")
                response.target_side = ""
                return response
            self.layout = int(request.layout)
            self.target_color = str(request.target_color)
            target_side = derive_target_side(self.layout, self.target_color)

        response.accepted = True
        response.reason = ""
        response.target_side = target_side
        self.log.info(
            f"/collect/set_config 수락: layout={self.layout} "
            f"target_color={self.target_color} target_side={target_side}")
        return response

    def _on_submit_result(self, request: SubmitResult.Request,
                          response: SubmitResult.Response) -> SubmitResult.Response:
        """/collect/submit_result — awaiting_result 일 때만 결과를 받는다.

        Args:
            request: success, final_distance, notes.
            response: accepted/reason 을 채워 돌려준다.

        Returns:
            채워진 response.
        """
        accepted, reason = self._do_submit_result(
            bool(request.success), float(request.final_distance), str(request.notes))
        response.accepted = accepted
        response.reason = reason
        return response

    def emergency_submit(self) -> tuple[bool, str]:
        """대시보드 없이 조이스틱 버튼으로 '성공'을 즉시 확정한다(비상구).

        Returns:
            (수락 여부, 사유/안내 문자열).
        """
        return self._do_submit_result(
            True, EMERGENCY_DEFAULT_DISTANCE, "dashboard-down fallback")

    def _do_submit_result(self, success: bool, final_distance: float,
                          notes: str) -> tuple[bool, str]:
        """결과를 확정한다 — /collect/submit_result 서비스와 비상 제출 버튼이 공유.

        awaiting_result 가 아니면 거부한다. 수락되면 낙관적으로 먼저
        state 를 idle 로 되돌린 뒤(중복 제출 방지) episode.json 을 쓰고
        verify_errand_session.py 를 돌린다 — 둘 다 몇 초 걸릴 수 있어 락 밖에서
        한다.

        Args:
            success: 성공 여부.
            final_distance: 최종 거리(m).
            notes: 비고.

        Returns:
            (수락 여부, 사유/안내 문자열).
        """
        with self._state_lock:
            if self.state != STATE_AWAITING_RESULT:
                return False, f"state={self.state} (awaiting_result 아님 — 이미 처리됐을 수 있다)"
            ep = self.last_closed
            if ep is None:
                self.state = STATE_IDLE
                return False, "내부 상태 불일치 — 제출할 에피소드가 없다"
            self.state = STATE_IDLE

        finalize_episode_json(ep, success, final_distance, None, notes, self.cfg)
        rc = run_verify(ep)
        verdict = "검수 OK" if rc == 0 else "재촬영 권장(위 사유 확인)"
        self.log.info(
            f"{ep.name} 결과 확정 success={success} "
            f"final_distance={final_distance} notes={notes!r} 검수={verdict}")
        return True, f"{ep.name} 저장 완료 — {verdict}"

    # ── 대시보드 연동: 상태 퍼블리셔 ───────────────────────────────
    def _publish_status(self) -> None:
        """/collect/status 에 배치/목표색/상태/카운터/센서 상태를 싣는다."""
        with self._state_lock:
            layout = self.layout
            target_color = self.target_color
            state = self.state
            last_ep = self.last_closed
            rejection = self._last_start_rejection
        target_side = (derive_target_side(layout, target_color)
                       if layout is not None and target_color is not None else None)
        counts = scan_counts_all(self.cfg.root)
        total = sum(sum(c.values()) for c in counts.values())
        payload: dict[str, Any] = {
            "layout": layout,
            "target_color": target_color,
            "target_side": target_side,
            "state": state,
            "counts": counts,
            "episode_count_total": total,
            "odom_ok": self.odom_fresh(),
            "camera_ok": self.camera_fresh(),
            "frame_hz": round(self.frame_hz(), 2),
            "last_episode_dir": str(last_ep.dir_path) if last_ep is not None else None,
            "last_start_rejection": rejection[1] if rejection is not None else None,
        }
        try:
            self._status_pub.publish(String(data=json.dumps(payload)))
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"/collect/status 발행 실패: {exc!r}")


# ──────────────────────────────────────────────────────────────────────
# 버튼 매핑
# ──────────────────────────────────────────────────────────────────────

def build_button_map(cfg: Config) -> tuple[dict[int, str], dict[int, str]]:
    """긴급정지·시작·종료·비상 제출 버튼을 매핑한다.

    배치/목표색 선택은 더 이상 조이스틱에 없다 — 대시보드가 ROS 서비스로
    설정한다. 로봇 위치 선택도 없다 — TASK1/2/3_BUTTON(3/6/7) 은 바인딩하지
    않는다.

    Args:
        cfg: 실행 설정.

    Returns:
        ({버튼: 액션 키}, {버튼: 설명}).
    """
    jg = load_joystick_module()
    reserved = reserved_buttons()
    if EMERGENCY_SUBMIT_BUTTON in reserved:
        raise SystemExit(
            f"내부 오류: EMERGENCY_SUBMIT_BUTTON={EMERGENCY_SUBMIT_BUTTON} 이 "
            f"예약 버튼과 겹친다 ({reserved[EMERGENCY_SUBMIT_BUTTON]}) — "
            "코드에서 상수를 바꿔야 한다.")
    action = {
        jg.STOP_BUTTON: "estop",
        jg.COLLECT_START_BUTTON: "start",
        jg.COLLECT_STOP_BUTTON: "stop",
        EMERGENCY_SUBMIT_BUTTON: "emergency_submit",
    }
    desc = {
        jg.STOP_BUTTON: "긴급 정지 (모터만 정지, 기록은 계속 + 이벤트 기록)",
        jg.COLLECT_START_BUTTON: "★ 에피소드 시작 (대시보드에서 배치/목표색 설정 후)",
        jg.COLLECT_STOP_BUTTON: "★ 에피소드 종료 (즉시 — 결과는 대시보드에서 입력)",
        EMERGENCY_SUBMIT_BUTTON: "대시보드 다운 비상구 — 결과대기 중 '성공'으로 즉시 제출",
    }
    return action, desc


def handle_button(rec: ColorMarkerRecorder, act: str, button: int) -> str:
    """버튼 하나를 처리한다.

    Args:
        rec: 기록 노드.
        act: build_button_map 이 준 액션 키.
        button: 버튼 번호.

    Returns:
        화면에 띄울 알림 문구.
    """
    jg = load_joystick_module()
    if act == "estop":
        jg.stop_serbot()
        rec.mark_emergency()
        return f"버튼 {button}: 긴급 정지 (이벤트 기록)"
    if act == "start":
        rec.request_start()
        return f"버튼 {button}: 에피소드 시작 요청"
    if act == "stop":
        rec.request_stop("done")
        return f"버튼 {button}: 에피소드 종료 요청"
    if act == "emergency_submit":
        ok, reason = rec.emergency_submit()
        return f"버튼 {button}: {'비상 성공 제출 완료' if ok else '비상 제출 무시'} — {reason}"
    return f"버튼 {button}"


def print_button_table(cfg: Config, desc: dict[int, str], joystick: Any) -> None:
    """조작 표를 터미널에 출력한다.

    Args:
        cfg: 실행 설정.
        desc: 버튼 설명 맵.
        joystick: pygame Joystick (실제 버튼 수 확인용).
    """
    print("")
    print("=" * 78)
    print("  조작 (원본 joystick_gotoobject.py 매핑 중 시작/종료/긴급정지만 유지"
          " + 목표색 버튼)")
    print("=" * 78)
    print(f"  {'버튼':<6} {'기능':<66}")
    print("  " + "-" * 74)
    for btn in sorted(desc):
        mark = "★" if "★" in desc[btn] else " "
        print(f"  {mark}{btn:<5} {desc[btn]}")
    used = set(desc)
    try:
        n_buttons = joystick.get_numbuttons()
    except Exception:  # noqa: BLE001
        n_buttons = 10
    free = [b for b in range(n_buttons) if b not in used]
    if free:
        print("  " + "-" * 74)
        print(f"  {'남음':<6} {', '.join(str(b) for b in free)} (미할당)")
    print("  " + "-" * 74)
    print("  Ctrl+C   기록 중이면 중단 처리 후 프로세스 종료")
    print("=" * 78)
    print("")


# ──────────────────────────────────────────────────────────────────────
# 누적 카운터
# ──────────────────────────────────────────────────────────────────────

EPISODE_DIR_RE = re.compile(r"^ep\d{4}_L([123])_(red|green|blue)$")


def scan_counts(root: Path, layout: int) -> dict[str, int]:
    """디스크에 이미 있는 성공 에피소드를 세어 색별 카운트를 만든다.

    실패(success != true) 에피소드는 세지 않는다 — 카운터는 "품질 좋은
    데모가 몇 개 모였는가"를 보기 위한 것이라, 실패한 시도는 다시 돌아야
    한다는 신호로 남겨야 한다.

    Args:
        root: --root 경로.
        layout: 현재 세션의 배치(1/2/3). 다른 배치 에피소드는 세지 않는다.

    Returns:
        {target_color: 개수}. 3칸 모두 채워서 돌려준다.
    """
    counts: dict[str, int] = {c: 0 for c in TARGET_COLORS}
    if not root.is_dir():
        return counts
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        m = EPISODE_DIR_RE.match(path.name)
        if not m or int(m.group(1)) != layout:
            continue
        ep_json = path / "episode.json"
        if not ep_json.is_file():
            continue
        try:
            data = json.loads(ep_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("success") is True:
            color = str(data.get("target_color"))
            if color in counts:
                counts[color] += 1
    return counts


def scan_counts_all(root: Path) -> dict[int, dict[str, int]]:
    """세 배치 전부의 색별 성공 카운트를 만든다 (대시보드 3×3 표용).

    Args:
        root: --root 경로.

    Returns:
        {layout: {target_color: 개수}}.
    """
    return {layout: scan_counts(root, layout) for layout in LAYOUTS}


def print_counts_table(counts: dict[str, int], cfg: Config,
                       layout: Optional[int]) -> None:
    """색별 누적 카운터(3칸)를 출력한다.

    Args:
        counts: scan_counts() 형식의 카운트.
        cfg: 실행 설정 (target_per_color 사용).
        layout: 표시할 배치 번호. None 이면 "미설정"으로 표시한다(대시보드가
            아직 배치를 설정하지 않은 상태).
    """
    layout_label = layout if layout is not None else "미설정"
    print(f"  [누적 카운터 — 배치 {layout_label}, 성공한 에피소드만] "
          f"(목표: 색당 {cfg.target_per_color})")
    cells = []
    for color in TARGET_COLORS:
        n = counts[color]
        mark = "*" if n >= cfg.target_per_color else ""
        cells.append(f"{color} {n:>2}/{cfg.target_per_color}{mark}")
    print("  " + "    ".join(cells))
    total = sum(counts.values())
    goal = cfg.target_per_color * len(TARGET_COLORS)
    print(f"  합계 {total}/{goal}  (다른 배치는 별도 표)")


# ──────────────────────────────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────────────────────────────

def checklist_lines(rec: ColorMarkerRecorder) -> list[str]:
    """수집 시작 직전 체크리스트. odom 항목만 자동으로 갱신된다.

    Args:
        rec: 기록 노드.

    Returns:
        출력할 줄 목록.
    """
    fresh = rec.odom_fresh()
    odom_box = "☑" if fresh else "☐"
    odom_tail = "" if fresh else " ← 미수신! 'bot'/'botcheck' 로 확인할 것"
    return [
        "  □ 3색 모두 보임",
        f"  {odom_box} odom 수신{odom_tail}",
        "  □ 로봇 위치 변경",
    ]


def print_status(cfg: Config, rec: ColorMarkerRecorder,
                 joystick: Any, xyz: tuple[float, float, float],
                 notice: str) -> None:
    """수집 상태 패널. 카운터는 배치가 바뀔 수 있으므로 매번 디스크에서 다시 센다.

    Args:
        cfg: 실행 설정.
        rec: 기록 노드.
        joystick: pygame Joystick.
        xyz: 데드존 적용된 (x, y, z) 스틱 값.
        notice: 최근 알림 한 줄.
    """
    jg = load_joystick_module()
    frame = "═" * 78
    # os.system("clear") 는 5Hz 로 매번 서브프로세스를 새로 띄워 터미널을
    # 통째로 지우므로 화면이 깜빡인다. 커서만 홈으로 되돌리고 그 아래를
    # 지우면(스크롤백은 건드리지 않음) 프로세스 생성 없이 그 자리에서
    # 다시 그릴 수 있어 깜빡임이 크게 줄어든다.
    sys.stdout.write("\x1b[H\x1b[J")

    x_pos, y_pos, z_pos = xyz
    lx, ly, az = jg.joystick_to_cmd_vel(x_pos, y_pos, z_pos)

    with rec._state_lock:
        ep = rec.current
        layout = rec.layout
        target_color = rec.target_color
        state = rec.state
        rejection = rec._last_start_rejection

    # collect_gotoobject.py 의 검수 요약 박스와 같은 스타일 — 위/아래 테두리만
    # 긋고 안쪽은 들여쓰기만 한 평문. 섹션마다 빈 줄로 나누던 기존 방식은
    # 한 화면에 다 안 들어와 스크롤이 생기고, 매 틱 지우고 다시 그리다 보니
    # 그 스크롤 어긋남이 "정신없다"는 느낌으로 이어졌다. 여기서는 조종·조건·
    # 체크리스트·기록상태·카운터를 한 프레임 안에 눌러 담아 한 번에 훑을 수
    # 있게 한다. 배치/목표색은 이제 대시보드가 설정하므로 여기는 읽기 전용
    # 표시만 한다 — 버튼 안내가 없다.
    print(f"╔{frame}╗")
    print(f"■ TIC-VLA 3색 마커 수집 | {joystick.get_name()} | "
          f"SerBOT {'실제' if jg.SERBOT_AVAILABLE else '더미'}({jg.bot.name})")
    print(f"  state: {state}"
          + (f"  |  배치 {layout} — {layout_desc(layout, ko=True)}"
             if layout is not None else "  |  배치 (미설정 — 대시보드에서 설정)"))
    print(f"  스틱 x={x_pos:+.2f} y={y_pos:+.2f} z={z_pos:+.2f} "
          f"→ cmd_vel vx={lx:+.2f} vy={ly:+.2f} wz={az:+.2f} → {cfg.cmd_vel_topic}")

    print(f"  목표색: {target_color or '(미설정 — 대시보드에서 설정)'}")
    if layout is not None and target_color:
        side = derive_target_side(layout, target_color)
        print(f"  target_side: {side}  |  instruction: {INSTRUCTIONS[target_color]!r}")
    print("  체크: " + "   ".join(line.strip() for line in checklist_lines(rec)))

    if ep is not None:
        dur = time.monotonic() - ep.t0_mono
        rows = count_csv_rows(ep.csv_path)
        print(f"  [기록 중] {ep.dir_path}  프레임 {ep.frames}  행 {rows}  "
              f"경과 {dur:5.1f}s  stall {ep.stall_ticks}")
    elif state == STATE_AWAITING_RESULT:
        print("  [결과 대기] — 대시보드(/collect)에서 성공/실패를 입력하세요"
              f" (또는 버튼 {EMERGENCY_SUBMIT_BUTTON}=비상 성공 제출)")
    else:
        print("  [대기 중] — 대시보드에서 배치/목표색 설정 후 8번으로 시작하세요"
              + (f"  ← 직전 시작 거부: {rejection[1]}" if rejection is not None else ""))
    counts = (scan_counts(cfg.root, layout) if layout is not None
             else {c: 0 for c in TARGET_COLORS})
    print_counts_table(counts, cfg, layout)

    footer = f"  8=시작  9=종료  0=긴급정지  {EMERGENCY_SUBMIT_BUTTON}=결과 비상제출(성공)"
    if notice:
        footer += f"  |  알림: {notice}"
    print(footer)
    print(f"╚{frame}╝")


# ──────────────────────────────────────────────────────────────────────
# 결과 확정 (입력은 대시보드/비상 제출 버튼에서 온다 — 콘솔 프롬프트 없음)
# ──────────────────────────────────────────────────────────────────────

def finalize_episode_json(ep: Episode, success: Optional[bool],
                          final_distance: Optional[float],
                          start_distance_approx: Optional[float],
                          notes: str, cfg: Config) -> None:
    """수기 입력을 반영해 episode.json 을 최종 확정한다.

    Args:
        ep: 대상 에피소드.
        success: 성공 여부. 치명 중단이면 False.
        final_distance: 최종 거리(m).
        start_distance_approx: 시작 거리 대략값(m).
        notes: 메모.
        cfg: 실행 설정.
    """
    payload = {
        "dataset": DATASET_NAME,
        "layout": ep.layout,
        "target_color": ep.target_color,
        "target_side": derive_target_side(ep.layout, ep.target_color),
        "instruction": INSTRUCTIONS[ep.target_color],
        "start_distance_approx": start_distance_approx,
        "success": success,
        "final_distance": final_distance,
        "notes": notes,
        "status": ep.status,
        "hz": cfg.hz,
        "num_frames": ep.frames,
        "num_csv_rows": count_csv_rows(ep.csv_path),
        "stall_ticks": ep.stall_ticks,
        "warnings": list(ep.warnings),
    }
    write_json_atomic(ep.json_path, payload)


# ──────────────────────────────────────────────────────────────────────
# verify_errand_session.py 재사용
# ──────────────────────────────────────────────────────────────────────

def run_verify(ep: Episode) -> int:
    """collect/verify_errand_session.py 를 그대로(수정 없이) 호출해 검수한다.

    verify_errand_session.py 는 session_YYMMDD.../segNN_object/... 형태를
    기대한다. 이 수집기는 세션/구간이 없는 단일 에피소드 구조라, 실제 저장물
    (ep_dir/rgb, trajectory.csv, episode.json)은 그대로 두고 검증에만 쓰는
    임시 폴더를 만들어 그 안에 session.json + ep_dir 로의 심볼릭 링크
    (seg00_<target_color>) 를 둔다. 검증이 끝나면 preview/ 만 ep_dir 로
    옮기고 임시 폴더는 지운다 — 실제 데이터셋 폴더 모양은 바뀌지 않는다.

    Args:
        ep: 방금 닫힌 에피소드.

    Returns:
        verify_errand_session.py 의 종료 코드 (0=OK, 1=재촬영 권장, 2=읽기 실패).
    """
    if not ep.dir_path.is_dir():
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="colormarker_verify_"))
    try:
        seg_link = tmp / f"seg00_{ep.target_color}"
        seg_link.symlink_to(ep.dir_path.resolve(), target_is_directory=True)
        instruction = INSTRUCTIONS[ep.target_color]
        session_payload = {
            "dataset": DATASET_NAME,
            "session_id": ep.name,
            "scenario": f"L{ep.layout}_{ep.target_color}",
            "hz": SAVE_HZ,
            "status": "completed" if ep.status == "done" else ep.status,
            "segment_plan": [
                {"index": 0, "object": ep.target_color, "instruction": instruction},
            ],
            "events": list(ep.events),
            "warnings": list(ep.warnings),
        }
        write_json_atomic(tmp / "session.json", session_payload)

        print("")
        print(f"== 자동 검수: {ep.dir_path} ==")
        result = subprocess.run([sys.executable, str(VERIFY_SCRIPT), str(tmp)])

        preview_src = tmp / "preview"
        if preview_src.is_dir():
            preview_dst = ep.dir_path / "preview"
            if preview_dst.exists():
                shutil.rmtree(preview_dst, ignore_errors=True)
            shutil.move(str(preview_src), str(preview_dst))
        return result.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """인자를 파싱한다.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        파싱 결과.
    """
    parser = argparse.ArgumentParser(
        description="TIC-VLA 3색 마커 단일 방문 에피소드 수집기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=default_root(),
                        help="저장 루트. 이 아래 바로 ep{NNNN}_... 폴더가 쌓인다")
    parser.add_argument("--target-per-color", type=int, default=8,
                        help="색당 목표 성공 에피소드 수 (배치당 = 이 값 × 3색)")
    parser.add_argument("--hz", type=float, default=SAVE_HZ,
                        help="저장 레이트. s01 의 SRC_HZ 와 맞출 것")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="저장 JPEG 품질")
    parser.add_argument("--odom-topic", default=ODOM_TOPIC_DEFAULT)
    parser.add_argument("--camera-topic", default=CAMERA_TOPIC_DEFAULT)
    parser.add_argument(
        "--cmd-vel-topic",
        default=os.environ.get("JOYSTICK_CMD_VEL_TOPIC", "/joystick_cmd_vel"),
        help="조이스틱 입력을 내보낼 Twist 토픽 (기존 스택 호환용)")
    parser.add_argument("--sensor-timeout", type=float, default=5.0,
                        help="시작 시 odom/카메라 수신 대기 최대 초")
    parser.add_argument("--odom-stall-sec", type=float, default=0.5,
                        help="이 시간 동안 odom 이 안 오면 에피소드 중단")
    parser.add_argument("--camera-stall-sec", type=float, default=0.5,
                        help="이 시간 동안 새 프레임이 안 오면 에피소드 중단")
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
    if args.target_per_color < 1:
        raise SystemExit("ERROR: --target-per-color 는 1 이상이어야 한다")
    if args.hz <= 0.0:
        raise SystemExit("ERROR: --hz 는 0보다 커야 한다")
    if args.hz != SAVE_HZ:
        print(f"[경고] --hz {args.hz} != s01 SRC_HZ {SAVE_HZ}. "
              "s01 상수도 같이 바꾸지 않으면 시간축이 틀어진다.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("ERROR: --jpeg-quality 는 1~100")

    root = Path(args.root).expanduser().resolve()
    if "GoToObject_v1" in root.parts:
        raise SystemExit(
            "ERROR: --root 가 GoToObject_v1 을 가리킨다 — 완전히 분리된 "
            "데이터셋이어야 한다. --root 를 확인할 것.")
    root.mkdir(parents=True, exist_ok=True)

    return Config(
        root=root,
        target_per_color=int(args.target_per_color),
        hz=float(args.hz),
        jpeg_quality=int(args.jpeg_quality),
        odom_topic=str(args.odom_topic),
        camera_topic=str(args.camera_topic),
        cmd_vel_topic=str(args.cmd_vel_topic),
        sensor_timeout=float(args.sensor_timeout),
        odom_stall_sec=float(args.odom_stall_sec),
        camera_stall_sec=float(args.camera_stall_sec),
    )


def preflight(rec: ColorMarkerRecorder, cfg: Config) -> int:
    """odom·카메라 수신을 확인한다. 실패하면 종료 코드를 준다.

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
        print("!" * 78)
        print("  odom 을 받지 못했다 — 시작하지 않는다.")
        print("  odometry_publisher 가 떠 있는지 확인할 것: "
              f"ros2 topic hz {cfg.odom_topic}")
        print("!" * 78)
        return EXIT_ODOM

    ok, why = rec.wait_for_camera(cfg.sensor_timeout)
    print(f"[점검] camera {cfg.camera_topic}: {'OK' if ok else '실패'} — {why}")
    if not ok:
        print("!" * 78)
        print("  카메라 프레임을 받지 못했거나 변환에 실패했다 — 시작하지 않는다.")
        print(f"  serbot_camera(camera_node) 확인: ros2 topic hz {cfg.camera_topic}")
        print("!" * 78)
        return EXIT_CAMERA
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
    print("[설정] 배치/목표색: 대시보드(/collect)에서 설정 — 여기엔 없다")
    print(f"[설정] 저장 루트: {cfg.root}")
    print(f"[설정] 저장     : {cfg.hz}Hz, 원본 해상도(리사이즈 없음), "
          f"JPEG q={cfg.jpeg_quality}")
    print(f"[설정] 색당 목표: {cfg.target_per_color} (성공 기준, 배치당 "
          f"{cfg.target_per_color * len(TARGET_COLORS)})")
    print("[안내] 접근 0.10 m/s 직행 → 1.5m 부터 서서히 감속(급정지 금지) "
          "→ 0.8m 정지 → 9번으로 즉시 종료. 정지 유지 구간 없음.")
    print("[안내] 로봇 시작 위치는 기록하지 않는다 — 매 에피소드 30~50cm "
          "아무렇게나 옮길 것 (같은 그림 반복 방지).")

    rclpy.init(args=None)
    rec = ColorMarkerRecorder(cfg)
    executor = SingleThreadedExecutor()
    executor.add_node(rec)
    spin_thread = threading.Thread(
        target=executor.spin, name="colormarker-executor", daemon=True)
    spin_thread.start()

    exit_code = EXIT_OK
    try:
        exit_code = preflight(rec, cfg)
        if exit_code != EXIT_OK:
            return exit_code

        joystick = jg.init_pygame_and_joystick()
        print_button_table(cfg, desc_map, joystick)

        num_axes = joystick.get_numaxes()
        clock = pygame.time.Clock()
        debug_interval = max(1, int(jg.UPDATE_HZ / jg.DEBUG_HZ))
        frame_count = 0
        notice = "대시보드(/collect)에서 배치/목표색을 설정한 뒤 8번으로 시작하세요"
        xyz = (0.0, 0.0, 0.0)

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
                notice = handle_button(rec, act, event.button)

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

            if rec.episode_closed.is_set():
                rec.episode_closed.clear()
                ep = rec.last_closed
                jg.stop_serbot()
                rec.publish_cmd_vel(0.0, 0.0, 0.0)
                if ep is not None:
                    if ep.status == "done":
                        # 결과 입력은 이제 대시보드(/collect/submit_result) 나
                        # 비상 제출 버튼이 한다 — 여기서는 상태를
                        # awaiting_result 로 넘긴 채 알림만 띄운다.
                        notice = (
                            f"{ep.name} 종료 — 대시보드에서 결과를 입력하세요 "
                            f"(또는 버튼 {EMERGENCY_SUBMIT_BUTTON}=비상 성공 제출)")
                    else:
                        finalize_episode_json(
                            ep, False, None, None,
                            f"치명 오류로 중단: {ep.status}", cfg)
                        rc = run_verify(ep)
                        notice = (f"{ep.name} 저장 완료(중단) — "
                                  f"{'검수 OK' if rc == 0 else '재촬영 권장(위 사유 확인)'}")

            if rec.fatal is not None:
                jg.stop_serbot()
                print(f"\n[중단] 센서 오류({rec.fatal})로 프로세스를 끝낸다.")
                break

            if frame_count % debug_interval == 0:
                print_status(cfg, rec, joystick, xyz, notice)
            frame_count += 1
            clock.tick(jg.UPDATE_HZ)

    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C — 기록 중이면 중단 처리한다.")
        ep = rec.stop_from_main("aborted_keyboard")
        if ep is not None:
            finalize_episode_json(
                ep, False, None, None, "Ctrl+C 중 중단", cfg)
            run_verify(ep)
    except Exception as exc:  # noqa: BLE001
        print("\n[오류] 예외 발생:", repr(exc))
        ep = rec.stop_from_main("aborted_exception")
        if ep is not None:
            finalize_episode_json(
                ep, False, None, None, f"예외로 중단: {exc!r}", cfg)
            run_verify(ep)
    finally:
        try:
            jg.stop_serbot()
        except Exception:  # noqa: BLE001
            pass
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

    print("")
    print("=" * 78)
    print("  수집 종료 — 배치별 누적")
    print("=" * 78)
    for layout, layout_counts in scan_counts_all(cfg.root).items():
        print_counts_table(layout_counts, cfg, layout)
    print("=" * 78)

    if rec.fatal == "odom":
        return EXIT_ODOM
    if rec.fatal == "camera":
        return EXIT_CAMERA
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
