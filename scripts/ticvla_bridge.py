#!/usr/bin/env python3
"""TIC-VLA <-> SerBot II 실시간 주행 브릿지 노드.

`TICVLA.predict()` 는 전체 13.85초가 걸리지만, 그중 느린 구간(①②③)은 전부
과거 프레임에만 의존한다. 현재 프레임은 ④ `extract_feature()` 에서만 들어온다.
그래서 ③이 만든 `past_key_values` 를 캐싱해 두고 ④⑤만 반복하면 약 500ms(2Hz)로
waypoint 를 뽑을 수 있다. 이 노드는 그 비동기 분리를 구현한다.

    [느린 스레드]  15초 주기로 predict() 실행 -> past_key_values 캐시 갱신
    [빠른 타이머]  2Hz 로 현재 프레임 + 캐시 -> waypoints -> Twist

캐시가 낡은 만큼은 모델이 제공하는 보정 입력으로 메운다.
`time_delay` 에 경과 시간을, `robot_state[3:5]` 에 그동안의 오도메트리 변위를 넣는다.

이 노드는 `/cmd_vel` 에 직접 발행하지 않는다. 반드시 `/ticvla/cmd_vel` 로 내보내고
`safety_mux.py` 가 타임아웃·속도제한·비상정지를 거쳐 모터로 중계한다.

기본값은 Shadow Mode 다. 아무 인자 없이 실행하면 계산만 하고 명령을 내보내지 않는다.

사용 예:

    # 계산만, 로봇은 움직이지 않음 (기본)
    python3 ticvla_bridge.py

    # 실제 발행 (safety_mux.py 가 함께 떠 있어야 한다)
    python3 ticvla_bridge.py --enable --speed-scale 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import statistics
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import color_target

# ----------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------
#: bero 모터 데드존 실측값. 매핑 테이블 최소 논제로 각속도 2.008 rad/s 에서
#: 유도된 최소 유효 전진 속도(m/s). 이보다 느린 명령은 cmd=0 으로 반올림된다.
MIN_EFFECTIVE_LINEAR: float = 0.095

#: 드라이버가 낼 수 있는 최대 속도(m/s). 참고용 상한.
MAX_DRIVER_LINEAR: float = 0.42

#: 액션 헤드 waypoint 스텝 간격(초). 10Hz, 총 30스텝 = 3초 horizon.
STEP_DT_SEC: float = 0.1

#: 모델이 요구하는 입력 이미지 한 변 크기.
MODEL_INPUT_SIZE: int = 448

#: VLM 이 실제로 생성하는 토큰 수(실측 131). 응답 텍스트는 assistant 메시지가 되어
#: prefill 입력에 들어가므로(ticvla.py:595), 이보다 상한을 낮추면 <answer> 블록이
#: 잘리고 action expert 가 받는 컨텍스트가 망가진다. scripts/ticvla_vlm_token_bench.py
#: 실측값(delayed 4장, 3회 평균):
#:     greedy 200 : 10.51s, 최종 변위 0.781m, 표준편차 0.000m, <answer> 완성 100%
#:     sample 200 : 10.65s, 최종 변위 0.520m, 표준편차 0.288m, <answer> 완성 100%
#:     greedy 112 :  9.27s, 최종 변위 0.546m, <answer> 잘림
#:     greedy  64 :  6.36s, 최종 변위 0.039m, <answer> 시작조차 못 함
#: 131 이상(140/160/200)은 시간과 출력이 완전히 동일했다.
MIN_SAFE_NEW_TOKENS: int = 131

#: chat 래핑을 두 번 걸지 않도록 vlm 객체에 남기는 표식.
CHAT_WRAPPED_FLAG: str = '_ticvla_bridge_chat_wrapped'


class GuardedCmdPublisher:
    """안전 조건이 깨졌을 때 Twist 를 0 으로 바꿔 발행하는 프록시.

    발행 지점이 여러 곳이어도 한 군데서 막을 수 있도록 퍼블리셔 자체를 감싼다.
    드롭하지 않고 0 을 넣는 이유는, 하위 노드가 명령 타임아웃을 기다리는 대신
    즉시 멈추게 하려는 것이다.

    forward_only 가 True 이면 후진(vx < 0) 을 발행 직전에 0 으로 클램프한다.
    """

    #: 후진 클램프 warn 로그 최소 간격(초).
    _FORWARD_CLAMP_LOG_PERIOD: float = 5.0

    def __init__(self, inner: Any, twist_cls: Any,
                 reason_fn: Any, logger: Any,
                 forward_only: bool = True) -> None:
        """프록시를 만든다.

        Args:
            inner: 원본 rclpy Publisher.
            twist_cls: geometry_msgs/Twist 클래스.
            reason_fn: 차단 사유를 돌려주는 콜러블. None 이면 통과.
            logger: rclpy 로거.
            forward_only: True 이면 linear.x 음수를 0 으로 클램프한다.
        """
        self._inner = inner
        self._Twist = twist_cls
        self._reason_fn = reason_fn
        self._log = logger
        self._forward_only = forward_only
        self._last_reason: Optional[str] = None
        #: 마지막 후진 클램프 warn 시각 (monotonic). 0.0 은 아직 없음.
        self._last_forward_clamp_log_at: float = 0.0

    def publish(self, msg: Any) -> None:
        """안전 조건을 확인한 뒤 발행한다.

        Args:
            msg: 발행하려는 geometry_msgs/Twist.
        """
        reason = self._reason_fn()
        if reason is None:
            if self._last_reason is not None:
                self._log.info(
                    f'명령 발행 재개 (직전 차단 사유: {self._last_reason})')
                self._last_reason = None
            # 안전 게이트 통과 후, 필요하면 후진만 0 으로 클램프한다.
            self._inner.publish(self._maybe_clamp_reverse(msg))
            return
        if reason != self._last_reason:
            self._log.warn(f'명령 차단 — 0 발행 | 사유: {reason}')
            self._last_reason = reason
        self._inner.publish(self._Twist())

    def _maybe_clamp_reverse(self, msg: Any) -> Any:
        """forward_only 일 때 vx 음수를 0 으로 바꾼 복사본을 돌려준다.

        원본 msg 는 호출자가 재사용할 수 있으므로 직접 변형하지 않는다.
        vy·angular.z 는 회피에 필요하므로 그대로 둔다.

        Args:
            msg: 발행하려는 geometry_msgs/Twist.

        Returns:
            클램프가 필요 없으면 원본, 필요하면 복사본.
        """
        if not self._forward_only or msg.linear.x >= 0.0:
            return msg
        original_vx = float(msg.linear.x)
        now = time.monotonic()
        if (self._last_forward_clamp_log_at == 0.0
                or now - self._last_forward_clamp_log_at
                >= self._FORWARD_CLAMP_LOG_PERIOD):
            self._log.warn(
                f'후진 차단 — vx={original_vx:.3f} 를 0 으로 클램프')
            self._last_forward_clamp_log_at = now
        out = self._Twist()
        out.linear.x = 0.0
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = msg.angular.z
        return out

    def __getattr__(self, name: str) -> Any:
        """프록시에 없는 속성은 원본 퍼블리셔로 넘긴다."""
        return getattr(self._inner, name)


#: 전처리 검증 허용 오차. 두 경로가 같은 픽셀에서 출발했다면 bfloat16 양자화
#: 간격(2.0 근처에서 약 0.0156)보다 작은 차이는 나올 수 없으므로, 이 값을 넘지
#: 않는다는 것은 사실상 비트 단위로 같다는 뜻이다.
PREPROCESS_TOLERANCE: float = 1e-3

#: --verify-preprocess 를 켰을 때 검증할 틱 수. 검증은 전처리를 세 번 하므로
#: 무한히 돌리지 않고 앞쪽 몇 틱만 확인한 뒤 스스로 멈춘다.
VERIFY_TICKS: int = 5

#: IMX219 CSI 카메라 GStreamer 파이프라인 템플릿.
GST_PIPELINE: str = (
    'nvarguscamerasrc sensor-id={sensor_id} ! '
    'video/x-raw(memory:NVMM),width={width},height={height},'
    'format=NV12,framerate={fps}/1 ! '
    'nvvidconv flip-method={flip_method} ! '
    'video/x-raw,format=BGRx ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink drop=1 max-buffers=1'
)

#: nvvidconv flip-method 값의 의미. 로그를 사람이 읽을 수 있게 하는 용도다.
FLIP_METHOD_LABELS: dict[int, str] = {
    0: '회전 없음',
    1: '반시계 90도',
    2: '180도',
    3: '시계 90도',
    4: '좌우 반전',
    5: '반시계 90도 + 좌우 반전',
    6: '상하 반전',
    7: '시계 90도 + 좌우 반전',
}

#: 이 로봇의 카메라는 물리적으로 180도 뒤집혀 장착되어 있다.
#: ros2_ws/src/serbot_camera/config/camera.yaml 의 flip_method 와 같은 값이어야 한다.
EXPECTED_FLIP_METHOD: int = 2

#: 카메라를 열었다고 인정하기 전에 첫 프레임 확보를 기다리는 최대 시간(초).
#: isOpened() 는 파이프라인 생성만 확인하므로 CaptureSession 실패를 잡지 못한다.
FIRST_FRAME_TIMEOUT_SEC: float = 3.0

#: 폴백 상태에서 경고를 다시 띄우는 간격(초).
FALLBACK_WARN_INTERVAL_SEC: float = 30.0

#: 카메라 소스 표기. jsonl 의 camera_source 필드에 그대로 들어간다.
SOURCE_LIVE: str = 'live'
SOURCE_FALLBACK: str = 'fallback'
SOURCE_NONE: str = 'none'

#: 시작 배너를 찍기 전에 카메라 공급원 확정을 기다리는 시간(초).
CAMERA_SOURCE_WAIT_SEC: float = 6.0

#: 카메라를 확보하지 못했을 때의 종료 코드 (기본 정책 — --allow-fallback 이면 대신
#: 정지 이미지 폴백으로 넘어가므로 이 코드가 나오지 않는다).
EXIT_NO_CAMERA: int = 2


def flip_method_text(flip_method: int) -> str:
    """flip-method 값을 "2(180도)" 형태의 문자열로 만든다.

    Args:
        flip_method: nvvidconv flip-method 값.

    Returns:
        값과 의미를 함께 담은 문자열. 모르는 값이면 의미 자리에 물음표를 넣는다.
    """
    return f'{flip_method}({FLIP_METHOD_LABELS.get(flip_method, "알 수 없음")})'


def apply_distributed_shim() -> list[str]:
    """Jetson PyTorch 의 비활성 torch.distributed 를 보완한다.

    Jetson 전용 PyTorch 는 USE_DISTRIBUTED=0 으로 빌드되어
    `torch.distributed.is_initialized` 등이 아예 존재하지 않는다. 반면 InternVL
    원격 코드는 디버그 출력 조건으로 이를 무조건 호출하므로 AttributeError 가 난다.
    분산 학습을 하지 않는 단일 프로세스에서는 "초기화되지 않음"이 정답이므로
    누락된 함수만 이 프로세스 안에서 채워 넣는다. torch 를 재설치하지 않는다.

    반드시 모델을 만들기 전에 호출해야 한다.

    Returns:
        보완한 속성 이름 목록.
    """
    import torch.distributed as dist

    patched: list[str] = []
    defaults = {
        'is_available': lambda: False,
        'is_initialized': lambda: False,
        'get_rank': lambda: 0,
        'get_world_size': lambda: 1,
    }
    for name, func in defaults.items():
        if not hasattr(dist, name):
            setattr(dist, name, func)
            patched.append(name)
    return patched


# ======================================================================
# 프레임 버퍼
# ======================================================================
@dataclass
class FrameRecord:
    """디스크에 저장된 프레임 한 장.

    Attributes:
        timestamp: 저장 시각(monotonic).
        path: jpg 파일 경로. 느린 경로 predict() 가 파일 경로를 요구한다.
        owned: 이 노드가 만든 파일인가 (링버퍼에서 삭제해도 되는가).
        array: 카메라 스레드가 갖고 있던 BGR 원본 배열. 빠른 경로가 JPEG
            인코딩/디코딩 왕복을 건너뛰는 데 쓴다. 메모리를 아끼기 위해 최신
            몇 장만 유지하고 오래된 것은 None 으로 비운다.
    """

    timestamp: float
    path: Path
    owned: bool
    array: Optional[np.ndarray] = None


class FrameStore:
    """스레드 안전 프레임 링 버퍼.

    모델의 `load_image()` 가 파일 경로를 요구하므로 프레임을 디스크(기본 /dev/shm)에
    저장하고 (시각, 경로) 목록을 유지한다. 용량을 넘으면 오래된 것부터 지운다.
    폴백 모드로 쓰는 기존 정지 이미지는 owned=False 라 삭제하지 않는다.
    """

    def __init__(self, capacity: int, keep_array: int = 4) -> None:
        """버퍼를 초기화한다.

        Args:
            capacity: 유지할 최대 프레임 개수.
            keep_array: numpy 원본 배열을 들고 있을 최신 프레임 수. 1280x720
                BGR 한 장이 약 2.8MB 이므로 필요한 만큼만 유지한다.
        """
        self._capacity = capacity
        self._keep_array = max(1, keep_array)
        self._lock = threading.Lock()
        self._records: deque[FrameRecord] = deque()
        self._save_times: deque[float] = deque()
        self._pinned: dict[Path, int] = {}
        self._pending_delete: set[Path] = set()

    def _unlink(self, path: Path) -> None:
        """파일을 지운다. 실패해도 무시한다.

        Args:
            path: 지울 파일 경로.
        """
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def add(self, record: FrameRecord) -> None:
        """프레임을 추가하고 용량을 넘으면 오래된 것을 지운다.

        사용 중(pin)인 파일은 지우지 않고 대기 목록에 넣는다. VLM 이 14초 동안
        같은 경로를 다시 열기 때문에, 그 사이 링버퍼가 파일을 지우면 predict() 가
        FileNotFoundError 로 실패한다.

        Args:
            record: 추가할 프레임 기록.
        """
        with self._lock:
            self._records.append(record)
            self._save_times.append(record.timestamp)
            while self._save_times and record.timestamp - self._save_times[0] > 5.0:
                self._save_times.popleft()
            # 최신 keep_array 장만 원본 배열을 유지하고 그보다 오래된 것은 비운다.
            # jpg 파일은 그대로 남으므로 느린 경로는 영향을 받지 않는다.
            if len(self._records) > self._keep_array:
                self._records[-(self._keep_array + 1)].array = None
            while len(self._records) > self._capacity:
                old = self._records.popleft()
                if not old.owned:
                    # 폴백용 원본 이미지는 건드리지 않는다.
                    continue
                if old.path in self._pinned:
                    self._pending_delete.add(old.path)
                else:
                    self._unlink(old.path)

    def pin(self, records: list[FrameRecord]) -> None:
        """프레임을 사용 중으로 표시해 삭제를 막는다.

        Args:
            records: 보호할 프레임 목록.
        """
        with self._lock:
            for record in records:
                self._pinned[record.path] = self._pinned.get(record.path, 0) + 1

    def unpin(self, records: list[FrameRecord]) -> None:
        """사용 중 표시를 해제하고, 밀린 삭제를 처리한다.

        Args:
            records: 해제할 프레임 목록.
        """
        with self._lock:
            for record in records:
                count = self._pinned.get(record.path, 0) - 1
                if count > 0:
                    self._pinned[record.path] = count
                    continue
                self._pinned.pop(record.path, None)
                if record.path in self._pending_delete:
                    self._pending_delete.discard(record.path)
                    self._unlink(record.path)

    def latest(self) -> Optional[FrameRecord]:
        """가장 최근 프레임을 돌려준다.

        Returns:
            최신 FrameRecord. 버퍼가 비었으면 None.
        """
        with self._lock:
            return self._records[-1] if self._records else None

    def pick_delayed(self, count: int, spacing_sec: float,
                     minimum: int) -> Optional[list[FrameRecord]]:
        """과거 프레임을 시간 간격을 두고 고른다.

        현재로부터 -count*spacing ... -spacing 시점에 가장 가까운 프레임을 찾는다.
        같은 프레임이 중복 선택되면 하나만 남긴다.

        Args:
            count: 원하는 프레임 개수.
            spacing_sec: 프레임 사이 목표 시간 간격(초).
            minimum: 이보다 적게 모이면 아직 대기해야 한다고 판단한다.

        Returns:
            시간 오름차순 FrameRecord 목록. 개수가 minimum 미만이면 None.
        """
        with self._lock:
            records = list(self._records)
        if not records:
            return None

        now = records[-1].timestamp
        picked: list[FrameRecord] = []
        seen: set[Path] = set()
        for step in range(count, 0, -1):
            target = now - spacing_sec * step
            nearest = min(records, key=lambda r: abs(r.timestamp - target))
            if nearest.path not in seen:
                seen.add(nearest.path)
                picked.append(nearest)

        if len(picked) < minimum:
            return None
        return picked

    def stats(self) -> tuple[int, float]:
        """버퍼 상태를 돌려준다.

        Returns:
            (보관 중인 프레임 수, 최근 5초 저장 FPS).
        """
        with self._lock:
            count = len(self._records)
            times = list(self._save_times)
        if len(times) < 2:
            return count, 0.0
        span = times[-1] - times[0]
        return count, (len(times) - 1) / span if span > 0 else 0.0


# ======================================================================
# 카메라 스레드
# ======================================================================
class _CameraSourceBase(threading.Thread):
    """카메라 프레임 소스 공통 로직 (스레드/폴백/상태조회).

    `CameraWorker`(직접 GStreamer 오픈)와 `RosTopicCameraSource`(ROS 토픽 구독)
    가 이 기반을 공유한다. 두 구현 모두 "실 카메라 획득"에 실패하면 같은 정책
    (기본은 치명 종료, --allow-fallback 을 명시했을 때만 정지 이미지 순환 폴백)을
    따라야 하므로 그 부분을 여기 한 군데에 둔다 — 갈라놓으면 한쪽만 고치는 실수가 난다.

    서브클래스는 `run()` 만 구현하면 된다. 카메라/토픽을 확보하면
    `self._source = SOURCE_LIVE; self._source_ready.set()` 을 하고 프레임마다
    `self._save_frame(frame)` 을 부르면 되고, 실패 시 `self._fatal(reason)` 또는
    `self._run_fallback()` 을 호출하면 나머지는 공통 로직이 처리한다.
    """

    def __init__(self, cfg: argparse.Namespace, store: FrameStore,
                 logger: Any, name: str,
                 on_fatal: Optional[Callable[[str], None]] = None) -> None:
        """공통 상태를 초기화한다.

        Args:
            cfg: 명령행 설정.
            store: 프레임을 넣을 버퍼.
            logger: ROS 로거.
            name: 스레드 이름 (로그 구분용).
            on_fatal: 폴백이 허용되지 않는데(기본값, --allow-fallback 미지정) 카메라를
                확보하지 못했을 때 부를 콜백. 이유 문자열을 받는다. None 이면 폴백으로 넘어간다.
        """
        super().__init__(name=name, daemon=True)
        self._cfg = cfg
        self._store = store
        self._log = logger
        self._on_fatal = on_fatal
        # 주의: threading.Thread 에는 내부 _stop() 메서드가 있다. 같은 이름을 쓰면
        # 스레드 종료 시 threading 이 Event 를 호출하려다 TypeError 로 죽는다.
        self._stop_event = threading.Event()
        self._seq = 0
        self._capture_fps = 0.0
        # 공급원은 전환 시점에만 바꾼다. 루프가 끝났다고 되돌리지 않는다.
        # 종료 중에 기록되는 레코드까지 실제 공급원을 가리켜야 하기 때문이다.
        self._source = SOURCE_NONE
        # 프레임 공급원이 확정되면(카메라 성공/폴백 진입/완전 실패) 세운다.
        self._source_ready = threading.Event()
        self._read_times: deque[float] = deque()

    # ── 상태 조회 ──────────────────────────────────────────────────────

    @property
    def capture_fps(self) -> float:
        """카메라에서 실제로 읽어들이는 FPS."""
        return self._capture_fps

    @property
    def is_fallback(self) -> bool:
        """폴백(정지 이미지) 모드로 동작 중인지 여부."""
        return self._source == SOURCE_FALLBACK

    @property
    def source(self) -> str:
        """버퍼에 프레임을 넣은 공급원. 'live' / 'fallback' / 'none'.

        'none' 은 아직 어느 쪽도 시작하지 못한 상태다. 이때는 프레임이 없어
        VLM 캐시도 만들어지지 않으므로 기록에 남을 일이 없다.
        """
        return self._source

    def wait_source(self, timeout: float) -> str:
        """프레임 공급원이 확정될 때까지 기다린다.

        시작 배너에 live/fallback 을 정확히 찍기 위한 것이다. 카메라 열기는
        nvargus 세션 생성까지 1초 안쪽이 걸린다.

        Args:
            timeout: 최대 대기 시간(초).

        Returns:
            확정된 공급원. 시간 안에 확정되지 않으면 'none'.
        """
        self._source_ready.wait(timeout)
        return self.source

    # ── 공통 내부 동작 (저장/폴백/치명 처리) ────────────────────────────

    def _save_frame(self, frame: np.ndarray) -> None:
        """프레임을 jpg 로 저장하고 버퍼에 등록한다.

        Args:
            frame: BGR 이미지 배열.
        """
        self._seq += 1
        path = Path(self._cfg.frame_dir) / f'frame_{self._seq:08d}.jpg'
        ok = cv2.imwrite(str(path), frame,
                         [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.jpeg_quality])
        if not ok:
            self._log.warn(f'프레임 저장 실패: {path}')
            return
        # 배열을 복사해 둔다. 호출자가 내부 버퍼를 재사용하더라도 보관 중인
        # 이전 프레임이 덮어써지지 않도록 하기 위한 것이다.
        self._store.add(FrameRecord(time.monotonic(), path, owned=True,
                                    array=frame.copy()))

    def _fallback_sources(self) -> list[Path]:
        """폴백 모드에서 순환 재생할 이미지 목록을 만든다.

        Returns:
            시간순 정렬된 이미지 경로 목록. 없으면 빈 목록.
        """
        directory = Path(self._cfg.image_dir)
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        )

    def _run_fallback(self) -> None:
        """폴백 루프. 정지 이미지를 save_fps 주기로 순환 등록한다."""
        sources = self._fallback_sources()
        if not sources:
            self._log.error(
                f'폴백 이미지가 없다: {self._cfg.image_dir} — 프레임 공급 중단')
            return

        self._source = SOURCE_FALLBACK
        self._source_ready.set()
        # 폴백 이미지는 이미 올바른 방향으로 저장된 것이므로 회전을 적용하지 않는다.
        # 여기에 flip 을 또 걸면 정방향 이미지가 거꾸로 뒤집힌다.
        self._warn_fallback(len(sources))
        self._log.warn(
            f'폴백 원본: {self._cfg.image_dir} ({len(sources)}장, '
            f'{self._cfg.save_fps:.1f}Hz 순환) — 이 경로에는 flip-method 를 '
            '적용하지 않는다 (저장된 이미지가 이미 정방향)')

        save_period = 1.0 / self._cfg.save_fps
        index = 0
        next_warn = time.monotonic() + FALLBACK_WARN_INTERVAL_SEC
        while not self._stop_event.is_set():
            # 폴백 상태가 길어질수록 실수로 방치될 위험이 커지므로 계속 상기시킨다.
            if time.monotonic() >= next_warn:
                next_warn = time.monotonic() + FALLBACK_WARN_INTERVAL_SEC
                self._warn_fallback(len(sources))

            source = sources[index % len(sources)]
            index += 1
            # 원본을 지우지 않도록 frame_dir 로 복사해 owned 파일로 다룬다.
            self._seq += 1
            target = Path(self._cfg.frame_dir) / f'frame_{self._seq:08d}.jpg'
            try:
                shutil.copyfile(source, target)
            except OSError as exc:
                self._log.warn(f'폴백 프레임 복사 실패: {exc}')
                time.sleep(save_period)
                continue
            # 폴백에서도 배열을 함께 실어 빠른 경로가 같은 코드를 타게 한다.
            # 디코딩 비용은 카메라 스레드가 부담한다.
            self._store.add(FrameRecord(time.monotonic(), target, owned=True,
                                        array=cv2.imread(str(target))))
            self._capture_fps = self._cfg.save_fps
            time.sleep(save_period)

    def _warn_fallback(self, count: int) -> None:
        """폴백 상태임을 배너로 알린다.

        입력이 잘못됐는데 출력은 멀쩡해 보이는 상황이라 WARN 한 줄로는 놓친다.
        측정 결과를 무효로 만드는 조건이므로 ERROR 로 눈에 띄게 남긴다.

        Args:
            count: 순환 재생 중인 이미지 장수.
        """
        line = '!' * 72
        self._log.error(line)
        self._log.error(f'  폴백 이미지 재생 중 — 실제 카메라 아님 ({count}장 순환). '
                        '이 데이터로는 장면을 판단할 수 없다.')
        self._log.error(line)

    def _fatal(self, reason: str) -> None:
        """카메라 확보 실패를 치명적 오류로 처리한다.

        Args:
            reason: 실패 원인.
        """
        self._log.error(f'카메라 확보 실패 — 카메라 필수가 기본 정책이므로 종료한다 '
                        f'(코드 {EXIT_NO_CAMERA}). 폴백 이미지로 측정하면 결과가 무효가 '
                        f'된다 — 그걸 감수하고서라도 계속하려면 --allow-fallback 을 '
                        f'명시할 것. ({reason})')
        # 콜백을 먼저 부른 뒤 이벤트를 세운다. 순서를 바꾸면 wait_source() 로
        # 깨어난 메인 스레드가 아직 비어 있는 실패 사유를 보고 추론을 시작한다.
        if self._on_fatal is not None:
            self._on_fatal(reason)
        self._source_ready.set()

    def stop(self) -> None:
        """스레드 종료를 요청한다."""
        self._stop_event.set()


class CameraWorker(_CameraSourceBase):
    """CSI 카메라를 직접 열어 프레임을 주기적으로 파일로 저장하는 스레드.

    카메라는 이 노드 하나만 연다. 대시보드/조이스틱/데이터수집과 카메라를
    나눠 써야 한다면 대신 `RosTopicCameraSource` 를 쓸 것(`--camera-topic`).
    열기에 실패하면 지정된 폴더의 정지 이미지를 순환 재생하는 폴백 모드로
    전환해 실내에서도 파이프라인을 시험할 수 있게 한다.
    """

    def __init__(self, cfg: argparse.Namespace, store: FrameStore,
                 logger: Any,
                 on_fatal: Optional[Callable[[str], None]] = None) -> None:
        """카메라 워커를 초기화한다.

        Args:
            cfg: 명령행 설정.
            store: 프레임을 넣을 버퍼.
            logger: ROS 로거.
            on_fatal: 폴백이 허용되지 않는데(기본값, --allow-fallback 미지정) 카메라를
                확보하지 못했을 때 부를 콜백. 이유 문자열을 받는다. None 이면 폴백으로 넘어간다.
        """
        super().__init__(cfg, store, logger, name='camera', on_fatal=on_fatal)
        self._cap: Optional[cv2.VideoCapture] = None

    # ── 내부 동작 ──────────────────────────────────────────────────────

    def _open_camera(self) -> bool:
        """GStreamer 파이프라인으로 CSI 카메라를 연다.

        회전은 nvvidconv 의 flip-method 로 GPU 에서 처리한다. numpy 로 뒤집으면
        매 프레임 CPU 비용이 붙고, 모델이 보는 영상과 저장되는 jpg 가 어긋날 수
        있으므로 파이프라인 단계에서 한 번에 바로잡는다.

        isOpened() 만으로는 판정하지 않는다. nvarguscamerasrc 가 다른 프로세스에
        점유되어 "Failed to create CaptureSession" 으로 실패해도 isOpened() 는
        True 를 돌려주기 때문이다. 실제로 프레임 한 장을 읽어 기대 해상도까지
        확인한 뒤에만 성공으로 본다.

        Returns:
            열기에 성공하면 True.
        """
        pipeline = GST_PIPELINE.format(
            sensor_id=self._cfg.sensor_id,
            width=self._cfg.cam_width,
            height=self._cfg.cam_height,
            fps=self._cfg.cam_fps,
            flip_method=self._cfg.flip_method,
        )
        self._log.info(
            f'카메라 열기 시도 | flip-method={flip_method_text(self._cfg.flip_method)} '
            f'| {pipeline}')
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            self._log.error('카메라 열기 실패: 파이프라인을 생성하지 못했다')
            return False

        ok, reason = self._probe_first_frame(cap)
        if not ok:
            cap.release()
            self._log.error(f'카메라 열기 실패: {reason}')
            return False

        self._cap = cap
        self._log.info(f'카메라 열기 성공 ({reason})')
        return True

    def _probe_first_frame(self, cap: cv2.VideoCapture) -> tuple[bool, str]:
        """첫 프레임을 실제로 읽어 해상도까지 확인한다.

        Args:
            cap: 열린 VideoCapture.

        Returns:
            (성공 여부, 사람이 읽을 설명). 실패 이유나 확인된 해상도를 담는다.
        """
        expected = (int(self._cfg.cam_height), int(self._cfg.cam_width))
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
        attempts = 0
        last = '프레임을 받지 못했다'

        while time.monotonic() < deadline and not self._stop_event.is_set():
            attempts += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                last = 'read() 가 프레임을 돌려주지 않았다'
                time.sleep(0.05)
                continue
            if frame.shape[:2] != expected:
                return False, (f'해상도가 다르다: 기대 {expected[1]}x{expected[0]}, '
                               f'실제 {frame.shape[1]}x{frame.shape[0]}')
            return True, (f'첫 프레임 {frame.shape[1]}x{frame.shape[0]}, '
                          f'{attempts}회 시도')

        if self._stop_event.is_set():
            return False, '종료 요청으로 중단'
        return False, (f'{FIRST_FRAME_TIMEOUT_SEC:.0f}초 안에 첫 프레임을 얻지 못했다 '
                       f'({attempts}회 시도, {last}) — 다른 프로세스가 카메라를 '
                       '점유했을 수 있다')

    def _run_camera(self) -> None:
        """카메라 루프. 계속 읽되 저장은 save_fps 주기로만 한다."""
        save_period = 1.0 / self._cfg.save_fps
        next_save = 0.0
        fails = 0

        while not self._stop_event.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 30:
                    # 이후 처리(폴백 전환 또는 종료)는 run() 이 정책에 따라 결정한다.
                    self._log.error('카메라 읽기 연속 실패 — 카메라 루프를 중단한다')
                    return
                time.sleep(0.01)
                continue
            fails = 0

            now = time.monotonic()
            self._read_times.append(now)
            while self._read_times and now - self._read_times[0] > 2.0:
                self._read_times.popleft()
            if len(self._read_times) > 1:
                span = self._read_times[-1] - self._read_times[0]
                self._capture_fps = (len(self._read_times) - 1) / span if span else 0.0

            if now >= next_save:
                next_save = now + save_period
                self._save_frame(frame)

    def run(self) -> None:
        """스레드 본체. 카메라를 열고, 실패하면 폴백으로 넘어간다.

        기본값은 폴백으로 넘어가지 않고 노드 종료를 요청하는 것이다.
        `--allow-fallback` 을 명시했을 때만 폴백으로 넘어간다.
        """
        opened = False
        try:
            opened = self._open_camera()
            if opened:
                self._source = SOURCE_LIVE
                self._source_ready.set()
                self._run_camera()
        except Exception as exc:  # noqa: BLE001
            self._log.error(f'카메라 루프 예외: {exc}')
        finally:
            self._release()

        if self._stop_event.is_set():
            self._source_ready.set()
            return

        if not self._cfg.allow_fallback:
            self._fatal('열기 실패' if not opened else '읽기 중단')
            return

        try:
            self._run_fallback()
        except Exception as exc:  # noqa: BLE001
            self._log.error(f'폴백 루프 예외: {exc}')
        finally:
            self._source_ready.set()

    def _release(self) -> None:
        """카메라 핸들을 해제한다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RosTopicCameraSource(_CameraSourceBase):
    """카메라를 직접 열지 않고 ROS Image 토픽을 구독해 채우는 프레임 소스.

    nvarguscamerasrc 는 한 프로세스만 열 수 있다(`serbot_camera/camera_node.py`
    참고). 웹 대시보드/조이스틱/데이터수집을 동시에 띄운 채 추론도 돌리려면
    이 노드도 카메라를 직접 열지 말고 `camera_node` 가 발행하는 토픽만
    구독해야 한다 — `--camera-topic` 으로 이 소스를 켠다.

    실제 수신은 rclpy 콜백(노드 executor 스레드)이 담당하고, 이 스레드는
    `_CameraSourceBase.run()` 계약대로 save_fps 주기로 최신 프레임을 꺼내
    디스크에 저장하는 일과 최초 수신 대기/끊김 감시만 한다.
    """

    def __init__(self, cfg: argparse.Namespace, node: Any, store: FrameStore,
                 logger: Any,
                 on_fatal: Optional[Callable[[str], None]] = None) -> None:
        """구독을 걸고 상태를 초기화한다.

        Args:
            cfg: 명령행 설정 (camera_topic, camera_timeout_sec 등).
            node: 구독을 등록할 rclpy 노드.
            store: 프레임을 넣을 버퍼.
            logger: ROS 로거.
            on_fatal: 폴백이 허용되지 않는데(기본값) 토픽에서 프레임을 못 받았을 때 부를 콜백.
        """
        super().__init__(cfg, store, logger, name='camera_ros', on_fatal=on_fatal)
        from cv_bridge import CvBridge
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from sensor_msgs.msg import Image as RosImage

        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_at: float = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = node.create_subscription(
            RosImage, cfg.camera_topic, self._on_image, sensor_qos)
        self._log.info(
            f'카메라 소스: ROS 토픽 구독 ({cfg.camera_topic}) — 이 노드는 카메라를 '
            '직접 열지 않는다 (대시보드/조이스틱/데이터수집과 동시 실행 가능)')

    def _on_image(self, msg: Any) -> None:
        """구독 콜백. 최신 프레임만 슬롯에 남긴다 (rclpy 콜백 스레드에서 실행).

        Args:
            msg: 구독한 sensor_msgs/Image.
        """
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self._log.error(f'카메라 토픽 프레임 변환 실패: {exc}')
            return
        now = time.monotonic()
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_at = now
        if self._source != SOURCE_LIVE:
            self._source = SOURCE_LIVE
            self._source_ready.set()
            self._log.info(
                f'카메라 토픽 첫 프레임 수신 ({frame.shape[1]}x{frame.shape[0]})')

    def run(self) -> None:
        """save_fps 주기로 최신 프레임을 꺼내 저장한다.

        첫 프레임을 `FIRST_FRAME_TIMEOUT_SEC` 안에 못 받으면 카메라 필수 정책
        (기본은 치명 종료, --allow-fallback 이면 폴백)을 따른다. 수신 중 끊기면(camera_timeout_sec
        이상 무수신) 경고만 반복하고 계속 기다린다 — camera_node 가 재연결하면
        바로 이어받을 수 있어야 하기 때문이다.
        """
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
        while (not self._stop_event.is_set() and self._source != SOURCE_LIVE
               and time.monotonic() < deadline):
            time.sleep(0.02)

        if self._stop_event.is_set():
            self._source_ready.set()
            return

        if self._source != SOURCE_LIVE:
            if not self._cfg.allow_fallback:
                self._fatal(
                    f'{self._cfg.camera_topic} 에서 {FIRST_FRAME_TIMEOUT_SEC:.0f}초 '
                    '안에 프레임을 받지 못했다 — camera_node 가 떠 있는지, '
                    "'ros2 topic hz " + self._cfg.camera_topic + "' 로 확인할 것")
                return
            try:
                self._run_fallback()
            finally:
                self._source_ready.set()
            return

        save_period = 1.0 / self._cfg.save_fps
        next_save = 0.0
        stale = False
        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._frame_lock:
                frame = self._latest_frame
                frame_at = self._latest_frame_at

            if frame is None or now - frame_at > self._cfg.camera_timeout_sec:
                if not stale:
                    stale = True
                    self._log.error(
                        f'{self._cfg.camera_topic} 프레임이 '
                        f'{self._cfg.camera_timeout_sec:.1f}초 이상 끊겼다 — '
                        'camera_node 상태를 확인할 것 (계속 대기한다)')
                time.sleep(0.1)
                continue
            if stale:
                stale = False
                self._log.info(f'{self._cfg.camera_topic} 프레임 재수신')

            self._read_times.append(now)
            while self._read_times and now - self._read_times[0] > 2.0:
                self._read_times.popleft()
            if len(self._read_times) > 1:
                span = self._read_times[-1] - self._read_times[0]
                self._capture_fps = (len(self._read_times) - 1) / span if span else 0.0

            if now >= next_save:
                next_save = now + save_period
                self._save_frame(frame)
            time.sleep(min(save_period, 0.05))

    def stop(self) -> None:
        """스레드 종료를 요청하고 구독을 해제한다."""
        super().stop()
        try:
            self._sub.destroy()
        except Exception:  # noqa: BLE001
            pass


# ======================================================================
# 빠른 루프 프로파일러
# ======================================================================
def cuda_sync() -> None:
    """CUDA 비동기 실행을 동기화한다.

    커널은 큐에 넣기만 하고 바로 반환하므로, 동기화 없이 시간을 재면 실제
    연산 시간이 다음 구간으로 밀린다. 계측할 때만 호출한다.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FastLoopProfiler:
    """빠른 루프의 구간별 시간과 GPU 메모리 변화를 모아 한 번에 보고한다.

    `--profile-fast` 가 꺼져 있으면 이 객체를 아예 만들지 않는다. 따라서
    계측 오버헤드(`cuda_sync()` 포함)가 실행 경로에 전혀 들어가지 않는다.

    한 틱 안에서 같은 구간이 여러 번 측정되면 합산한다. 한 번도 측정되지 않은
    구간은 0 으로 남겨, 그 연산이 아예 일어나지 않았다는 사실을 드러낸다.
    """

    #: 표에 출력할 구간 순서.
    SECTIONS: tuple[str, ...] = (
        't_frame_get',
        't_preprocess',
        't_vision_encode',
        't_llm_forward',
        't_kv_prepare',
        't_state_build',
        't_action_expert',
        't_postprocess',
    )

    #: 이 값보다 작으면 action expert 는 병목이 아니라고 판정한다(초).
    ACTION_EXPERT_LIMIT_SEC: float = 0.020

    def __init__(self, samples: int, logger: Any, out_path: Path) -> None:
        """계측기를 초기화한다.

        Args:
            samples: 표를 출력할 때까지 모을 틱 수.
            logger: ROS 로거.
            out_path: 결과 JSON 저장 경로.
        """
        self._target = samples
        self._log = logger
        self._out = out_path
        self._times: dict[str, list[float]] = {name: [] for name in self.SECTIONS}
        self._mems: dict[str, list[int]] = {name: [] for name in self.SECTIONS}
        self._acc_time: dict[str, float] = {}
        self._acc_mem: dict[str, int] = {}
        self._totals: list[float] = []
        self._ticks = 0
        self._reported = False
        self._facts: dict[str, Any] = {}
        self._calls: dict[str, int] = {}
        # 느린 스레드도 같은 VLM 메서드를 쓴다. 빠른 틱을 실행 중인 스레드에서
        # 온 호출만 세기 위해 thread-local 플래그로 구분한다.
        self._local = threading.local()
        self._lock = threading.Lock()

    # ── 스레드 구분 ────────────────────────────────────────────────────

    def enter_tick(self) -> None:
        """이 스레드가 빠른 틱을 실행 중이라고 표시한다."""
        self._local.active = True

    def is_active(self) -> bool:
        """현재 스레드가 빠른 틱 안에 있는지 여부.

        Returns:
            빠른 틱 실행 중이면 True.
        """
        return bool(getattr(self._local, 'active', False))

    # ── 측정 ───────────────────────────────────────────────────────────

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """구간을 감싸 소요 시간과 GPU 메모리 변화를 기록한다.

        Args:
            name: SECTIONS 중 하나.

        Yields:
            None.
        """
        cuda_sync()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.add(name, time.perf_counter() - started,
                     torch.cuda.memory_allocated() - before)

    def add(self, name: str, seconds: float, mem_delta: int = 0) -> None:
        """구간 측정값을 현재 틱에 누적한다.

        Args:
            name: 구간 이름.
            seconds: 소요 시간.
            mem_delta: GPU 할당량 변화(바이트).
        """
        self._acc_time[name] = self._acc_time.get(name, 0.0) + seconds
        self._acc_mem[name] = self._acc_mem.get(name, 0) + mem_delta

    def count(self, name: str) -> None:
        """빠른 루프에서 불린 메서드의 호출 횟수를 센다.

        Args:
            name: 메서드 이름.
        """
        with self._lock:
            self._calls[name] = self._calls.get(name, 0) + 1

    def fact(self, key: str, value: Any) -> None:
        """조사 항목을 기록한다. 처음 들어온 값만 남긴다.

        Args:
            key: 항목 이름.
            value: 값.
        """
        self._facts.setdefault(key, value)

    def cancel_tick(self) -> None:
        """예외로 중단된 틱을 버린다. 부분 측정값이 통계를 오염시키지 않게 한다."""
        self._local.active = False
        self._acc_time.clear()
        self._acc_mem.clear()

    def end_tick(self, total_sec: float) -> None:
        """한 틱을 마감한다. 목표 표본에 도달하면 표를 출력한다.

        Args:
            total_sec: 이 틱의 빠른 경로 전체 소요 시간.
        """
        self._local.active = False
        if self._reported:
            return
        self._ticks += 1
        self._totals.append(total_sec)
        for name in self.SECTIONS:
            # 측정되지 않은 구간은 0 으로 남긴다 (연산이 없었다는 증거).
            self._times[name].append(self._acc_time.get(name, 0.0))
            self._mems[name].append(self._acc_mem.get(name, 0))
        self._acc_time.clear()
        self._acc_mem.clear()
        if self._ticks >= self._target:
            try:
                self._report()
            except Exception as exc:  # noqa: BLE001
                self._log.error(f'프로파일 보고 실패: {exc}')
            self._reported = True

    # ── 보고 ───────────────────────────────────────────────────────────

    @staticmethod
    def _stats(samples: list[float]) -> dict[str, float]:
        """평균/중앙값/최소/최대를 계산한다.

        Args:
            samples: 표본 목록.

        Returns:
            통계 딕셔너리.
        """
        if not samples:
            return {'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
        return {
            'mean': statistics.fmean(samples),
            'median': statistics.median(samples),
            'min': min(samples),
            'max': max(samples),
        }

    def _report(self) -> None:
        """구간별 통계 표와 판정을 출력하고 JSON 으로 저장한다."""
        total_mean = statistics.fmean(self._totals) if self._totals else 0.0
        stats = {name: self._stats(self._times[name]) for name in self.SECTIONS}
        mem_mean = {name: (statistics.fmean(self._mems[name]) if self._mems[name] else 0.0)
                    for name in self.SECTIONS}

        self._log.info('=' * 84)
        self._log.info(f'빠른 루프 구간별 프로파일 ({self._ticks} 틱, '
                       f'전체 평균 {total_mean * 1000:.1f}ms)')
        self._log.info('=' * 84)
        self._log.info(f'{"구간":<17}{"평균":>10}{"중앙":>10}{"최소":>10}'
                       f'{"최대":>10}{"비율":>8}{"GPU 증감":>12}')
        self._log.info('-' * 84)
        for name in self.SECTIONS:
            row = stats[name]
            ratio = (row['mean'] / total_mean * 100.0) if total_mean > 0 else 0.0
            self._log.info(
                f'{name:<17}'
                f'{row["mean"] * 1000:>9.1f}ms'
                f'{row["median"] * 1000:>9.1f}ms'
                f'{row["min"] * 1000:>9.1f}ms'
                f'{row["max"] * 1000:>9.1f}ms'
                f'{ratio:>7.1f}%'
                f'{mem_mean[name] / 1e6:>+11.2f}MB')
        self._log.info('-' * 84)

        self._report_verdict(stats, total_mean)
        self._report_facts()
        self._save(stats, mem_mean, total_mean)

    def _report_verdict(self, stats: dict[str, dict[str, float]],
                        total_mean: float) -> None:
        """판정 문구를 출력한다.

        Args:
            stats: 구간별 통계.
            total_mean: 전체 평균 소요 시간(초).
        """
        action = stats['t_action_expert']['mean']
        llm = stats['t_llm_forward']['mean']
        biggest = max(self.SECTIONS, key=lambda name: stats[name]['mean'])
        biggest_ratio = (stats[biggest]['mean'] / total_mean * 100.0
                         if total_mean > 0 else 0.0)

        self._log.info('[판정]')
        if action < self.ACTION_EXPERT_LIMIT_SEC:
            self._log.info(
                f'  Action Expert 는 병목이 아니다({action * 1000:.1f}ms). '
                f'나머지 {(total_mean - action) * 1000:.1f}ms 가 VLM 재계산이다.')
        else:
            self._log.info(
                f'  Action Expert 가 {action * 1000:.1f}ms 로 병목 후보다.')
        self._log.info(
            f'  최대 구간: {biggest} — {stats[biggest]["mean"] * 1000:.1f}ms '
            f'({biggest_ratio:.1f}%)')
        if llm > 0.0:
            self._log.info(
                f'  빠른 루프에서 language model 이 매 틱 실행되고 있다 '
                f'({llm * 1000:.1f}ms).')
        else:
            self._log.info(
                '  language model 은 빠른 루프에서 실행되지 않는다 (t_llm_forward=0).')

    def _report_facts(self) -> None:
        """조사 항목과 호출 횟수를 출력한다."""
        self._log.info('[조사 항목]')
        for key, value in sorted(self._facts.items()):
            self._log.info(f'  {key} = {value}')
        with self._lock:
            calls = dict(self._calls)
        per_tick = {name: round(n / max(1, self._ticks), 2)
                    for name, n in sorted(calls.items())}
        self._log.info(f'  빠른 루프 내 호출 횟수(틱당) = {per_tick}')

    def _save(self, stats: dict[str, dict[str, float]],
              mem_mean: dict[str, float], total_mean: float) -> None:
        """측정 원본과 통계를 JSON 으로 저장한다.

        Args:
            stats: 구간별 통계.
            mem_mean: 구간별 평균 GPU 메모리 변화.
            total_mean: 전체 평균 소요 시간(초).
        """
        with self._lock:
            calls = dict(self._calls)
        payload = {
            'ticks': self._ticks,
            'total_sec': self._totals,
            'total_mean_sec': total_mean,
            'sections': {
                name: {
                    'samples_sec': self._times[name],
                    'mem_delta_bytes': self._mems[name],
                    'stats_sec': stats[name],
                    'mem_delta_mean_bytes': mem_mean[name],
                    'ratio_percent': (stats[name]['mean'] / total_mean * 100.0
                                      if total_mean > 0 else 0.0),
                }
                for name in self.SECTIONS
            },
            'facts': self._facts,
            'fast_loop_calls': calls,
        }
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding='utf-8')
        self._log.info(f'프로파일 JSON 저장: {self._out}')


# ======================================================================
# 모델 래퍼
# ======================================================================
@dataclass
class VlmCache:
    """느린 루프가 만들어 둔 VLM 상태 캐시."""

    past: tuple
    captured_at: float
    response: str
    odom_x: float
    odom_y: float
    odom_yaw: float


class ModelRunner:
    """TICVLA 모델 로딩과 두 종류의 추론 경로를 담당한다.

    공식 저장소 코드는 수정하지 않고 두 곳을 래핑한다.

        - `model.vlm.forward` : `predict()` 가 반환하지 않는 `past_key_values` 를
          가로챈다.
        - `model.vlm.chat`    : `predict()` 가 내부에 하드코딩한 생성 설정
          (ticvla.py:579, max_new_tokens=200 / do_sample=True / temperature=0.7)
          을 CLI 로 받은 설정으로 갈아끼운다.
    """

    def __init__(self, cfg: argparse.Namespace, logger: Any) -> None:
        """모델을 로딩하고 forward / chat 래핑을 건다.

        Args:
            cfg: 명령행 설정.
            logger: ROS 로거.

        Raises:
            FileNotFoundError: 모델 또는 체크포인트 경로가 없는 경우.
            RuntimeError: CUDA 를 쓸 수 없는 경우.
        """
        self._cfg = cfg
        self._log = logger
        self._capture: dict[str, Any] = {}
        self.generation_config = self._build_generation_config()

        model_path = Path(cfg.model_path)
        ckpt_path = Path(cfg.ckpt_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f'모델 경로가 없다: {model_path}')
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'체크포인트가 없다: {ckpt_path}')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 를 쓸 수 없다. venv 와 실행 환경을 확인할 것.')

        patched = apply_distributed_shim()
        if patched:
            self._log.warn(
                f'torch.distributed 비활성 빌드 — {", ".join(patched)} 보완 적용')

        # 지연 import: shim 이 적용된 뒤에 InternVL 원격 코드가 로드되어야 한다.
        from ticvla.models.ticvla import TICVLA
        from ticvla.utils.vision import build_transform, dynamic_preprocess, load_image

        self._load_image = load_image
        # 메모리 전처리는 공식 함수를 그대로 재사용한다. 직접 구현하면 리샘플링
        # 커널이나 정규화가 미세하게 달라져 waypoint 가 바뀔 수 있다.
        self._dynamic_preprocess = dynamic_preprocess
        self._transform = build_transform(input_size=MODEL_INPUT_SIZE)

        # 검증은 앞쪽 몇 틱만 하고 스스로 멈춘다.
        self._verify_left = VERIFY_TICKS if cfg.verify_preprocess else 0
        self._verify_worst: dict[str, float] = {}

        self._log.info(f'모델 로딩 중: {model_path}')
        self.model = TICVLA(model_path=str(model_path), action_horizon_steps=30)
        self._load_checkpoint(ckpt_path)
        self.model.to('cuda')
        self.model.eval()
        self._wrap_vlm_forward()
        self._wrap_vlm_chat()
        self._log.info('모델 준비 완료 (bfloat16 / cuda)')
        self._log.info(
            f'VLM 생성 설정 교체 (predict() 하드코딩 대체) | '
            f'{"sample" if cfg.do_sample else "greedy"} | {self.generation_config}')
        if cfg.max_new_tokens < MIN_SAFE_NEW_TOKENS:
            self._log.warn(
                f'max_new_tokens={cfg.max_new_tokens} 은 실측 하한 '
                f'{MIN_SAFE_NEW_TOKENS} 미만입니다. <answer> 블록이 잘려 '
                'waypoint 가 붕괴할 수 있습니다.')

    # ── 생성 설정 ──────────────────────────────────────────────────────

    def _build_generation_config(self) -> dict[str, Any]:
        """CLI 설정으로 VLM 텍스트 생성 설정을 만든다.

        Returns:
            `vlm.chat` 에 넘길 생성 설정. greedy 일 때는 temperature 키를 넣지
            않는다 (transformers 가 쓰이지 않는 인자라고 경고한다).
        """
        config: dict[str, Any] = {
            'max_new_tokens': self._cfg.max_new_tokens,
            'do_sample': self._cfg.do_sample,
        }
        if self._cfg.do_sample:
            config['temperature'] = self._cfg.temperature
        return config

    # ── 로딩 ───────────────────────────────────────────────────────────

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        """체크포인트에서 VLM / action expert 가중치를 불러온다.

        `TICVLATester.__init__` 와 동일한 접두사 규칙을 따른다.

        Args:
            ckpt_path: 체크포인트 파일 경로.

        Raises:
            KeyError: 체크포인트에 state_dict 가 없는 경우.
        """
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        if 'state_dict' not in checkpoint:
            raise KeyError('체크포인트에 state_dict 가 없다')
        state_dict = checkpoint['state_dict']

        vlm_sd = {k[len('model.vlm.'):]: v for k, v in state_dict.items()
                  if k.startswith('model.vlm.')}
        act_sd = {k[len('model.action_expert.'):]: v for k, v in state_dict.items()
                  if k.startswith('model.action_expert.')}

        vlm_missing, vlm_unexpected = self.model.vlm.load_state_dict(
            vlm_sd, strict=False)
        act_missing, act_unexpected = self.model.action_expert.load_state_dict(
            act_sd, strict=False)
        self._log.info(
            f'checkpoint 로드 | VLM {len(vlm_sd)}키 '
            f'(missing {len(vlm_missing)}, unexpected {len(vlm_unexpected)}) | '
            f'action {len(act_sd)}키 '
            f'(missing {len(act_missing)}, unexpected {len(act_unexpected)})'
        )
        del state_dict, checkpoint, vlm_sd, act_sd

    def _wrap_vlm_forward(self) -> None:
        """vlm.forward 를 감싸 past_key_values 를 가로챈다.

        predict() 안에서 prefill 용으로 vlm(...) 이 한 번 호출된다. chat() 은
        language_model.generate 를 타므로 이 래퍼를 거치지 않는다.
        """
        original = self.model.vlm.forward
        capture = self._capture

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            capture['vlm_out'] = result
            return result

        self.model.vlm.forward = wrapper

    def _wrap_vlm_chat(self) -> None:
        """vlm.chat 을 감싸 predict() 가 넘기는 생성 설정을 교체한다.

        predict() 는 생성 설정을 함수 안에 하드코딩하고 있어서(ticvla.py:579)
        바깥에서 바꿀 방법이 없다. 공식 코드를 수정하지 않기 위해 호출 시점에
        인자를 갈아끼운다. predict() 는 4번째 위치 인자로 넘기지만 키워드로
        오는 경우도 함께 처리한다. 같은 객체에 두 번 걸지 않도록 표식을 남긴다.
        """
        if getattr(self.model.vlm, CHAT_WRAPPED_FLAG, False):
            return

        original = self.model.vlm.chat
        runner = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """생성 설정을 브릿지 설정으로 바꿔 원본 chat 을 호출한다."""
            config = dict(runner.generation_config)
            if 'generation_config' in kwargs:
                kwargs['generation_config'] = config
            elif len(args) > 3:
                positional = list(args)
                positional[3] = config
                args = tuple(positional)
            else:
                kwargs['generation_config'] = config
            return original(*args, **kwargs)

        self.model.vlm.chat = wrapper
        setattr(self.model.vlm, CHAT_WRAPPED_FLAG, True)

    # ── 추론 ───────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """모델이 올라간 디바이스."""
        return self.model.device

    def run_slow(self, delayed_paths: list[str], current_path: str,
                 instruction: str, robot_state: torch.Tensor,
                 time_delay: float) -> tuple[tuple, str]:
        """느린 경로. predict() 를 돌려 past_key_values 와 응답 텍스트를 얻는다.

        Args:
            delayed_paths: 과거 프레임 경로 목록.
            current_path: 현재 프레임 경로.
            instruction: 자연어 지시문.
            robot_state: (5,) bfloat16 텐서 [vx, vy, yaw_speed, dx, dy].
            time_delay: 지연 시간(초).

        Returns:
            (past_key_values 튜플, VLM 응답 텍스트).

        Raises:
            RuntimeError: prefill 출력이 캡처되지 않은 경우.
        """
        self._capture.pop('vlm_out', None)
        with torch.inference_mode():
            response, _waypoints, _prompt = self.model.predict(
                delayed_image_paths=delayed_paths,
                current_image_path=current_path,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=time_delay,
            )

        outputs = self._capture.get('vlm_out')
        if outputs is None:
            raise RuntimeError('vlm.forward 출력을 캡처하지 못했다')

        past = getattr(outputs, 'past_key_values', None)
        # DynamicCache 는 predict() 내부와 같은 방식으로 튜플로 변환한다.
        if past is not None and hasattr(past, 'layers'):
            past = tuple((layer.keys, layer.values) for layer in past.layers)
        elif past is None:
            past = tuple()
        return past, response

    def dump_model_inputs(self, paths: list[str], out_dir: Path,
                          vlm_runs: int, camera_source: str) -> list[str]:
        """predict() 에 넣는 이미지를 모델이 보는 상태 그대로 저장한다.

        저장하는 것은 정규화 직전 단계다. 구체적으로 `load_image()` 안에서
        PIL 로 열어 448x448 로 리사이즈한 타일이며, ToTensor/Normalize 만
        적용되지 않은 상태다. 정규화 결과는 사람이 볼 수 없는 값이라 그 앞
        단계를 남긴다.

        따라서 저장된 jpg 는 아래가 모두 반영된 "모델이 실제로 본" 그림이다.
            - GStreamer nvvidconv flip-method 회전
            - 448x448 리사이즈(BICUBIC)
        원본 카메라 프레임이 아니다.

        Args:
            paths: predict() 에 넘기는 이미지 경로. delayed(오래된→최신) 뒤에
                current 를 붙인 순서로 주며, 마지막 항목이 현재 프레임이다.
            out_dir: 저장 디렉터리.
            vlm_runs: 몇 번째 VLM 갱신인가. 파일명에 들어간다.
            camera_source: 'live' 또는 'fallback'. 파일명에 들어간다.

        Returns:
            저장에 성공한 파일 경로 목록. 실패한 항목은 빠진다.
        """
        saved: list[str] = []
        for index, path in enumerate(paths):
            target = out_dir / f'vlm{vlm_runs:03d}_{camera_source}_i{index}.jpg'
            try:
                image = Image.open(path).convert('RGB')
                tiles = self._dynamic_preprocess(
                    image, image_size=MODEL_INPUT_SIZE, use_thumbnail=True,
                    max_num=1)
                tiles[0].save(target, quality=self._cfg.jpeg_quality)
            except (OSError, ValueError, IndexError) as exc:
                self._log.warn(f'모델 입력 저장 실패 ({path}): {exc}')
                continue
            saved.append(str(target))
        return saved

    # ── 빠른 경로 단계 ─────────────────────────────────────────────────

    def _preprocess_file(self, current_path: str) -> torch.Tensor:
        """프레임 파일을 읽어 모델 입력 텐서로 만든다 (기존 경로).

        모델의 `load_image()` 가 파일 경로를 받으므로 디스크(tmpfs)를 거친다.
        JPEG 디코딩이 포함되어 메모리 경로보다 느리다.

        Args:
            current_path: 현재 프레임 경로.

        Returns:
            (타일수, 3, 448, 448) bfloat16 CUDA 텐서. max_num=1 이라 타일 1장.
        """
        return self._load_image(
            current_path, input_size=MODEL_INPUT_SIZE, max_num=1,
        ).to(torch.bfloat16).to(self.device)

    def _preprocess_array(self, frame: np.ndarray) -> torch.Tensor:
        """메모리에 있는 BGR 배열을 모델 입력 텐서로 만든다.

        `load_image()` 와 같은 단계를 그대로 밟는다. 다른 점은 JPEG 파일을 열지
        않고 이미 갖고 있는 배열에서 PIL 이미지를 만든다는 것 하나뿐이다.
        max_num=1 이면 `dynamic_preprocess` 는 448x448 로 리사이즈한 한 장만
        돌려주므로(use_thumbnail 은 타일이 2장 이상일 때만 동작) 결과 모양도 같다.

        Args:
            frame: cv2 가 준 BGR 배열. PIL 은 RGB 를 기대하므로 반드시 변환한다.

        Returns:
            (1, 3, 448, 448) bfloat16 CUDA 텐서.
        """
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tiles = self._dynamic_preprocess(
            image, image_size=MODEL_INPUT_SIZE, use_thumbnail=True, max_num=1)
        pixel = torch.stack([self._transform(tile) for tile in tiles])
        return pixel.to(torch.bfloat16).to(self.device)

    def _fast_preprocess(self, frame: FrameRecord) -> tuple[torch.Tensor, str]:
        """빠른 경로 전처리. 배열이 살아 있으면 메모리에서 바로 만든다.

        Args:
            frame: 현재 프레임 레코드.

        Returns:
            (입력 텐서, 어느 경로를 썼는지 나타내는 설명 문자열).
        """
        if self._verify_left > 0:
            self._verify_preprocess(frame)

        # 로컬 참조로 잡아 둔다. 카메라 스레드가 그 사이 레코드의 array 를 비워도
        # 이미 붙잡은 배열 객체는 살아 있다.
        array = frame.array
        if array is not None:
            return self._preprocess_array(array), '메모리 BGR 배열 (JPEG 왕복 없음)'
        return self._preprocess_file(str(frame.path)), '파일 경로 -> load_image()'

    # ── 전처리 검증 ────────────────────────────────────────────────────

    def _verify_preprocess(self, frame: FrameRecord) -> None:
        """파일 경로 방식과 메모리 방식의 전처리 결과를 비교해 로그로 남긴다.

        두 결과가 완전히 같을 수는 없다. 파일 경로는 손실 압축인 JPEG 를 거치기
        때문이다. 그래서 원인을 나눠서 본다.

            transform : 같은 JPEG 를 디코딩한 픽셀을 두 파이프라인에 각각 넣은
                차이. 전처리 구현이 동일한지 보는 값이며 0 이어야 한다.
            jpeg : 원본 배열 기반 결과와 JPEG 기반 결과의 차이. JPEG 압축 손실
                자체이므로 0 이 될 수 없고, 메모리 경로가 오히려 센서 원본에
                더 가깝다.

        Args:
            frame: 현재 프레임 레코드.
        """
        self._verify_left -= 1
        try:
            reference = self._preprocess_file(str(frame.path))
            # JPEG 를 load_image() 와 같은 방식으로 디코딩한 뒤 RGB->BGR 로 되돌려
            # 메모리 경로에 넣는다. 채널 교환은 무손실이라 비교에 영향이 없다.
            decoded = np.asarray(Image.open(frame.path).convert('RGB'))
            as_bgr = np.ascontiguousarray(decoded[:, :, ::-1])
            transform_diff = self._max_abs_diff(
                reference, self._preprocess_array(as_bgr))
            self._track_worst('transform', transform_diff)

            verdict = 'OK' if transform_diff <= PREPROCESS_TOLERANCE else 'MISMATCH'
            message = (f'전처리 검증 [{VERIFY_TICKS - self._verify_left}/'
                       f'{VERIFY_TICKS}] transform 최대오차 {transform_diff:.3e} '
                       f'({verdict}, 허용 {PREPROCESS_TOLERANCE:.1e})')

            if frame.array is not None:
                jpeg_diff = self._max_abs_diff(
                    reference, self._preprocess_array(frame.array))
                self._track_worst('jpeg', jpeg_diff)
                message += f' | jpeg 손실차 {jpeg_diff:.3e}'
            else:
                message += ' | 배열 없음(폴백 경로)'

            if transform_diff > PREPROCESS_TOLERANCE:
                self._log.warn(message)
            else:
                self._log.info(message)

            if self._verify_left == 0:
                self._log_verify_summary()
        except Exception as exc:  # noqa: BLE001 - 검증 실패가 주행을 막으면 안 된다
            self._verify_left = 0
            self._log.warn(f'전처리 검증 실패(무시하고 계속): {exc}')

    @staticmethod
    def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
        """두 텐서의 최대 절대 오차를 float32 로 계산한다.

        Args:
            left: 비교 대상 텐서.
            right: 비교 대상 텐서.

        Returns:
            최대 절대 오차. 모양이 다르면 무한대를 돌려준다.
        """
        if left.shape != right.shape:
            return float('inf')
        return (left.float() - right.float()).abs().max().item()

    def _track_worst(self, key: str, value: float) -> None:
        """검증 항목별 최댓값을 갱신한다.

        Args:
            key: 항목 이름.
            value: 이번 측정값.
        """
        self._verify_worst[key] = max(self._verify_worst.get(key, 0.0), value)

    def _log_verify_summary(self) -> None:
        """검증 종료 요약을 남긴다."""
        worst = self._verify_worst.get('transform', 0.0)
        jpeg = self._verify_worst.get('jpeg')
        tail = f', JPEG 손실차 최대 {jpeg:.3e}' if jpeg is not None else ''
        if worst <= PREPROCESS_TOLERANCE:
            self._log.info(
                f'전처리 검증 종료: 두 경로의 transform 결과가 일치한다 '
                f'(최대 {worst:.3e} <= {PREPROCESS_TOLERANCE:.1e}{tail}). '
                '메모리 경로를 계속 사용한다.')
        else:
            self._log.error(
                f'전처리 검증 종료: transform 결과가 다르다 '
                f'(최대 {worst:.3e} > {PREPROCESS_TOLERANCE:.1e}{tail}). '
                '메모리 경로의 waypoint 를 신뢰할 수 없다.')

    def _fast_vision(self, pixel: torch.Tensor) -> torch.Tensor:
        """현재 프레임을 비전 인코더로 통과시켜 임베딩을 만든다.

        `extract_feature()` 는 InternViT 비전 백본과 mlp1 투영을 실행한다.
        language model 은 타지 않는다.

        Args:
            pixel: 전처리된 픽셀 텐서.

        Returns:
            (1, L, H) 이미지 임베딩.
        """
        embeds = self.model.vlm.extract_feature(pixel)
        return embeds.reshape(-1, embeds.shape[-1]).unsqueeze(0)

    def _fast_state(self, robot_state: torch.Tensor,
                    time_delay: float) -> torch.Tensor:
        """robot_state 와 time_delay 를 붙여 (1, 6, 1) 상태 토큰을 만든다.

        Args:
            robot_state: (5,) bfloat16 텐서.
            time_delay: 캐시 캡처 이후 경과 시간(초).

        Returns:
            (1, 6, 1) bfloat16 CUDA 텐서.
        """
        delay = torch.tensor([time_delay], device=self.device,
                             dtype=torch.bfloat16)
        return torch.cat([robot_state, delay]).unsqueeze(0).unsqueeze(-1)

    def run_fast(self, frame: FrameRecord, robot_state: torch.Tensor,
                 time_delay: float, past: tuple,
                 profiler: Optional[FastLoopProfiler] = None) -> torch.Tensor:
        """빠른 경로. 현재 프레임 인코딩 + action expert 만 실행한다.

        Args:
            frame: 현재 프레임 레코드. 배열이 있으면 파일을 읽지 않는다.
            robot_state: (5,) bfloat16 텐서.
            time_delay: 캐시 캡처 이후 경과 시간(초).
            past: 캐시된 past_key_values. 매 틱 재계산하지 않고 그대로 재사용한다.
            profiler: 계측기. None 이면 계측 없이 원래 경로로 실행한다.

        Returns:
            (1, 30, 2) waypoint 텐서.
        """
        with torch.inference_mode():
            if profiler is None:
                pixel, _source = self._fast_preprocess(frame)
                embeds = self._fast_vision(pixel)
                state = self._fast_state(robot_state, time_delay)
                return self.model.action_expert(embeds, state, kv_cache=past)
            return self._run_fast_profiled(
                frame, robot_state, time_delay, past, profiler)

    def _run_fast_profiled(self, frame: FrameRecord, robot_state: torch.Tensor,
                           time_delay: float, past: tuple,
                           profiler: FastLoopProfiler) -> torch.Tensor:
        """구간별 시간을 재면서 빠른 경로를 실행한다.

        Args:
            frame: 현재 프레임 레코드.
            robot_state: (5,) bfloat16 텐서.
            time_delay: 캐시 캡처 이후 경과 시간(초).
            past: 캐시된 past_key_values.
            profiler: 계측기.

        Returns:
            (1, 30, 2) waypoint 텐서.
        """
        with profiler.section('t_preprocess'):
            pixel, source = self._fast_preprocess(frame)
        profiler.fact('vision_input_shape', tuple(pixel.shape))
        profiler.fact('vision_input_dtype', str(pixel.dtype))
        profiler.fact('image_source', source)

        with profiler.section('t_vision_encode'):
            embeds = self._fast_vision(pixel)
        profiler.fact('image_embeds_shape', tuple(embeds.shape))

        # past_key_values 는 느린 스레드가 만들 때 이미 튜플로 변환해 둔다.
        # 여기서는 변환도 재계산도 하지 않고 그대로 넘긴다는 사실을 기록한다.
        with profiler.section('t_kv_prepare'):
            needs_convert = past is not None and hasattr(past, 'layers')
            profiler.fact('kv_needs_conversion_in_fast_path', needs_convert)
            profiler.fact('kv_reuse', '느린 스레드 캐시를 그대로 재사용 (재계산 없음)')
            profiler.fact('kv_layers', len(past) if past is not None else 0)
            if past:
                profiler.fact('kv_last_layer_value_shape', tuple(past[-1][1].shape))
            profiler.fact('kv_object_id', id(past))

        with profiler.section('t_state_build'):
            state = self._fast_state(robot_state, time_delay)

        with profiler.section('t_action_expert'):
            waypoints = self.model.action_expert(embeds, state, kv_cache=past)
        return waypoints

    # ── 계측 훅 ────────────────────────────────────────────────────────

    def install_profile_hooks(self, profiler: FastLoopProfiler) -> None:
        """빠른 루프에서 어떤 VLM 메서드가 불리는지 세고 LLM 시간을 잰다.

        느린 스레드도 같은 메서드를 쓰므로, 빠른 틱을 실행 중인 스레드에서 온
        호출만 센다(계측기가 thread-local 플래그로 구분한다). language model 이
        빠른 루프에서 실제로 도는지 추정하지 않고 측정으로 확인하기 위한 훅이다.

        Args:
            profiler: 계측기.
        """
        vlm = self.model.vlm
        targets: list[tuple[Any, str, str, bool]] = [
            (vlm, 'extract_feature', 'vlm.extract_feature', False),
            (getattr(vlm, 'vision_model', None), 'forward',
             'vlm.vision_model.forward', False),
            (getattr(vlm, 'language_model', None), 'forward',
             'vlm.language_model.forward', True),
            (getattr(vlm, 'mlp1', None), 'forward', 'vlm.mlp1.forward', False),
        ]
        installed: list[str] = []
        for owner, attr, label, is_llm in targets:
            if owner is None or not hasattr(owner, attr):
                continue
            self._install_counter(owner, attr, label, is_llm, profiler)
            installed.append(label)
        self._log.info(f'빠른 루프 계측 훅 설치: {", ".join(installed)}')

    @staticmethod
    def _install_counter(owner: Any, attr: str, label: str, is_llm: bool,
                         profiler: FastLoopProfiler) -> None:
        """메서드를 감싸 빠른 틱 안에서의 호출만 세고, 필요하면 시간도 잰다.

        Args:
            owner: 대상 객체.
            attr: 감쌀 메서드 이름.
            label: 보고에 쓸 이름.
            is_llm: True 면 소요 시간을 t_llm_forward 에 더한다.
            profiler: 계측기.
        """
        original = getattr(owner, attr)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """빠른 틱 안에서 온 호출만 계측하고 원본을 호출한다."""
            if not profiler.is_active():
                return original(*args, **kwargs)
            profiler.count(label)
            if not is_llm:
                return original(*args, **kwargs)
            cuda_sync()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            cuda_sync()
            profiler.add('t_llm_forward', time.perf_counter() - started)
            return result

        setattr(owner, attr, wrapper)


# ======================================================================
# ROS 2 노드
# ======================================================================
class TicvlaBridge:
    """카메라·모델·ROS 를 엮는 브릿지 본체.

    rclpy Node 를 상속하지 않고 합성으로 감싼다. 모델 로딩 실패 같은 상황에서
    노드 수명주기를 바깥에서 단순하게 다루기 위해서다.
    """

    def __init__(self, node: Any, cfg: argparse.Namespace) -> None:
        """구독자·발행자·워커를 구성한다.

        Args:
            node: 이미 만들어진 rclpy Node.
            cfg: 명령행 설정.
        """
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        # STEP 5: 대시보드 명령 토픽용
        from std_msgs.msg import Bool
        from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                               HistoryPolicy, DurabilityPolicy)

        self._Twist = Twist
        self._String = String
        self._Bool = Bool
        self.node = node
        self.cfg = cfg
        self.log = node.get_logger()

        # ── 상태 ──
        self._store = FrameStore(cfg.buffer_size, keep_array=cfg.keep_array)
        self._cache: Optional[VlmCache] = None
        self._cache_lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._stop = threading.Event()

        self._odom_lock = threading.Lock()
        self._odom_ok = False
        self._odom_warned = False
        self._odom_vx = 0.0
        self._odom_vy = 0.0
        self._odom_wz = 0.0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0

        self._last_cmd: Optional[Any] = None
        self._last_cmd_at = 0.0
        self._last_fast_ms = 0.0
        self._last_status: dict[str, Any] = {}
        self._deadzone_blocked = False
        # ── STEP 7: 도착 판정 ──
        #: 도착 후보에 들어온 monotonic 시각. 0.0 이면 후보가 아니다.
        self._arrive_candidate_since: float = 0.0
        #: 도착 후보 연속 유지 시간(초). status / JSONL 에 노출한다.
        self._arrive_elapsed: float = 0.0
        #: lookahead 지점까지 변위(m). status / JSONL 에 노출한다.
        self._last_disp: float = 0.0
        #: ARRIVED 확정 여부. arrive_latch 면 해제 전까지 유지한다.
        self._arrived: bool = False
        self._published_state = ''
        self._log_at: dict[str, float] = {}
        self._log_lock = threading.Lock()
        self._vlm_runs = 0
        self._vlm_fails = 0
        # ── STEP 5: 외부 명령 상태 ──
        #: 웹/CLI 로 갱신되는 자연어 지시문. CLI 값이 초기값이다.
        self._instruction: str = cfg.instruction
        #: 비상정지 래치. True 가 되면 명시적 False 발행 전까지 풀리지 않는다.
        self._estop: bool = False
        #: enable 게이트. --require-enable 이 아니면 처음부터 열려 있다.
        self._enabled: bool = not cfg.require_enable
        #: 마지막 heartbeat 수신 시각 (monotonic). 0.0 은 아직 못 받음.
        self._heartbeat_at: float = 0.0
        #: 위 4개를 함께 보호한다.
        self._cmd_state_lock = threading.Lock()
        self._fatal_reason: Optional[str] = None

        Path(cfg.frame_dir).mkdir(parents=True, exist_ok=True)
        self._dump_dir: Optional[Path] = None
        if cfg.dump_frames:
            self._dump_dir = Path(cfg.dump_frames)
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(
                f'모델 입력 프레임 저장: {self._dump_dir} '
                '(전처리 후 448x448 RGB, flip 적용 상태)')
        self._jsonl = open(cfg.log_json, 'a', encoding='utf-8') if cfg.log_json else None

        # ── 통신 ──
        self._cmd_pub = node.create_publisher(Twist, cfg.out_topic, 10)
        # STEP 5: 모든 발행 경로를 안전 게이트로 통과시킨다.
        self._cmd_pub = GuardedCmdPublisher(
            self._cmd_pub, Twist, self._safety_block_reason, self.log,
            forward_only=cfg.forward_only)
        self._status_pub = node.create_publisher(String, cfg.status_topic, 10)
        node.create_subscription(Odometry, cfg.odom_topic, self._on_odom, 10)

        # ── STEP 5: 대시보드 명령 구독 ──
        # 대시보드(web_dashboard_node.py:140~168) 와 QoS 를 반드시 일치시킨다.
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 비상정지는 latched. 브리지가 나중에 떠도 마지막 True 를 받아야 한다.
        estop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(
            String, cfg.instruction_topic, self._on_instruction, command_qos)
        node.create_subscription(
            Bool, cfg.estop_topic, self._on_estop, estop_qos)
        node.create_subscription(
            Bool, cfg.enable_topic, self._on_enable, command_qos)
        if cfg.heartbeat_timeout > 0.0:
            node.create_subscription(
                Bool, cfg.heartbeat_topic, self._on_heartbeat, command_qos)
            self.log.info(
                f'heartbeat 감시 켜짐 | {cfg.heartbeat_topic} | '
                f'{cfg.heartbeat_timeout:.1f}s 무수신이면 0 발행')
        self.log.info(
            f'외부 명령 구독 | instruction={cfg.instruction_topic} '
            f'estop={cfg.estop_topic} enable={cfg.enable_topic} | '
            f'초기 지시문: {cfg.instruction!r}')

        # ── 색상 타겟팅 (--color-nav) ──
        # VLM/action expert 대신 HSV 색상 검출로 직접 조향하는 독립 경로.
        # 사유: 언어 조건화가 KV 캐시 마지막 레이어 값 하나로만 들어가는 얇은
        # 경로라 "3개 중 지정색" 같은 다중 객체 선택을 학습한 적이 없다
        # (실측: instruction 에 색을 넣어도 <think> 가 한 색만 언급하고 나머지를
        # 구분하지 못함). 색 3개가 고정 팔레트일 때는 고전 CV 가 훨씬 안정적이다.
        self._color_ranges = (color_target.load_ranges_json(cfg.color_ranges_json)
                              if cfg.color_ranges_json else color_target.DEFAULT_RANGES)
        self._color_servo_cfg = color_target.ServoConfig(
            kp_ang=cfg.color_kp_ang, max_wz=cfg.color_max_wz,
            max_vx=cfg.max_speed, approach_area=cfg.color_approach_area,
            stop_area=cfg.color_stop_area, steer_sign=cfg.color_steer_sign)
        self._color_last_target: Optional[str] = None
        self._color_dump_at: float = 0.0
        if cfg.color_nav:
            self.log.info(
                f'색상 타겟팅 켜짐(--color-nav) | 색상={[r.name for r in self._color_ranges]} | '
                f'min_area={cfg.color_min_area:.0f} approach={cfg.color_approach_area:.0f} '
                f'stop={cfg.color_stop_area:.0f} | VLM/action expert 경로는 쓰지 않는다')

        # ── 모델 ──
        # --color-nav --no-vlm 이면 VLM 로딩/추론을 통째로 건너뛴다 (GPU 불필요,
        # 즉시 기동). color-nav 없이 no-vlm 만 주면 아무것도 주행을 만들지 못하므로
        # parse_args() 에서 미리 막는다.
        self.model: Optional[ModelRunner] = None
        if not (cfg.color_nav and cfg.no_vlm):
            self.model = ModelRunner(cfg, self.log)
        else:
            self.log.info('--no-vlm — VLM 모델 로딩을 건너뛴다 (color-nav 전용)')

        # ── 빠른 루프 계측 (기본 꺼짐) ──
        self._profiler: Optional[FastLoopProfiler] = None
        if cfg.profile_fast and self.model is not None:
            self._profiler = FastLoopProfiler(
                cfg.profile_samples, self.log, Path(cfg.profile_json))
            self.model.install_profile_hooks(self._profiler)
            self.log.info(
                f'빠른 루프 프로파일 켜짐 — {cfg.profile_samples} 틱 모은 뒤 표를 '
                '1회 출력한다 (계측 오버헤드가 있으니 상시 사용 금지)')

        # ── 워커 ──
        if cfg.camera_topic:
            # camera_node(serbot_camera) 가 이미 카메라를 쥐고 있다는 뜻이다.
            # 대시보드/조이스틱/데이터수집과 동시에 띄우려면 이 노드는 카메라를
            # 직접 열지 말고 그 토픽만 구독해야 한다 — flip-method 는 camera_node
            # 쪽에서 이미 적용되므로 여기서는 의미가 없다.
            self._camera = RosTopicCameraSource(
                cfg, node, self._store, self.log, on_fatal=self._on_camera_fatal)
        else:
            # 카메라 스레드가 파이프라인 로그를 찍기 전에 회전 설정을 먼저 알린다.
            self._log_flip_setting()
            self._camera = CameraWorker(cfg, self._store, self.log,
                                        on_fatal=self._on_camera_fatal)
        self._camera.start()

        # 시작 배너에 live/fallback 을 정확히 찍기 위해 공급원 확정을 기다린다.
        # 카메라 필수가 기본값이므로 여기서 실패가 드러나 VLM 로딩 전에 빠질 수 있다.
        source = self._camera.wait_source(CAMERA_SOURCE_WAIT_SEC)

        self._vlm_thread = threading.Thread(
            target=self._vlm_loop, name='vlm', daemon=True)
        # 이미 카메라 확보에 실패했다면 추론을 시작하지 않는다. main() 이 곧 종료한다.
        if self._fatal_reason is None:
            if self.model is not None:
                self._vlm_thread.start()
            node.create_timer(1.0 / cfg.rate, self._fast_tick)

        mode = 'SHADOW (발행 안 함)' if cfg.shadow else f'ENABLE -> {cfg.out_topic}'
        self.log.info(f'ticvla_bridge 시작 | {mode}')
        self._log_camera_source(source)
        self.log.info(
            f'빠른 루프 {cfg.rate:.1f}Hz | VLM 주기 {cfg.vlm_period:.1f}s | '
            f'lookahead {cfg.lookahead_step} 스텝 | 속도배율 {cfg.speed_scale:.2f} | '
            f'최대 {cfg.max_speed:.3f} m/s | 데드존정책 {cfg.deadzone_policy} | '
            f'도착판정 disp<{cfg.arrive_threshold:.3f}m '
            f'hold={cfg.arrive_hold:.1f}s latch={cfg.arrive_latch}'
        )
        if cfg.max_speed < MIN_EFFECTIVE_LINEAR and cfg.deadzone_policy == 'zero':
            self.log.warn(
                f'max_speed({cfg.max_speed:.3f}) 가 데드존({MIN_EFFECTIVE_LINEAR}) '
                '보다 작다. 이 설정으로는 바퀴가 절대 돌지 않는다.')

    # ── 카메라 상태 ────────────────────────────────────────────────────

    @property
    def fatal_reason(self) -> Optional[str]:
        """치명적 오류 사유. None 이면 정상이다."""
        return self._fatal_reason

    def _on_camera_fatal(self, reason: str) -> None:
        """카메라 워커가 알린 치명적 실패를 기록한다.

        여기서 바로 종료하지 않는다. 정지 명령 발행과 자원 정리는 메인 스레드가
        해야 하므로 사유만 남기고 spin 루프가 빠지게 한다.

        Args:
            reason: 실패 원인.
        """
        self._fatal_reason = reason
        self._stop.set()

    def _log_camera_source(self, source: str) -> None:
        """현재 프레임 공급원을 시작 배너에 명시한다.

        Args:
            source: 'live' / 'fallback' / 'none'.
        """
        if self._fatal_reason is not None:
            self.log.error(f'카메라 소스: 없음 — {self._fatal_reason}. 종료한다.')
        elif source == SOURCE_LIVE:
            self.log.info(
                f'카메라 소스: live (CSI sensor-id={self.cfg.sensor_id}, '
                f'{self.cfg.cam_width}x{self.cfg.cam_height})')
        elif source == SOURCE_FALLBACK:
            self.log.error(
                f'카메라 소스: fallback — {self.cfg.image_dir} 의 정지 이미지다. '
                '실제 장면이 아니므로 측정 결과를 신뢰하면 안 된다 '
                '(--allow-fallback 으로 명시적으로 허용된 상태 — 정식 측정에는 쓰지 말 것).')
        else:
            self.log.warn(
                f'카메라 소스: 미확정 ({CAMERA_SOURCE_WAIT_SEC:.0f}초 안에 결정되지 '
                '않았다). 이후 로그에서 live/fallback 을 확인할 것.')

    # ── 카메라 회전 설정 ───────────────────────────────────────────────

    def _log_flip_setting(self) -> None:
        """적용될 카메라 회전 설정을 알리고, 의심스러운 값이면 경고한다.

        이 로봇의 카메라는 물리적으로 뒤집혀 장착되어 있다. 회전을 빼먹으면
        모델이 상하좌우가 뒤집힌 영상을 보게 되어 전진/후진과 좌/우 판단이
        모두 무효가 되지만, 영상 자체는 정상으로 보이므로 알아채기 어렵다.
        """
        flip = self.cfg.flip_method
        self.log.info(
            f'카메라 회전: nvvidconv flip-method={flip_method_text(flip)} '
            f'(폴백 이미지 경로에는 적용하지 않음)')

        if flip == 0:
            self.log.warn(
                'flip-method=0 입니다. 이 로봇의 카메라는 180도 뒤집혀 장착되어 '
                '있어 보통 2 가 맞습니다. 의도한 설정인지 확인하세요.')
        elif flip not in FLIP_METHOD_LABELS:
            self.log.warn(
                f'flip-method={flip} 는 nvvidconv 가 아는 값(0~7)이 아닙니다. '
                '카메라 열기가 실패할 수 있습니다.')
        elif flip != EXPECTED_FLIP_METHOD:
            self.log.warn(
                f'flip-method={flip_method_text(flip)} 는 serbot_camera 의 설정값 '
                f'{flip_method_text(EXPECTED_FLIP_METHOD)} 와 다릅니다. '
                '의도한 설정인지 확인하세요.')

    # ── 로그 억제 ──────────────────────────────────────────────────────

    def _log_throttled(self, key: str, message: str, level: str = 'info',
                       interval: float = 5.0) -> None:
        """같은 종류의 로그가 짧은 간격으로 반복되는 것을 막는다.

        속도가 데드존 경계 근처에서 흔들리면 상태가 매 틱 뒤집혀 로그가 도배된다.
        상태 변화 자체는 그대로 추적하고 출력만 억제한다.

        rclpy 로거는 호출 지점(파일 경로 + 라인 번호)을 키로 설정을 캐시한다.
        따라서 한 줄에서 severity 를 변수로 바꿔가며 호출하면 두 번째 호출에서
        "Logger severity cannot be changed between calls." 예외가 난다.
        아래 세 호출은 그래서 반드시 물리적으로 다른 줄이어야 한다.
        같은 이유로 rclpy 의 throttle_duration_sec 에 의존하지 않고
        time.monotonic() 으로 간격을 직접 관리한다.

        Args:
            key: 로그 종류 구분자.
            message: 출력할 내용.
            level: 심각도. 'error' / 'warn' / 그 외는 info 로 처리한다.
            interval: 같은 key 로그의 최소 간격(초).
        """
        now = time.monotonic()
        # 느린 스레드와 빠른 타이머가 함께 호출하므로 딕셔너리를 잠근다.
        with self._log_lock:
            if now - self._log_at.get(key, -1e9) < interval:
                return
            self._log_at[key] = now

        if level == 'error':
            self.log.error(message)
        elif level == 'warn':
            self.log.warn(message)
        else:
            self.log.info(message)

    # ── STEP 5: 외부 명령 콜백 ────────────────────────────────────────

    def _on_instruction(self, msg: Any) -> None:
        """자연어 지시문을 교체한다.

        빈 문자열은 무시한다. 실수로 지시문을 지우면 모델이 기본값도 아닌
        빈 프롬프트를 받게 되므로 방어한다.

        Args:
            msg: std_msgs/String. data 가 새 지시문.
        """
        text = (msg.data or '').strip()
        if not text:
            self.log.warn('빈 instruction 수신 — 무시한다')
            return
        with self._cmd_state_lock:
            if text == self._instruction:
                return
            old = self._instruction
            self._instruction = text
        self.log.info(f'instruction 교체: {old!r} -> {text!r}')

    def _on_estop(self, msg: Any) -> None:
        """비상정지를 래치한다.

        True 는 즉시 걸리고, False 는 명시적 해제로만 취급한다.
        대시보드 버튼은 True 만 보내므로 해제는 사람이 직접 발행해야 한다.

        Args:
            msg: std_msgs/Bool.
        """
        value = bool(msg.data)
        with self._cmd_state_lock:
            if value == self._estop:
                return
            self._estop = value
        if value:
            self.log.error('비상정지 수신 — 0 발행으로 고정한다')
        else:
            self.log.warn('비상정지 해제 요청 — 명령 발행을 재개한다')

    def _on_enable(self, msg: Any) -> None:
        """주행 허용 게이트를 갱신한다.

        enable=True 재발행은 ARRIVED 래치 해제로도 쓴다. 값이 이미 True 여도
        도착 래치가 걸려 있으면 풀어 주행을 재개한다.

        Args:
            msg: std_msgs/Bool.
        """
        value = bool(msg.data)
        cleared_arrive = False
        gated = False
        with self._cmd_state_lock:
            # True 재발행으로 ARRIVED 래치를 푼다 (프로세스 재시작 없이 데모 재시작).
            if value and self._arrived:
                self._arrived = False
                self._arrive_candidate_since = 0.0
                self._arrive_elapsed = 0.0
                cleared_arrive = True
            if value != self._enabled:
                self._enabled = value
                gated = True
        if cleared_arrive:
            self.log.info(
                'enable True 재수신 — ARRIVED 래치 해제, 명령 발행을 재개한다')
        if gated:
            self.log.info(f'enable 게이트: {value}')

    def _on_heartbeat(self, msg: Any) -> None:
        """대시보드 생존 신호를 기록한다.

        Args:
            msg: std_msgs/Bool. 내용은 보지 않고 수신 시각만 쓴다.
        """
        with self._cmd_state_lock:
            self._heartbeat_at = time.monotonic()

    def _current_instruction(self) -> str:
        """현재 지시문을 스레드 안전하게 읽는다.

        Returns:
            느린 경로에 넘길 자연어 지시문.
        """
        with self._cmd_state_lock:
            return self._instruction

    def _safety_block_reason(self) -> Optional[str]:
        """명령을 0 으로 죽여야 하는 이유를 돌려준다.

        Returns:
            차단 사유 문자열. 차단할 이유가 없으면 None.
        """
        now = time.monotonic()
        with self._cmd_state_lock:
            if self._estop:
                return 'estop'
            if not self._enabled:
                return 'disabled'
            timeout = self.cfg.heartbeat_timeout
            if timeout > 0.0:
                if self._heartbeat_at == 0.0:
                    return 'heartbeat_none'
                if now - self._heartbeat_at > timeout:
                    return 'heartbeat_lost'
        return None

    # ── 오도메트리 ─────────────────────────────────────────────────────

    def _on_odom(self, msg: Any) -> None:
        """오도메트리에서 속도와 위치를 갱신한다.

        Args:
            msg: nav_msgs/Odometry 메시지.
        """
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        twist = msg.twist.twist
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2)

        with self._odom_lock:
            self._odom_ok = True
            self._odom_x = float(position.x)
            self._odom_y = float(position.y)
            self._odom_yaw = math.atan2(siny, cosy)
            self._odom_vx = float(twist.linear.x)
            self._odom_vy = float(twist.linear.y)
            self._odom_wz = float(twist.angular.z)

    def _odom_snapshot(self) -> tuple[bool, float, float, float, float, float, float]:
        """현재 오도메트리 값을 한 번에 읽는다.

        Returns:
            (수신 여부, x, y, yaw, vx, vy, wz).
        """
        with self._odom_lock:
            return (self._odom_ok, self._odom_x, self._odom_y, self._odom_yaw,
                    self._odom_vx, self._odom_vy, self._odom_wz)

    def _build_robot_state(self, cache: Optional[VlmCache]) -> torch.Tensor:
        """모델 입력용 robot_state [vx, vy, yaw_speed, dx, dy] 를 만든다.

        dx, dy 는 캐시를 캡처한 시점부터 지금까지의 변위다. 오도메트리는 odom
        프레임 기준이므로, 캡처 당시 로봇 자세로 회전시켜 로봇 기준 변위로 바꾼다.

        Args:
            cache: 현재 VLM 캐시. None 이면 변위를 0 으로 둔다.

        Returns:
            (5,) bfloat16 CUDA 텐서.
        """
        available, x, y, _yaw, vx, vy, wz = self._odom_snapshot()
        if not available:
            if not self._odom_warned:
                self.log.warn(
                    f'{self.cfg.odom_topic} 수신 없음 — robot_state 를 0 으로 둔다 '
                    '(이 경고는 한 번만 표시)')
                self._odom_warned = True
            return torch.zeros(5, dtype=torch.bfloat16,
                               device=self.model.device)

        dx = dy = 0.0
        if cache is not None:
            world_dx = x - cache.odom_x
            world_dy = y - cache.odom_y
            cos_a = math.cos(-cache.odom_yaw)
            sin_a = math.sin(-cache.odom_yaw)
            dx = world_dx * cos_a - world_dy * sin_a
            dy = world_dx * sin_a + world_dy * cos_a

        return torch.tensor([vx, vy, wz, dx, dy], dtype=torch.bfloat16,
                            device=self.model.device)

    # ── 느린 루프 ──────────────────────────────────────────────────────

    def _vlm_loop(self) -> None:
        """느린 스레드 본체. 주기적으로 predict() 를 돌려 캐시를 갱신한다."""
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._vlm_once()
            except Exception as exc:  # noqa: BLE001
                self._vlm_fails += 1
                self.log.error(f'VLM 루프 실패({self._vlm_fails}회) — 캐시 유지: {exc}')

            # 남은 시간만큼 쉬되, 종료 요청에는 즉시 반응한다.
            remaining = self.cfg.vlm_period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _vlm_once(self) -> None:
        """느린 루프 1회. 프레임을 골라 predict() 를 돌리고 캐시를 교체한다."""
        delayed = self._store.pick_delayed(
            self.cfg.num_delayed, self.cfg.delayed_spacing, self.cfg.min_delayed)
        latest = self._store.latest()
        if delayed is None or latest is None:
            self.log.info('프레임이 아직 부족하다 — VLM 실행 대기')
            return

        with self._cache_lock:
            cache = self._cache
        robot_state = self._build_robot_state(cache)
        _ok, ox, oy, oyaw, _vx, _vy, _wz = self._odom_snapshot()

        # predict() 는 14초 가까이 걸리고 그동안 같은 경로를 다시 연다.
        # 링버퍼가 그 파일을 지우지 못하도록 사용하는 동안 고정해 둔다.
        in_use = delayed + [latest]
        self._store.pin(in_use)
        # 저장은 pin 안에서 한다. 링버퍼가 파일을 지우기 전에 열어야 한다.
        frame_paths = self._dump_inputs([str(r.path) for r in in_use])
        # STEP 5: 느린 경로 진입 시점에 한 번 읽어 두 분기가 같은 값을 쓰게 한다.
        instruction = self._current_instruction()
        started = time.monotonic()
        try:
            # 느린 경로는 GPU 를 오래 쓴다. 정책에 따라 배타적으로 잡는다.
            if self.cfg.gpu_lock == 'exclusive':
                with self._gpu_lock:
                    past, response = self.model.run_slow(
                        [str(r.path) for r in delayed], str(latest.path),
                        instruction, robot_state, 0.0)
            else:
                past, response = self.model.run_slow(
                    [str(r.path) for r in delayed], str(latest.path),
                    instruction, robot_state, 0.0)
        finally:
            self._store.unpin(in_use)
        elapsed = time.monotonic() - started

        with self._cache_lock:
            self._cache = VlmCache(past=past, captured_at=time.monotonic(),
                                   response=response, odom_x=ox, odom_y=oy,
                                   odom_yaw=oyaw)
        self._vlm_runs += 1
        self.log.info(
            f'VLM 캐시 갱신 #{self._vlm_runs} ({elapsed:.2f}s, '
            f'delayed {len(delayed)}장)')
        # 부호 검증에서 "모델이 장애물을 실제로 봤는가" 를 사람이 판단할 자료다.
        # 줄바꿈을 합쳐 한 줄로 남긴다. 텍스트 로그에서 정규식으로 뽑기 쉽게 하려는
        # 것이고, 원문 그대로는 JSONL 쪽에 남는다.
        self.log.info(f'VLM 응답 #{self._vlm_runs}: {" ".join(response.split())}')
        self._write_vlm_jsonl(response, elapsed, frame_paths, instruction)

    def _dump_inputs(self, paths: list[str]) -> list[str]:
        """--dump-frames 가 켜져 있으면 모델 입력 이미지를 저장한다.

        추론 직전에 부른다. 파일명에 쓰는 갱신 번호는 이번 추론이 성공했을 때
        붙을 번호(현재값 + 1)라서 로그의 "VLM 캐시 갱신 #n" 과 맞는다.

        Args:
            paths: predict() 에 넘기는 경로. 마지막 항목이 현재 프레임이다.

        Returns:
            저장한 파일 경로 목록. 비활성이면 빈 목록.
        """
        if self._dump_dir is None:
            return []
        return self.model.dump_model_inputs(
            paths, self._dump_dir, self._vlm_runs + 1, self._camera_source())

    def _camera_source(self) -> str:
        """현재 프레임 공급원 문자열.

        Returns:
            'live' / 'fallback' / 'none'.
        """
        return self._camera.source

    # ── 빠른 루프 ──────────────────────────────────────────────────────

    def _fast_tick(self) -> None:
        """빠른 타이머 콜백. waypoint 를 뽑아 Twist 로 바꿔 발행한다."""
        try:
            self._fast_tick_inner()
        except Exception as exc:  # noqa: BLE001
            # 어떤 실패도 노드를 죽이지 않는다. 안전한 상태(정지)로 수렴시킨다.
            if self._profiler is not None:
                self._profiler.cancel_tick()
            self.log.error(f'빠른 루프 예외 — 정지 발행: {exc}')
            self._publish(self._Twist(), reason='error')

    def _fast_tick_inner(self) -> None:
        """빠른 루프 실제 처리."""
        if self.cfg.color_nav:
            # VLM 캐시를 기다리지 않는다 — 이 경로는 애초에 VLM 을 거치지 않는다.
            self._color_tick()
            return

        with self._cache_lock:
            cache = self._cache
        if cache is None:
            # 캐시가 없으면 아무것도 내보내지 않는다. safety_mux 가 정지시킨다.
            self._publish_status(None, None, 'waiting_vlm')
            return

        profiler = self._profiler
        # 프레임 획득 구간: 링버퍼 조회 + GPU 락 확보 + pin 까지 포함한다.
        frame_started = time.perf_counter() if profiler is not None else 0.0

        latest = self._store.latest()
        if latest is None:
            self._publish_status(cache, None, 'no_frame')
            return

        # GPU 경합 회피: 느린 스레드가 쓰는 중이면 건너뛴다.
        acquired = True
        if self.cfg.gpu_lock == 'exclusive':
            acquired = self._gpu_lock.acquire(blocking=False)
        if not acquired:
            self._reuse_or_zero(cache)
            return

        self._store.pin([latest])
        if profiler is not None:
            profiler.enter_tick()
            profiler.add('t_frame_get', time.perf_counter() - frame_started)
        try:
            time_delay = time.monotonic() - cache.captured_at
            if profiler is None:
                robot_state = self._build_robot_state(cache)
            else:
                with profiler.section('t_state_build'):
                    robot_state = self._build_robot_state(cache)
            started = time.monotonic()
            waypoints = self.model.run_fast(
                latest, robot_state, time_delay, cache.past, profiler)
            self._last_fast_ms = (time.monotonic() - started) * 1000.0
        finally:
            self._store.unpin([latest])
            if self.cfg.gpu_lock == 'exclusive':
                self._gpu_lock.release()

        if profiler is None:
            cmd, info = self._waypoints_to_twist(waypoints)
        else:
            with profiler.section('t_postprocess'):
                cmd, info = self._waypoints_to_twist(waypoints)
            # 전체 시간은 프레임 획득부터 후처리까지를 합쳐서 본다.
            profiler.end_tick(time.perf_counter() - frame_started)
        if cmd is None:
            self._publish(self._Twist(), reason='invalid_waypoints')
            self._publish_status(cache, info, 'invalid')
            return

        self._last_cmd = cmd
        self._last_cmd_at = time.monotonic()
        # ARRIVED 확정이면 상태 문자열에 남겨 대시보드가 구분할 수 있게 한다.
        arrived = bool(info.get('arrived'))
        reason = 'arrived' if arrived else 'ok'
        state = 'arrived' if arrived else 'ok'
        self._publish(cmd, reason=reason)
        self._publish_status(cache, info, state)
        self._write_jsonl(cache, info)

    # ── 색상 타겟팅 (--color-nav) ──────────────────────────────────────

    def _color_tick(self) -> None:
        """색상 기반 시각 서보 한 틱. VLM/action expert 를 거치지 않는다.

        도착 판정은 `_waypoints_to_twist()` 와 같은 2단 규약(순간 후보 ->
        `--arrive-hold` 초 연속 유지 -> 래치)을 그대로 재사용한다. 그래서
        `--enable` 재발행으로 래치를 푸는 `_on_enable()` 동작도 그대로 적용된다.
        """
        now = time.monotonic()

        if self._arrived and self.cfg.arrive_latch:
            # 래치돼 있으면 카메라를 다시 볼 필요도 없다 — 그대로 정지 유지.
            self._publish(self._Twist(), reason='color_arrived')
            info = {'arrived': True, 'disp': self._last_disp,
                    'arrive_elapsed': self._arrive_elapsed,
                    'color': {'target': self._color_last_target}}
            self._publish_status(None, info, 'color_arrived')
            return

        latest = self._store.latest()
        if latest is None or latest.array is None:
            self._publish(self._Twist(), reason='color_no_frame')
            self._publish_status(
                None, {'color': {'target': self._color_last_target, 'found': False}},
                'color_no_frame')
            return

        frame = latest.array
        h, w = frame.shape[:2]

        instruction = self._current_instruction()
        parsed = color_target.parse_target_color(instruction, self._color_ranges)
        if parsed is not None:
            self._color_last_target = parsed
        target = self._color_last_target

        blobs = color_target.detect_blobs(
            frame, self._color_ranges, min_area=self.cfg.color_min_area)
        blob = blobs.get(target) if target else None
        servo = color_target.compute_servo(blob, target, w, h, self._color_servo_cfg)

        if (self.cfg.color_debug_dir
                and now - self._color_dump_at >= self.cfg.color_debug_period):
            self._color_dump_at = now
            self._dump_color_debug(frame, blobs, target)

        twist = self._Twist()
        if not servo.found:
            # 목표색을 못 찾음. 기본은 제자리 정지 — --color-search-wz 를 주면
            # 그 각속도로 제자리 탐색 회전을 한다(기본 0.0 = 더 안전한 정지).
            twist.angular.z = self.cfg.color_search_wz
            reason = 'color_lost'
            self._arrive_candidate_since = 0.0
            self._arrive_elapsed = 0.0
        elif servo.arrived:
            # compute_servo() 가 이미 면적 임계값을 넘겼다고 본 순간(후보) 상태.
            if self._arrive_candidate_since <= 0.0:
                self._arrive_candidate_since = now
            self._arrive_elapsed = now - self._arrive_candidate_since
            if (self._arrive_elapsed >= self.cfg.arrive_hold
                    and not self._arrived):
                self._arrived = True
                self.log.info(
                    f'색상 ARRIVED 확정 — target={target} '
                    f'{self._arrive_elapsed:.1f}s 연속')
            reason = 'color_arrived' if self._arrived else 'color_hold'
        else:
            self._arrive_candidate_since = 0.0
            self._arrive_elapsed = 0.0
            twist.linear.x = servo.vx
            twist.angular.z = servo.wz
            reason = 'color_track'

        self._last_disp = (
            0.0 if blob is None
            else max(0.0, (self.cfg.color_stop_area - blob.area)
                     / self.cfg.color_stop_area))

        self._publish(twist, reason=reason)
        info: dict[str, Any] = {
            'cmd': {'vx': twist.linear.x, 'vy': 0.0, 'wz': twist.angular.z},
            'arrived': self._arrived,
            'disp': self._last_disp,
            'arrive_elapsed': self._arrive_elapsed,
            'color': {
                'target': target,
                'found': servo.found,
                'area': (blob.area if blob else None),
                'cx': (blob.cx if blob else None),
                'cy': (blob.cy if blob else None),
                'detected': sorted(blobs.keys()),
            },
        }
        self._publish_status(None, info, reason)
        if self._jsonl is not None:
            record = dict(info)
            record['event'] = 'tick'
            record['t'] = time.time()
            record['instruction'] = instruction
            record['camera_source'] = self._camera_source()
            self._append_jsonl(record)

    def _dump_color_debug(self, frame: np.ndarray, blobs: dict[str, color_target.Blob],
                          target: Optional[str]) -> None:
        """탐지 결과를 주석 이미지로 저장한다 (튜닝용, `--color-debug-dir`).

        Args:
            frame: 원본 BGR 프레임.
            blobs: 이번 틱에서 찾은 색상 블롭들.
            target: 현재 목표 색.
        """
        try:
            out_dir = Path(self.cfg.color_debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated = color_target.annotate(frame, blobs, target)
            path = out_dir / f'color_{int(time.time())}.jpg'
            cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as exc:  # noqa: BLE001
            self._log_throttled('color_debug_dump_fail',
                                f'색상 디버그 이미지 저장 실패: {exc}', level='warn')

    def _reuse_or_zero(self, cache: VlmCache) -> None:
        """GPU 를 못 잡았을 때 직전 명령을 재사용하거나 정지한다.

        Args:
            cache: 현재 VLM 캐시 (상태 보고용).
        """
        # 도착 래치 중에는 직전 명령을 재사용하지 않는다.
        if self._arrived and self.cfg.arrive_latch:
            self._publish(self._Twist(), reason='arrived')
            self._publish_status(cache, {
                'disp': self._last_disp,
                'arrive_elapsed': self._arrive_elapsed,
                'arrived': True,
                'deadzone_blocked': False,
            }, 'arrived')
            return
        age = time.monotonic() - self._last_cmd_at
        if self._last_cmd is not None and age <= self.cfg.stale_limit:
            self._publish(self._last_cmd, reason='reuse')
            self._publish_status(cache, self._last_status.get('waypoint_final'),
                                 'gpu_busy_reuse')
        else:
            self._publish(self._Twist(), reason='stale')
            self._publish_status(cache, None, 'gpu_busy_stale')

    # ── waypoint -> Twist ──────────────────────────────────────────────

    def _waypoints_to_twist(
        self, waypoints: torch.Tensor,
    ) -> tuple[Optional[Any], dict[str, Any]]:
        """waypoint 텐서를 Twist 명령으로 변환하고 2단 도착/데드존 판정을 한다.

        판정 순서:
          1) ARRIVED 래치 → 0
          2) disp < arrive_threshold → 도착 후보(0), 타이머 누적
          3) speed < deadzone → deadzone 정책, 도착 타이머 리셋
          4) 그 외 → 그대로 발행, 도착 타이머 리셋

        Args:
            waypoints: (1, T, 2) 텐서. 현재 프레임 기준 상대 변위 [dx, dy].

        Returns:
            (Twist 또는 None, 상태 정보 딕셔너리). NaN/Inf 면 None 을 돌려준다.
        """
        values = waypoints.detach().to(torch.float32).cpu()[0]
        total_steps = int(values.shape[0])

        if bool(torch.isnan(values).any()) or bool(torch.isinf(values).any()):
            self.log.error('waypoint 에 NaN/Inf 포함 — 명령을 만들지 않는다')
            return None, {'nan_inf': True}

        index = max(1, min(self.cfg.lookahead_step, total_steps - 1))
        dx = float(values[index, 0])
        dy = float(values[index, 1])
        horizon = index * STEP_DT_SEC
        # 변위 기준. 속도는 lookahead 에 흔들리므로 도착 판정에는 쓰지 않는다.
        disp = math.hypot(dx, dy)
        self._last_disp = disp

        final_dx = float(values[-1, 0])
        final_dy = float(values[-1, 1])
        now = time.monotonic()

        def _info(vx: float, vy: float, wz: float, *, blocked: bool,
                  limited: bool, arrive_elapsed: float) -> dict[str, Any]:
            """status / JSONL 공통 필드를 만든다."""
            return {
                'lookahead_step': index,
                'target': {'dx': dx, 'dy': dy},
                'waypoint_final': {
                    'dx': final_dx,
                    'dy': final_dy,
                    'distance_m': math.hypot(final_dx, final_dy),
                },
                'cmd': {'vx': vx, 'vy': vy, 'wz': wz},
                'speed_limited': limited,
                'deadzone_blocked': blocked,
                'disp': disp,
                'arrive_elapsed': round(arrive_elapsed, 3),
                'arrived': self._arrived,
            }

        # ── 1) ARRIVED 래치: 확정 후 계속 0 ──────────────────────────
        if self._arrived and self.cfg.arrive_latch:
            self._arrive_elapsed = max(
                self._arrive_elapsed,
                now - self._arrive_candidate_since
                if self._arrive_candidate_since > 0.0 else self._arrive_elapsed)
            cmd = self._Twist()
            return cmd, _info(0.0, 0.0, 0.0, blocked=False, limited=False,
                              arrive_elapsed=self._arrive_elapsed)

        # 목표 지점까지의 평균 속도를 그대로 명령으로 쓴다.
        vx = dx / horizon * self.cfg.speed_scale
        vy = dy / horizon * self.cfg.speed_scale

        speed = math.hypot(vx, vy)
        limited = False
        if speed > self.cfg.max_speed and speed > 0.0:
            # 방향은 유지하고 크기만 줄인다.
            ratio = self.cfg.max_speed / speed
            vx, vy = vx * ratio, vy * ratio
            speed = self.cfg.max_speed
            limited = True

        blocked = False
        arrive_elapsed = 0.0

        # ── 2단 판정: 도착(변위) → 데드존(속도) ──────────────────────
        if disp < self.cfg.arrive_threshold:
            # 도착 후보. 복도 중간 감속과 구분하려면 연속 유지 시간이 필요하다.
            if self._arrive_candidate_since <= 0.0:
                self._arrive_candidate_since = now
            arrive_elapsed = now - self._arrive_candidate_since
            self._arrive_elapsed = arrive_elapsed
            vx = vy = 0.0
            speed = 0.0
            if (not self._arrived
                    and arrive_elapsed >= self.cfg.arrive_hold):
                self._arrived = True
                self.log.info(
                    f'ARRIVED 확정 — {arrive_elapsed:.1f}s 연속, '
                    f'disp={disp:.4f}m (임계 {self.cfg.arrive_threshold:.3f}m)'
                    + (' | 래치 ON (enable True 재발행으로 해제)'
                       if self.cfg.arrive_latch else ' | 래치 OFF'))
        else:
            # 변위가 임계 이상이면 도착 타이머를 반드시 리셋한다.
            # 안 그러면 복도 중간에서 잠깐 느려질 때마다 카운터가 쌓인다.
            if self._arrive_candidate_since > 0.0 or self._arrive_elapsed > 0.0:
                self._arrive_candidate_since = 0.0
                self._arrive_elapsed = 0.0
            if self._arrived and not self.cfg.arrive_latch:
                self._arrived = False
                self.log.info(
                    f'도착 조건 해제 — disp={disp:.4f}m, 주행을 재개한다')

            # 데드존은 도착 판정과 독립. 도착이 아닐 때만 적용한다.
            if 0.0 < speed < MIN_EFFECTIVE_LINEAR:
                if self.cfg.deadzone_policy == 'boost':
                    ratio = MIN_EFFECTIVE_LINEAR / speed
                    vx, vy = vx * ratio, vy * ratio
                    speed = MIN_EFFECTIVE_LINEAR
                else:
                    vx = vy = 0.0
                    blocked = True

        if blocked != self._deadzone_blocked:
            # 상태가 바뀔 때만, 그것도 억제 간격을 두고 로그를 남긴다.
            if blocked:
                self._log_throttled(
                    'deadzone',
                    f'속도 {speed:.4f} m/s 가 데드존({MIN_EFFECTIVE_LINEAR}) 미만 — '
                    '0 발행 (--deadzone-policy boost 로 바꾸면 최소값으로 올린다)',
                    level='warn')
            else:
                self._log_throttled('deadzone', '데드존 해제 — 정상 명령 발행')
            self._deadzone_blocked = blocked

        cmd = self._Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = self._heading_rate(dx, dy) if self.cfg.use_heading else 0.0

        return cmd, _info(vx, vy, cmd.angular.z, blocked=blocked,
                          limited=limited, arrive_elapsed=arrive_elapsed)

    def _heading_rate(self, dx: float, dy: float) -> float:
        """진행 방향으로 몸통을 돌리는 각속도를 계산한다 (기본 비활성).

        옴니휠이라 회전 없이도 이동할 수 있으므로 기본값은 0 이다.
        `--use-heading` 을 켰을 때만 이 값을 쓴다.

        Args:
            dx: 목표 지점 전진 변위.
            dy: 목표 지점 좌측 변위.

        Returns:
            제한된 각속도(rad/s).
        """
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        error = math.atan2(dy, dx)
        rate = error * self.cfg.heading_gain
        return max(-self.cfg.max_yaw, min(self.cfg.max_yaw, rate))

    # ── 발행 ───────────────────────────────────────────────────────────

    def _publish(self, cmd: Any, reason: str) -> None:
        """제어 명령을 발행한다. Shadow Mode 면 발행하지 않는다.

        Args:
            cmd: 보낼 Twist.
            reason: 상태 전이 로그용 사유 문자열.
        """
        if reason != self._published_state:
            self._log_throttled(
                'cmd_state',
                f'명령 상태 전이: {self._published_state or "init"} -> {reason} '
                f'(vx={cmd.linear.x:+.3f}, vy={cmd.linear.y:+.3f})')
            self._published_state = reason
        if self.cfg.shadow:
            return
        self._cmd_pub.publish(cmd)

    def _publish_status(self, cache: Optional[VlmCache],
                        info: Optional[dict[str, Any]], state: str) -> None:
        """모니터링용 상태 JSON 을 발행한다.

        Args:
            cache: 현재 VLM 캐시.
            info: waypoint/명령 정보.
            state: 현재 루프 상태 문자열.
        """
        buffered, save_fps = self._store.stats()
        odom_ok = self._odom_snapshot()[0]
        payload: dict[str, Any] = {
            'state': state,
            'shadow': self.cfg.shadow,
            'vlm_age_sec': (round(time.monotonic() - cache.captured_at, 2)
                            if cache else None),
            'vlm_response': cache.response if cache else None,
            'vlm_runs': self._vlm_runs,
            'vlm_fails': self._vlm_fails,
            'camera_fps': round(self._camera.capture_fps, 1),
            'camera_fallback': self._camera.is_fallback,
            'frame_save_fps': round(save_fps, 1),
            'frames_buffered': buffered,
            'fast_loop_ms': round(self._last_fast_ms, 1),
            'waypoint_final': (info or {}).get('waypoint_final'),
            'cmd': (info or {}).get('cmd'),
            'deadzone_blocked': (info or {}).get('deadzone_blocked', False),
            'disp': (info or {}).get('disp', self._last_disp),
            'arrive_elapsed': (info or {}).get(
                'arrive_elapsed', self._arrive_elapsed),
            'arrived': (info or {}).get('arrived', self._arrived),
            'odom_available': odom_ok,
            'color': (info or {}).get('color'),
        }
        self._last_status = payload
        message = self._String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._status_pub.publish(message)

    def _write_jsonl(self, cache: VlmCache, info: dict[str, Any]) -> None:
        """추론 기록을 JSONL 로 남긴다.

        Args:
            cache: 현재 VLM 캐시.
            info: waypoint/명령 정보.
        """
        if self._jsonl is None:
            return
        record = dict(info)
        record['event'] = 'tick'
        record['t'] = time.time()
        record['vlm_age_sec'] = time.monotonic() - cache.captured_at
        record['fast_loop_ms'] = self._last_fast_ms
        # 폴백 데이터를 나중에 자동으로 걸러낼 수 있도록 공급원을 함께 남긴다.
        record['camera_source'] = self._camera_source()
        self._append_jsonl(record)

    def _write_vlm_jsonl(self, response: str, elapsed: float,
                         frame_paths: list[str],
                         instruction: str) -> None:
        """VLM 응답 원문을 JSONL 에 별도 레코드로 남긴다.

        빠른 틱 레코드마다 응답을 넣으면 같은 텍스트가 매 초 반복되어 파일이
        커진다. 갱신 시점에 한 번만 기록한다.

        Args:
            response: VLM 응답 원문.
            elapsed: predict() 소요 시간(초).
            frame_paths: --dump-frames 로 저장한 모델 입력 경로. 비활성이면 빈 목록.
            instruction: 이번 추론에 쓴 자연어 지시문.
        """
        if self._jsonl is None:
            return
        self._append_jsonl({
            'event': 'vlm_refresh',
            'vlm_runs': self._vlm_runs,
            'response': response,
            'elapsed_sec': elapsed,
            'instruction': instruction,
            'disp': self._last_disp,
            'arrived': self._arrived,
            'camera_source': self._camera_source(),
            'frame_paths': frame_paths,
            't': time.time(),
        })

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        """레코드 한 줄을 JSONL 에 붙여 쓴다.

        Args:
            record: 기록할 딕셔너리.
        """
        if self._jsonl is None:
            return
        try:
            self._jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._jsonl.flush()
        except OSError as exc:
            self.log.warn(f'JSONL 기록 실패: {exc}')

    # ── 종료 ───────────────────────────────────────────────────────────

    def shutdown(self) -> bool:
        """정지 명령을 내보내고 워커·카메라·GPU 를 정리한다.

        rclpy 컨텍스트가 살아 있는 동안 호출되어야 정지 명령이 실제로 나간다.

        Returns:
            모든 스레드가 제때 끝났으면 True. VLM 스레드가 predict() 중이라
            남아 있으면 False (호출 측이 인터프리터 종료를 건너뛰어야 한다).
        """
        print('종료 중 — 정지 명령 발행', flush=True)
        self._stop.set()

        if not self.cfg.shadow:
            for _ in range(self.cfg.zero_burst):
                try:
                    self._cmd_pub.publish(self._Twist())
                except Exception as exc:  # noqa: BLE001
                    print(f'정지 명령 발행 실패: {exc}', file=sys.stderr)
                    break
                time.sleep(0.02)

        self._camera.stop()
        self._camera.join(timeout=3.0)
        clean = True
        if self._vlm_thread.is_alive():
            # 진행 중인 predict() 는 14초 가까이 걸린다. 그만큼 기다리면 Ctrl+C
            # 반응이 느려지므로 짧게만 기다리고 남으면 호출 측에 알린다.
            self._vlm_thread.join(timeout=self.cfg.join_timeout)
            if self._vlm_thread.is_alive():
                clean = False
                print('VLM 스레드가 predict() 중 — 인터프리터 정리를 건너뛴다',
                      file=sys.stderr)

        if self._jsonl is not None:
            self._jsonl.close()

        with self._cache_lock:
            self._cache = None
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        self.node.destroy_node()
        return clean


# ======================================================================
# CLI / main
# ======================================================================
def parse_args(argv: list[str]) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: 인자 리스트.

    Returns:
        파싱된 Namespace. --enable 이 주어지면 shadow 가 꺼진다.
    """
    home = Path.home() / 'TIC-VLA'
    parser = argparse.ArgumentParser(
        description='TIC-VLA <-> SerBot II 주행 브릿지 (기본 Shadow Mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 안전
    parser.add_argument('--shadow', action='store_true', default=True,
                        help='계산만 하고 명령을 발행하지 않는다 (기본 동작)')
    parser.add_argument('--enable', action='store_true',
                        help='실제로 명령을 발행한다. --shadow 를 해제')
    # 주행
    parser.add_argument('--allow-reverse', dest='forward_only',
                        action='store_false', default=True,
                        help='vx 음수(후진)를 허용한다. 기본은 0 으로 클램프한다. '
                             '전방 카메라만 있어 후방은 관측되지 않으므로 기본 금지다')
    parser.add_argument('--instruction-topic', default='/ticvla/instruction',
                        help='자연어 지시문을 받는 토픽 (std_msgs/String)')
    parser.add_argument('--estop-topic', default='/ticvla/emergency_stop',
                        help='비상정지 토픽 (std_msgs/Bool, TRANSIENT_LOCAL)')
    parser.add_argument('--enable-topic', default='/ticvla/enable',
                        help='주행 허용 게이트 토픽 (std_msgs/Bool)')
    parser.add_argument('--require-enable', action='store_true',
                        help='켜면 /ticvla/enable 에 True 가 오기 전까지 0 만 발행한다')
    parser.add_argument('--heartbeat-topic', default='/ticvla/heartbeat',
                        help='대시보드 생존 신호 토픽 (std_msgs/Bool)')
    parser.add_argument('--heartbeat-timeout', type=float, default=0.0,
                        help='이 시간(초) 동안 heartbeat 가 없으면 0 을 발행한다. '
                             '0 이면 감시하지 않는다 (기본)')
    parser.add_argument('--instruction', default='Move forward safely and efficiently.',
                        help='자연어 지시문')
    parser.add_argument('--rate', type=float, default=2.0, help='빠른 루프 Hz')
    parser.add_argument('--vlm-period', type=float, default=15.0,
                        help='느린 루프 주기(초). 실측 predict() 13.85s')
    parser.add_argument('--lookahead-step', type=int, default=5,
                        help='목표 waypoint 스텝 (5 = 0.5초 앞)')
    parser.add_argument('--speed-scale', type=float, default=1.0, help='속도 배율')
    parser.add_argument('--max-speed', type=float, default=0.20,
                        help='최대 속도 m/s')
    parser.add_argument('--deadzone-policy', choices=('zero', 'boost'),
                        default='zero',
                        help=f'데드존({MIN_EFFECTIVE_LINEAR} m/s) 미만일 때 처리')
    parser.add_argument('--arrive-threshold', type=float, default=0.05,
                        help='lookahead 지점까지의 변위(m)가 이 값 미만이면 '
                             '도착 후보로 본다. 속도가 아니라 변위 기준이므로 '
                             'lookahead 값에 흔들리지 않는다')
    parser.add_argument('--arrive-hold', type=float, default=3.0,
                        help='도착 후보 상태가 이 시간(초) 이상 연속되면 ARRIVED 확정')
    parser.add_argument('--arrive-latch', action='store_true', default=True,
                        help='ARRIVED 를 래치한다. 데모에서는 도착이 종료 조건이므로 기본 켜짐')
    parser.add_argument('--no-arrive-latch', dest='arrive_latch',
                        action='store_false',
                        help='ARRIVED 를 래치하지 않고 조건이 풀리면 주행을 재개한다')
    parser.add_argument('--stale-limit', type=float, default=2.0,
                        help='직전 명령 재사용 허용 시간(초)')
    parser.add_argument('--use-heading', action='store_true',
                        help='진행 방향으로 몸통을 돌리는 각속도를 계산 (기본 꺼짐)')
    parser.add_argument('--heading-gain', type=float, default=0.8,
                        help='--use-heading 일 때 방향 오차 P 게인')
    parser.add_argument('--max-yaw', type=float, default=0.4,
                        help='--use-heading 일 때 최대 각속도 rad/s')
    # 프레임
    parser.add_argument('--num-delayed', type=int, default=4,
                        help='delayed 프레임 수')
    parser.add_argument('--min-delayed', type=int, default=2,
                        help='이보다 적게 모이면 VLM 실행을 미룬다')
    parser.add_argument('--delayed-spacing', type=float, default=1.0,
                        help='delayed 프레임 간 목표 시간 간격(초)')
    parser.add_argument('--frame-dir', default='/dev/shm/ticvla_frames',
                        help='프레임 저장 경로 (SD 수명 보호를 위해 tmpfs 권장)')
    parser.add_argument('--image-dir', default=str(home / 'logs/test_seq'),
                        help='--allow-fallback 일 때 순환 재생할 폴백 이미지 폴더')
    parser.add_argument('--allow-fallback', action='store_true',
                        help='카메라를 확보하지 못해도 종료(코드 2)하지 않고 '
                             '--image-dir 의 정지 이미지로 폴백한다. 측정에서는 '
                             '절대 쓰지 말 것 — 폴백 이미지로 재면 결과가 무효다. '
                             '기본값은 폴백 없이 즉시 종료(카메라 필수)다')
    parser.add_argument('--require-camera', action='store_true',
                        help='(deprecated, no-op) 카메라 필수가 이제 기본 동작이라 '
                             '이 플래그는 아무 효과가 없다. 폴백을 쓰려면 '
                             '--allow-fallback 을 대신 쓸 것')
    parser.add_argument('--dump-frames', default=None, metavar='DIR',
                        help='VLM 추론에 실제로 넣은 이미지를 이 디렉터리에 저장한다 '
                             '(전처리 후 448x448 RGB, flip 적용 상태). 장면을 눈으로 '
                             '확인하는 검증 자료')
    parser.add_argument('--buffer-size', type=int, default=120,
                        help='유지할 프레임 개수 (save-fps 기준 보관 시간이 '
                             'vlm-period 보다 길어야 안전하다)')
    parser.add_argument('--keep-array', type=int, default=4,
                        help='numpy 원본 배열을 들고 있을 최신 프레임 수. 빠른 '
                             '루프가 JPEG 왕복 없이 전처리하는 데 쓴다 '
                             '(1280x720 BGR 한 장에 약 2.8MB)')
    parser.add_argument('--verify-preprocess', action='store_true',
                        help=f'앞쪽 {VERIFY_TICKS} 틱에서 파일 경로와 메모리 '
                             '전처리 결과의 최대 절대 오차를 비교해 로그로 남긴다')
    parser.add_argument('--save-fps', type=float, default=5.0,
                        help='프레임을 디스크에 저장하는 주기 Hz')
    parser.add_argument('--jpeg-quality', type=int, default=90,
                        help='저장 JPEG 품질')
    parser.add_argument(
        '--camera-topic', default=None, metavar='TOPIC',
        help='주어지면 카메라를 직접 열지 않고 이 sensor_msgs/Image 토픽을 '
             '구독한다 (예: /camera/image_raw_fast, serbot_camera/camera_node.py '
             '가 발행). nvarguscamerasrc 는 한 프로세스만 열 수 있으므로, 웹 '
             '대시보드/조이스틱/데이터수집과 카메라를 동시에 쓰려면 반드시 '
             '이 옵션으로 실행할 것 — --sensor-id/--cam-*/--flip-method 는 '
             '이때 camera_node 쪽 설정을 따르므로 무시된다')
    parser.add_argument(
        '--camera-timeout-sec', type=float, default=2.0,
        help='--camera-topic 모드에서 이 시간(초) 이상 프레임이 끊기면 '
             '경고를 반복하며 대기한다(치명 종료하지 않음 — camera_node 재연결을 '
             '기다린다)')
    parser.add_argument('--sensor-id', type=int, default=0, help='CSI 센서 번호')
    parser.add_argument('--cam-width', type=int, default=1280)
    parser.add_argument('--cam-height', type=int, default=720)
    parser.add_argument('--cam-fps', type=int, default=15)
    parser.add_argument('--flip-method', type=int, default=EXPECTED_FLIP_METHOD,
                        help='nvvidconv 회전 설정. 0 없음 / 1 반시계90 / 2 180도 / '
                             '3 시계90 / 4 좌우반전 / 6 상하반전. 이 로봇은 카메라가 '
                             '180도 뒤집혀 장착되어 있어 2 가 기본값이다 '
                             '(serbot_camera 의 flip_method 와 같은 값)')
    # 토픽
    parser.add_argument('--out-topic', default='/ticvla/cmd_vel',
                        help='제어 명령 토픽. safety_mux 가 이걸 받는다')
    parser.add_argument('--status-topic', default='/ticvla/status')
    parser.add_argument('--odom-topic', default='/wheel/odom',
                        help='오도메트리 토픽. EKF 사용 시 /odometry/filtered')
    # 모델
    parser.add_argument('--model-path', default=str(home / 'models/InternVL3-1B'))
    parser.add_argument('--ckpt-path',
                        default=str(home / 'checkpoints/TIC-VLA-model.ckpt'))
    parser.add_argument('--do-sample', action='store_true',
                        help='VLM 텍스트 생성에 sampling 을 쓴다. 기본은 greedy — '
                             '실측상 속도는 같고(10.51 vs 10.65s) 결정론적이며 '
                             '변위도 크다(0.781 vs 0.520m)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='--do-sample 일 때만 쓰는 sampling 온도')
    parser.add_argument('--max-new-tokens', type=int, default=200,
                        help=f'VLM 생성 토큰 상한. 실측 하한 {MIN_SAFE_NEW_TOKENS} '
                             '미만이면 <answer> 가 잘려 waypoint 가 붕괴한다')
    parser.add_argument('--gpu-lock', choices=('exclusive', 'shared'),
                        default='exclusive',
                        help='exclusive: VLM 실행 중 빠른 루프를 건너뛴다. '
                             'shared: 두 루프가 GPU 를 동시에 쓴다')
    # 계측
    parser.add_argument('--profile-fast', action='store_true',
                        help='빠른 루프를 구간별로 계측한다. 켜면 구간마다 '
                             'cuda synchronize 가 들어가 느려지므로 상시 사용 금지')
    parser.add_argument('--profile-samples', type=int, default=20,
                        help='--profile-fast 일 때 표를 출력하기까지 모을 틱 수')
    parser.add_argument('--profile-json',
                        default=str(home / 'logs/fast_loop_profile.json'),
                        help='구간별 프로파일 결과 저장 경로')
    # 색상 타겟팅 (--color-nav) — VLM/action expert 경로를 완전히 대체한다.
    # 배경: KV 캐시 마지막 레이어 값 하나로만 조건화되는 얇은 언어 경로라
    # "3개 중 지정색" 같은 다중 객체 선택을 학습한 적이 없다 (자세한 근거는
    # color_target.py 모듈 docstring 참고).
    parser.add_argument('--color-nav', action='store_true',
                        help='HSV 색상 검출로 직접 조향한다 (VLM/action expert 를 '
                             '건너뛴다). red/green/blue(빨강/초록/파랑) 중 '
                             'instruction 에서 목표색을 골라 그 물체로 접근')
    parser.add_argument('--no-vlm', action='store_true',
                        help='--color-nav 전용. VLM 모델 로딩/느린 루프를 아예 '
                             '건너뛴다 (GPU 불필요, 기동이 빠르다)')
    parser.add_argument('--color-ranges-json', default=None, metavar='FILE',
                        help='HSV 임계값 JSON (color_target.save_ranges_json() 형식). '
                             '없으면 color_target.DEFAULT_RANGES(원색 기준 출발값)를 '
                             '쓴다 — 실제 물체/조명에서는 `color_target.py test`로 '
                             '재조정할 것')
    parser.add_argument('--color-min-area', type=float, default=1500.0,
                        help='이보다 작은 색상 연결영역(px^2)은 노이즈로 버린다')
    parser.add_argument('--color-approach-area', type=float, default=6000.0,
                        help='이 면적 이상부터 감속을 시작한다(목표에 가까워짐 신호)')
    parser.add_argument('--color-stop-area', type=float, default=20000.0,
                        help='이 면적 이상이면 도착 후보로 본다 (--arrive-hold 초 '
                             '연속 유지되면 --arrive-latch 규약대로 확정/래치)')
    parser.add_argument('--color-kp-ang', type=float, default=1.2,
                        help='화면 중심 오차(-1..1) -> 각속도 비례 게인')
    parser.add_argument('--color-max-wz', type=float, default=0.6,
                        help='색상 추적 중 최대 각속도 rad/s')
    parser.add_argument('--color-steer-sign', type=float, default=1.0,
                        choices=(1.0, -1.0),
                        help='조향 부호. 카메라 마운트/좌우 정의가 기대와 반대로 '
                             '나오면 -1 로 뒤집는다')
    parser.add_argument('--color-search-wz', type=float, default=0.0,
                        help='목표색을 못 찾았을 때 제자리에서 도는 탐색 각속도 '
                             'rad/s. 0(기본)이면 그 자리에 정지 — 더 안전한 기본값')
    parser.add_argument('--color-debug-dir', default=None, metavar='DIR',
                        help='탐지 결과(박스+라벨)를 주기적으로 이 폴더에 jpg 로 '
                             '저장한다 (실주행 중 튜닝 확인용)')
    parser.add_argument('--color-debug-period', type=float, default=1.0,
                        help='--color-debug-dir 저장 주기(초)')
    # 기타
    parser.add_argument('--log-json', default=None,
                        help='추론 기록 JSONL 저장 경로')
    parser.add_argument('--zero-burst', type=int, default=20,
                        help='종료 시 발행할 0 속도 메시지 개수')
    parser.add_argument('--join-timeout', type=float, default=3.0,
                        help='종료 시 VLM 스레드를 기다릴 시간(초)')

    cfg = parser.parse_args(argv)
    if cfg.enable:
        cfg.shadow = False
    if cfg.no_vlm and not cfg.color_nav:
        parser.error('--no-vlm 은 --color-nav 와 함께만 쓴다 '
                     '(그렇지 않으면 아무 경로도 주행 명령을 만들지 못한다)')
    if cfg.require_camera:
        print('[DEPRECATED] --require-camera 는 더 이상 아무 효과가 없다 '
              '(카메라 필수가 이제 기본 동작이다). 이 플래그는 다음 버전에서 '
              '제거될 수 있으니 명령행에서 빼도 된다.', file=sys.stderr)
    return cfg


def main() -> int:
    """노드를 실행한다.

    rclpy.init() 이후 SIGINT/SIGTERM 을 직접 잡아 플래그만 세운다. rclpy 기본
    핸들러는 컨텍스트를 즉시 닫아 종료 시점의 정지 명령 발행이 실패하기 때문이다.

    Returns:
        종료 코드.
    """
    import rclpy

    rclpy.init()
    cfg = parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    node = rclpy.create_node('ticvla_bridge')

    # TicvlaBridge 생성자가 --camera-topic 모드에서 첫 프레임을 기다리며
    # 블로킹되는데(RosTopicCameraSource.wait_source), 구독 콜백은 spin 이
    # 돌아야만 실행된다. spin 을 생성자 이후에 시작하면 그 대기 동안 콜백이
    # 한 번도 실행되지 못해 카메라 필수 기본 정책이 항상 타임아웃으로 죽는다.
    # 그래서 노드 생성 직후, TicvlaBridge 생성 전에 spin 을 백그라운드
    # 스레드로 먼저 돌리기 시작한다.
    requested = {'stop': False}

    def _spin_loop() -> None:
        while rclpy.ok() and not requested['stop']:
            rclpy.spin_once(node, timeout_sec=0.05)

    spin_thread = threading.Thread(target=_spin_loop, name='spin', daemon=True)
    spin_thread.start()

    bridge: Optional[TicvlaBridge] = None
    try:
        bridge = TicvlaBridge(node, cfg)
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(f'초기화 실패: {exc}')
        requested['stop'] = True
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    def _on_signal(signum: int, frame: object) -> None:
        """종료 요청 플래그만 세운다. 실제 정리는 메인 루프 종료 후 수행."""
        requested['stop'] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    exit_code = 0
    try:
        while (rclpy.ok() and not requested['stop']
               and bridge.fatal_reason is None):
            time.sleep(0.05)
        requested['stop'] = True
        spin_thread.join(timeout=1.0)
        if bridge.fatal_reason is not None:
            # 정지 명령 발행과 자원 정리는 아래 shutdown() 이 처리한다.
            exit_code = EXIT_NO_CAMERA
    except Exception as exc:  # noqa: BLE001
        print(f'실행 중 오류: {exc}', file=sys.stderr)
    finally:
        requested['stop'] = True
        spin_thread.join(timeout=1.0)
        clean = True
        try:
            clean = bridge.shutdown()
        except Exception as exc:  # noqa: BLE001
            print(f'종료 처리 중 오류: {exc}', file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        if not clean:
            # 데몬 스레드가 predict() 중이면 인터프리터 종료 과정에서 CUDA/C++
            # 객체 소멸이 abort 를 낸다. 정지 명령과 자원 해제는 이미 끝났으므로
            # 종료 코드를 확정한 뒤 즉시 프로세스를 끊는다.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
