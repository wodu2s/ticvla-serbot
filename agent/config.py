#!/usr/bin/env python3
"""ReAct 에이전트 설정(config/react_agent.yaml) 로더."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class RetryConfig:
    """Gemini 호출이 5xx/네트워크 오류로 실패했을 때의 재시도 설정.

    429(쿼터)는 이 설정과 무관하게 무조건 즉시 실패 처리한다 — 쿼터 초과는
    기다린다고 풀리는 문제가 아니기 때문이다.
    """

    initial_backoff_sec: float = 1.0
    backoff_multiplier: float = 2.0
    #: 첫 시도부터 누적해서 이 시간(초)을 넘기면 더 재시도하지 않고 포기한다.
    max_total_wait_sec: float = 60.0


@dataclass
class GeminiConfig:
    """Gemini 호출 설정."""

    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 1024
    #: 요청 하나가 이 시간(초)을 넘기면 httpx 가 타임아웃 예외를 낸다. 이게
    #: 없으면 연결이 느려지거나 멈췄을 때 예외 자체가 안 나서 retry 로직이
    #: 발동하지 못하고 generate() 가 무한정 블로킹된다.
    request_timeout_sec: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class LimitsConfig:
    """안전장치 상한."""

    max_goto_calls: int = 10
    max_total_steps: int = 25


@dataclass
class TicVlaConfig:
    """--dry-run 이 아닐 때만 쓰는 TIC-VLA 연동 설정.

    ``model_path``/``ckpt_path``/``frame_dir`` 는 dry-run 에서는 쓰이지 않으므로
    null 을 허용한다. live 모드에서 null 이면 로딩 시점에 에러를 낸다.
    """

    model_path: Optional[str] = None
    ckpt_path: Optional[str] = None
    frame_dir: Optional[str] = None
    num_delayed: int = 4


@dataclass
class DirectDriveConfig:
    """--direct(ReAct 단독 주행, TIC-VLA 없음) 모드 설정.

    Gemini 가 카메라 프레임을 직접 보고, goto/rotate 는 odom 폐루프 단순
    cmd_vel 동작으로 실행한다. 속도는 cmd_vel_mux 가 한 번 더 클램프한다.
    """

    camera_topic: str = "/camera/image_raw_fast"
    odom_topic: str = "/wheel/odom"
    #: cmd_vel_mux 의 모델 입력 토픽. mux 가 0.5s 무수신이면 0 을 내보낸다.
    cmd_vel_topic: str = "/ticvla/cmd_vel"
    #: 이 시간(초) 동안 새 프레임/odom 이 없으면 동작을 중단한다.
    sensor_stale_sec: float = 1.0
    linear_speed: float = 0.08
    angular_speed: float = 0.35
    #: rotate[left|right] 한 번의 회전각(도).
    rotate_deg: float = 45.0
    #: 카메라 수평 화각(도). bbox 중심 → 방위각 환산에만 쓴다.
    hfov_deg: float = 62.0
    #: goto 한 번 안에서 (위치 추정 → 회전 → 전진) 반복 상한.
    max_servo_iters: int = 6
    #: 한 반복에서 전진하는 최대 거리(m).
    max_step_m: float = 0.6
    #: 추정 거리가 이 값 이하이거나 bbox 높이 비율이 arrive_bbox_h 이상이면 도착.
    stop_distance_m: float = 0.5
    arrive_bbox_h: float = 0.45
    #: 방위각이 이 값(도) 이하면 회전 없이 바로 전진한다.
    heading_tolerance_deg: float = 6.0
    #: 동작 하나(회전/전진)의 시간 상한(초). 넘으면 정지하고 실패 처리.
    motion_timeout_sec: float = 20.0
    #: 스텝별 프레임 저장 디렉터리(상대경로면 저장소 루트 기준). 실행마다 하위 폴더 생성.
    frame_save_dir: str = "logs/react_direct"


@dataclass
class ReactAgentConfig:
    """ReAct 루프 전체 설정."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    ticvla: TicVlaConfig = field(default_factory=TicVlaConfig)
    direct: DirectDriveConfig = field(default_factory=DirectDriveConfig)
    fewshot_path: str = "prompts/react_fewshot.txt"
    log_dir: str = "logs/react"


def load_config(config_path: str | Path) -> ReactAgentConfig:
    """YAML 설정 파일을 읽어 ``ReactAgentConfig`` 로 만든다.

    Args:
        config_path: 설정 파일 경로.

    Returns:
        채워진 설정 객체. 파일에 없는 필드는 dataclass 기본값을 쓴다.

    Raises:
        FileNotFoundError: 설정 파일이 없는 경우.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ReAct 에이전트 설정 파일이 없다: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = ReactAgentConfig()
    if "gemini" in raw:
        gemini_raw = dict(raw["gemini"])
        retry_raw = gemini_raw.pop("retry", None)
        cfg.gemini = GeminiConfig(**gemini_raw)
        if retry_raw:
            cfg.gemini.retry = RetryConfig(**retry_raw)
    if "limits" in raw:
        cfg.limits = LimitsConfig(**raw["limits"])
    if "ticvla" in raw:
        cfg.ticvla = TicVlaConfig(**raw["ticvla"])
    if "direct" in raw:
        cfg.direct = DirectDriveConfig(**raw["direct"])
    if "fewshot_path" in raw:
        cfg.fewshot_path = raw["fewshot_path"]
    if "log_dir" in raw:
        cfg.log_dir = raw["log_dir"]
    return cfg
