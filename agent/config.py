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
class ReactAgentConfig:
    """ReAct 루프 전체 설정."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    ticvla: TicVlaConfig = field(default_factory=TicVlaConfig)
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
    if "fewshot_path" in raw:
        cfg.fewshot_path = raw["fewshot_path"]
    if "log_dir" in raw:
        cfg.log_dir = raw["log_dir"]
    return cfg
