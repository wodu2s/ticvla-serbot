#!/usr/bin/env python3
"""TIC-VLA 호출 어댑터.

ReAct 루프는 이 인터페이스(``TicVlaAdapter``)만 알면 된다. 구현체는 두 가지다.

- ``DryRunAdapter``: TIC-VLA 를 전혀 호출하지 않고 표준입력으로 관측을 받는다
  (요구사항 7, ``--dry-run``). 루프 로직만 검증할 때 쓴다.
- ``LiveTicVlaAdapter``: ``third_party/TIC-VLA`` 의 ``TICVLA.predict()`` 를
  직접 호출해 실제 ``<think>`` 관측을 받아온다.

주의 — ``LiveTicVlaAdapter`` 의 ``goto``/``rotate`` 는 TIC-VLA 추론만 수행하고
실제 로봇을 구동하지는 않는다(cmd_vel/odom/safety_mux 연동과 "도착" 판정은
``scripts/ticvla_bridge.py`` 쪽 ROS 노드의 몫이며 이번 작업 범위 밖이다). 이 점을
호출할 때마다 로그로 남긴다 — 실제로 움직였다고 착각하지 않게 하기 위해서다.
"""
from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from agent.config import TicVlaConfig

#: scripts/analyze_sign_test.py:58 와 동일한 패턴을 그대로 재사용한다.
#: 닫는 태그 없이 잘린 응답도 잡기 위해 ``(?:</think>|\Z)`` 폴백을 둔다.
_RE_THINK = re.compile(r'<think>(.*?)(?:</think>|\Z)', re.DOTALL)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TicVlaAdapter(ABC):
    """ReAct 루프가 의존하는 TIC-VLA 호출 인터페이스."""

    @abstractmethod
    def initial_observation(self) -> str:
        """루프 시작 시 첫 관측 o_0 을 만든다."""

    @abstractmethod
    def goto(self, target: str) -> str:
        """``goto[target]`` 을 수행하고 새 관측을 돌려준다. 블로킹 호출이다."""

    @abstractmethod
    def rotate(self, direction: str) -> str:
        """``rotate[left|right]`` 를 수행하고 새 관측을 돌려준다. 블로킹 호출이다."""

    @abstractmethod
    def stop(self) -> None:
        """루프가 비정상 종료(예: LLM 장애)할 때 안전하게 멈추는 훅.

        예외를 던지면 안 된다 — 이미 실패 처리 중에 불리므로, 여기서 또
        예외가 나면 트레이스백 없이 종료한다는 목표 자체가 깨진다.
        """


class DryRunAdapter(TicVlaAdapter):
    """TIC-VLA 를 호출하지 않고 표준입력으로 관측을 받는 어댑터."""

    def _read(self, label: str) -> str:
        """사람에게 라벨을 보여주고 한 줄(또는 EOF 까지)을 관측으로 받는다.

        Args:
            label: 프롬프트에 보여줄 안내 문구.

        Returns:
            입력받은 관측 텍스트. 빈 줄이면 빈 문자열.

        Raises:
            EOFError: 표준입력이 끊긴 경우 — 루프를 계속할 수 없으므로 그대로 올린다.
        """
        print(label, file=sys.stderr, end='', flush=True)
        line = sys.stdin.readline()
        if line == '':
            raise EOFError('--dry-run: 표준입력이 끊겼다(EOF) — 관측을 더 받을 수 없다')
        return line.rstrip('\n')

    def initial_observation(self) -> str:
        return self._read('[dry-run] 초기 관측 o_0 > ')

    def goto(self, target: str) -> str:
        return self._read(f"[dry-run] goto[{target}] 이후 관측 > ")

    def rotate(self, direction: str) -> str:
        return self._read(f"[dry-run] rotate[{direction}] 이후 관측 > ")

    def stop(self) -> None:
        """TIC-VLA 를 호출한 적이 없으므로 할 일이 없다."""


class LiveTicVlaAdapter(TicVlaAdapter):
    """``TICVLA.predict()`` 를 직접 호출해 실제 ``<think>`` 관측을 받는 어댑터."""

    #: predict() 기본 instruction (ticvla.py:537 과 동일한 중립 지시문).
    _NEUTRAL_INSTRUCTION = 'Move forward safely and efficiently.'

    def __init__(self, cfg: TicVlaConfig, logger: Any) -> None:
        """TICVLA 모델을 로딩한다.

        third_party/TIC-VLA 쪽 코드는 import 만 하고 수정하지 않는다.
        체크포인트 로딩은 ``scripts/ticvla_smoke_test.py`` 의 기존 로직을
        그대로 재사용한다(``ModelRunner`` 전체를 끌어오지 않는 이유는, 이
        어댑터가 과거 프레임 KV 캐시 재사용(2Hz 빠른 경로)을 쓰지 않고 매
        스텝 ``predict()`` 를 통째로 다시 돌리기 때문 — 그 최적화가 애초에
        필요 없다).

        Args:
            cfg: ``model_path``/``ckpt_path``/``frame_dir``/``num_delayed``.
            logger: 로그 출력 대상. ``.info``/``.warning`` 을 쓴다.

        Raises:
            ValueError: ``model_path``/``ckpt_path``/``frame_dir`` 가 비어 있는 경우.
            FileNotFoundError: 모델/체크포인트 경로가 실제로 없는 경우.
            RuntimeError: CUDA 를 쓸 수 없는 경우.
        """
        if not cfg.model_path or not cfg.ckpt_path or not cfg.frame_dir:
            raise ValueError(
                '--dry-run 이 아니면 config 의 ticvla.model_path / ticvla.ckpt_path '
                '/ ticvla.frame_dir 를 모두 채워야 한다.')

        self._cfg = cfg
        self._log = logger
        self._frame_dir = Path(cfg.frame_dir)
        self._stub_warned = False

        sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
        from ticvla_smoke_test import (  # noqa: E402  (지연 import, 경로 삽입 뒤)
            apply_torch_distributed_shim,
            load_checkpoint_into_model,
            select_frames,
        )

        apply_torch_distributed_shim()

        import torch  # noqa: E402
        from ticvla.models.ticvla import TICVLA  # noqa: E402

        self._torch = torch
        self._select_frames = select_frames

        model_path = Path(cfg.model_path)
        ckpt_path = Path(cfg.ckpt_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f'모델 경로가 없다: {model_path}')
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'체크포인트가 없다: {ckpt_path}')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 를 쓸 수 없다 — live 모드는 GPU 가 필요하다.')

        # action_horizon_steps 는 CLAUDE.md 에 적힌 대로 30 고정이어야 사전학습
        # action_chunk_embed 와 shape 이 맞는다.
        self._model = TICVLA(model_path=str(model_path), action_horizon_steps=30)
        load_checkpoint_into_model(self._model, ckpt_path)
        self._model.to('cuda')
        self._model.eval()

    def _warn_stub(self) -> None:
        """실제 로봇 구동(ROS cmd_vel)은 없다는 것을 처음 한 번 크게 경고한다."""
        if self._stub_warned:
            return
        self._stub_warned = True
        self._log.warning(
            '이 어댑터는 TIC-VLA 추론만 수행한다 — goto/rotate 가 실제로 로봇을 '
            '움직이지 않는다. cmd_vel/odom/safety_mux 연동과 "도착" 판정은 '
            '별도 작업(scripts/ticvla_bridge.py 쪽 ROS 통합)이 맡는다.')

    def _extract_think(self, response: str) -> str:
        """VLM 원문 응답에서 ``<think>`` 블록만 뽑는다.

        체크포인트가 ``<think>`` 태그를 학습하지 않았으면 태그가 아예 없을 수
        있다 — 그 경우 루프를 죽이지 않고 원문 전체를 관측으로 돌려주되 경고를
        남긴다.

        Args:
            response: ``TICVLA.predict()`` 가 돌려준 원문 텍스트.

        Returns:
            ``<think>`` 블록 내용(태그 없으면 원문 전체, 앞뒤 공백 제거).
        """
        match = _RE_THINK.search(response)
        if match is None:
            self._log.warning(
                f'응답에 <think> 태그가 없다 — 원문을 그대로 관측으로 쓴다: '
                f'{response!r}')
            return response.strip()
        return match.group(1).strip()

    def _infer(self, instruction: str) -> str:
        """프레임 디렉터리에서 최신 프레임을 골라 ``predict()`` 한 번을 돌린다.

        Args:
            instruction: TIC-VLA 에 줄 자연어 지시문.

        Returns:
            추출된 ``<think>`` 관측 텍스트.
        """
        delayed_paths, current_path = self._select_frames(
            self._frame_dir, self._cfg.num_delayed)
        robot_state = self._torch.zeros(
            5, dtype=self._torch.bfloat16, device=self._model.device)
        with self._torch.inference_mode():
            response, _waypoints, _prompt = self._model.predict(
                delayed_image_paths=delayed_paths,
                current_image_path=current_path,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=0.0,
            )
        return self._extract_think(response)

    def initial_observation(self) -> str:
        return self._infer(self._NEUTRAL_INSTRUCTION)

    def goto(self, target: str) -> str:
        self._warn_stub()
        return self._infer(f'Go to {target}.')

    def rotate(self, direction: str) -> str:
        self._warn_stub()
        # 실제 회전이 없으므로 같은 프레임을 다시 보는 것과 다르지 않을 수
        # 있다 — 위 _warn_stub() 경고가 그 한계를 명시한다.
        return self._infer(self._NEUTRAL_INSTRUCTION)

    def stop(self) -> None:
        """LLM 장애 등으로 루프가 중단될 때 불린다.

        이 어댑터는 애초에 cmd_vel 을 발행하지 않으므로(위 클래스 docstring
        참고) 여기서 멈출 실제 모터 명령이 없다 — 그래도 "정지 훅이 호출됐다"는
        사실 자체를 로그로 남긴다. cmd_vel 을 실제로 발행하는 ROS 통합판을
        만들 때는 여기서 0 속도를 발행해야 한다.
        """
        self._log.warning(
            'stop() 호출됨(LLM 장애로 인한 종료로 추정) — 이 어댑터는 cmd_vel 을 '
            '발행한 적이 없으므로 멈출 모터 명령이 없다. 실차 ROS 통합판에서는 '
            '여기서 안전 정지(0 속도 발행)를 구현해야 한다.')
