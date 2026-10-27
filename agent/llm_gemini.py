#!/usr/bin/env python3
"""Gemini API(google-genai) 얇은 래퍼.

모델명/생성 설정은 agent.config.GeminiConfig 로 뺀다. API 키는 환경변수
``GEMINI_API_KEY`` 하나만 쓴다(요구사항 2).

Gemini 호출이 재시도 상한 안에서도 실패하면 ``LLMUnavailable`` 로 변환해
올린다 — 원래 예외(``ClientError``/``ServerError``/네트워크 오류)가 그대로
새면 react_loop 가 트레이스백과 함께 죽고, 실차에서는 로봇이 마지막 명령대로
계속 움직이는 안전 문제가 된다.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from agent.config import GeminiConfig, RetryConfig

logger = logging.getLogger('llm_gemini')


class LLMUnavailable(RuntimeError):
    """Gemini 호출이 (재시도를 포함해) 끝내 실패했을 때 올라오는 전용 예외.

    react_loop 는 이것만 잡으면 되고, 원래 ``ClientError``/``ServerError``/
    httpx 예외 종류를 알 필요가 없다.

    Attributes:
        status_code: HTTP 상태 코드. 응답 자체가 없는 네트워크 레벨 오류이거나
            Gemini 응답이 비어 있는 경우는 None.
        reason: 사람이 읽을 원인 메시지.
    """

    def __init__(self, status_code: Optional[int], reason: str) -> None:
        self.status_code = status_code
        self.reason = reason
        super().__init__(f'Gemini 호출 실패 (status={status_code}): {reason}')


class GeminiClient:
    """ReAct 루프가 매 스텝 호출하는 Gemini 텍스트 생성 클라이언트."""

    def __init__(self, cfg: GeminiConfig) -> None:
        """클라이언트를 만든다.

        Args:
            cfg: 모델명/생성 설정/재시도 설정.

        Raises:
            RuntimeError: ``GEMINI_API_KEY`` 환경변수가 없는 경우.
            ImportError: ``google-genai`` 패키지가 설치되어 있지 않은 경우.
        """
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError(
                'GEMINI_API_KEY 환경변수가 없다. '
                'export GEMINI_API_KEY=... 로 설정한 뒤 다시 실행할 것.')

        from google import genai
        from google.genai import types

        self._cfg = cfg
        # HttpOptions.timeout 이 없으면 연결이 멈췄을 때 예외가 아예 안 나서
        # generate() 가 무한정 블로킹되고, 아래 재시도 로직도 발동하지 못한다.
        http_options = types.HttpOptions(
            timeout=int(cfg.request_timeout_sec * 1000))
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._generation_config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )

    def _should_retry(self, elapsed_sec: float, next_backoff_sec: float,
                       retry_cfg: RetryConfig) -> bool:
        """다음 백오프까지 기다려도 총 대기 상한을 넘지 않는지 본다.

        Args:
            elapsed_sec: 첫 시도 이후 지금까지 흐른 시간.
            next_backoff_sec: 이번에 기다리려는 시간.
            retry_cfg: 재시도 설정.

        Returns:
            재시도해도 되면 True.
        """
        return elapsed_sec + next_backoff_sec <= retry_cfg.max_total_wait_sec

    def generate(self, prompt: str) -> str:
        """프롬프트 전체를 보내고 생성된 텍스트를 받는다.

        매 호출마다 누적된 컨텍스트 전체(few-shot + 이력)를 ``prompt`` 하나로
        받는다 — ReAct 루프 쪽에서 대화 히스토리를 들고 있지 않고 매번
        통째로 다시 보내는 설계와 맞물린다.

        상태코드별 처리:
            - 429(쿼터 초과): 기다려도 안 풀리므로 재시도 없이 즉시 실패.
            - 4xx(그 외 클라이언트 오류): 요청 자체가 잘못된 것이므로 마찬가지로
              재시도 없이 즉시 실패.
            - 5xx(서버 오류, 503 포함): 일시적 과부하로 보고 지수 백오프로
              재시도하되, 총 대기 시간이 ``retry.max_total_wait_sec`` 를 넘으면
              포기한다.
            - 네트워크 오류(연결 실패/타임아웃 등 응답 자체가 없는 경우): 5xx 와
              같은 정책(지수 백오프 + 총 대기 상한)을 적용한다.

        Args:
            prompt: few-shot 예시 + 지금까지의 (행동, 관측) 이력을 담은 전체 텍스트.

        Returns:
            Gemini 가 생성한 원문 텍스트.

        Raises:
            LLMUnavailable: 위 정책에 따라 재시도 후에도(또는 재시도 없이)
                실패한 경우.
        """
        import httpx
        from google.genai import errors

        retry_cfg = self._cfg.retry
        started = time.monotonic()
        backoff = retry_cfg.initial_backoff_sec
        attempt = 0

        while True:
            attempt += 1
            try:
                response = self._client.models.generate_content(
                    model=self._cfg.model,
                    contents=prompt,
                    config=self._generation_config,
                )
            except errors.ClientError as exc:
                # 429(쿼터)를 포함한 모든 4xx 는 기다려도 풀리지 않는 오류다.
                if exc.code == 429:
                    logger.error(
                        'Gemini 429(쿼터 초과) — 재시도하지 않고 즉시 실패 처리한다.')
                raise LLMUnavailable(exc.code, exc.message or str(exc)) from exc
            except errors.ServerError as exc:
                elapsed = time.monotonic() - started
                if not self._should_retry(elapsed, backoff, retry_cfg):
                    raise LLMUnavailable(exc.code, exc.message or str(exc)) from exc
                logger.warning(
                    f'Gemini {exc.code}(서버 오류, 일시적 과부하로 추정) — '
                    f'{backoff:.1f}초 뒤 재시도(시도 {attempt}): {exc.message}')
                time.sleep(backoff)
                backoff *= retry_cfg.backoff_multiplier
                continue
            except errors.APIError as exc:
                # 4xx/5xx 외의 뜻밖의 상태코드. 재시도해서 좋아질 근거가 없다.
                raise LLMUnavailable(exc.code, exc.message or str(exc)) from exc
            except httpx.HTTPError as exc:
                # 응답 자체를 못 받은 경우(연결 실패/타임아웃 등) — 일시적일
                # 가능성이 높으므로 서버 오류와 같은 재시도 정책을 쓴다.
                elapsed = time.monotonic() - started
                if not self._should_retry(elapsed, backoff, retry_cfg):
                    raise LLMUnavailable(None, str(exc)) from exc
                logger.warning(
                    f'Gemini 네트워크 오류 — {backoff:.1f}초 뒤 재시도'
                    f'(시도 {attempt}): {exc}')
                time.sleep(backoff)
                backoff *= retry_cfg.backoff_multiplier
                continue

            text = response.text
            if not text:
                raise LLMUnavailable(None, f'Gemini 응답이 비어 있다: {response!r}')
            return text
