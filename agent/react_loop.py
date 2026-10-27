#!/usr/bin/env python3
"""TIC-VLA 위에 얹는 상위 ReAct(Yao et al., 2022) 에이전트 루프.

TIC-VLA 는 "보이는 목표로 이동"만 할 수 있다. 이 루프는 그 위에서 목표가
시야에 없을 때 think[...]/goto[...]/rotate[left|right]/done[found|fail] 중
하나를 매 스텝 고르는 판단층이다.

관측 o_t 는 TIC-VLA 추론이 만드는 <think> 장면 서술 텍스트이고, 컨텍스트는
(행동, 관측) 쌍을 텍스트로 누적해 매 호출마다 전체를 다시 Gemini 에 보낸다.

사용 예:
    # 루프 로직만 검증 (TIC-VLA 호출 없음, 관측을 손으로 입력)
    GEMINI_API_KEY=... python3 -m agent.react_loop --goal "Find the gray bin." --dry-run

    # 실제 TIC-VLA 추론 (goto/rotate 는 추론만 하고 로봇을 움직이지는 않는다 —
    # agent/tic_vla_adapter.py 의 LiveTicVlaAdapter 설명 참고)
    GEMINI_API_KEY=... venvs/ticvla/bin/python3 -m agent.react_loop \\
        --goal "Find the gray bin."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.actions import ActionParseError, ParsedAction, parse_action
from agent.config import ReactAgentConfig, load_config
from agent.llm_gemini import GeminiClient, LLMUnavailable
from agent.tic_vla_adapter import DryRunAdapter, LiveTicVlaAdapter, TicVlaAdapter

logger = logging.getLogger('react_loop')

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StepRecord:
    """JSONL 로그 한 줄에 대응하는 레코드(요구사항 6).

    ``failure_type``/``status_code`` 는 LLM 호출 자체가 불가능해서 강제
    종료했을 때만 채워진다 — 로그 집계에서 "알고리즘 실패"(모델이 스스로
    done[fail] 했거나 상한에 걸린 경우)와 "LLM 인프라 장애"를 이 필드로
    구분한다.
    """

    step: int
    timestamp: str
    action: Optional[str]
    argument: str
    valid: bool
    observation: str
    llm_raw_response: str
    failure_type: Optional[str] = None
    status_code: Optional[int] = None


class JsonlLogger:
    """스텝마다 (행동, 관측, LLM 원본 응답, 타임스탬프)를 JSONL 로 남긴다."""

    def __init__(self, log_dir: str, run_label: str) -> None:
        """로그 파일을 연다.

        Args:
            log_dir: 로그 디렉터리(상대경로면 저장소 루트 기준).
            run_label: 파일명에 붙일 이번 실행 라벨(타임스탬프).
        """
        directory = Path(log_dir)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'react_{run_label}.jsonl'
        self._fh = self.path.open('a', encoding='utf-8')

    def write(self, record: StepRecord) -> None:
        """레코드 한 줄을 JSONL 로 쓰고 즉시 flush 한다."""
        self._fh.write(json.dumps(record.__dict__, ensure_ascii=False) + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _utcnow_iso() -> str:
    """UTC ISO-8601 타임스탬프."""
    return datetime.now(timezone.utc).isoformat()


def _load_fewshot(fewshot_path: str) -> str:
    """few-shot 예시 파일을 읽는다.

    요구사항 1: 예시 문장을 코드에 하드코딩하지 않는다. 파일이 없으면
    명확한 에러로 종료하되, 내용이 비어 있는 것은 허용한다.

    Args:
        fewshot_path: 저장소 루트 기준 상대경로 또는 절대경로.

    Returns:
        파일 내용(빈 문자열일 수 있음).

    Raises:
        FileNotFoundError: 파일 자체가 없는 경우.
    """
    path = Path(fewshot_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f'few-shot 예시 파일이 없다: {path} — prompts/react_fewshot.txt 를 '
            '만들 것(내용은 비어 있어도 된다).')
    return path.read_text(encoding='utf-8')


#: 행동 공간 자체는 사용자가 준 스펙을 그대로 옮긴 것이다(예시 문장이 아니다).
_ACTION_SPACE_SPEC = """\
You control a mobile robot together with TIC-VLA, a low-level policy that can \
only drive toward a target that is already visible in the current view. At \
each turn you receive an Observation: a <think> scene-description text \
produced by TIC-VLA's own reasoning about what it currently sees. Based on \
the Goal and the history of (Action, Observation) pairs below, choose exactly \
one Action from:

  think[<your reasoning>]   No environment call. The system will reply with \
observation "OK." and move to the next turn. Use this to reason without acting.
  goto[<target>]            Delegates to TIC-VLA to drive toward <target>. \
<target> MUST be a string that appears verbatim in the most recent \
Observation, otherwise this step is rejected. This call blocks until TIC-VLA \
finishes.
  rotate[left|right]        Rotate in place, then obtain a new observation.
  done[found|fail]          End the episode.

Respond with exactly one action in the form name[argument]."""


def _build_prompt(fewshot_text: str, goal: str, history: list[str]) -> str:
    """few-shot + 목표 + 누적 이력으로 전체 프롬프트를 만든다.

    요구사항: 컨텍스트는 (행동, 관측) 쌍을 텍스트로 누적하고, 매 호출마다
    전체를 다시 보낸다 — history 를 자르지 않고 그대로 이어 붙인다.

    Args:
        fewshot_text: ``prompts/react_fewshot.txt`` 내용(비어 있을 수 있음).
        goal: 이번 에피소드의 목표.
        history: 지금까지의 "Action: ...\\nObservation: ..." 블록 목록.

    Returns:
        Gemini 에 보낼 전체 프롬프트.
    """
    sections = [_ACTION_SPACE_SPEC]
    if fewshot_text.strip():
        sections.append(fewshot_text.strip())
    sections.append(f'Goal: {goal}')
    sections.extend(history)
    sections.append('Action:')
    return '\n\n'.join(sections)


class ReactLoop:
    """ReAct 스텝 루프 본체."""

    def __init__(self, cfg: ReactAgentConfig, goal: str, adapter: TicVlaAdapter,
                 llm: GeminiClient, fewshot_text: str, jsonl: JsonlLogger) -> None:
        self._cfg = cfg
        self._goal = goal
        self._adapter = adapter
        self._llm = llm
        self._fewshot_text = fewshot_text
        self._jsonl = jsonl
        self._history: list[str] = []
        # goto 그라운딩 검사 기준. think[...]/무효 스텝의 더미 관측("OK.", 에러
        # 안내문)은 여기 반영하지 않는다 — 안 그러면 think 바로 다음 goto 가
        # 항상 "직전 관측에 없음"으로 무효 처리되어 think+goto 조합이 망가진다.
        self._last_scene_observation = ''

    def _record(self, step: int, action: Optional[str], argument: str,
                valid: bool, observation: str, raw_response: str,
                failure_type: Optional[str] = None,
                status_code: Optional[int] = None) -> None:
        """스텝 기록을 JSONL 에 남긴다."""
        self._jsonl.write(StepRecord(
            step=step, timestamp=_utcnow_iso(), action=action,
            argument=argument, valid=valid, observation=observation,
            llm_raw_response=raw_response, failure_type=failure_type,
            status_code=status_code))

    def _append_history(self, action_text: str, observation: str) -> None:
        """(행동, 관측) 한 쌍을 컨텍스트 이력에 추가한다.

        goto 그라운딩 기준(``_last_scene_observation``)은 건드리지 않는다 —
        그건 실제 TIC-VLA 관측이 나온 지점에서 호출자가 직접 갱신한다.
        """
        self._history.append(f'Action: {action_text}\nObservation: {observation}')

    def run(self) -> str:
        """루프를 끝까지 돌린다.

        Returns:
            최종 판정. ``"found"``/``"fail"``/``"llm_unavailable"`` 중 하나.
            마지막 값은 모델이 선택한 알고리즘적 실패가 아니라 Gemini 호출
            자체가 불가능해서 강제 종료했다는 뜻이다(요구사항 2/4).
        """
        initial_obs = self._adapter.initial_observation()
        self._record(step=0, action='init', argument='', valid=True,
                     observation=initial_obs, raw_response='')
        self._append_history('(episode start)', initial_obs)
        self._last_scene_observation = initial_obs
        logger.info(f'초기 관측: {initial_obs}')

        goto_calls = 0
        step = 0
        while True:
            step += 1
            if step > self._cfg.limits.max_total_steps:
                logger.warning(
                    f'전체 스텝 상한({self._cfg.limits.max_total_steps}) 도달 — '
                    'done[fail] 로 강제 종료한다.')
                self._record(step=step, action='done', argument='fail',
                            valid=True, observation='[AUTO] max_total_steps 도달',
                            raw_response='')
                return 'fail'

            prompt = _build_prompt(self._fewshot_text, self._goal, self._history)
            try:
                raw_response = self._llm.generate(prompt)
            except LLMUnavailable as exc:
                logger.error(
                    f'[스텝 {step}] Gemini 호출 불가 — done[fail] 로 안전 종료한다: '
                    f'{exc}')
                self._adapter.stop()
                self._record(step=step, action='done', argument='fail',
                            valid=True, observation=f'[AUTO] LLM 불가: {exc.reason}',
                            raw_response='', failure_type='llm_unavailable',
                            status_code=exc.status_code)
                return 'llm_unavailable'

            try:
                action: ParsedAction = parse_action(raw_response)
            except ActionParseError as exc:
                logger.warning(f'[스텝 {step}] 행동 파싱 실패 — 무효 처리: {exc}')
                note = f'ERROR: {exc}'
                self._record(step=step, action=None, argument='', valid=False,
                            observation=note, raw_response=raw_response)
                self._append_history('(unparseable — please answer with exactly '
                                     'one action in the form name[argument])', note)
                continue

            if action.kind == 'think':
                observation = 'OK.'
                self._append_history(f'think[{action.argument}]', observation)
                self._record(step=step, action='think', argument=action.argument,
                            valid=True, observation=observation,
                            raw_response=raw_response)
                continue

            if action.kind == 'rotate':
                observation = self._adapter.rotate(action.argument)
                self._append_history(f'rotate[{action.argument}]', observation)
                self._last_scene_observation = observation
                self._record(step=step, action='rotate', argument=action.argument,
                            valid=True, observation=observation,
                            raw_response=raw_response)
                logger.info(f'[스텝 {step}] rotate[{action.argument}] → {observation}')
                continue

            if action.kind == 'goto':
                if action.argument not in self._last_scene_observation:
                    logger.warning(
                        f"[스텝 {step}] goto[{action.argument}] 무효 — 직전 관측에 "
                        "없는 문자열. 스텝은 소비하되 TIC-VLA 는 호출하지 않는다.")
                    note = (f"INVALID: '{action.argument}' does not appear in the "
                           "most recent Observation. Choose a target that is "
                           "literally present there, or think[...]/rotate[...] first.")
                    self._record(step=step, action='goto', argument=action.argument,
                                valid=False, observation=note,
                                raw_response=raw_response)
                    self._append_history(f'goto[{action.argument}]', note)
                    continue

                if goto_calls >= self._cfg.limits.max_goto_calls:
                    logger.warning(
                        f'goto 호출 상한({self._cfg.limits.max_goto_calls}) 도달 — '
                        'done[fail] 로 강제 종료한다.')
                    self._record(step=step, action='done', argument='fail',
                                valid=True, observation='[AUTO] max_goto_calls 도달',
                                raw_response=raw_response)
                    return 'fail'

                goto_calls += 1
                observation = self._adapter.goto(action.argument)
                self._append_history(f'goto[{action.argument}]', observation)
                self._last_scene_observation = observation
                self._record(step=step, action='goto', argument=action.argument,
                            valid=True, observation=observation,
                            raw_response=raw_response)
                logger.info(f'[스텝 {step}] goto[{action.argument}] → {observation}')
                continue

            # action.kind == 'done'
            logger.info(f'[스텝 {step}] done[{action.argument}] — 에피소드 종료')
            self._record(step=step, action='done', argument=action.argument,
                        valid=True, observation='', raw_response=raw_response)
            return action.argument


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', required=True,
                        help='이번 에피소드의 목표(자연어).')
    parser.add_argument('--config', default='config/react_agent.yaml',
                        help='YAML 설정 파일 경로(기본: config/react_agent.yaml).')
    parser.add_argument('--dry-run', action='store_true',
                        help='TIC-VLA 를 호출하지 않고 표준입력으로 관측을 받는다.')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 진입점."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    args = _parse_args(argv)

    try:
        cfg = load_config(args.config)
        fewshot_text = _load_fewshot(cfg.fewshot_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    run_label = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl = JsonlLogger(cfg.log_dir, run_label)
    logger.info(f'JSONL 로그: {jsonl.path}')

    try:
        llm = GeminiClient(cfg.gemini)
    except (RuntimeError, ImportError) as exc:
        logger.error(str(exc))
        return 1

    adapter: TicVlaAdapter
    if args.dry_run:
        adapter = DryRunAdapter()
    else:
        try:
            adapter = LiveTicVlaAdapter(cfg.ticvla, logger)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error(f'TIC-VLA 어댑터 초기화 실패: {exc}')
            return 1

    loop = ReactLoop(cfg, args.goal, adapter, llm, fewshot_text, jsonl)
    try:
        verdict = loop.run()
    except EOFError as exc:
        logger.error(str(exc))
        return 1
    finally:
        jsonl.close()

    logger.info(f'종료 판정: {verdict}')
    # 0=정상(found), 1=알고리즘적 실패(모델이 스스로 fail 했거나 상한 도달),
    # 2=LLM 인프라 장애로 인한 중단 — 로그 집계에서 셋을 구분해야 한다(요구사항 4).
    if verdict == 'found':
        return 0
    if verdict == 'llm_unavailable':
        return 2
    return 1


if __name__ == '__main__':
    sys.exit(main())
