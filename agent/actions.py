#!/usr/bin/env python3
"""ReAct 행동 공간 파싱.

행동 공간은 4개뿐이다: ``think[...]``, ``goto[<target>]``,
``rotate[left|right]``, ``done[found|fail]``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: LLM 원문 응답에서 첫 번째로 나오는 행동 하나만 집는다.
#: 모델이 행동 앞에 다른 추론 문장을 붙이더라도 상관없게 ``re.search`` 를 쓴다.
_ACTION_PATTERN = re.compile(
    r'\b(think|goto|rotate|done)\[(.*?)\]', re.DOTALL)

_ROTATE_DIRECTIONS = ('left', 'right')
_DONE_VERDICTS = ('found', 'fail')


class ActionParseError(ValueError):
    """LLM 응답에서 유효한 행동을 뽑지 못했을 때."""


@dataclass
class ParsedAction:
    """파싱된 행동 한 개.

    Attributes:
        kind: ``think``/``goto``/``rotate``/``done`` 중 하나.
        argument: 대괄호 안 인자 (앞뒤 공백 제거됨).
    """

    kind: str
    argument: str


def parse_action(raw_response: str) -> ParsedAction:
    """LLM 원문 응답에서 행동 하나를 뽑는다.

    Args:
        raw_response: Gemini 가 돌려준 원문 텍스트.

    Returns:
        파싱된 행동.

    Raises:
        ActionParseError: 4개 행동 패턴이 하나도 없거나, ``rotate``/``done`` 의
            인자가 허용된 값(``left``/``right``, ``found``/``fail``)이 아닌 경우.
    """
    match = _ACTION_PATTERN.search(raw_response)
    if match is None:
        raise ActionParseError(
            '응답에서 think[...]/goto[...]/rotate[left|right]/done[found|fail] '
            f'패턴을 찾지 못했다: {raw_response!r}')

    kind, argument = match.group(1), match.group(2).strip()

    if kind == 'rotate' and argument not in _ROTATE_DIRECTIONS:
        raise ActionParseError(
            f"rotate 인자는 {_ROTATE_DIRECTIONS} 중 하나여야 한다: {argument!r}")
    if kind == 'done' and argument not in _DONE_VERDICTS:
        raise ActionParseError(
            f"done 인자는 {_DONE_VERDICTS} 중 하나여야 한다: {argument!r}")

    return ParsedAction(kind=kind, argument=argument)
