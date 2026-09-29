#!/usr/bin/env python3
"""STEP 4 부호 검증 로그를 장면별로 비교해 좌표계 부호가 맞는지 판정한다.

카메라 flip-method 를 2 로 고친 뒤 처음 하는 측정이므로, 모델이 뽑는 waypoint 의
부호가 로봇 좌표계와 실제로 일치하는지 확인하는 것이 목적이다.

부호 규약은 ROS REP-103 을 따른다. x 는 전진, y 는 왼쪽이 양수다. 따라서
왼쪽에 장애물이 있으면 오른쪽으로 피해야 하므로 dy 가 음수여야 한다.

입력은 ticvla_bridge.py 가 --log-json 으로 남기는 JSONL 이다. 같은 stem 의
.jsonl 이 없으면 tee 로 남긴 .log 를 텍스트 폴백으로 시도한다. 텍스트 로그에는
틱마다의 waypoint 가 없어 부호 판정이 SKIP 되므로 JSONL 을 쓰는 것이 좋다.

사용 예:
    python scripts/analyze_sign_test.py
    python scripts/analyze_sign_test.py --logs logs/a.jsonl logs/b.jsonl \\
        --labels 열린통로,벽앞
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: 저장소 루트. 상대경로 로그를 찾는 기준이다.
ROOT: Path = Path(__file__).resolve().parent.parent

#: 기본 로그 경로. ticvla_bridge.py --log-json 으로 남긴 파일을 기대한다.
DEFAULT_LOGS: tuple[str, ...] = (
    'logs/sign_scene1.jsonl',
    'logs/sign_scene2.jsonl',
    'logs/sign_scene3.jsonl',
    'logs/sign_scene4.jsonl',
)

#: 기본 장면 이름.
DEFAULT_LABELS: str = '열린통로,벽앞,왼쪽장애물,오른쪽장애물'

#: 텍스트 폴백에서 VLM 응답 줄을 뽑는 정규식.
RE_VLM_RESPONSE: re.Pattern[str] = re.compile(r'VLM 응답 #(\d+):\s*(.+?)\s*$')

#: 텍스트 폴백에서 VLM 갱신 소요 시간을 뽑는 정규식.
RE_VLM_ELAPSED: re.Pattern[str] = re.compile(
    r'VLM 캐시 갱신 #(\d+)\s*\(([0-9.]+)s')

#: 텍스트 폴백에서 명령 속도를 뽑는 정규식.
RE_CMD: re.Pattern[str] = re.compile(
    r'vx=([+-]?\d+\.\d+),\s*vy=([+-]?\d+\.\d+)')

#: 응답 안의 사고/답변 블록. 닫는 태그가 없는 잘린 응답도 잡는다.
RE_THINK: re.Pattern[str] = re.compile(r'<think>(.*?)(?:</think>|\Z)', re.DOTALL)
RE_ANSWER: re.Pattern[str] = re.compile(r'<answer>(.*?)(?:</answer>|\Z)', re.DOTALL)


# ======================================================================
# 표 정렬 도우미
# ======================================================================
def display_width(text: str) -> int:
    """터미널에 표시되는 문자 폭을 센다.

    한글은 두 칸을 차지하므로 len() 으로 자리를 맞추면 고정폭 표가 어긋난다.

    Args:
        text: 대상 문자열.

    Returns:
        표시 폭.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
               for ch in text)


def pad(text: str, width: int, align: str = 'left') -> str:
    """표시 폭을 기준으로 문자열을 채운다.

    Args:
        text: 대상 문자열.
        width: 목표 표시 폭.
        align: 'left' 또는 'right'.

    Returns:
        공백으로 채운 문자열.
    """
    space = ' ' * max(0, width - display_width(text))
    return text + space if align == 'left' else space + text


def signed(value: float) -> str:
    """부호를 항상 붙여 소수점 4자리로 만든다.

    Args:
        value: 값.

    Returns:
        "+0.0512" 형태. nan 이면 하이픈.
    """
    return '-' if math.isnan(value) else f'{value:+.4f}'


def plain(value: float) -> str:
    """소수점 4자리로 만든다. 부호는 붙이지 않는다.

    표준편차나 거리처럼 음수가 될 수 없는 값에 쓴다. 여기에 + 를 붙이면
    부호가 의미를 갖는 dx/dy 와 구분이 되지 않는다.

    Args:
        value: 값.

    Returns:
        "0.0083" 형태. nan 이면 하이픈.
    """
    return '-' if math.isnan(value) else f'{value:.4f}'


def percent(value: float) -> str:
    """비율을 백분율 문자열로 만든다.

    Args:
        value: 0.0~1.0 비율.

    Returns:
        "100%" 형태. nan 이면 하이픈.
    """
    return '-' if math.isnan(value) else f'{value * 100:.0f}%'


# ======================================================================
# 자료 구조
# ======================================================================
@dataclass
class Tick:
    """빠른 루프 한 틱의 결과.

    Attributes:
        dx: waypoint 마지막 스텝의 전후 변위(m). 양수가 전진.
        dy: waypoint 마지막 스텝의 좌우 변위(m). 양수가 왼쪽.
        distance: waypoint 마지막 스텝까지의 직선 거리(m).
        speed: 발행된 명령 속도 크기(m/s).
        deadzone_blocked: 데드존에 걸려 0 이 발행되었는가. 브릿지 판정을 그대로 쓴다.
        speed_limited: 최대 속도 제한에 걸렸는가.
        vlm_age_sec: 이 틱이 쓴 VLM 캐시의 나이(초).
    """

    dx: float
    dy: float
    distance: float
    speed: float
    deadzone_blocked: bool
    speed_limited: bool
    vlm_age_sec: float


@dataclass
class Response:
    """VLM 갱신 한 번의 응답.

    Attributes:
        vlm_runs: 몇 번째 갱신인가.
        elapsed_sec: predict() 소요 시간(초).
        text: 응답 원문.
    """

    vlm_runs: int
    elapsed_sec: float
    text: str


@dataclass
class Scene:
    """장면 하나의 측정 결과.

    Attributes:
        label: 장면 이름.
        source: 실제로 읽은 파일. 없으면 None.
        ticks: 틱 목록.
        responses: VLM 응답 목록.
        parse_failures: JSON 해석에 실패해 건너뛴 줄 수.
        text_fallback: 텍스트 로그로 대체했는가.
    """

    label: str
    source: Optional[Path] = None
    ticks: list[Tick] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    parse_failures: int = 0
    text_fallback: bool = False

    def values(self, key: str) -> list[float]:
        """틱에서 특정 항목만 뽑아 nan 을 걸러 돌려준다.

        Args:
            key: Tick 의 속성 이름.

        Returns:
            유한한 값 목록.
        """
        found = [float(getattr(t, key)) for t in self.ticks]
        return [v for v in found if not math.isnan(v)]

    def ratio(self, key: str) -> float:
        """불리언 항목이 True 인 틱의 비율을 구한다.

        Args:
            key: Tick 의 불리언 속성 이름.

        Returns:
            비율. 틱이 없으면 nan.
        """
        if not self.ticks:
            return float('nan')
        return sum(1 for t in self.ticks if getattr(t, key)) / len(self.ticks)

    @property
    def deadzone_pass_rate(self) -> float:
        """데드존을 통과한(0 으로 깎이지 않은) 틱의 비율."""
        blocked = self.ratio('deadzone_blocked')
        return float('nan') if math.isnan(blocked) else 1.0 - blocked

    @property
    def has_waypoints(self) -> bool:
        """부호 판정에 쓸 dx/dy 를 확보했는가."""
        return bool(self.values('dx'))


@dataclass
class CheckResult:
    """부호 판정 하나의 결과.

    Attributes:
        scene_name: 장면 표기(예: '장면1 열린통로').
        condition: 판정 조건 문구.
        status: 'PASS' / 'FAIL' / 'SKIP'.
        detail: 실측 수치 설명.
        noisy: 평균이 표준편차보다 작아 노이즈 수준인가.
        key: 어느 판정인지 구분하는 번호(1~4).
    """

    scene_name: str
    condition: str
    status: str
    detail: str
    noisy: bool
    key: int


# ======================================================================
# 통계
# ======================================================================
def mean_std(values: list[float]) -> tuple[float, float]:
    """평균과 표본표준편차를 구한다.

    Args:
        values: 값 목록.

    Returns:
        (평균, 표준편차). 값이 없으면 (nan, nan), 하나면 표준편차 0.
    """
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def mean_of(values: list[float]) -> float:
    """평균만 구한다.

    Args:
        values: 값 목록.

    Returns:
        평균. 값이 없으면 nan.
    """
    return statistics.fmean(values) if values else float('nan')


# ======================================================================
# 파싱
# ======================================================================
def classify(record: dict[str, Any]) -> str:
    """레코드 종류를 판정한다.

    event 키가 없는 구버전 파일도 읽을 수 있어야 하므로 키 존재로 되짚는다.

    Args:
        record: JSONL 레코드.

    Returns:
        'tick', 'vlm_refresh', 또는 알 수 없으면 빈 문자열.
    """
    event = record.get('event')
    if isinstance(event, str) and event:
        return event
    if 'waypoint_final' in record:
        return 'tick'
    if 'response' in record:
        return 'vlm_refresh'
    return ''


def as_float(source: Any, key: str) -> float:
    """딕셔너리에서 실수를 꺼낸다. 없거나 형식이 틀리면 nan.

    Args:
        source: 딕셔너리로 기대되는 값.
        key: 키 이름.

    Returns:
        실수 또는 nan.
    """
    if not isinstance(source, dict):
        return float('nan')
    try:
        return float(source[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def tick_from_record(record: dict[str, Any]) -> Optional[Tick]:
    """tick 레코드를 Tick 으로 바꾼다.

    dx/dy 는 waypoint_final 을 쓴다. target 은 lookahead 한 지점만 보는 값이라
    장면 사이의 의도 차이가 덜 드러난다.

    Args:
        record: JSONL 레코드.

    Returns:
        Tick. waypoint 가 없거나 NaN 기록이면 None.
    """
    if record.get('nan_inf'):
        return None
    final = record.get('waypoint_final')
    if not isinstance(final, dict):
        return None

    cmd = record.get('cmd')
    vx = as_float(cmd, 'vx')
    vy = as_float(cmd, 'vy')
    speed = (math.hypot(vx, vy)
             if not math.isnan(vx) and not math.isnan(vy) else float('nan'))
    return Tick(
        dx=as_float(final, 'dx'),
        dy=as_float(final, 'dy'),
        distance=as_float(final, 'distance_m'),
        speed=speed,
        deadzone_blocked=bool(record.get('deadzone_blocked', False)),
        speed_limited=bool(record.get('speed_limited', False)),
        vlm_age_sec=as_float(record, 'vlm_age_sec'),
    )


def response_from_record(record: dict[str, Any]) -> Optional[Response]:
    """vlm_refresh 레코드를 Response 로 바꾼다.

    Args:
        record: JSONL 레코드.

    Returns:
        Response. 응답 문자열이 비었으면 None.
    """
    text = record.get('response')
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        runs = int(record.get('vlm_runs', 0))
    except (TypeError, ValueError):
        runs = 0
    return Response(vlm_runs=runs,
                    elapsed_sec=as_float(record, 'elapsed_sec'),
                    text=text.strip())


def parse_jsonl(path: Path, scene: Scene) -> None:
    """JSONL 로그를 읽어 장면을 채운다.

    Args:
        path: JSONL 경로.
        scene: 결과를 채울 장면.
    """
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            scene.parse_failures += 1
            continue
        if not isinstance(record, dict):
            scene.parse_failures += 1
            continue

        kind = classify(record)
        if kind == 'tick':
            tick = tick_from_record(record)
            if tick is not None:
                scene.ticks.append(tick)
        elif kind == 'vlm_refresh':
            response = response_from_record(record)
            if response is not None:
                scene.responses.append(response)
        # 그 밖의 레코드는 조용히 건너뛴다.


def parse_text(path: Path, scene: Scene) -> None:
    """텍스트 로그에서 뽑을 수 있는 것만 뽑는다.

    브릿지는 틱마다의 waypoint 를 텍스트로 남기지 않으므로 dx/dy 는 얻을 수 없다.
    여기서 얻는 것은 VLM 응답과 상태 전이 시점의 명령 속도다.

    Args:
        path: 텍스트 로그 경로.
        scene: 결과를 채울 장면.
    """
    elapsed: dict[int, float] = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        found = RE_VLM_ELAPSED.search(line)
        if found:
            elapsed[int(found.group(1))] = float(found.group(2))
            continue

        found = RE_VLM_RESPONSE.search(line)
        if found:
            runs = int(found.group(1))
            scene.responses.append(Response(
                vlm_runs=runs,
                elapsed_sec=elapsed.get(runs, float('nan')),
                text=found.group(2).strip()))
            continue

        found = RE_CMD.search(line)
        if found:
            speed = math.hypot(float(found.group(1)), float(found.group(2)))
            scene.ticks.append(Tick(
                dx=float('nan'), dy=float('nan'), distance=float('nan'),
                speed=speed, deadzone_blocked=speed == 0.0,
                speed_limited=False, vlm_age_sec=float('nan')))


def resolve(pattern: str) -> tuple[Optional[Path], list[Path]]:
    """로그 경로를 찾는다. 없으면 같은 stem 의 .log 를 시도한다.

    Args:
        pattern: 경로. 상대경로는 저장소 루트 기준.

    Returns:
        (찾은 경로 또는 None, 찾아본 경로 목록).
    """
    raw = Path(pattern)
    primary = raw if raw.is_absolute() else ROOT / raw
    tried = [primary]
    if primary.is_file():
        return primary, tried

    fallback = primary.with_suffix('.log')
    if fallback != primary:
        tried.append(fallback)
        if fallback.is_file():
            return fallback, tried
    return None, tried


def load_scenes(patterns: list[str], labels: list[str]) -> list[Scene]:
    """경로와 이름을 짝지어 장면 목록을 만든다.

    Args:
        patterns: 로그 경로 목록.
        labels: 장면 이름 목록.

    Returns:
        장면 목록. 파일을 못 찾은 장면도 빈 상태로 포함한다.

    Raises:
        SystemExit: 파일을 하나도 찾지 못한 경우.
    """
    scenes: list[Scene] = []
    missing: list[tuple[str, list[Path]]] = []

    for index, pattern in enumerate(patterns):
        label = labels[index] if index < len(labels) else f'장면{index + 1}'
        found, tried = resolve(pattern)
        scene = Scene(label=label, source=found)
        if found is None:
            missing.append((label, tried))
        else:
            scene.text_fallback = found.suffix.lower() != '.jsonl'
            if scene.text_fallback:
                parse_text(found, scene)
            else:
                parse_jsonl(found, scene)
        scenes.append(scene)

    if missing:
        print('경고: 다음 장면의 로그를 찾지 못했다.', file=sys.stderr)
        for label, tried in missing:
            print(f'  {label}:', file=sys.stderr)
            for path in tried:
                print(f'    - {path} (없음)', file=sys.stderr)

    if all(scene.source is None for scene in scenes):
        print('', file=sys.stderr)
        print('오류: 읽을 로그가 하나도 없다. STEP 4 측정을 먼저 하거나 --logs 로 '
              '실제 경로를 지정할 것.', file=sys.stderr)
        print('  예: python scripts/ticvla_bridge.py --shadow '
              '--log-json logs/sign_scene1.jsonl 2>&1 | tee logs/sign_scene1.log',
              file=sys.stderr)
        raise SystemExit(2)

    return scenes


# ======================================================================
# 출력 1: 장면별 요약표
# ======================================================================
#: 요약표 컬럼 정의. (헤더, 표시폭, 정렬).
COLUMNS: tuple[tuple[str, int, str], ...] = (
    ('장면', 14, 'left'),
    ('틱수', 5, 'right'),
    ('dx평균', 9, 'right'),
    ('dx표준편차', 11, 'right'),
    ('dy평균', 9, 'right'),
    ('dy표준편차', 11, 'right'),
    ('최종변위평균', 13, 'right'),
    ('속도평균', 9, 'right'),
    ('데드존통과율', 13, 'right'),
    ('제한율', 7, 'right'),
    ('vlm_age평균', 12, 'right'),
)

#: 표 너비. 컬럼 폭 합계에 구분 공백을 더한 값.
TABLE_WIDTH: int = sum(width for _, width, _ in COLUMNS) + len(COLUMNS) - 1


def print_summary(scenes: list[Scene]) -> None:
    """장면별 요약표를 출력한다.

    Args:
        scenes: 장면 목록.
    """
    failures = [(s.label, s.parse_failures) for s in scenes if s.parse_failures]
    if failures:
        print()
        for label, count in failures:
            print(f'  파싱 실패 {count}줄 ({label})')

    print()
    print('=' * TABLE_WIDTH)
    print('1. 장면별 요약')
    print('=' * TABLE_WIDTH)
    print(' '.join(pad(name, width, align) for name, width, align in COLUMNS))
    print('-' * TABLE_WIDTH)

    for scene in scenes:
        dx_mean, dx_std = mean_std(scene.values('dx'))
        dy_mean, dy_std = mean_std(scene.values('dy'))
        cells = (
            (scene.label, 'left', ''),
            (str(len(scene.ticks)), 'right', ''),
            (signed(dx_mean), 'right', ''),
            (plain(dx_std), 'right', ''),
            (signed(dy_mean), 'right', ''),
            (plain(dy_std), 'right', ''),
            (plain(mean_of(scene.values('distance'))), 'right', ''),
            (plain(mean_of(scene.values('speed'))), 'right', ''),
            (percent(scene.deadzone_pass_rate), 'right', ''),
            (percent(scene.ratio('speed_limited')), 'right', ''),
            (plain(mean_of(scene.values('vlm_age_sec'))), 'right', ''),
        )
        print(' '.join(pad(text, COLUMNS[i][1], align)
                       for i, (text, align, _) in enumerate(cells)))

    print('-' * TABLE_WIDTH)
    print('  dx/dy 는 waypoint 마지막 스텝의 상대 변위(m). x 전진 양수, y 왼쪽 양수.')
    print('  데드존통과율/제한율은 브릿지가 기록한 deadzone_blocked, speed_limited 를 '
          '그대로 집계한 값이다.')

    ages = [(s.label, max(s.values('vlm_age_sec'), default=float('nan')))
            for s in scenes if s.values('vlm_age_sec')]
    if ages:
        shown = ', '.join(f'{label} {value:.1f}s' for label, value in ages)
        print(f'  vlm_age 최대: {shown}')

    fallbacks = [s.label for s in scenes if s.text_fallback]
    if fallbacks:
        print(f'  텍스트 폴백 사용: {", ".join(fallbacks)} — 이 장면은 waypoint 기록이 '
              '없어 부호 판정이 SKIP 된다')

    absent = [s.label for s in scenes if s.source is None]
    if absent:
        print(f'  로그 없음: {", ".join(absent)}')


# ======================================================================
# 출력 2: 부호 판정
# ======================================================================
def scene_at(scenes: list[Scene], index: int) -> Optional[Scene]:
    """장면 목록에서 index 번째를 안전하게 꺼낸다.

    Args:
        scenes: 장면 목록.
        index: 0 부터 시작하는 번호.

    Returns:
        장면 또는 None.
    """
    return scenes[index] if index < len(scenes) else None


def evaluate(scenes: list[Scene]) -> list[CheckResult]:
    """부호 판정 네 가지를 수행한다.

    Args:
        scenes: 장면 목록. 순서가 장면1~4 에 대응한다.

    Returns:
        판정 결과 목록.
    """
    results: list[CheckResult] = []
    stats: dict[int, tuple[float, float, float, float]] = {}
    for index in range(4):
        scene = scene_at(scenes, index)
        if scene is None:
            continue
        dx_mean, dx_std = mean_std(scene.values('dx'))
        dy_mean, dy_std = mean_std(scene.values('dy'))
        stats[index] = (dx_mean, dx_std, dy_mean, dy_std)

    def name_of(index: int) -> str:
        """'장면N 이름' 표기를 만든다."""
        scene = scene_at(scenes, index)
        label = scene.label if scene is not None else ''
        return f'장면{index + 1} {label}'.strip()

    def skipped(index: int, condition: str, key: int) -> CheckResult:
        """자료가 없어 판정하지 못한 결과를 만든다."""
        scene = scene_at(scenes, index)
        if scene is None:
            reason = '로그 경로가 지정되지 않았다'
        elif scene.source is None:
            reason = '로그 파일이 없다'
        else:
            reason = f'{scene.source.name} 에 waypoint 기록이 없다'
        return CheckResult(name_of(index), condition, 'SKIP',
                           f'측정 없음 ({reason})', False, key)

    # ① 열린 통로에서는 앞으로 나아가야 한다.
    condition = 'dx > 0'
    scene = scene_at(scenes, 0)
    if scene is None or not scene.has_waypoints:
        results.append(skipped(0, condition, 1))
    else:
        dx_mean, dx_std, _, _ = stats[0]
        results.append(CheckResult(
            name_of(0), condition,
            'PASS' if dx_mean > 0.0 else 'FAIL',
            f'dx평균 {signed(dx_mean)}, 표준편차 {plain(dx_std)}',
            abs(dx_mean) < dx_std, 1))

    # ② 벽 앞에서는 열린 통로보다 덜 전진해야 한다.
    condition = 'dx평균(장면2) < dx평균(장면1)'
    scene = scene_at(scenes, 1)
    first = scene_at(scenes, 0)
    if (scene is None or not scene.has_waypoints
            or first is None or not first.has_waypoints):
        results.append(skipped(1, condition, 2))
    else:
        dx2, std2, _, _ = stats[1]
        dx1, _, _, _ = stats[0]
        results.append(CheckResult(
            name_of(1), condition,
            'PASS' if dx2 < dx1 else 'FAIL',
            f'dx평균 {signed(dx2)}, 표준편차 {plain(std2)} '
            f'(장면1 {signed(dx1)}, 차이 {signed(dx2 - dx1)})',
            abs(dx2 - dx1) < std2, 2))

    # ③ 왼쪽 장애물 → 오른쪽으로 피한다 → dy 음수.
    condition = 'dy < 0'
    scene = scene_at(scenes, 2)
    if scene is None or not scene.has_waypoints:
        results.append(skipped(2, condition, 3))
    else:
        _, _, dy_mean, dy_std = stats[2]
        results.append(CheckResult(
            name_of(2), condition,
            'PASS' if dy_mean < 0.0 else 'FAIL',
            f'dy평균 {signed(dy_mean)}, 표준편차 {plain(dy_std)}',
            abs(dy_mean) < dy_std, 3))

    # ④ 오른쪽 장애물 → 왼쪽으로 피한다 → dy 양수.
    condition = 'dy > 0'
    scene = scene_at(scenes, 3)
    if scene is None or not scene.has_waypoints:
        results.append(skipped(3, condition, 4))
    else:
        _, _, dy_mean, dy_std = stats[3]
        results.append(CheckResult(
            name_of(3), condition,
            'PASS' if dy_mean > 0.0 else 'FAIL',
            f'dy평균 {signed(dy_mean)}, 표준편차 {plain(dy_std)}',
            abs(dy_mean) < dy_std, 4))

    return results


def print_checks(results: list[CheckResult]) -> None:
    """부호 판정 결과를 출력한다.

    Args:
        results: 판정 결과 목록.
    """
    print()
    print('=' * TABLE_WIDTH)
    print('2. 부호 판정')
    print('=' * TABLE_WIDTH)
    for result in results:
        print(f'  {pad(result.scene_name, 22)} '
              f'{pad(result.condition, 30)} : {pad(result.status, 4)} '
              f' ({result.detail})')
        if result.noisy:
            print('       ※ 평균이 표준편차보다 작다. 장면 차이가 노이즈 수준일 '
                  '수 있음.')
    print()


# ======================================================================
# 출력 3: VLM 응답 원문
# ======================================================================
def print_responses(scenes: list[Scene]) -> None:
    """장면별 VLM 응답을 순서대로 출력한다.

    장애물을 실제로 언급했는지는 사람이 읽고 판단해야 한다.

    Args:
        scenes: 장면 목록.
    """
    print('=' * TABLE_WIDTH)
    print('3. VLM 응답 원문 (장애물 언급 여부는 직접 확인할 것)')
    print('=' * TABLE_WIDTH)

    for scene in scenes:
        print()
        header = f'── {scene.label} (응답 {len(scene.responses)}개) '
        print(header + '─' * max(0, TABLE_WIDTH - display_width(header)))
        if not scene.responses:
            print('  응답 기록이 없다. ticvla_bridge.py 가 갱신마다 남기는 '
                  'vlm_refresh 레코드나')
            print('  "VLM 응답 #n" 로그가 파일에 포함되어야 한다.')
            continue

        for response in scene.responses:
            elapsed = ('' if math.isnan(response.elapsed_sec)
                       else f', {response.elapsed_sec:.2f}s')
            print(f'  [#{response.vlm_runs}{elapsed}]')
            print_blocks(response.text)
    print()


def print_blocks(text: str) -> None:
    """응답에서 think/answer 블록을 나눠 출력한다.

    Args:
        text: 응답 원문.
    """
    think = RE_THINK.search(text)
    answer = RE_ANSWER.search(text)
    if think is None and answer is None:
        print('    [원문] (think/answer 태그 없음)')
        print(indent(text.strip()))
        return

    if think is not None:
        truncated = '' if '</think>' in text else '   ← 닫는 태그 없음 (잘렸다)'
        print(f'    [think]{truncated}')
        print(indent(think.group(1).strip()))
    if answer is not None:
        truncated = '' if '</answer>' in text else '   ← 닫는 태그 없음 (잘렸다)'
        print(f'    [answer]{truncated}')
        print(indent(answer.group(1).strip()))


def indent(text: str, prefix: str = '      ') -> str:
    """여러 줄 텍스트에 들여쓰기를 붙인다.

    Args:
        text: 원본 텍스트.
        prefix: 각 줄 앞에 붙일 문자열.

    Returns:
        들여쓴 텍스트.
    """
    return '\n'.join(prefix + line for line in text.splitlines()) or prefix


# ======================================================================
# 출력 4: 전체 판정
# ======================================================================
#: 판정 번호별 실패 시 의심 지점.
SUSPICION: dict[int, str] = {
    1: '_waypoints_to_twist 의 x 부호 확인',
    2: '부호 문제 아님. 모델이 벽을 인식 못 했을 가능성. 해당 장면 VLM 응답 확인',
    3: '좌표 변환 계층의 y 부호 반전 확인',
    4: '좌표 변환 계층의 y 부호 반전 확인',
}


def print_verdict(results: list[CheckResult]) -> bool:
    """전체 판정 문구를 출력한다.

    Args:
        results: 판정 결과 목록.

    Returns:
        네 항목 모두 PASS 면 True.
    """
    print('=' * TABLE_WIDTH)
    print('4. 전체 판정')
    print('=' * TABLE_WIDTH)

    skipped = [r for r in results if r.status == 'SKIP']
    failed = [r for r in results if r.status == 'FAIL']

    if skipped:
        print(f'  측정 미완료 — {len(skipped)}개 장면의 판정을 하지 못했다.')
        for result in skipped:
            print(f'    - {result.scene_name}: {result.detail}')
        print()

    if not failed and not skipped:
        print('  좌표계 부호 정상. 실주행 진행 가능.')
        noisy = [r for r in results if r.noisy]
        if noisy:
            print()
            print('  단, 다음 항목은 평균이 표준편차보다 작아 확신하기 어렵다.')
            for result in noisy:
                print(f'    - {result.scene_name}: {result.detail}')
            print('  장애물을 더 가깝게 두고 다시 측정하면 판정이 뚜렷해진다.')
        return True

    if failed:
        print(f'  실패 {len(failed)}건 — 실주행을 진행하면 안 된다.')
        for result in failed:
            print(f'    - {result.scene_name} ({result.condition})')
            print(f'      실측: {result.detail}')
            print(f'      의심: {SUSPICION.get(result.key, "원인 미분류")}')
        print()

        keys = {r.key for r in failed}
        if {3, 4} <= keys:
            print('  장면3 과 장면4 가 함께 실패했다. 좌우 회피가 양쪽 모두 반대이므로 '
                  'y 부호 규약')
            print('  자체가 뒤집힌 것이다. 좌표 변환 계층에서 dy 부호를 한 번만 '
                  '반전시킬 것.')
        elif keys == {2}:
            print('  부호 판정은 모두 통과했고 장면2 만 실패했다. 좌표계 문제가 '
                  '아니므로 해당 장면의')
            print('  VLM 응답에 벽 언급이 있는지 위 3번 항목에서 확인할 것.')

    return False


# ======================================================================
# 진입점
# ======================================================================
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 해석한다.

    Args:
        argv: 인자 목록. None 이면 sys.argv 를 쓴다.

    Returns:
        해석된 설정.
    """
    parser = argparse.ArgumentParser(
        description='STEP 4 부호 검증 로그를 장면별로 비교해 좌표계 부호를 판정한다',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logs', nargs='+', default=list(DEFAULT_LOGS),
                        help='장면별 JSONL 경로. 없으면 같은 이름의 .log 를 '
                             '텍스트 폴백으로 시도한다')
    parser.add_argument('--labels', default=DEFAULT_LABELS,
                        help='쉼표로 구분한 장면 이름')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """부호 검증 분석을 실행한다.

    Args:
        argv: 명령행 인자.

    Returns:
        네 항목 모두 PASS 면 0, 아니면 1.
    """
    cfg = parse_args(argv)
    labels = [name.strip() for name in cfg.labels.split(',') if name.strip()]
    if len(labels) < len(cfg.logs):
        print(f'경고: 로그 {len(cfg.logs)}개에 이름 {len(labels)}개만 주어졌다. '
              '남는 장면은 기본 이름을 쓴다.', file=sys.stderr)

    scenes = load_scenes(cfg.logs, labels)

    print()
    print('STEP 4 부호 검증 분석')
    print('  기준: x 전진 양수 / y 왼쪽 양수 (ROS REP-103)')
    for index, scene in enumerate(scenes, 1):
        source = scene.source.name if scene.source else '(없음)'
        print(f'  장면{index} {scene.label}: {source}')

    print_summary(scenes)
    results = evaluate(scenes)
    print_checks(results)
    print_responses(scenes)
    return 0 if print_verdict(results) else 1


if __name__ == '__main__':
    sys.exit(main())
