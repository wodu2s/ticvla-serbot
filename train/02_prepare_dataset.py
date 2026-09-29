#!/usr/bin/env python3
"""raw 3색 마커 에피소드 → TIC-VLA 학습용 GND_json 변환기 (수집 → 학습 사이 층).

변환 자체(윈도우 슬라이싱, FLU 오프셋 계산, waypoint/history/future 생성)는
third_party/TIC-VLA/data/s01_batch_json_generation.py 가 그대로 한다.
이 스크립트는 그것을 감싸는 얇은 층이다 — s01 을 재구현하지 않는다.

입력: ~/TIC-VLA/data/GoToColorMarker_v1/raw/ep{NNNN}_L{layout}_{color}/
      (rgb/rgb_00000.jpg…, trajectory.csv, episode.json — joystick_colormarker.py 산출물)

third_party/TIC-VLA 는 읽기 전용으로 취급한다. 실행 파일을 고쳐야 하는 경우
(s01 의 모듈 상수)에는 **임시 사본을 만들어 상수만 치환**한 뒤 그 사본을 실행한다.
원본 s01_batch_json_generation.py 는 절대 건드리지 않는다.

────────────────────────────────────────────────────────────────────────
결정 1 — 출력 디렉토리 이름은 GND_json 이어야 한다 (자유롭지 않음)
────────────────────────────────────────────────────────────────────────
third_party/TIC-VLA/ticvla/data/vlm_data.py 의 `_detect_dataset_info()` 는
샘플 경로 문자열에 "/GND/GND_json" 또는 "/GND_json/" 이 있는지만 보고
데이터셋 종류를 하드코딩으로 판정한다(GND_json / SCAND_json / DynaNav_json
셋 중 하나가 아니면 `_infer_robot_type()` 이 ValueError 를 던진다 — 실제로
직접 그 함수를 읽고 확인했다). 그래서 출력을 반드시
    ~/TIC-VLA/data/GoToColorMarker_v1/GND/GND_json/
아래에 둔다. 'ColorMarker_json' 같은 임의 이름을 쓰면 학습 첫 배치에서 죽는다.

────────────────────────────────────────────────────────────────────────
결정 2 — GND_data 디렉토리는 만들지 않는다 (s04 도 돌리지 않는다) ★
────────────────────────────────────────────────────────────────────────
s01 이 각 프레임 json 의 "img" 값에 넣는 경로는 **원본 rgb/ 파일의 절대경로**
그대로다(`rgb_abs_root = rgb_dir.resolve().as_posix()`, s01:283-284). s04 는
이 절대경로들을 GND_json/GND_data 상대경로로 "다시 쓰는" 스크립트인데, 실제
이미지 파일을 복사하지는 않는다(`copy_non_json_siblings` 는 instruction_*.txt
/cot_*.txt 같은 텍스트 사이드카만 복사하고 .jpg 는 건드리지 않는다 — 직접
읽고 확인했다). 즉 s04 를 돌려도 GND_data 밑에 실제 이미지가 생기지 않는다
— 우리가 따로 복사/심볼릭링크를 만들어야 그 폴더가 의미를 갖는다.

그런데 로더(`vlm_data.py::_remap_image_path`)는 이렇게 동작한다:

    if os.path.isabs(img_path_normalized) and os.path.exists(img_path_normalized):
        return img_path_normalized   # ← 존재하는 절대경로면 바로 쓰고 끝

즉 "img" 가 **존재하는 절대경로**면 GND_data/데이터셋 판정 로직 전체를
건너뛰고 그 경로를 그대로 쓴다. s01 이 이미 그런 절대경로(원본 raw/epNNNN/rgb/
아래)를 넣어주므로, GND_data 를 만들어 이미지를 복사하는 것은 디스크만
두 배로 먹을 뿐 아무 기능적 이득이 없다 — 그래서 만들지 않는다.

트레이드오프(반드시 알아야 할 것): 이 방식은 raw/ 폴더가 학습이 끝날 때까지
**그 자리에 그대로** 있어야 함을 전제한다. raw/ 를 지우거나 다른 경로로
옮기면 이미 생성된 GND_json 의 절대경로가 깨진다. 학습을 다른 머신으로
옮길 때는 GND/ 뿐 아니라 raw/ 도 함께 복사해야 한다. 이 사실을
PREP_REPORT.md 에도 남긴다.

────────────────────────────────────────────────────────────────────────
결정 3 — s02(GPT 지시문 생성)는 아예 돌리지 않는다
────────────────────────────────────────────────────────────────────────
s02_batch_instruction.py 는 이미지를 보고 GPT 로 지시문 문장 자체를
**새로 생성**한다 — 우리 데이터처럼 "이 에피소드의 목표색은 이미 정해져
있고 그 문장이 episode.json 에 있다"는 상황과 맞지 않는다(GPT 가 색을
잘못 언급하면 이 데이터셋의 존재 이유인 색 판별 학습이 조용히 무효화된다).
그래서 s02 를 호출하지 않고, 이 스크립트가 직접 episode.json 의
"instruction" 값을 읽어 s01 이 각 프레임 json 에 남긴 instruction_file
경로에 그대로 써 넣는다(§ instruction 처리 참고). 문장은 절대 스크립트에
하드코딩하지 않는다 — 하드코딩하면 색과 문장이 어긋나도 조용히 지나간다.

────────────────────────────────────────────────────────────────────────
결정 4 — --skip-annotation 일 때 "cot" 키를 제거한다 (빈 파일을 까는 대신)
────────────────────────────────────────────────────────────────────────
학습 데이터로더 `_load_annotation()` (vlm_data.py:318-377) 은:

    cot_file = data.get('cot')
    if cot_file and isinstance(cot_file, str):        # ← 키가 없으면 통째로 건너뜀
        ...
        except FileNotFoundError:
            self._consecutive_missing_count += 1
            if self._consecutive_missing_count >= 11:
                raise FileNotFoundError(...)           # ← 폴더당 11번째에서 죽는다

"cot" 키 자체를 지우면 `if cot_file and ...` 가 False 라서 그 블록 전체를
건너뛴다 — 파일을 열어보지도 않고, consecutive_missing_count 도 절대
건드리지 않는다. 빈 cot_*.txt 를 까는 방법도 크래시는 막지만(빈 문자열은
falsy 라 결과적으로 같은 브랜치로 감), 프레임마다 실제 파일 열기 시도가
남고 디스크에 의미 없는 빈 파일이 수백 개 생긴다. 그래서 **키 제거**를
택했다. (이 사실은 PREP_REPORT.md 에도 다시 적는다.)

부작용: cot 가 없으면 `_build_messages` 가 다른 프롬프트를 쓴다
("Use reasoning to predict future target waypoints" → "Return the future
target waypoints"). stage-2(Action Expert) 손실은 smooth_l1(waypoint) 뿐이고
<think> 텍스트는 loss 에 안 들어가므로 1차 학습에서는 문제없지만, 1차/2차
결과를 비교하려면 이 조건(어노테이션 유무)을 고정해야 한다.
(참고: 학습을 teacher_answer=drop 모드로 돌리면 assistant 메시지 자체가
빠지므로 이 -100/코트 유무 자체가 무관해진다 — third_party 학습 스크립트
쪽 설정이라 여기서 제어하지 않는다.)

────────────────────────────────────────────────────────────────────────
금지 (지킨 것)
────────────────────────────────────────────────────────────────────────
- third_party/TIC-VLA 의 파일은 읽기만 한다. s01 은 임시 사본으로만 실행한다.
- data/GoToObject_v1 (이전 회색 통 데이터셋)은 이 스크립트 어디에서도 경로로
  등장하지 않는다.
- instruction 문장은 스크립트에 하드코딩하지 않는다 — 항상 episode.json 에서
  읽는다.
- 검수 실패는 조용히 넘기지 않는다 — 스킵하되 항상 요약표에 남긴다.

CLI
---
    python3 train/02_prepare_dataset.py --episodes 3
    python3 train/02_prepare_dataset.py --episodes all --skip-annotation
    python3 train/02_prepare_dataset.py --episodes all --force
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────────────────────────────

#: ~/TIC-VLA
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 읽기 전용으로 취급하는 업스트림 레포.
THIRD_PARTY_REPO = PROJECT_ROOT / "third_party" / "TIC-VLA"
S01_SCRIPT = THIRD_PARTY_REPO / "data" / "s01_batch_json_generation.py"
S03_SCRIPT = THIRD_PARTY_REPO / "data" / "s03_batch_annotate.py"

DATASET_NAME = "GoToColorMarker_v1"
DATA_ROOT = PROJECT_ROOT / "data" / DATASET_NAME
RAW_ROOT = DATA_ROOT / "raw"
GND_ROOT = DATA_ROOT / "GND"
GND_JSON_DIR = GND_ROOT / "GND_json"
REPORT_PATH = DATA_ROOT / "PREP_REPORT.md"
INSUFFICIENT_FUTURE_LIST = GND_ROOT / "insufficient_future_samples.txt"

# ──────────────────────────────────────────────────────────────────────
# 검수 상수
# ──────────────────────────────────────────────────────────────────────

TRAJECTORY_HEADER = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]
TARGET_COLORS = ("red", "green", "blue")
SIDES = ("left", "center", "right")

#: 목표 프레임 간격과 허용 폭(ms). verify_errand_session.py 와 같은 기준.
TARGET_INTERVAL_MS = 100.0
INTERVAL_TOL_MS = 20.0

#: 쿼터니언 정규화 허용 오차. verify_errand_session.py 와 같은 기준.
QUAT_NORM_TOL = 0.05

#: 감속 확인에 쓰는 구간 길이(초, 10Hz 기준 30샘플).
DECEL_WINDOW_S = 3.0
DECEL_WINDOW_N = int(DECEL_WINDOW_S * 10)

#: episode.json 필수 필드.
REQUIRED_EPISODE_FIELDS = (
    "layout", "target_color", "target_side", "instruction",
    "success", "final_distance", "notes",
)

#: rgb_00000.jpg 처럼 '접두_숫자.확장자' 인지 — policy_data.get_numeric_key 와
#: 정확히 같은 규칙(stem.split('_', 1)[1] 이 float 로 파싱돼야 한다).
def is_numeric_key_safe(stem: str) -> bool:
    """stem 이 policy_data.get_numeric_key 요구사항을 만족하는지.

    Args:
        stem: 확장자를 뺀 파일명(예: "rgb_00000").

    Returns:
        '_' 뒤가 float 로 파싱되면 True.
    """
    if "_" not in stem:
        return False
    numeric_part = stem.split("_", 1)[1]
    try:
        float(numeric_part)
        return True
    except ValueError:
        return False


# ──────────────────────────────────────────────────────────────────────
# 소소한 유틸
# ──────────────────────────────────────────────────────────────────────

def yaw_from_quat(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray,
                  qw: np.ndarray) -> np.ndarray:
    """쿼터니언에서 yaw(rad) 를 뽑는다 (verify_errand_session.py 와 동일 공식).

    Args:
        qx, qy, qz, qw: 쿼터니언 성분 배열.

    Returns:
        unwrap 된 yaw 배열.
    """
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny, cosy))


def body_forward_speed(x: np.ndarray, y: np.ndarray,
                       yaw: np.ndarray) -> np.ndarray:
    """동체 전방 속도(스텝당, m/step) — 월드 x 단조성이 아니라 FLU 전방 투영.

    좌회전만 해도 월드 x 는 줄 수 있으므로, 후진/감속 판정은 항상 이
    값(동체 전방 변위)을 써야 한다.

    Args:
        x, y: 월드 좌표.
        yaw: unwrap 된 yaw 배열(같은 길이).

    Returns:
        길이 len(x)-1 인 스텝별 전방 변위(m/step, 10Hz 이므로 ×10 하면 m/s).
    """
    dx, dy = np.diff(x), np.diff(y)
    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    return dx * np.cos(yaw_mid) + dy * np.sin(yaw_mid)


def write_json_atomic(path: Path, payload: dict) -> None:
    """json 을 임시 파일에 쓴 뒤 replace 한다.

    Args:
        path: 최종 경로.
        payload: 직렬화할 dict.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────────
# 입력 검수 (에피소드 단위)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeCheck:
    """에피소드 하나의 입력 검수 결과."""

    name: str
    ep_dir: Path
    ok: bool = True
    fails: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    instruction: str = ""
    target_color: str = ""
    layout: Optional[int] = None
    n_frames: int = 0


def check_episode(ep_dir: Path) -> EpisodeCheck:
    """에피소드 하나를 검수한다. 실패해도 예외를 던지지 않고 fails 에 쌓는다.

    Args:
        ep_dir: raw/ep{NNNN}_L{layout}_{color} 폴더.

    Returns:
        검수 결과.
    """
    chk = EpisodeCheck(name=ep_dir.name, ep_dir=ep_dir)

    ep_json_path = ep_dir / "episode.json"
    if not ep_json_path.is_file():
        chk.fails.append("episode.json 없음")
        chk.ok = False
        return chk
    try:
        meta = json.loads(ep_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        chk.fails.append(f"episode.json 파싱 실패: {exc}")
        chk.ok = False
        return chk

    missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in meta]
    if missing:
        chk.fails.append(f"episode.json 필수 필드 없음: {missing}")
    layout = meta.get("layout")
    color = str(meta.get("target_color", ""))
    side = str(meta.get("target_side", ""))
    instruction = str(meta.get("instruction", ""))
    chk.instruction = instruction
    chk.target_color = color
    chk.layout = layout if isinstance(layout, int) else None

    if color not in TARGET_COLORS:
        chk.fails.append(f"target_color 이상함: {color!r}")
    if side not in SIDES:
        chk.fails.append(f"target_side 이상함: {side!r}")
    if not isinstance(layout, int) or layout not in (1, 2, 3):
        chk.fails.append(f"layout 이 1/2/3 정수가 아님: {layout!r}")

    # ── instruction 색 일치 ★ 불일치는 즉시 실패 ──
    if color in TARGET_COLORS and instruction:
        mentioned = {c for c in TARGET_COLORS if c in instruction.lower()}
        if mentioned != {color}:
            chk.fails.append(
                f"instruction 색 불일치: target_color={color!r} "
                f"instruction={instruction!r} (언급된 색={sorted(mentioned)})")
    elif not instruction:
        chk.fails.append("instruction 이 비어 있음")

    if chk.fails:
        chk.ok = False
        # episode.json 자체가 깨졌으면 트래젝토리/이미지 검사는 의미가 적지만
        # 계속 진행해서 요약표에 모든 문제를 한 번에 보여준다.

    # ── rgb / trajectory.csv ──
    rgb_dir = ep_dir / "rgb"
    csv_path = ep_dir / "trajectory.csv"
    if not rgb_dir.is_dir():
        chk.fails.append("rgb/ 폴더 없음")
        chk.ok = False
        return chk
    if not csv_path.is_file():
        chk.fails.append("trajectory.csv 없음")
        chk.ok = False
        return chk

    images = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.jpeg")) + \
        sorted(rgb_dir.glob("*.png"))
    images = sorted(images)

    # ★ 파일명 패턴 — get_numeric_key 가 요구하는 형식인지.
    bad_names = [p.name for p in images if not is_numeric_key_safe(p.stem)]
    if bad_names:
        chk.fails.append(
            f"파일명이 '접두_숫자.확장자' 형식이 아님 ({len(bad_names)}개, "
            f"예: {bad_names[:3]}) — 학습 첫 배치에서 크래시한다")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        import csv as csv_mod
        reader = csv_mod.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            chk.fails.append("trajectory.csv 가 비었다")
            chk.ok = False
            return chk
        rows = [r for r in reader if r]

    if header != TRAJECTORY_HEADER:
        chk.fails.append(f"trajectory.csv 헤더 불일치: {header}")
        chk.ok = False
        return chk

    n_rows = len(rows)
    chk.n_frames = n_rows
    if len(images) != n_rows:
        chk.fails.append(f"rgb {len(images)}장 ≠ csv {n_rows}행")

    if n_rows < 2:
        chk.fails.append(f"csv 행이 {n_rows}개뿐 — 검사를 더 진행할 수 없다")
        chk.ok = not chk.fails
        return chk

    idx = {c: header.index(c) for c in TRAJECTORY_HEADER}
    try:
        data = {c: np.array([float(r[idx[c]]) for r in rows], dtype=np.float64)
               for c in TRAJECTORY_HEADER}
    except (ValueError, IndexError) as exc:
        chk.fails.append(f"trajectory.csv 숫자 파싱 실패: {exc}")
        chk.ok = not chk.fails
        return chk

    t = data["time"]
    dt_ms = np.diff(t) * 1000.0
    if dt_ms.size:
        med = float(np.median(dt_ms))
        if abs(med - TARGET_INTERVAL_MS) > INTERVAL_TOL_MS:
            chk.fails.append(
                f"프레임 간격 중앙값 {med:.1f}ms "
                f"(허용 {TARGET_INTERVAL_MS:.0f}±{INTERVAL_TOL_MS:.0f}ms)")

    # ── 쿼터니언 정규화 (odom 환각 탐지) ──
    qnorm = np.sqrt(data["qx"] ** 2 + data["qy"] ** 2
                    + data["qz"] ** 2 + data["qw"] ** 2)
    max_err = float(np.abs(qnorm - 1.0).max())
    if max_err > QUAT_NORM_TOL:
        chk.fails.append(
            f"쿼터니언 비정규화 (|q| 최대 오차 {max_err:.3f}) — odom 의심")

    # ── 감속 확인 (경고 수준) ──
    if n_rows >= DECEL_WINDOW_N * 2 + 1:
        yaw = yaw_from_quat(data["qx"], data["qy"], data["qz"], data["qw"])
        fwd = body_forward_speed(data["x"], data["y"], yaw)  # m/step, len n-1
        last = fwd[-DECEL_WINDOW_N:]
        prev = fwd[-2 * DECEL_WINDOW_N:-DECEL_WINDOW_N]
        mean_last, mean_prev = float(last.mean()), float(prev.mean())
        if mean_last >= mean_prev * 0.95:
            chk.warns.append(
                f"감속 신호 약함/없음 — 마지막 {DECEL_WINDOW_S:.0f}s 평균속도 "
                f"{mean_last * 10:.3f}m/s ≥ 그 앞 구간 {mean_prev * 10:.3f}m/s")
        # 급정지(계단식으로 0) 탐지: 마지막 구간 안에서 한 스텝 만에 크게 꺾이고
        # 이후 거의 정지하는 패턴.
        if last.size >= 5:
            drop = -np.diff(last)
            biggest = float(drop.max()) if drop.size else 0.0
            tail_mean = float(np.abs(last[-3:]).mean())
            if biggest > 0.003 and tail_mean < 0.001:
                chk.warns.append(
                    f"급정지 패턴 의심 — 마지막 구간에서 한 스텝 최대 감속 "
                    f"{biggest * 10:.3f}m/s, 종료 직전 평균속도 "
                    f"{tail_mean * 10:.4f}m/s (계단식 정지)")
    else:
        chk.warns.append("에피소드가 짧아 감속 구간(6s)을 확인할 수 없다")

    chk.ok = not chk.fails
    return chk


# ──────────────────────────────────────────────────────────────────────
# s01 임시 사본 (상수 치환)
# ──────────────────────────────────────────────────────────────────────

#: (정규식, 치환문자열, 사람이 읽을 이름) — 값이 이미 같아도 항상 강제한다.
#: third_party 가 git pull 등으로 원본(30Hz/10s/True)으로 되돌아가도
#: 조용히 잘못된 설정으로 새지 않도록 하기 위함이다.
S01_PATCHES: tuple[tuple[str, str, str], ...] = (
    (r"^SRC_HZ\s*=\s*[0-9.]+", "SRC_HZ = 10.0", "SRC_HZ"),
    (r"^WINDOW_SLIDE_SECONDS\s*=\s*[0-9.]+", "WINDOW_SLIDE_SECONDS = 2.0",
     "WINDOW_SLIDE_SECONDS"),
    (r"^USE_CAMERA_OPTICAL_TO_FLU\s*=\s*(True|False)",
     "USE_CAMERA_OPTICAL_TO_FLU = False", "USE_CAMERA_OPTICAL_TO_FLU"),
)


def make_patched_s01(tmp_dir: Path) -> Path:
    """s01 을 임시 사본으로 복사하고 모듈 상수만 치환한다. 원본은 건드리지 않는다.

    Args:
        tmp_dir: 사본을 둘 임시 디렉토리.

    Raises:
        RuntimeError: 원본에서 치환 대상 줄을 정확히 하나씩 찾지 못하면
            (= 원본 형식이 바뀌었다는 뜻) 조용히 넘어가지 않고 즉시 실패한다.

    Returns:
        치환된 사본 경로.
    """
    if not S01_SCRIPT.is_file():
        raise RuntimeError(f"s01 스크립트를 찾을 수 없다: {S01_SCRIPT}")
    src = S01_SCRIPT.read_text(encoding="utf-8")
    for pattern, replacement, label in S01_PATCHES:
        rx = re.compile(pattern, re.MULTILINE)
        n = len(rx.findall(src))
        if n != 1:
            raise RuntimeError(
                f"s01 에서 {label} 줄을 정확히 1개 찾지 못했다(찾은 개수={n}) "
                "— 원본 s01_batch_json_generation.py 형식이 바뀐 것 같다. "
                "이 스크립트의 S01_PATCHES 정규식을 다시 확인할 것.")
        src = rx.sub(replacement, src, count=1)
    patched = tmp_dir / "s01_patched.py"
    patched.write_text(src, encoding="utf-8")
    return patched


def run_s01(patched_s01: Path, ep_dir: Path, output_dir: Path) -> None:
    """패치된 s01 을 단일 에피소드(단일 scene)로 돌린다.

    Args:
        patched_s01: make_patched_s01() 이 만든 사본.
        ep_dir: 입력 에피소드 폴더(trajectory.csv 가 직접 있어야 단일 scene 모드).
        output_dir: GND_JSON_DIR.

    Raises:
        subprocess.CalledProcessError: s01 이 실패하면.
    """
    subprocess.run(
        [sys.executable, str(patched_s01),
         "--input_dir", str(ep_dir), "--output_dir", str(output_dir)],
        check=True,
    )


def ensure_meta_json(ep_dir: Path, instruction: str) -> None:
    """s01 이 요구하는 meta.json(비어있지 않은 instruction) 을 만든다.

    s01 은 episode.json 을 모른다 — meta.json 또는 instruction.txt 가 없으면
    아무것도 하지 않고 바로 FileNotFoundError 를 던진다. episode.json 의
    instruction 을 그대로 복사해 넣는다(문장을 새로 만들지 않는다).

    Args:
        ep_dir: 에피소드 폴더.
        instruction: episode.json 의 instruction 값.
    """
    meta_path = ep_dir / "meta.json"
    write_json_atomic(meta_path, {"instruction": instruction})


# ──────────────────────────────────────────────────────────────────────
# instruction / cot 후처리
# ──────────────────────────────────────────────────────────────────────

def apply_instruction_and_cot(window_dirs: list[Path], instruction: str,
                              skip_annotation: bool) -> tuple[int, int]:
    """윈도우 폴더들의 모든 프레임 json 에 instruction 을 쓰고 cot 를 처리한다.

    Args:
        window_dirs: 이 에피소드가 만든 윈도우 폴더 목록.
        instruction: episode.json 에서 읽은 지시문(에피소드 전 프레임 동일).
        skip_annotation: True 면 "cot" 키를 제거한다(결정 4 참고).

    Returns:
        (instruction 파일 작성 수, cot 키 제거 수).
    """
    n_instr = 0
    n_cot_stripped = 0
    for wd in window_dirs:
        for jf in sorted(wd.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            instr_path = data.get("instruction_file")
            if instr_path:
                Path(instr_path).parent.mkdir(parents=True, exist_ok=True)
                Path(instr_path).write_text(instruction, encoding="utf-8")
                n_instr += 1
            if skip_annotation and "cot" in data:
                del data["cot"]
                jf.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
                n_cot_stripped += 1
    return n_instr, n_cot_stripped


def run_s03_annotation(json_folder: Path) -> None:
    """s03(OpenAI CoT 어노테이션)을 전체 GND_json 폴더에 대해 돌린다.

    s03 자체가 cot 파일이 이미 있는 프레임은 건너뛰는 재개(resume) 로직을
    가지고 있어(첫 프레임 cot 존재 여부로 스킵 판단), 여러 번 실행해도
    안전하다. OpenAI API 키/비용이 필요하다 — --skip-annotation 없이 돌릴 때만
    호출된다.

    Args:
        json_folder: GND_JSON_DIR.
    """
    if not S03_SCRIPT.is_file():
        raise RuntimeError(f"s03 스크립트를 찾을 수 없다: {S03_SCRIPT}")
    subprocess.run(
        [sys.executable, str(S03_SCRIPT),
         "--json_folder", str(json_folder), "--call_gpt", "true"],
        check=True,
    )


# ──────────────────────────────────────────────────────────────────────
# 출력 검수 (전체)
# ──────────────────────────────────────────────────────────────────────

WINDOW_SUFFIX_RE = re.compile(r"_\d+s_\d+s$")


def episode_name_from_window(window_name: str) -> str:
    """윈도우 폴더 이름에서 원본 에피소드 이름을 복원한다.

    Args:
        window_name: 예 "ep0001_L1_blue_0s_20s".

    Returns:
        예 "ep0001_L1_blue".
    """
    return WINDOW_SUFFIX_RE.sub("", window_name)


def get_numeric_key(path: Path) -> float:
    """policy_data.py 의 get_numeric_key 를 그대로 재현한다.

    Args:
        path: json 파일 경로.

    Returns:
        정렬 키.
    """
    stem = path.stem
    numeric_str = stem.split("_", 1)[1]
    return float(numeric_str)


@dataclass
class OutputReport:
    """출력 검수 결과 전체."""

    n_window_dirs: int = 0
    n_json_total: int = 0
    n_training_samples: int = 0
    color_counts: dict[str, int] = field(default_factory=dict)
    side_counts: dict[str, int] = field(default_factory=dict)
    color_side_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    guidance_valid: int = 0
    guidance_invalid: int = 0
    insufficient_future_samples: list[str] = field(default_factory=list)
    future_y_pos: int = 0
    future_y_neg: int = 0
    future_y_zero: int = 0
    delayed_idx_risk_dirs: list[tuple[str, float]] = field(default_factory=list)


def build_output_report(gnd_json_dir: Path, raw_root: Path) -> OutputReport:
    """GND_JSON_DIR 전체를 스캔해 학습 관점 검수 지표를 계산한다.

    _find_samples (every-5th-frame 필터)를 그대로 재현한다 — json 총 개수가
    아니라 이 필터를 통과한 수가 실제 학습 샘플 수다.

    Args:
        gnd_json_dir: GND_JSON_DIR.
        raw_root: raw/ 폴더 (에피소드별 target_color/target_side 조회용).

    Returns:
        검수 결과.
    """
    rep = OutputReport()
    if not gnd_json_dir.is_dir():
        return rep

    window_dirs = sorted(p for p in gnd_json_dir.iterdir() if p.is_dir())
    rep.n_window_dirs = len(window_dirs)

    # 에피소드 메타 캐시 (target_color/target_side).
    ep_meta_cache: dict[str, dict] = {}

    def episode_meta(ep_name: str) -> Optional[dict]:
        if ep_name not in ep_meta_cache:
            ej = raw_root / ep_name / "episode.json"
            if ej.is_file():
                try:
                    ep_meta_cache[ep_name] = json.loads(ej.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    ep_meta_cache[ep_name] = {}
            else:
                ep_meta_cache[ep_name] = {}
        return ep_meta_cache[ep_name] or None

    for color in TARGET_COLORS:
        rep.color_counts[color] = 0
    for side in SIDES:
        rep.side_counts[side] = 0
    for color in TARGET_COLORS:
        for side in SIDES:
            rep.color_side_counts[(color, side)] = 0

    for wd in window_dirs:
        json_files = sorted(wd.glob("*.json"))
        rep.n_json_total += len(json_files)
        if not json_files:
            continue

        # delayed_idx 음수 위험 — 폴더별 최소 timestamp 가 0 근처인지.
        timestamps = []
        for jf in json_files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
                timestamps.append(float(d.get("timestamp", 0.0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if timestamps:
            min_ts = min(timestamps)
            if min_ts > 0.15:
                rep.delayed_idx_risk_dirs.append((wd.name, min_ts))

        # _find_samples 재현: 폴더 안에서 숫자키로 정렬 후 5개마다 1개.
        try:
            sorted_files = sorted(json_files, key=get_numeric_key)
        except (ValueError, IndexError):
            # 파일명이 깨졌으면(입력 검수에서 이미 걸러졌어야 함) 이 폴더는
            # 학습 샘플 계산에서 제외하고 넘어간다 — 조용히 무시하지 않기
            # 위해 delayed_idx_risk_dirs 에 사유를 남긴다.
            rep.delayed_idx_risk_dirs.append((wd.name, float("nan")))
            continue
        selected = sorted_files[0::5]
        rep.n_training_samples += len(selected)

        ep_name = episode_name_from_window(wd.name)
        meta = episode_meta(ep_name)
        color = str(meta.get("target_color")) if meta else None
        side = str(meta.get("target_side")) if meta else None
        if color in TARGET_COLORS:
            rep.color_counts[color] += len(selected)
        if side in SIDES:
            rep.side_counts[side] += len(selected)
        if color in TARGET_COLORS and side in SIDES:
            rep.color_side_counts[(color, side)] += len(selected)

        for jf in selected:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            future = d.get("future", [])
            if len(future) > 89:
                rep.guidance_valid += 1
            else:
                rep.guidance_invalid += 1
                rep.insufficient_future_samples.append(str(jf))

            y = None
            if len(future) > 29 and isinstance(future[29], dict):
                off = future[29].get("offset")
                if off and len(off) >= 2:
                    y = float(off[1])
            elif future and isinstance(future[-1], dict):
                off = future[-1].get("offset")
                if off and len(off) >= 2:
                    y = float(off[1])
            if y is None:
                continue
            if y > 1e-4:
                rep.future_y_pos += 1
            elif y < -1e-4:
                rep.future_y_neg += 1
            else:
                rep.future_y_zero += 1

    return rep


# ──────────────────────────────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────────────────────────────

def write_report(checks: list[EpisodeCheck], processed: list[str],
                 skipped_existing: list[str], skip_annotation: bool,
                 n_instr_written: int, n_cot_stripped: int,
                 out: OutputReport) -> None:
    """PREP_REPORT.md 를 쓴다.

    Args:
        checks: 이번 실행에서 검수한 에피소드 전체(성공/실패 모두).
        processed: 실제로 s01 을 돌린 에피소드 이름 목록.
        skipped_existing: 이미 준비돼 있어 건너뛴(--force 없음) 에피소드.
        skip_annotation: --skip-annotation 여부.
        n_instr_written: 이번 실행에서 쓴 instruction 파일 수.
        n_cot_stripped: 이번 실행에서 제거한 "cot" 키 수.
        out: build_output_report() 결과 (GND_json 전체 기준).
    """
    lines: list[str] = []
    a = lines.append
    a(f"# {DATASET_NAME} 데이터 준비 리포트\n")
    a(f"- raw 루트: `{RAW_ROOT}`")
    a(f"- 출력(GND_json): `{GND_JSON_DIR}`")
    a(f"- 어노테이션(s03/CoT): {'건너뜀 (--skip-annotation)' if skip_annotation else '실행함'}")
    a("")

    a("## 결정 요약 (전체 근거는 스크립트 상단 독스트링)\n")
    a("- GND_json 이름은 하드코딩 요구사항이라 고정 (`_infer_robot_type`).")
    a("- GND_data 디렉토리·s04 는 실행하지 않음 — img 경로가 이미 존재하는 절대경로라 "
      "`_remap_image_path` 가 그대로 쓴다. **raw/ 폴더를 학습 중 이동·삭제하면 깨진다.**")
    if skip_annotation:
        a(f"- cot 처리: **\"cot\" 키를 제거**함 (빈 파일을 까는 대신). "
          f"이번 실행에서 {n_cot_stripped}개 프레임 json 에서 제거. "
          "프롬프트가 'Return the future target waypoints' 문구로 바뀐다.")
    else:
        a("- cot 처리: s03 을 실제로 호출해 OpenAI 로 생성함(비용 발생).")
    a("")

    a("## 입력 검수 — 에피소드별\n")
    a("| 에피소드 | 판정 | 색 | 실패 사유 | 경고 |")
    a("|---|---|---|---|---|")
    n_ok = n_fail = 0
    for c in checks:
        verdict = "OK" if c.ok else "SKIP"
        if c.ok:
            n_ok += 1
        else:
            n_fail += 1
        fails = "; ".join(c.fails) if c.fails else ""
        warns = "; ".join(c.warns) if c.warns else ""
        a(f"| {c.name} | {verdict} | {c.target_color or '?'} | {fails} | {warns} |")
    a("")
    a(f"통과 {n_ok} / 실패(스킵) {n_fail} / 총 {len(checks)}")
    a("")
    if skipped_existing:
        a(f"이미 준비돼 있어 재생성을 건너뛴 에피소드(`--force` 로 재생성): "
          f"{', '.join(skipped_existing)}")
        a("")
    a(f"이번 실행에서 s01 을 실제로 돌린 에피소드: {len(processed)}개, "
      f"instruction 파일 {n_instr_written}개 작성.")
    a("")

    a("## 출력 검수 — 전체 (GND_json 누적 기준, 이번 실행뿐 아니라 전체)\n")
    a(f"- 윈도우 폴더 수: {out.n_window_dirs}")
    a(f"- json 총 개수: {out.n_json_total}")
    a(f"- ★ 실제 학습 샘플 수 (`_find_samples`: 폴더별 정렬 후 5개마다 1개): "
      f"**{out.n_training_samples}**")
    a("")

    a("### 색별 / target_side 별 샘플 분포 (학습 샘플 기준)\n")
    a("설계상 각 색이 left/center/right 를 고르게 거쳐야 한다. "
      "한쪽으로 치우쳐 있으면 배치(layout)를 안 바꾸고 수집했다는 뜻이다.\n")
    a("| 색 \\ side | left | center | right | 합계 |")
    a("|---|---|---|---|---|")
    skew_warns = []
    for color in TARGET_COLORS:
        row = [str(out.color_side_counts[(color, s)]) for s in SIDES]
        total = sum(out.color_side_counts[(color, s)] for s in SIDES)
        a(f"| {color} | " + " | ".join(row) + f" | {total} |")
        nz = [out.color_side_counts[(color, s)] for s in SIDES if out.color_side_counts[(color, s)] > 0]
        if total > 0 and len(nz) < 2:
            skew_warns.append(color)
    a("")
    if skew_warns:
        a(f"**⚠️ 경고: {', '.join(skew_warns)} 색이 side 1종류로 치우쳐 있다 — "
          "배치를 바꿔가며 수집했는지 확인할 것.**")
        a("")

    a("### guidance_waypoint 유효 비율\n")
    total_g = out.guidance_valid + out.guidance_invalid
    ratio = (out.guidance_valid / total_g * 100.0) if total_g else 0.0
    a(f"- 유효(future > 89 스텝, 즉 9초 초과): {out.guidance_valid} "
      f"({ratio:.1f}%) / 무효: {out.guidance_invalid}")
    a("- ※ 이 비율은 **time_delay=0(지연 없음)을 가정한 근사값**이다. 실제 학습은 "
      "매 스텝 0~10s 사이 무작위 지연(delayed frame)을 쓰므로 지연이 클수록 더 "
      "이른 프레임의 future 를 보게 돼 실제 유효율은 이 값과 다를 수 있다.")
    a("- 무효 샘플이 몰리는 구간은 대개 에피소드 끝(도착 직전 감속 구간)이다 — "
      f"목록: `{INSUFFICIENT_FUTURE_LIST}`")
    a("- teacher_answer=drop 모드로 학습하면 assistant 메시지 자체가 프롬프트에서 "
      "빠지므로 -100 센티넬 문자열도 들어가지 않는다 — 그 경우 이 항목은 무관하다.")
    a("")

    a("### future y 성분 부호 분포 (좌/우 균형)\n")
    total_y = out.future_y_pos + out.future_y_neg + out.future_y_zero
    if total_y:
        a(f"- 양(좌, +y): {out.future_y_pos} ({out.future_y_pos / total_y * 100:.1f}%)")
        a(f"- 음(우, -y): {out.future_y_neg} ({out.future_y_neg / total_y * 100:.1f}%)")
        a(f"- 0: {out.future_y_zero}")
    else:
        a("- (샘플 없음)")
    a("")

    a("### delayed_idx 음수 위험 (폴더별 최소 timestamp)\n")
    a("음수면 파이썬 인덱싱이 조용히 리스트 끝을 집어 미래 프레임을 과거로 "
      "착각한다 — 최소 timestamp 가 0 근처(≤0.15s)가 아닌 폴더만 표로 남긴다.\n")
    if out.delayed_idx_risk_dirs:
        a("| 윈도우 폴더 | 최소 timestamp(s) |")
        a("|---|---|")
        for name, ts in out.delayed_idx_risk_dirs:
            ts_str = "파일명 파싱 실패" if ts != ts else f"{ts:.3f}"  # NaN 체크
            a(f"| {name} | {ts_str} |")
    else:
        a("(문제 없음 — 모든 폴더의 최소 timestamp 가 0 근처)")
    a("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    if out.insufficient_future_samples:
        INSUFFICIENT_FUTURE_LIST.parent.mkdir(parents=True, exist_ok=True)
        INSUFFICIENT_FUTURE_LIST.write_text(
            "\n".join(out.insufficient_future_samples), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# CLI / main
# ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """인자를 파싱한다.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        파싱 결과.
    """
    parser = argparse.ArgumentParser(
        description="raw 3색 마커 에피소드를 TIC-VLA 학습용 GND_json 으로 변환",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", default="all",
                        help="처리할 에피소드 수(정수) 또는 'all'. "
                             "raw/ 를 이름순으로 정렬해 앞에서부터 이만큼 고른다")
    parser.add_argument("--skip-annotation", action="store_true",
                        help="s03(OpenAI CoT 생성)을 건너뛴다 — 'cot' 키를 제거한다")
    parser.add_argument("--force", action="store_true",
                        help="이미 GND_json 에 결과가 있는 에피소드도 지우고 다시 만든다")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """엔트리포인트.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        종료 코드 (실패한 에피소드가 있어도 나머지가 처리됐으면 0).
    """
    args = parse_args(argv)

    if "GoToObject_v1" in str(RAW_ROOT):
        raise SystemExit("ERROR: RAW_ROOT 가 GoToObject_v1 을 가리킨다 — 코드가 잘못됐다.")
    if not RAW_ROOT.is_dir():
        raise SystemExit(f"ERROR: raw 폴더가 없다: {RAW_ROOT}")

    ep_dirs = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
    if not ep_dirs:
        raise SystemExit(f"ERROR: {RAW_ROOT} 에 에피소드 폴더가 없다.")

    if args.episodes != "all":
        try:
            n = int(args.episodes)
        except ValueError:
            raise SystemExit(
                f"ERROR: --episodes 는 정수 또는 'all' 이어야 한다: {args.episodes!r}")
        ep_dirs = ep_dirs[:n]

    print(f"[설정] 대상 에피소드: {len(ep_dirs)}개")
    print(f"[설정] --skip-annotation={args.skip_annotation} --force={args.force}")

    # ── 1) 입력 검수 ──
    print("\n[1/4] 입력 검수 중...")
    checks = [check_episode(d) for d in ep_dirs]
    for c in checks:
        tag = "✓" if c.ok else "✗"
        print(f"  {tag} {c.name}" + (f" — {'; '.join(c.fails)}" if c.fails else ""))
        for w in c.warns:
            print(f"      ! {w}")

    passing = [c for c in checks if c.ok]
    if not passing:
        print("\n통과한 에피소드가 없다 — s01 을 돌리지 않는다.")
        out = build_output_report(GND_JSON_DIR, RAW_ROOT)
        write_report(checks, [], [], args.skip_annotation, 0, 0, out)
        print(f"\n리포트: {REPORT_PATH}")
        return 1

    # ── 2) s01 (패치된 사본) ──
    print(f"\n[2/4] s01 실행 중... ({len(passing)}개 에피소드)")
    GND_JSON_DIR.mkdir(parents=True, exist_ok=True)
    processed: list[str] = []
    skipped_existing: list[str] = []
    total_instr = 0
    total_cot_stripped = 0

    with tempfile.TemporaryDirectory(prefix="ticvla_s01_patched_") as td:
        patched_s01 = make_patched_s01(Path(td))
        print(f"  (s01 임시 사본: {patched_s01}, 원본은 건드리지 않음)")

        for chk in passing:
            existing = sorted(GND_JSON_DIR.glob(f"{chk.name}_*s_*s"))
            if existing and not args.force:
                print(f"  - {chk.name}: 이미 준비됨 ({len(existing)}개 윈도우) — 건너뜀")
                skipped_existing.append(chk.name)
                continue
            if existing and args.force:
                for wd in existing:
                    shutil.rmtree(wd)
                print(f"  - {chk.name}: --force, 기존 {len(existing)}개 윈도우 삭제 후 재생성")

            ensure_meta_json(chk.ep_dir, chk.instruction)
            try:
                run_s01(patched_s01, chk.ep_dir, GND_JSON_DIR)
            except subprocess.CalledProcessError as exc:
                print(f"  ✗ {chk.name}: s01 실패 (rc={exc.returncode}) — 스킵")
                chk.ok = False
                chk.fails.append(f"s01 실패 (rc={exc.returncode})")
                continue

            window_dirs = sorted(GND_JSON_DIR.glob(f"{chk.name}_*s_*s"))
            if not window_dirs:
                print(f"  ✗ {chk.name}: s01 이 윈도우를 하나도 만들지 않음 (에피소드가 "
                      "너무 짧을 수 있다: 최소 150프레임/15초 필요) — 스킵")
                chk.ok = False
                chk.fails.append("s01 산출 윈도우 0개 (에피소드 길이 부족)")
                continue

            n_instr, n_cot = apply_instruction_and_cot(
                window_dirs, chk.instruction, args.skip_annotation)
            total_instr += n_instr
            total_cot_stripped += n_cot
            processed.append(chk.name)
            print(f"  ✓ {chk.name}: 윈도우 {len(window_dirs)}개, "
                  f"instruction {n_instr}개 작성")

    # ── 3) 어노테이션(옵션) ──
    if not args.skip_annotation:
        print("\n[3/4] s03(OpenAI CoT) 실행 중... (비용 발생)")
        run_s03_annotation(GND_JSON_DIR)
    else:
        print("\n[3/4] --skip-annotation — s03 건너뜀 (cot 키 이미 제거함)")

    # ── 4) 출력 검수 + 리포트 ──
    print("\n[4/4] 출력 검수 중...")
    out = build_output_report(GND_JSON_DIR, RAW_ROOT)
    write_report(checks, processed, skipped_existing, args.skip_annotation,
                total_instr, total_cot_stripped, out)

    print(f"\n윈도우 폴더 {out.n_window_dirs}개, json {out.n_json_total}개, "
          f"실제 학습 샘플 {out.n_training_samples}개")
    print(f"리포트: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
