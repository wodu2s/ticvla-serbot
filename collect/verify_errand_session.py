#!/usr/bin/env python3
"""verify_errand_session.py — 심부름 시연 세션 즉시 검수.

joystick_errand.py 가 만든 세션 폴더 하나를 받아, 보고에 쓸 만한 품질인지
구간별로 검사하고 이벤트 시점 프레임을 preview/ 에 뽑아 둔다.

    python3 collect/verify_errand_session.py <session_dir>

기대하는 입력 구조
------------------
    session_YYMMDD_HHMMSS_<scenario>/
      session.json
      seg00_<object>/{rgb/rgb_00000.jpg…, trajectory.csv, episode.json}
      seg01_<object>/…

검사 항목
---------
구간별
  1. rgb 파일 수 == trajectory.csv 행 수 (+ 번호 0..N-1 연속)
  2. 프레임 간격 중앙값이 100ms ± 20ms  → 10Hz 가 실제로 지켜졌는가
  3. 총 이동거리 / 소요시간 / 평균속도  → 0.10 m/s 근처인가
  4. 전진 단조성 — 후진 환각 패턴 검출
     odom x 는 월드 좌표라 회전하면 정상 주행에도 줄어든다. 그래서
     '월드 x 감소'와 '동체 전방 변위 감소'를 따로 센다. 후진 판정은
     헤딩에 투영한 동체 전방 변위(dx·cos yaw + dy·sin yaw)로 한다.
  5. FLU 부호 — 좌회전(yaw 증가) 구간에서 좌측(+y) 으로 움직였는가
  6. 쿼터니언 정규화 / 정지 상태 반복 (odom 환각 검출)

세션 전체
  - session.json 이벤트를 구간·프레임에 매핑 (기록된 frame 과 CSV 시각 재계산 비교)
  - obstacle_detected / replan / new_order / emergency_stop 시점 프레임을
    preview/ 에 캡션 붙여 저장
  - 구간별 첫 / 중간 / 마지막 프레임도 같은 폴더에 저장

마지막 한 줄: [OK] 또는 [재촬영 권장: 사유]
종료 코드 0 = OK, 1 = 재촬영 권장, 2 = 세션을 읽을 수 없음
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────
# 기준값
# ──────────────────────────────────────────────────────────────────────

#: 목표 프레임 간격과 허용 폭 (ms).
TARGET_INTERVAL_MS = 100.0
INTERVAL_TOL_MS = 20.0

#: 목표 평균속도(m/s). joystick_gotoobject 의 MAX_LINEAR_X_MPS 기본값.
TARGET_SPEED = 0.10
#: 경고 밴드 / 실패 밴드.
SPEED_WARN_RATIO = 0.3
SPEED_FAIL_RATIO = 0.6

#: 한 스텝 동체 전방 변위가 이보다 더 음수면 후진으로 센다(m).
#: 10Hz·0.10m/s 면 정상 스텝이 0.010m 다. 2mm 로 두면 0.02m/s 짜리 느린
#: 후진도 잡히고, 정지 중 엔코더 잡음은 누적 기준에서 걸러진다.
BACKWARD_STEP_TOL = 0.002

#: 월드 x 단조성은 정보용이라 양자화 수준만 넘으면 감소로 센다(m).
WORLD_X_STEP_TOL = 0.0005
#: 후진 누적거리가 총거리의 이 비율을 넘으면 실패.
BACKWARD_FAIL_RATIO = 0.02
#: 후진 누적거리 절대 하한(m). 이보다 작으면 비율과 무관하게 통과.
BACKWARD_FAIL_MIN_M = 0.05

#: 회전 판정 임계 (rad/step). 0.005rad ≈ 0.29도.
YAW_STEP_TOL = 0.005
#: 부호 판정을 내리려면 좌회전 스텝이 최소 이만큼 있어야 한다.
MIN_TURN_STEPS = 5

#: 구간이 이만큼도 움직이지 않았으면 실패.
MIN_DISTANCE_M = 0.20
#: 구간 최소 길이(초).
MIN_DURATION_S = 2.0

#: preview 를 뽑을 이벤트 종류.
PREVIEW_EVENT_TYPES = (
    "obstacle_detected", "replan", "new_order", "emergency_stop")

#: s01 이 요구하는 컬럼. joystick_errand.TRAJECTORY_HEADER 와 같아야 한다.
EXPECTED_HEADER = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]

IMAGE_PATTERN = "rgb_%05d.jpg"


# ──────────────────────────────────────────────────────────────────────
# 결과 구조
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SegmentReport:
    """구간 하나의 검사 결과."""

    index: int
    name: str
    object_name: str
    instruction: str = ""
    n_images: int = 0
    n_rows: int = 0
    interval_med_ms: float = float("nan")
    interval_min_ms: float = float("nan")
    interval_max_ms: float = float("nan")
    distance_m: float = 0.0
    duration_s: float = 0.0
    speed_mps: float = 0.0
    net_disp_m: float = 0.0
    world_x_back_m: float = 0.0
    world_x_back_steps: int = 0
    body_back_m: float = 0.0
    body_back_steps: int = 0
    body_back_run: int = 0
    yaw_total_deg: float = 0.0
    left_steps: int = 0
    left_lateral_m: float = 0.0
    left_world_dy_m: float = 0.0
    right_steps: int = 0
    right_lateral_m: float = 0.0
    turn_verdict: str = "n/a"
    stall_ticks: int = 0
    fails: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)


@dataclass
class EventRow:
    """이벤트 하나의 매핑 결과."""

    order: int
    t: float
    kind: str
    seg_index: Optional[int]
    seg_name: str
    recorded_frame: Optional[int]
    csv_frame: Optional[int]
    csv_time: Optional[float]
    seg_t: Optional[float]
    image: Optional[Path]
    preview: Optional[Path]
    note: str = ""


# ──────────────────────────────────────────────────────────────────────
# 읽기
# ──────────────────────────────────────────────────────────────────────

def load_session(session_dir: Path) -> dict:
    """session.json 을 읽는다.

    Args:
        session_dir: 세션 폴더.

    Raises:
        SystemExit: 폴더나 json 이 없거나 깨졌으면 종료한다.

    Returns:
        session.json 내용.
    """
    if not session_dir.is_dir():
        raise SystemExit(f"ERROR: 세션 폴더가 없다: {session_dir}")
    path = session_dir / "session.json"
    if not path.is_file():
        raise SystemExit(
            f"ERROR: session.json 이 없다: {path}\n"
            "  joystick_errand.py 가 만든 세션 폴더를 지정할 것.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: session.json 파싱 실패: {exc}") from exc


def find_segment_dirs(session_dir: Path) -> list[Path]:
    """segNN_<object> 폴더를 번호순으로 찾는다.

    Args:
        session_dir: 세션 폴더.

    Returns:
        구간 폴더 목록.
    """
    dirs = [p for p in session_dir.iterdir()
            if p.is_dir() and p.name.startswith("seg") and "_" in p.name]
    return sorted(dirs, key=lambda p: p.name)


def read_trajectory(path: Path) -> tuple[Optional[dict], list[str]]:
    """trajectory.csv 를 numpy 배열 dict 로 읽는다.

    Args:
        path: CSV 경로.

    Returns:
        (컬럼별 배열 dict, 오류 목록). 읽을 수 없으면 (None, 오류).
    """
    errors: list[str] = []
    if not path.is_file():
        return None, [f"trajectory.csv 없음: {path.name}"]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, ["trajectory.csv 가 비었다"]
        rows = [r for r in reader if r]

    if header != EXPECTED_HEADER:
        errors.append(f"헤더 불일치: {','.join(header)}")
        missing = [c for c in EXPECTED_HEADER if c not in header]
        if missing:
            return None, errors + [f"필수 컬럼 없음: {missing}"]

    idx = {c: header.index(c) for c in EXPECTED_HEADER}
    data: dict[str, np.ndarray] = {}
    try:
        for col in EXPECTED_HEADER:
            j = idx[col]
            data[col] = np.array([float(r[j]) for r in rows], dtype=np.float64)
    except (ValueError, IndexError) as exc:
        return None, errors + [f"숫자 파싱 실패: {exc}"]
    return data, errors


def yaw_from_quat(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray,
                  qw: np.ndarray) -> np.ndarray:
    """쿼터니언에서 yaw(rad)를 뽑아 unwrap 한다.

    Args:
        qx, qy, qz, qw: 쿼터니언 성분 배열.

    Returns:
        unwrap 된 yaw 배열.
    """
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny, cosy))


# ──────────────────────────────────────────────────────────────────────
# 구간 검사
# ──────────────────────────────────────────────────────────────────────

def analyze_segment(seg_dir: Path, index: int, target_speed: float,
                    backward_tol: float = BACKWARD_STEP_TOL) -> SegmentReport:
    """구간 하나를 검사한다.

    Args:
        seg_dir: 구간 폴더.
        index: 구간 번호.
        target_speed: 기대 평균속도(m/s).
        backward_tol: 후진으로 셀 한 스텝 변위 임계(m).

    Returns:
        검사 결과.
    """
    obj = seg_dir.name.split("_", 1)[1] if "_" in seg_dir.name else "?"
    rep = SegmentReport(index=index, name=seg_dir.name, object_name=obj)

    ep_path = seg_dir / "episode.json"
    if ep_path.is_file():
        try:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
            rep.instruction = str(ep.get("instruction", ""))
            rep.stall_ticks = int(ep.get("stall_ticks", 0) or 0)
            if str(ep.get("status", "")) not in ("done", ""):
                rep.fails.append(f"episode.json status={ep.get('status')}")
        except json.JSONDecodeError as exc:
            rep.warns.append(f"episode.json 파싱 실패: {exc}")
    else:
        rep.warns.append("episode.json 없음")

    # ── 1. 프레임 수 vs 행 수 ──
    rgb_dir = seg_dir / "rgb"
    images = sorted(rgb_dir.glob("rgb_*.jpg")) if rgb_dir.is_dir() else []
    rep.n_images = len(images)
    if not rgb_dir.is_dir():
        rep.fails.append("rgb/ 폴더 없음")

    data, errors = read_trajectory(seg_dir / "trajectory.csv")
    for err in errors:
        rep.fails.append(err)
    if data is None:
        return rep

    t = data["time"]
    rep.n_rows = int(t.size)
    if rep.n_images != rep.n_rows:
        rep.fails.append(f"rgb {rep.n_images}장 ≠ csv {rep.n_rows}행")
    if rep.n_rows == 0:
        rep.fails.append("행이 0개")
        return rep
    expected_names = [IMAGE_PATTERN % i for i in range(rep.n_images)]
    if [p.name for p in images] != expected_names:
        rep.fails.append("rgb 파일 번호가 0..N-1 연속이 아니다")

    # ── 2. 프레임 간격 ──
    if rep.n_rows >= 2:
        dt_ms = np.diff(t) * 1000.0
        rep.interval_med_ms = float(np.median(dt_ms))
        rep.interval_min_ms = float(dt_ms.min())
        rep.interval_max_ms = float(dt_ms.max())
        off = abs(rep.interval_med_ms - TARGET_INTERVAL_MS)
        if off > INTERVAL_TOL_MS:
            rep.fails.append(
                f"프레임 간격 중앙값 {rep.interval_med_ms:.1f}ms "
                f"(허용 {TARGET_INTERVAL_MS:.0f}±{INTERVAL_TOL_MS:.0f}ms)")
        if rep.interval_max_ms > TARGET_INTERVAL_MS * 3.0:
            rep.warns.append(
                f"최대 간격 {rep.interval_max_ms:.0f}ms — 프레임 누락 구간 있음")
    else:
        rep.fails.append("행이 1개 — 간격을 볼 수 없다")
        return rep

    # ── 3. 거리 / 시간 / 속도 ──
    x, y = data["x"], data["y"]
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    rep.distance_m = float(step.sum())
    rep.duration_s = float(t[-1] - t[0])
    rep.net_disp_m = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    rep.speed_mps = rep.distance_m / rep.duration_s if rep.duration_s > 0 else 0.0

    if rep.distance_m < MIN_DISTANCE_M:
        rep.fails.append(
            f"총 이동거리 {rep.distance_m:.3f}m < {MIN_DISTANCE_M}m — 사실상 정지")
    if rep.duration_s < MIN_DURATION_S:
        rep.fails.append(f"소요시간 {rep.duration_s:.1f}s < {MIN_DURATION_S}s")
    if rep.speed_mps > 0:
        ratio = abs(rep.speed_mps - target_speed) / target_speed
        if ratio > SPEED_FAIL_RATIO:
            rep.fails.append(
                f"평균속도 {rep.speed_mps:.3f}m/s — 목표 {target_speed:.2f} 대비 "
                f"{ratio * 100:.0f}% 벗어남")
        elif ratio > SPEED_WARN_RATIO:
            rep.warns.append(
                f"평균속도 {rep.speed_mps:.3f}m/s (목표 {target_speed:.2f})")

    # ── 4. 전진 단조성 ──
    yaw = yaw_from_quat(data["qx"], data["qy"], data["qz"], data["qw"])
    rep.yaw_total_deg = float(np.degrees(yaw[-1] - yaw[0]))

    qnorm = np.sqrt(data["qx"] ** 2 + data["qy"] ** 2
                    + data["qz"] ** 2 + data["qw"] ** 2)
    if np.abs(qnorm - 1.0).max() > 0.05:
        rep.fails.append(
            f"쿼터니언이 정규화되지 않았다 (|q| 최대 오차 "
            f"{np.abs(qnorm - 1.0).max():.3f}) — odom 이 의심스럽다")

    back_x = dx < -WORLD_X_STEP_TOL
    rep.world_x_back_steps = int(back_x.sum())
    rep.world_x_back_m = float(-dx[back_x].sum()) if rep.world_x_back_steps else 0.0

    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    fwd = dx * np.cos(yaw_mid) + dy * np.sin(yaw_mid)
    lat = -dx * np.sin(yaw_mid) + dy * np.cos(yaw_mid)

    back_b = fwd < -backward_tol
    rep.body_back_steps = int(back_b.sum())
    rep.body_back_m = float(-fwd[back_b].sum()) if rep.body_back_steps else 0.0
    rep.body_back_run = longest_run(back_b)

    limit = max(BACKWARD_FAIL_MIN_M, rep.distance_m * BACKWARD_FAIL_RATIO)
    if rep.body_back_m > limit:
        rep.fails.append(
            f"후진 누적 {rep.body_back_m:.3f}m ({rep.body_back_steps}스텝, "
            f"연속 최대 {rep.body_back_run}) > 허용 {limit:.3f}m")
    elif rep.body_back_steps:
        rep.warns.append(
            f"후진 스텝 {rep.body_back_steps}개, 누적 {rep.body_back_m:.3f}m "
            f"(허용 {limit:.3f}m 이내)")
    if rep.world_x_back_steps and not rep.body_back_steps:
        rep.warns.append(
            f"월드 x 감소 {rep.world_x_back_steps}스텝 "
            f"({rep.world_x_back_m:.3f}m) — yaw {rep.yaw_total_deg:+.0f}도 "
            "회전 때문이며 동체 기준 후진은 없음")

    # ── 5. FLU 부호 (좌회전에서 +y) ──
    # 전진 중인 스텝만 본다. 후진하며 좌회전하면 동체 기준으로는 오른쪽으로
    # 밀리는 것이 정상이라, 섞으면 부호 판정이 뒤집혀 보인다.
    fwd_step = fwd > backward_tol
    dyaw = np.diff(yaw)
    left = (dyaw > YAW_STEP_TOL) & fwd_step
    right = (dyaw < -YAW_STEP_TOL) & fwd_step
    rep.left_steps = int(left.sum())
    rep.right_steps = int(right.sum())
    rep.left_lateral_m = float(lat[left].sum()) if rep.left_steps else 0.0
    rep.left_world_dy_m = float(dy[left].sum()) if rep.left_steps else 0.0
    rep.right_lateral_m = float(lat[right].sum()) if rep.right_steps else 0.0

    if rep.left_steps < MIN_TURN_STEPS:
        rep.turn_verdict = "n/a"
        rep.warns.append(
            f"전진 중 좌회전 스텝 {rep.left_steps}개 — FLU 부호를 판정할 수 없다")
    elif rep.left_lateral_m > 0:
        rep.turn_verdict = "OK"
    else:
        rep.turn_verdict = "역부호"
        rep.fails.append(
            f"좌회전 {rep.left_steps}스텝인데 동체 좌측 변위가 "
            f"{rep.left_lateral_m:+.3f}m — FLU 부호가 뒤집혔을 수 있다")

    if rep.stall_ticks:
        rep.warns.append(
            f"카메라 미수신 tick {rep.stall_ticks}회 — s01 이 시간축을 "
            "행 인덱스로 재생성하므로 그만큼 압축된다")
    return rep


def longest_run(mask: np.ndarray) -> int:
    """True 가 연속으로 가장 길게 이어진 개수.

    Args:
        mask: bool 배열.

    Returns:
        최대 연속 길이.
    """
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ──────────────────────────────────────────────────────────────────────
# 이벤트 매핑
# ──────────────────────────────────────────────────────────────────────

def map_events(session: dict, seg_dirs: list[Path]) -> list[EventRow]:
    """session.json 이벤트를 구간·프레임에 매핑한다.

    이벤트에 적힌 frame 을 그대로 믿지 않고 seg_t 로 CSV 시각을 다시 찾아
    비교한다. 둘이 어긋나면 그 자체가 기록 이상 신호다.

    Args:
        session: session.json 내용.
        seg_dirs: 구간 폴더 목록.

    Returns:
        매핑 결과 목록.
    """
    times: dict[int, np.ndarray] = {}
    for i, seg_dir in enumerate(seg_dirs):
        data, _ = read_trajectory(seg_dir / "trajectory.csv")
        if data is not None and data["time"].size:
            times[i] = data["time"]

    rows: list[EventRow] = []
    for order, ev in enumerate(session.get("events", [])):
        seg_index = ev.get("segment")
        seg_index = int(seg_index) if isinstance(seg_index, (int, float)) else None
        seg_name = ""
        if seg_index is not None and 0 <= seg_index < len(seg_dirs):
            seg_name = seg_dirs[seg_index].name
        seg_t = ev.get("seg_t")
        seg_t = float(seg_t) if isinstance(seg_t, (int, float)) else None
        recorded = ev.get("frame")
        recorded = int(recorded) if isinstance(recorded, (int, float)) else None

        csv_frame: Optional[int] = None
        csv_time: Optional[float] = None
        note = ""
        arr = times.get(seg_index) if seg_index is not None else None
        if arr is not None and seg_t is not None:
            csv_frame = int(np.argmin(np.abs(arr - seg_t)))
            csv_time = float(arr[csv_frame])
            if recorded is not None and abs(csv_frame - recorded) > 1:
                note = f"frame 불일치(기록 {recorded})"
        elif seg_index is None:
            note = "구간 밖 이벤트"
        elif arr is None:
            note = "구간 CSV 없음"

        image: Optional[Path] = None
        if seg_index is not None and 0 <= seg_index < len(seg_dirs):
            frame_idx = csv_frame if csv_frame is not None else recorded
            if frame_idx is not None:
                cand = seg_dirs[seg_index] / "rgb" / (IMAGE_PATTERN % frame_idx)
                if cand.is_file():
                    image = cand
                elif not note:
                    note = "프레임 이미지 없음"

        rows.append(EventRow(
            order=order, t=float(ev.get("t", 0.0)), kind=str(ev.get("type", "?")),
            seg_index=seg_index, seg_name=seg_name, recorded_frame=recorded,
            csv_frame=csv_frame, csv_time=csv_time, seg_t=seg_t,
            image=image, preview=None, note=note))
    return rows


# ──────────────────────────────────────────────────────────────────────
# preview 저장
# ──────────────────────────────────────────────────────────────────────

def load_font() -> Any:
    """캡션용 폰트. DejaVuSans 가 있으면 쓰고 없으면 기본 폰트.

    Returns:
        PIL ImageFont.
    """
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, 14)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def save_preview(src: Path, dst: Path, caption: str, annotate: bool,
                 font: Any) -> bool:
    """프레임 하나를 preview 로 저장한다.

    Args:
        src: 원본 프레임 경로.
        dst: 저장 경로.
        caption: 아래쪽 띠에 넣을 ASCII 캡션.
        annotate: 캡션을 넣을지 여부.
        font: 캡션 폰트.

    Returns:
        저장 성공 여부.
    """
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if not annotate:
                im.save(dst, format="JPEG", quality=95)
                return True
            bar = 22
            canvas = Image.new("RGB", (im.width, im.height + bar), (0, 0, 0))
            canvas.paste(im, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, im.height + 4), caption, fill=(255, 255, 255), font=font)
            canvas.save(dst, format="JPEG", quality=95)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [preview] 저장 실패 {dst.name}: {exc!r}")
        return False


def save_previews(session_dir: Path, seg_dirs: list[Path],
                  reports: list[SegmentReport], events: list[EventRow],
                  annotate: bool) -> tuple[Path, int]:
    """이벤트 시점과 구간 첫/중간/마지막 프레임을 preview/ 에 저장한다.

    Args:
        session_dir: 세션 폴더.
        seg_dirs: 구간 폴더 목록.
        reports: 구간 검사 결과.
        events: 이벤트 매핑 결과.
        annotate: 캡션을 넣을지 여부.

    Returns:
        (preview 폴더, 저장한 파일 수).
    """
    out = session_dir / "preview"
    out.mkdir(exist_ok=True)
    font = load_font()
    saved = 0

    for ev in events:
        if ev.kind not in PREVIEW_EVENT_TYPES or ev.image is None:
            continue
        frame = ev.csv_frame if ev.csv_frame is not None else ev.recorded_frame
        dst = out / (f"event{ev.order:02d}_{ev.kind}_seg{ev.seg_index:02d}"
                     f"_f{frame:05d}.jpg")
        caption = (f"{ev.kind}  {ev.seg_name}  t={ev.t:.1f}s  frame={frame}")
        if save_preview(ev.image, dst, caption, annotate, font):
            ev.preview = dst
            saved += 1

    for rep, seg_dir in zip(reports, seg_dirs):
        if rep.n_images == 0:
            continue
        picks = {
            "first": 0,
            "mid": rep.n_images // 2,
            "last": rep.n_images - 1,
        }
        for label, idx in picks.items():
            src = seg_dir / "rgb" / (IMAGE_PATTERN % idx)
            if not src.is_file():
                continue
            dst = out / f"{rep.name}_{label}_f{idx:05d}.jpg"
            caption = (f"{rep.name} {label}  frame={idx}/{rep.n_images - 1}  "
                       f"{rep.instruction}")
            if save_preview(src, dst, caption, annotate, font):
                saved += 1
    return out, saved


# ──────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────

def fmt(value: float, spec: str) -> str:
    """NaN 을 '-' 로 보여주는 포맷.

    Args:
        value: 값.
        spec: 포맷 스펙 (예 ".1f").

    Returns:
        문자열.
    """
    if value != value:  # NaN
        return "-"
    return format(value, spec)


def print_session_head(session_dir: Path, session: dict) -> list[str]:
    """세션 메타를 출력하고 세션 수준 실패 사유를 돌려준다.

    Args:
        session_dir: 세션 폴더.
        session: session.json 내용.

    Returns:
        실패 사유 목록.
    """
    fails: list[str] = []
    print("=" * 92)
    print(f"  세션 검수: {session_dir.name}")
    print("=" * 92)
    print(f"  경로     : {session_dir}")
    print(f"  시나리오 : {session.get('scenario', '?')}   "
          f"dataset={session.get('dataset', '?')}   hz={session.get('hz', '?')}")
    status = str(session.get("status", "?"))
    print(f"  상태     : {status}")
    plan = session.get("segment_plan") or []
    if plan:
        print("  구간계획 : " + " → ".join(
            f"seg{p.get('index', i):02d}:{p.get('object', '?')}"
            for i, p in enumerate(plan)))
    if status.startswith("aborted"):
        fails.append(f"세션이 중단됨(status={status})")
    elif status == "completed_partial":
        fails.append("계획한 구간을 다 못 돌았다(completed_partial)")
    elif status == "recording":
        fails.append("session.json 이 recording 상태 — 수집이 정상 종료되지 않았다")
    for warn in session.get("warnings") or []:
        print(f"  ! 세션 경고: {warn}")
    return fails


def print_segment_table(reports: list[SegmentReport], target_speed: float) -> None:
    """구간 검사 결과를 표로 출력한다.

    Args:
        reports: 구간 검사 결과.
        target_speed: 기대 평균속도.
    """
    print("")
    print("-" * 92)
    print("  구간 검사   (dt_med=프레임 간격 중앙값, back=동체기준 후진, "
          "turn=좌회전 FLU 부호)")
    print("-" * 92)
    print(f"  {'seg':<4}{'object':<11}{'imgs':>6}{'rows':>6}{'dt_med':>9}"
          f"{'dist':>8}{'dur':>8}{'speed':>9}{'back':>7}{'turn':>8}  판정")
    for r in reports:
        verdict = "FAIL" if r.fails else ("warn" if r.warns else "OK")
        print(f"  {r.index:<4}{r.object_name:<11}{r.n_images:>6}{r.n_rows:>6}"
              f"{fmt(r.interval_med_ms, '.1f') + 'ms':>9}"
              f"{r.distance_m:>7.2f}m{r.duration_s:>7.1f}s"
              f"{r.speed_mps:>8.3f} {r.body_back_steps:>6}{r.turn_verdict:>8}"
              f"  {verdict}")
    print(f"  (기대: dt_med {TARGET_INTERVAL_MS:.0f}±{INTERVAL_TOL_MS:.0f}ms, "
          f"speed ≈ {target_speed:.2f} m/s, back=0)")

    for r in reports:
        if not (r.fails or r.warns):
            continue
        print("")
        print(f"  [seg{r.index:02d}_{r.object_name}] {r.instruction}")
        print(f"    간격 min/med/max = {fmt(r.interval_min_ms, '.0f')}/"
              f"{fmt(r.interval_med_ms, '.1f')}/{fmt(r.interval_max_ms, '.0f')} ms"
              f"   경로 {r.distance_m:.2f}m / 직선 {r.net_disp_m:.2f}m"
              f"   yaw {r.yaw_total_deg:+.0f}도")
        print(f"    후진: 동체 {r.body_back_steps}스텝 {r.body_back_m:.3f}m "
              f"(연속 최대 {r.body_back_run}) / 월드 x 감소 "
              f"{r.world_x_back_steps}스텝 {r.world_x_back_m:.3f}m")
        print(f"    회전: 좌 {r.left_steps}스텝 동체+y {r.left_lateral_m:+.3f}m "
              f"(월드 dy {r.left_world_dy_m:+.3f}m) / "
              f"우 {r.right_steps}스텝 동체+y {r.right_lateral_m:+.3f}m")
        for msg in r.fails:
            print(f"    FAIL {msg}")
        for msg in r.warns:
            print(f"    warn {msg}")


def print_event_table(events: list[EventRow]) -> list[str]:
    """이벤트 매핑을 표로 출력한다.

    Args:
        events: 이벤트 매핑 결과.

    Returns:
        경고 사유 목록.
    """
    warns: list[str] = []
    print("")
    print("-" * 92)
    print("  이벤트 매핑   (csv_f = seg_t 로 다시 찾은 프레임, rec_f = 수집 시 기록값)")
    print("-" * 92)
    if not events:
        print("  이벤트 없음")
        return warns
    print(f"  {'t(s)':>8}  {'type':<20}{'seg':<12}{'rec_f':>6}{'csv_f':>7}"
          f"{'csv_t':>8}  프레임 / 비고")
    for ev in events:
        seg = ev.seg_name or "-"
        rec = "-" if ev.recorded_frame is None else str(ev.recorded_frame)
        csvf = "-" if ev.csv_frame is None else str(ev.csv_frame)
        csvt = "-" if ev.csv_time is None else f"{ev.csv_time:.2f}"
        tail = ev.image.name if ev.image else "-"
        if ev.preview is not None:
            tail += f"  → preview/{ev.preview.name}"
        if ev.note:
            tail += f"  ({ev.note})"
        print(f"  {ev.t:>8.2f}  {ev.kind:<20}{seg:<12}{rec:>6}{csvf:>7}"
              f"{csvt:>8}  {tail}")
        if ev.note.startswith("frame 불일치"):
            warns.append(f"{ev.kind}@{ev.t:.1f}s {ev.note}")
        if ev.kind in PREVIEW_EVENT_TYPES and ev.image is None:
            warns.append(f"{ev.kind}@{ev.t:.1f}s 프레임 이미지를 찾지 못했다")

    marks = [e for e in events if e.kind in PREVIEW_EVENT_TYPES]
    print(f"  수기/긴급 이벤트 {len(marks)}건 " + (
        ", ".join(f"{k}={sum(1 for e in marks if e.kind == k)}"
                  for k in PREVIEW_EVENT_TYPES
                  if any(e.kind == k for e in marks)) or "없음"))
    return warns


def main(argv: Optional[list[str]] = None) -> int:
    """엔트리포인트.

    Args:
        argv: 인자 목록. None 이면 sys.argv.

    Returns:
        0 = OK, 1 = 재촬영 권장, 2 = 세션을 읽을 수 없음.
    """
    parser = argparse.ArgumentParser(
        description="심부름 시연 세션 즉시 검수 (joystick_errand.py 출력물)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("session_dir", help="session_YYMMDD_HHMMSS_<scenario> 폴더")
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED,
                        help="기대 평균속도(m/s)")
    parser.add_argument("--backward-step-tol", type=float,
                        default=BACKWARD_STEP_TOL,
                        help="후진으로 셀 한 스텝 동체 전방 변위 임계(m)")
    parser.add_argument("--no-preview", action="store_true",
                        help="preview/ 이미지 저장을 건너뛴다")
    parser.add_argument("--no-annotate", action="store_true",
                        help="preview 이미지에 캡션 띠를 넣지 않는다")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir).expanduser().resolve()
    session = load_session(session_dir)

    seg_dirs = find_segment_dirs(session_dir)
    fails = print_session_head(session_dir, session)
    if not seg_dirs:
        print("")
        print("  구간 폴더(segNN_<object>)가 없다.")
        print("")
        print(f"[재촬영 권장: 구간 폴더가 없다 — {session_dir.name}]")
        return 1

    reports = [analyze_segment(d, i, args.target_speed, args.backward_step_tol)
               for i, d in enumerate(seg_dirs)]
    print_segment_table(reports, args.target_speed)

    events = map_events(session, seg_dirs)
    if not args.no_preview:
        out, saved = save_previews(
            session_dir, seg_dirs, reports, events, not args.no_annotate)
        event_warns = print_event_table(events)
        print("")
        print(f"  preview: {saved}장 → {out}")
    else:
        event_warns = print_event_table(events)
        print("")
        print("  preview: 저장 생략(--no-preview)")

    # ── 판정 ──
    warns = list(event_warns)
    for r in reports:
        for msg in r.fails:
            fails.append(f"seg{r.index:02d}_{r.object_name}: {msg}")
        for msg in r.warns:
            warns.append(f"seg{r.index:02d}_{r.object_name}: {msg}")

    planned = len(session.get("segment_plan") or [])
    if planned and len(seg_dirs) != planned:
        fails.append(f"구간 폴더 {len(seg_dirs)}개 ≠ 계획 {planned}개")

    print("")
    print("=" * 92)
    if fails:
        print(f"[재촬영 권장: {fails[0]}]")
        if len(fails) > 1:
            print(f"  그 외 실패 {len(fails) - 1}건:")
            for msg in fails[1:]:
                print(f"    - {msg}")
        if warns:
            print(f"  경고 {len(warns)}건:")
            for msg in warns:
                print(f"    - {msg}")
        print("=" * 92)
        return 1

    if warns:
        print(f"[OK]  경고 {len(warns)}건 — 보고에는 쓸 수 있다")
        for msg in warns:
            print(f"    - {msg}")
    else:
        print("[OK]")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            sys.exit(2)
        raise
