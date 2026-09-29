#!/usr/bin/env python3
"""HSV 기반 3색 객체 탐지 + 시각 서보(visual servo) 명령 계산.

배경
----
`ticvla_bridge.py` 가 쓰는 VLM+action expert 경로는 언어 조건화가 매우 얇다
(`ticvla/models/ticvla.py`의 `_process_kv_cache` 가 VLM 마지막 레이어 value
벡터 하나로만 action expert 를 조건화한다). 학습 CoT 라벨(`data/s03_batch_annotate.py`)도
"critical object 하나"만 서술하도록 설계돼 있고, 실제 학습 데이터(`data/GoToObject_v1`)도
전부 회색 통(graybin) 뿐이라 색을 구분해 타깃을 고르는 능력 자체가 없다.
그래서 "3개 중 지정색으로 가라" 는 지금 체크포인트로는 프롬프트만 바꿔서 될 일이
아니다 (실측: instruction 을 바꿔도 <think> 가 색 이름 하나만 언급하고 3개를
구분하지 못함).

이 모듈은 그 자리를 고전 컴퓨터비전으로 채운다 — VLM 을 거치지 않고 카메라
프레임에서 직접 3색 블롭을 찾아 조향한다. 순수 함수 위주로 짜서 ROS/torch 없이
단위 테스트 및 오프라인 캘리브레이션이 가능하다.

캘리브레이션 워크플로우 (헤드리스 Jetson 기준, GUI 불필요)
--------------------------------------------------------
1) 로봇을 세워두고 3개 객체를 카메라 앞에 둔 채 정지 프레임을 수집한다::

       python3 scripts/color_target.py capture \\
           --out logs/color_calib --seconds 10 --sensor-id 0

2) 실시간으로 탐지가 되는지 즉석에서 본다(ROS 불필요, 명령 발행 없음)::

       python3 scripts/color_target.py live --debug-dir logs/color_calib_live

   `logs/color_calib_live/*.jpg` 를 scp 로 내려받아 박스가 각 물체를 제대로
   잡는지 눈으로 확인한다. 잘못 잡으면 3)으로 임계값을 조정한다.

3) 저장된 프레임에 대해 임계값을 바꿔가며 오프라인으로 반복 검증한다::

       python3 scripts/color_target.py test --dir logs/color_calib \\
           --annotated-dir logs/color_calib_annotated \\
           --json config/color_ranges.json

   `--json` 파일이 없으면 `DEFAULT_RANGES` 로 시작해서, 잘 맞을 때까지 해당
   JSON 을 직접 수정하며 재실행한다. `probe` 로 특정 픽셀의 HSV 값을 찍어보면
   임계값을 잡기 쉽다::

       python3 scripts/color_target.py probe --image logs/color_calib/frame_0001.jpg \\
           --x 640 --y 360 --box 15

4) 값이 굳어지면 `ticvla_bridge.py --color-nav --color-ranges-json config/color_ranges.json`
   로 실제 주행에 연결한다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ======================================================================
# 색상 임계값
# ======================================================================
@dataclass
class ColorRange:
    """단일 색상의 HSV 임계값. OpenCV HSV 범위: H 0-179, S/V 0-255.

    Attributes:
        name: 색 이름 (내부 키. 'red' / 'green' / 'blue').
        lower1, upper1: 기본 임계 구간.
        lower2, upper2: 두 번째 구간(선택). 빨강처럼 H=0 근처를 감싸는 색에 쓴다.
        keywords: instruction 파싱에 쓸 키워드(다국어 지원).
    """
    name: str
    lower1: tuple[int, int, int]
    upper1: tuple[int, int, int]
    lower2: Optional[tuple[int, int, int]] = None
    upper2: Optional[tuple[int, int, int]] = None
    keywords: tuple[str, ...] = ()


#: 기본 임계값 — 채도 높은 원색 물체 기준 출발점이다. 실제 조명/물체에서는
#: 반드시 `probe`/`test` 로 재조정할 것. 그대로 실주행에 쓰지 않는다.
DEFAULT_RANGES: list[ColorRange] = [
    ColorRange('red', (0, 120, 80), (10, 255, 255),
               (170, 120, 80), (180, 255, 255),
               keywords=('red', '빨강', '빨간', '빨간색')),
    ColorRange('green', (40, 80, 60), (85, 255, 255),
               keywords=('green', '초록', '초록색', '녹색')),
    ColorRange('blue', (95, 100, 60), (130, 255, 255),
               keywords=('blue', '파랑', '파란', '파란색')),
]


def load_ranges_json(path: str | Path) -> list[ColorRange]:
    """JSON 파일에서 색상 임계값을 읽는다. 없는 필드는 DEFAULT_RANGES 값으로 채운다.

    Args:
        path: JSON 파일 경로. `save_ranges_json()` 이 만든 형식.

    Returns:
        ColorRange 목록.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    defaults = {r.name: r for r in DEFAULT_RANGES}
    out = []
    for entry in data:
        base = defaults.get(entry.get('name', ''))
        merged = dict(asdict(base)) if base else {}
        merged.update(entry)
        merged['lower1'] = tuple(merged['lower1'])
        merged['upper1'] = tuple(merged['upper1'])
        merged['lower2'] = tuple(merged['lower2']) if merged.get('lower2') else None
        merged['upper2'] = tuple(merged['upper2']) if merged.get('upper2') else None
        merged['keywords'] = tuple(merged.get('keywords', ()))
        out.append(ColorRange(**merged))
    return out


def save_ranges_json(path: str | Path, ranges: list[ColorRange]) -> None:
    """색상 임계값을 JSON 으로 저장한다 (캘리브레이션 템플릿 생성용).

    Args:
        path: 저장할 경로.
        ranges: 저장할 ColorRange 목록.
    """
    Path(path).write_text(
        json.dumps([asdict(r) for r in ranges], ensure_ascii=False, indent=2),
        encoding='utf-8')


# ======================================================================
# 탐지
# ======================================================================
@dataclass
class Blob:
    """탐지된 색상 영역 하나."""
    color: str
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def _mask_for(hsv: np.ndarray, r: ColorRange) -> np.ndarray:
    """단일 색상의 이진 마스크를 만든다.

    Args:
        hsv: cv2.cvtColor(..., COLOR_BGR2HSV) 결과.
        r: 색상 임계값.

    Returns:
        0/255 이진 마스크.
    """
    mask = cv2.inRange(hsv, np.array(r.lower1), np.array(r.upper1))
    if r.lower2 is not None and r.upper2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(r.lower2), np.array(r.upper2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def detect_blobs(frame_bgr: np.ndarray, ranges: list[ColorRange] = DEFAULT_RANGES,
                  min_area: float = 1500.0) -> dict[str, Blob]:
    """각 색상별로 가장 큰 연결 영역 하나씩을 찾는다.

    Args:
        frame_bgr: 카메라 원본 BGR 프레임. `CameraWorker` 가 GPU 단(nvvidconv
            flip-method)에서 이미 좌우/상하 보정을 끝낸 프레임을 그대로 넣는다 —
            여기서 다시 뒤집지 않는다.
        ranges: 탐지할 색상 임계값 목록.
        min_area: 이보다 작은 연결 영역은 노이즈로 버린다(px^2).

    Returns:
        색 이름 -> Blob. 임계값을 만족하는 영역이 없으면 그 색은 빠진다.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out: dict[str, Blob] = {}
    for r in ranges:
        mask = _mask_for(hsv, r)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        cx = m['m10'] / m['m00'] if m['m00'] else x + w / 2.0
        cy = m['m01'] / m['m00'] if m['m00'] else y + h / 2.0
        out[r.name] = Blob(r.name, float(cx), float(cy), float(area), (x, y, w, h))
    return out


# ======================================================================
# instruction -> 목표색 파싱
# ======================================================================
_IGNORE_PAT = re.compile(r'\bignore\b|무시')


def parse_target_color(instruction: str,
                        ranges: list[ColorRange] = DEFAULT_RANGES) -> Optional[str]:
    """지시문에서 목표 색을 뽑는다.

    "Go to the red object, ignore the green and blue ones." 같은 문형을
    가정한다. 'ignore'(또는 '무시') 앞부분에서 가장 먼저 등장하는 색 키워드를
    목표로 본다. 그 구간에 색이 없으면 문장 전체에서 처음 등장하는 색을 쓴다.

    한계: "무시하지 말고 빨강 말고 파랑으로 가"처럼 부정이 겹치는 복잡한 문장은
    못 푼다. 데모 수준의 단순 문형 전용.

    Args:
        instruction: 자연어 지시문.
        ranges: 색 이름 -> 키워드 매핑.

    Returns:
        목표 색 이름(ranges 의 name) 또는 못 찾으면 None.
    """
    text = instruction.lower()
    m = _IGNORE_PAT.search(text)
    head = text[:m.start()] if m else text
    for scope in (head, text):
        best_pos, best_name = None, None
        for r in ranges:
            for kw in r.keywords:
                pos = scope.find(kw.lower())
                if pos != -1 and (best_pos is None or pos < best_pos):
                    best_pos, best_name = pos, r.name
        if best_name is not None:
            return best_name
    return None


# ======================================================================
# 시각 서보 (블롭 -> vx, wz)
# ======================================================================
@dataclass
class ServoConfig:
    """시각 서보 게인/한계.

    Attributes:
        kp_ang: 화면 중심 오차(-1..1) -> wz(rad/s) 비례 게인.
        max_wz: 최대 각속도.
        max_vx: 최대 전진 속도.
        approach_area: 이 면적(px^2) 이상부터 감속을 시작한다 (가까워짐 신호).
        stop_area: 이 면적 이상이면 도착으로 본다.
        min_vx: 감속 하한.
        steer_sign: 조향 부호. 카메라 마운트/좌우 정의가 꼬였을 때 -1 로 뒤집는다.
    """
    kp_ang: float = 1.2
    max_wz: float = 0.6
    max_vx: float = 0.20
    approach_area: float = 6000.0
    stop_area: float = 20000.0
    min_vx: float = 0.0
    steer_sign: float = 1.0


@dataclass
class ServoCommand:
    """시각 서보 결과."""
    vx: float
    wz: float
    arrived: bool
    found: bool
    color: Optional[str]
    blob: Optional[Blob]


def compute_servo(blob: Optional[Blob], color: Optional[str],
                   frame_w: int, frame_h: int,
                   cfg: ServoConfig = ServoConfig()) -> ServoCommand:
    """블롭 위치/크기로부터 전진/회전 명령을 만든다.

    부호 규약: ROS REP103 대로 angular.z > 0 은 좌회전(CCW). 블롭이 화면
    오른쪽(cx > 중심)에 있으면 우회전해야 하므로 wz 는 음수가 나온다.

    Args:
        blob: 목표색 블롭. 못 찾았으면 None.
        color: 목표 색 이름 (로그/디버그용, 계산에는 안 쓴다).
        frame_w: 프레임 가로 크기(px).
        frame_h: 프레임 세로 크기(px).
        cfg: 게인/한계.

    Returns:
        ServoCommand. blob 이 None 이면 vx=wz=0, found=False.
    """
    if blob is None:
        return ServoCommand(0.0, 0.0, False, False, color, None)

    err_x = (blob.cx - frame_w / 2.0) / (frame_w / 2.0)  # -1(왼쪽) .. +1(오른쪽)
    wz = float(np.clip(-cfg.kp_ang * err_x * cfg.steer_sign, -cfg.max_wz, cfg.max_wz))

    if blob.area >= cfg.stop_area:
        vx = 0.0
        arrived = True
    else:
        arrived = False
        if blob.area <= cfg.approach_area:
            vx = cfg.max_vx
        else:
            span = max(cfg.stop_area - cfg.approach_area, 1.0)
            frac = (blob.area - cfg.approach_area) / span
            vx = cfg.max_vx * (1.0 - frac)
        vx = float(np.clip(vx, cfg.min_vx, cfg.max_vx))
        # 목표가 화면 가장자리로 크게 벗어나 있으면 회전을 우선하고 전진은 줄인다.
        vx *= max(0.0, 1.0 - min(abs(err_x), 1.0) * 0.6)

    return ServoCommand(vx, wz, arrived, True, color, blob)


def annotate(frame_bgr: np.ndarray, blobs: dict[str, Blob],
             target: Optional[str] = None) -> np.ndarray:
    """탐지 결과를 그린 사본을 만든다 (캘리브레이션/디버그용).

    Args:
        frame_bgr: 원본 프레임 (변경하지 않는다).
        blobs: `detect_blobs()` 결과.
        target: 목표색이면 굵은 테두리로 강조한다.

    Returns:
        박스/라벨이 그려진 사본.
    """
    draw_colors = {'red': (0, 0, 255), 'green': (0, 200, 0), 'blue': (255, 0, 0)}
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
    for name, b in blobs.items():
        color = draw_colors.get(name, (255, 255, 255))
        thickness = 4 if name == target else 2
        x, y, bw, bh = b.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        cv2.circle(out, (int(b.cx), int(b.cy)), 5, color, -1)
        cv2.putText(out, f'{name} area={int(b.area)}', (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if target and target not in blobs:
        cv2.putText(out, f'target={target} NOT FOUND', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


# ======================================================================
# 카메라 (bridge 와 같은 파이프라인 — 독립 실행용, torch/rclpy 의존 없음)
# ======================================================================
_GST_PIPELINE = (
    'nvarguscamerasrc sensor-id={sensor_id} ! '
    'video/x-raw(memory:NVMM),width={width},height={height},'
    'format=NV12,framerate={fps}/1 ! '
    'nvvidconv flip-method={flip_method} ! '
    'video/x-raw,format=BGRx ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink drop=1 max-buffers=1'
)


def open_camera(sensor_id: int, width: int, height: int, fps: int,
                 flip_method: int) -> cv2.VideoCapture:
    """`ticvla_bridge.py` 와 동일한 GStreamer 파이프라인으로 CSI 카메라를 연다.

    주의: nvarguscamerasrc 는 보통 단일 소비자만 허용한다. `ticvla_bridge.py`
    가 이미 카메라를 쥐고 있으면 이 함수는 실패한다 — 캘리브레이션은 bridge를
    끄고 실행할 것.

    Args:
        sensor_id: CSI 센서 번호.
        width, height, fps: 캡처 해상도/프레임레이트.
        flip_method: nvvidconv flip-method.

    Returns:
        연 VideoCapture.
    """
    pipeline = _GST_PIPELINE.format(sensor_id=sensor_id, width=width, height=height,
                                    fps=fps, flip_method=flip_method)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f'카메라를 열지 못했다: {pipeline}')
    return cap


# ======================================================================
# CLI
# ======================================================================
def _cmd_capture(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.sensor_id, args.cam_width, args.cam_height,
                       args.cam_fps, args.flip_method)
    print(f'{args.seconds}초 동안 캡처 -> {out_dir}')
    start = time.monotonic()
    n = 0
    try:
        while time.monotonic() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            n += 1
            cv2.imwrite(str(out_dir / f'frame_{n:04d}.jpg'), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            time.sleep(max(0.0, 1.0 / args.save_fps))
    finally:
        cap.release()
    print(f'{n}장 저장 완료')


def _cmd_probe(args: argparse.Namespace) -> None:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f'이미지를 읽지 못했다: {args.image}', file=sys.stderr)
        sys.exit(1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    half = args.box // 2
    h, w = frame.shape[:2]
    x0, x1 = max(0, args.x - half), min(w, args.x + half + 1)
    y0, y1 = max(0, args.y - half), min(h, args.y + half + 1)
    bgr_patch = frame[y0:y1, x0:x1].reshape(-1, 3)
    hsv_patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    print(f'좌표 ({args.x},{args.y}) 주변 {args.box}x{args.box} 평균/범위:')
    print(f'  BGR mean={bgr_patch.mean(axis=0)} min={bgr_patch.min(axis=0)} max={bgr_patch.max(axis=0)}')
    print(f'  HSV mean={hsv_patch.mean(axis=0)} min={hsv_patch.min(axis=0)} max={hsv_patch.max(axis=0)}')
    print('  -> 이 HSV min/max 근처로 lower1/upper1 을 잡으면 시작점이 된다 '
          '(S/V 는 여유를 좀 더 두는 게 보통 안전).')


def _cmd_test(args: argparse.Namespace) -> None:
    ranges = load_ranges_json(args.json) if args.json else DEFAULT_RANGES
    files = sorted(Path(args.dir).glob('*.jpg'))
    if not files:
        print(f'{args.dir} 에 jpg 가 없다', file=sys.stderr)
        sys.exit(1)
    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else None
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        blobs = detect_blobs(frame, ranges, min_area=args.min_area)
        summary = ', '.join(
            f'{name}: area={int(b.area)} cx={b.cx:.0f} cy={b.cy:.0f}'
            for name, b in blobs.items()) or '(탐지 없음)'
        print(f'{f.name}: {summary}')
        if annotated_dir:
            cv2.imwrite(str(annotated_dir / f.name), annotate(frame, blobs))
    if annotated_dir:
        print(f'주석 이미지 저장: {annotated_dir}')


def _cmd_live(args: argparse.Namespace) -> None:
    ranges = load_ranges_json(args.json) if args.json else DEFAULT_RANGES
    cap = open_camera(args.sensor_id, args.cam_width, args.cam_height,
                       args.cam_fps, args.flip_method)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    print('Ctrl+C 로 종료. ROS 로는 아무것도 보내지 않는다 (탐지만 확인).')
    if args.instruction:
        target = parse_target_color(args.instruction, ranges)
        print(f'instruction={args.instruction!r} -> target={target!r}')
    else:
        target = None
    last_dump = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            blobs = detect_blobs(frame, ranges, min_area=args.min_area)
            summary = ', '.join(
                f'{name}:{int(b.area)}@({b.cx:.0f},{b.cy:.0f})'
                for name, b in blobs.items()) or '(탐지 없음)'
            print(summary)
            now = time.monotonic()
            if debug_dir and now - last_dump >= args.debug_period:
                last_dump = now
                cv2.imwrite(str(debug_dir / f'live_{int(now)}.jpg'),
                            annotate(frame, blobs, target))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    def _cam_args(p: argparse.ArgumentParser) -> None:
        p.add_argument('--sensor-id', type=int, default=0)
        p.add_argument('--cam-width', type=int, default=1280)
        p.add_argument('--cam-height', type=int, default=720)
        p.add_argument('--cam-fps', type=int, default=15)
        p.add_argument('--flip-method', type=int, default=2)

    p_capture = sub.add_parser('capture', help='정지 프레임을 디스크에 저장')
    _cam_args(p_capture)
    p_capture.add_argument('--out', required=True)
    p_capture.add_argument('--seconds', type=float, default=10.0)
    p_capture.add_argument('--save-fps', type=float, default=2.0)
    p_capture.set_defaults(func=_cmd_capture)

    p_probe = sub.add_parser('probe', help='저장된 이미지의 한 지점 HSV 값을 출력')
    p_probe.add_argument('--image', required=True)
    p_probe.add_argument('--x', type=int, required=True)
    p_probe.add_argument('--y', type=int, required=True)
    p_probe.add_argument('--box', type=int, default=11, help='평균 낼 정사각형 한 변(px)')
    p_probe.set_defaults(func=_cmd_probe)

    p_test = sub.add_parser('test', help='저장된 프레임들에 대해 탐지를 검증')
    p_test.add_argument('--dir', required=True)
    p_test.add_argument('--json', default=None, help='ColorRange 목록 JSON (없으면 기본값)')
    p_test.add_argument('--min-area', type=float, default=1500.0)
    p_test.add_argument('--annotated-dir', default=None)
    p_test.set_defaults(func=_cmd_test)

    p_live = sub.add_parser('live', help='카메라 실시간 탐지 (ROS 무관, 명령 미발행)')
    _cam_args(p_live)
    p_live.add_argument('--json', default=None)
    p_live.add_argument('--min-area', type=float, default=1500.0)
    p_live.add_argument('--debug-dir', default=None)
    p_live.add_argument('--debug-period', type=float, default=1.0)
    p_live.add_argument('--instruction', default=None,
                        help='이 지시문으로 parse_target_color() 결과도 같이 보여준다')
    p_live.set_defaults(func=_cmd_live)

    p_template = sub.add_parser('init-json', help='DEFAULT_RANGES 를 JSON 템플릿으로 저장')
    p_template.add_argument('--out', required=True)
    p_template.set_defaults(func=lambda a: save_ranges_json(a.out, DEFAULT_RANGES))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
