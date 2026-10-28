#!/usr/bin/env python3
"""GoToColorMarker_v1 수집 잔량 표 — 배치×색 성공 수와 남은 개수, marker 4종 현황.

joystick_colormarker.py 의 scan_counts() 와 같은 기준(success=true 만 셈)이지만
rclpy/pygame 의존이 없어 ROS 를 source 하지 않은 셸에서도 바로 돈다.

사용 예:
    python3 collect/remaining_colormarker.py              # 색당 목표 16
    python3 collect/remaining_colormarker.py --target 8
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "GoToColorMarker_v1" / "raw"
COLORS = ("red", "green", "blue")
#: joystick_colormarker.EPISODE_DIR_RE 와 같은 규칙(ep{NNNN}_L{layout}_{color}).
EPISODE_DIR_RE = re.compile(r"^ep\d{4}_L([123])_(red|green|blue)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target", type=int, default=16,
                        help="배치×색당 목표 성공 에피소드 수")
    args = parser.parse_args()

    legacy: Counter[tuple[int, str]] = Counter()
    legacy_fail: Counter[tuple[int, str]] = Counter()
    marker: Counter[str] = Counter()
    unfinished: list[str] = []
    for ep in sorted(p for p in args.root.iterdir() if p.is_dir()):
        ej = ep / "episode.json"
        if not ej.is_file():
            unfinished.append(f"{ep.name}(episode.json 없음)")
            continue
        try:
            meta = json.loads(ej.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            unfinished.append(f"{ep.name}(episode.json 파싱 실패)")
            continue
        if meta.get("status") == "recording" or meta.get("success") is None:
            unfinished.append(f"{ep.name}(결과 미입력)")
            continue
        if meta.get("episode_type"):
            key = meta["episode_type"]
            if meta.get("target_color"):
                key += f"/{meta['target_color']}"
            if meta.get("landmark_name"):
                key += f"/{meta['landmark_name']}"
            marker[key + ("" if meta.get("success") else " (실패)")] += 1
            continue
        m = EPISODE_DIR_RE.match(ep.name)
        if not m:
            continue
        k = (int(m.group(1)), m.group(2))
        if meta.get("success") is True:
            legacy[k] += 1
        else:
            legacy_fail[k] += 1

    print(f"[3색 배치 수집 잔량] root={args.root}  목표 {args.target}/배치·색 (성공 기준)")
    print(f"{'배치':<5}" + "".join(f"{c:>16}" for c in COLORS) + f"{'남은 합':>10}")
    total_left = 0
    todo: list[str] = []
    for layout in (1, 2, 3):
        cells, left_sum = [], 0
        for c in COLORS:
            n = legacy[(layout, c)]
            left = max(0, args.target - n)
            left_sum += left
            fail = legacy_fail[(layout, c)]
            cells.append(f"{n:>2}/{args.target}" + (f" (실패{fail})" if fail else "")
                         + (" ✓" if left == 0 else f" -{left}"))
            if left:
                todo.append(f"배치{layout} {c} ×{left}")
        total_left += left_sum
        print(f"L{layout:<4}" + "".join(f"{s:>16}" for s in cells) + f"{left_sum:>10}")
    print(f"→ 남은 성공 에피소드: {total_left}개")
    if todo:
        print("  순서 제안(배치 전환 최소화): " + ", ".join(todo))

    print("\n[marker 에피소드 (--episode-type)]")
    if marker:
        for k, v in sorted(marker.items()):
            print(f"  {k:<40} {v}")
    else:
        print("  아직 없음")

    if unfinished:
        print("\n[⚠ 결과 미확정 폴더 — 대시보드에서 결과 입력하거나 확인할 것]")
        for u in unfinished:
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
