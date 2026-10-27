#!/usr/bin/env python3
"""고정 프레임에 대해 TIC-VLA <think> 를 여러 지시문 × 반복으로 뽑아 JSONL 로 남긴다.

로봇은 전혀 움직이지 않는다. 같은 단일 프레임을 ``agent.tic_vla_adapter.LiveTicVlaAdapter``
에 반복해서 흘려보내며, 서로 다른 지시문(프레임에 보이는 물체 2개 + 보이지 않는 물체 2개)
× 반복 횟수만큼 원문 응답을 모은다 — 모델 로딩/프레임 선택/<think> 추출은 전부
``LiveTicVlaAdapter`` 를 그대로 재사용하고, 여기서는 새로 만들지 않는다
(추출 정규식도 ``tic_vla_adapter._RE_THINK`` — ``scripts/analyze_sign_test.py:58`` 와
동일한 패턴 — 를 그대로 가져다 진단에만 쓴다).

두 종류의 프레임 디렉터리(``frame_type``)를 돈다:
    * ``scene``: 실제로 물체가 보이는 낮 프레임(``data/GoToColorMarker_v1`` 에서 가져온
      원본 1장을 ``num_delayed+1`` 장 복제).
    * ``dark``:  대조군. 조명 없이 찍힌 암흑 프레임(동일 방식으로 복제).

둘 다 모델을 한 번만 로딩한 뒤 ``LiveTicVlaAdapter._frame_dir`` 속성만 바꿔가며
재사용한다 — GPU 에 모델을 두 번 올리지 않기 위해서다.

사용 예:
    source ~/.gemini_env  # GEMINI_API_KEY 는 안 쓰지만 관례상 같이 소싱
    venvs/ticvla/bin/python3 -m agent.dump_think_static \\
        --config config/react_agent.yaml \\
        --scene-frame-dir logs/think_dump/scene_frame \\
        --dark-frame-dir logs/think_dump/static_frame \\
        --repeats 5 \\
        --out-dir logs/think_dump
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.config import TicVlaConfig, load_config
from agent.tic_vla_adapter import LiveTicVlaAdapter

logger = logging.getLogger('dump_think_static')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 그룹 a(프레임에 보이는 물체) / 그룹 b(프레임에 없는 물체) 지시문.
#: 두 frame_type(scene/dark) 모두 동일한 지시문 세트를 쓴다 — 사람이 scene 프레임을
#: 보고 확정한 값이다(2026-10-26).
INSTRUCTIONS: list[tuple[str, str]] = [
    ('present', 'Go to the red cooler.'),
    ('present', 'Go to the blue cooler.'),
    ('absent', 'Find the gray bin.'),
    ('absent', 'Find the cup.'),
]


@dataclass
class DumpRecord:
    """JSONL 한 줄."""

    timestamp: str
    model_path: str
    ckpt_path: str
    frame_dir: str
    frame_type: str
    frame_file: str
    instruction: str
    group: str
    repeat_idx: int
    think_text: str
    raw_output: str
    observation_degraded: bool
    latency_s: float


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_degraded(raw_output: str) -> bool:
    """``<think>`` 블록이 없거나 닫는 태그 없이 잘렸으면 ``True``.

    ``LiveTicVlaAdapter._extract_think`` 가 쓰는 것과 똑같은 두 태그 문자열만
    보고 판단한다 — 새 정규식을 만들지 않는다(금지 사항).
    """
    if '<think>' not in raw_output:
        return True
    if '</think>' not in raw_output:
        return True
    return False


def _run_frame_type(adapter: LiveTicVlaAdapter, frame_dir: Path, frame_type: str,
                     repeats: int) -> list[DumpRecord]:
    """한 frame_type(scene/dark)에 대해 지시문 × repeats 만큼 _infer() 를 돌린다.

    ``LiveTicVlaAdapter._infer`` 를 직접 호출한다 — ``goto()``/``rotate()`` 공개
    메서드는 지시문을 ``f'Go to {target}.'`` 형식으로 강제로 바꿔버려서,
    "Find the cup." 같은 그룹 b 지시문을 있는 그대로 보낼 수 없기 때문이다.
    모델 로딩·프레임 선택·<think> 추출 로직은 손대지 않고 그대로 재사용한다.

    Args:
        adapter: 이미 모델이 로딩된 어댑터 인스턴스(공유 재사용).
        frame_dir: 이번 frame_type 이 읽을 프레임 디렉터리.
        frame_type: ``'scene'`` 또는 ``'dark'`` (로그 필드용 라벨일 뿐).
        repeats: 지시문당 반복 횟수.

    Returns:
        이번 frame_type 에서 나온 레코드 리스트(지시문 수 × repeats 개).
    """
    adapter._frame_dir = frame_dir  # type: ignore[attr-defined]
    records: list[DumpRecord] = []

    for group, instruction in INSTRUCTIONS:
        for repeat_idx in range(repeats):
            delayed_paths, current_path = adapter._select_frames(  # type: ignore[attr-defined]
                adapter._frame_dir, adapter._cfg.num_delayed)  # type: ignore[attr-defined]
            robot_state = adapter._torch.zeros(  # type: ignore[attr-defined]
                5, dtype=adapter._torch.bfloat16, device=adapter._model.device)  # type: ignore[attr-defined]

            start = time.monotonic()
            with adapter._torch.inference_mode():  # type: ignore[attr-defined]
                raw_response, _waypoints, _prompt = adapter._model.predict(  # type: ignore[attr-defined]
                    delayed_image_paths=delayed_paths,
                    current_image_path=current_path,
                    instruction=instruction,
                    robot_state=robot_state,
                    time_delay=0.0,
                )
            latency_s = time.monotonic() - start
            think_text = adapter._extract_think(raw_response)  # type: ignore[attr-defined]

            record = DumpRecord(
                timestamp=_utcnow_iso(),
                model_path=str(adapter._cfg.model_path),  # type: ignore[attr-defined]
                ckpt_path=str(adapter._cfg.ckpt_path),  # type: ignore[attr-defined]
                frame_dir=str(frame_dir),
                frame_type=frame_type,
                frame_file=current_path,
                instruction=instruction,
                group=group,
                repeat_idx=repeat_idx,
                think_text=think_text,
                raw_output=raw_response,
                observation_degraded=_is_degraded(raw_response),
                latency_s=latency_s,
            )
            records.append(record)
            logger.info(
                f'[{frame_type}] {instruction!r} #{repeat_idx} '
                f'({latency_s:.1f}s, degraded={record.observation_degraded}): '
                f'{think_text[:80]!r}')
    return records


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/react_agent.yaml',
                        help='model_path/ckpt_path/num_delayed 를 가져올 설정 파일.')
    parser.add_argument('--scene-frame-dir', required=True,
                        help='물체가 실제로 보이는 고정 프레임 디렉터리.')
    parser.add_argument('--dark-frame-dir', required=True,
                        help='대조군(암흑) 고정 프레임 디렉터리.')
    parser.add_argument('--repeats', type=int, default=5,
                        help='지시문당 반복 횟수(기본 5).')
    parser.add_argument('--out-dir', default='logs/think_dump',
                        help='결과 JSONL 을 저장할 디렉터리.')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    args = _parse_args(argv)

    cfg = load_config(args.config)
    if not cfg.ticvla.model_path or not cfg.ticvla.ckpt_path:
        logger.error('config 의 ticvla.model_path/ckpt_path 가 비어 있다.')
        return 1

    scene_dir = Path(args.scene_frame_dir)
    dark_dir = Path(args.dark_frame_dir)

    run_label = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'think_static_{run_label}.jsonl'

    # 초기화는 두 frame_type 중 아무 쪽이나 상관없다 — 실제로 어느 디렉터리를
    # 읽을지는 _run_frame_type() 이 매번 adapter._frame_dir 를 바꿔서 정한다.
    ticvla_cfg = TicVlaConfig(
        model_path=cfg.ticvla.model_path,
        ckpt_path=cfg.ticvla.ckpt_path,
        frame_dir=str(scene_dir),
        num_delayed=cfg.ticvla.num_delayed,
    )

    logger.info(f'JSONL 출력: {out_path}')
    logger.info(f'모델: {ticvla_cfg.model_path}')
    logger.info(f'체크포인트: {ticvla_cfg.ckpt_path}')

    try:
        adapter = LiveTicVlaAdapter(ticvla_cfg, logger)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error(f'TIC-VLA 어댑터 초기화 실패: {exc}')
        return 1

    with out_path.open('w', encoding='utf-8') as fh:
        meta = {
            'meta': True,
            'run_timestamp': _utcnow_iso(),
            'model_path': ticvla_cfg.model_path,
            'ckpt_path': ticvla_cfg.ckpt_path,
            'scene_frame_dir': str(scene_dir),
            'dark_frame_dir': str(dark_dir),
            'num_delayed': ticvla_cfg.num_delayed,
            'repeats': args.repeats,
            'instructions': INSTRUCTIONS,
        }
        fh.write(json.dumps(meta, ensure_ascii=False) + '\n')
        fh.flush()

        for frame_type, frame_dir in (('scene', scene_dir), ('dark', dark_dir)):
            records = _run_frame_type(adapter, frame_dir, frame_type, args.repeats)
            for record in records:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
                fh.flush()

    logger.info(f'완료: {out_path}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
