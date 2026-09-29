#!/usr/bin/env python3
"""TIC-VLA predict() 구간별 지연시간 측정.

공식 코드를 수정하지 않고 주요 함수를 래핑해 시간을 잰다.
느린 루프(VLM)와 빠른 루프(action)를 분리했을 때의 실제 제어 주기를 확인한다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path('/home/soda/TIC-VLA')
sys.path.insert(0, str(ROOT / 'scripts'))

from ticvla_smoke_test import (  # noqa: E402
    apply_torch_distributed_shim,
    load_checkpoint_into_model,
    select_frames,
)

# Jetson 용 torch 는 USE_DISTRIBUTED=0 이라 torch.distributed.is_initialized 가 없다.
# InternVL 원격 코드가 이를 무조건 호출하므로 모델을 만들기 전에 보완해 둔다.
apply_torch_distributed_shim()

from ticvla.models.ticvla import TICVLA  # noqa: E402

TIMINGS: dict[str, list[float]] = {}
CAPTURED: dict[str, Any] = {}


def _sync() -> None:
    """CUDA 비동기 실행을 동기화한다 (정확한 시간 측정을 위해)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap(obj: Any, attr: str, key: str, capture: str | None = None) -> None:
    """객체의 메서드를 시간 측정 래퍼로 교체한다.

    Args:
        obj: 대상 객체.
        attr: 감쌀 메서드 이름.
        key: TIMINGS 에 기록할 키.
        capture: 반환값을 CAPTURED 에 저장할 키. None 이면 저장하지 않는다.
    """
    original = getattr(obj, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _sync()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        _sync()
        TIMINGS.setdefault(key, []).append(time.perf_counter() - started)
        if capture is not None:
            CAPTURED[capture] = result
        return result

    setattr(obj, attr, wrapper)


def main() -> int:
    """프로파일링을 실행한다."""
    print('모델 로딩 중...')
    model = TICVLA(model_path=str(ROOT / 'models/InternVL3-1B'),
                   action_horizon_steps=30)
    load_checkpoint_into_model(model, ROOT / 'checkpoints/TIC-VLA-model.ckpt')
    model.to('cuda')
    model.eval()

    # 구간별 래핑. vlm.forward 는 past_key_values 를 얻기 위해 반환값도 캡처한다.
    wrap(model, 'load_images', '1_load_delayed')
    wrap(model.vlm, 'chat', '2_vlm_chat')
    wrap(model.vlm, 'forward', '3_vlm_prefill', capture='vlm_out')
    wrap(model.vlm, 'extract_feature', '4_extract_current')
    wrap(model.action_expert, 'forward', '5_action_expert')

    delayed, current = select_frames(ROOT / 'logs/test_seq', 4)
    robot_state = torch.zeros(5, dtype=torch.bfloat16, device=model.device)

    print('warmup 1회...')
    with torch.inference_mode():
        model.predict(delayed_image_paths=delayed, current_image_path=current,
                      instruction='Move forward safely and efficiently.',
                      robot_state=robot_state, time_delay=0.0)
    TIMINGS.clear()

    print('전체 predict() 3회 측정...')
    full: list[float] = []
    for _ in range(3):
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            model.predict(delayed_image_paths=delayed, current_image_path=current,
                          instruction='Move forward safely and efficiently.',
                          robot_state=robot_state, time_delay=0.0)
        _sync()
        full.append(time.perf_counter() - started)

    # ── 빠른 루프 단독 측정: 캐시된 past_key_values 재사용 ──────────────
    past = CAPTURED['vlm_out'].past_key_values
    if past is not None and hasattr(past, 'layers'):
        past = tuple((layer.keys, layer.values) for layer in past.layers)

    from ticvla.utils.vision import load_image
    fast: list[float] = []
    print('빠른 루프(④+⑤만) 10회 측정...')
    for _ in range(10):
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            pixel = load_image(current, input_size=448, max_num=1).to(
                torch.bfloat16).to(model.device)
            embeds = model.vlm.extract_feature(pixel)
            embeds = embeds.reshape(-1, embeds.shape[-1]).unsqueeze(0)
            state = torch.cat([
                robot_state,
                torch.tensor([0.0], device=model.device, dtype=torch.bfloat16),
            ]).unsqueeze(0).unsqueeze(-1)
            model.action_expert(embeds, state, kv_cache=past)
        _sync()
        fast.append(time.perf_counter() - started)

    # ── 결과 출력 ──────────────────────────────────────────────────────
    total = sum(full) / len(full)
    print()
    print('=' * 62)
    print('predict() 구간별 지연시간 (3회 평균)')
    print('=' * 62)
    for key in sorted(TIMINGS):
        samples = TIMINGS[key]
        mean = sum(samples) / len(samples)
        print(f'  {key:<20} {mean:>8.3f}s  ({mean / total * 100:>5.1f}%)  '
              f'x{len(samples)}')
    print(f"  {'전체':<20} {total:>8.3f}s")
    print()
    print('=' * 62)
    print('빠른 루프 단독 (④ extract_feature + ⑤ action_expert)')
    print('=' * 62)
    fast_mean = sum(fast[1:]) / len(fast[1:])
    print(f'  평균 {fast_mean * 1000:>8.1f} ms  '
          f'→ 이론상 {1 / fast_mean:>5.1f} Hz')
    print(f'  min / max : {min(fast) * 1000:.1f} / {max(fast) * 1000:.1f} ms')
    print(f'  느린 루프 대비 {total / fast_mean:>5.1f}배 빠름')

    out = ROOT / 'logs/latency_profile.json'
    out.write_text(json.dumps({
        'stages': {k: v for k, v in TIMINGS.items()},
        'full_predict_sec': full,
        'fast_loop_sec': fast,
    }, indent=2), encoding='utf-8')
    print(f'\nJSON 저장: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
