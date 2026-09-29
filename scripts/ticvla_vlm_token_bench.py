#!/usr/bin/env python3
"""VLM 생성 설정(max_new_tokens / do_sample)에 따른 지연시간과 출력 변화 측정.

`TICVLA.predict()` 는 생성 설정을 내부에 하드코딩하고 있다.

    ticvla/models/ticvla.py:579
    generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.7)

공식 코드를 수정하지 않고 바꾸려면 `model.vlm.chat` 을 감싸 4번째 인자를 갈아끼운다.

주의: 생성된 응답 텍스트는 그대로 assistant 메시지가 되어 prefill 입력에 들어간다
(ticvla.py:595). 따라서 max_new_tokens 를 줄이면 속도만 변하는 것이 아니라
past_key_values 와 waypoint 까지 달라진다. 그래서 시간과 함께 waypoint 도 같이 잰다.

사용 예:

    python3 ticvla_vlm_token_bench.py --repeat 3
"""
from __future__ import annotations

import argparse
import json
import statistics
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

# InternVL 원격 코드가 torch.distributed 를 호출하므로 모델 생성 전에 보완한다.
apply_torch_distributed_shim()

from ticvla.models.ticvla import TICVLA  # noqa: E402

#: 측정할 생성 설정. 200/sample 이 현재 공식 기본값이다.
CONFIGS: list[dict[str, Any]] = [
    {'label': '200 / sample (현행)',
     'config': {'max_new_tokens': 200, 'do_sample': True, 'temperature': 0.7}},
    {'label': '200 / greedy',
     'config': {'max_new_tokens': 200, 'do_sample': False}},
    {'label': ' 64 / greedy',
     'config': {'max_new_tokens': 64, 'do_sample': False}},
    {'label': ' 64 / sample',
     'config': {'max_new_tokens': 64, 'do_sample': True, 'temperature': 0.7}},
]


def sync() -> None:
    """CUDA 비동기 실행을 동기화한다 (정확한 시간 측정을 위해)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Instrumented:
    """vlm.chat / vlm.forward 를 감싸 생성 설정을 교체하고 시간을 재는 래퍼.

    Attributes:
        override: chat 에 강제로 넣을 생성 설정. None 이면 원래 값을 쓴다.
        chat_sec: 마지막 chat 호출 소요 시간.
        prefill_sec: 마지막 forward(prefill) 호출 소요 시간.
        response: 마지막 chat 이 만든 응답 텍스트.
    """

    def __init__(self, model: Any) -> None:
        """래핑을 설치한다.

        Args:
            model: TICVLA 인스턴스.
        """
        self.override: dict[str, Any] | None = None
        self.chat_sec = 0.0
        self.prefill_sec = 0.0
        self.response = ''

        original_chat = model.vlm.chat
        original_forward = model.vlm.forward

        def chat_wrapper(*args: Any, **kwargs: Any) -> Any:
            """생성 설정을 갈아끼우고 chat 시간을 잰다."""
            if self.override is not None:
                if 'generation_config' in kwargs:
                    kwargs['generation_config'] = dict(self.override)
                else:
                    # predict() 는 generation_config 를 4번째 위치 인자로 넘긴다.
                    args = list(args)
                    args[3] = dict(self.override)
                    args = tuple(args)
            sync()
            started = time.perf_counter()
            result = original_chat(*args, **kwargs)
            sync()
            self.chat_sec = time.perf_counter() - started
            self.response = result if isinstance(result, str) else str(result)
            return result

        def forward_wrapper(*args: Any, **kwargs: Any) -> Any:
            """prefill 시간을 잰다."""
            sync()
            started = time.perf_counter()
            result = original_forward(*args, **kwargs)
            sync()
            self.prefill_sec = time.perf_counter() - started
            return result

        model.vlm.chat = chat_wrapper
        model.vlm.forward = forward_wrapper


def measure(model: Any, probe: Instrumented, delayed: list[str], current: str,
            instruction: str, config: dict[str, Any],
            repeat: int) -> dict[str, Any]:
    """한 가지 생성 설정으로 predict() 를 여러 번 돌려 통계를 낸다.

    Args:
        model: TICVLA 인스턴스.
        probe: 설치된 래퍼.
        delayed: 과거 프레임 경로 목록.
        current: 현재 프레임 경로.
        instruction: 자연어 지시문.
        config: 생성 설정.
        repeat: 반복 횟수.

    Returns:
        측정 결과 딕셔너리.
    """
    probe.override = config
    totals: list[float] = []
    chats: list[float] = []
    prefills: list[float] = []
    tokens: list[int] = []
    finals: list[tuple[float, float]] = []
    responses: list[str] = []
    robot_state = torch.zeros(5, dtype=torch.bfloat16, device=model.device)

    for _ in range(repeat):
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            _response, waypoints, _prompt = model.predict(
                delayed_image_paths=delayed,
                current_image_path=current,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=0.0,
            )
        sync()
        totals.append(time.perf_counter() - started)
        chats.append(probe.chat_sec)
        prefills.append(probe.prefill_sec)
        tokens.append(len(model.tokenizer(probe.response)['input_ids']))
        responses.append(probe.response)
        values = waypoints.detach().to(torch.float32).cpu()[0]
        finals.append((float(values[-1, 0]), float(values[-1, 1])))

    distances = [(x ** 2 + y ** 2) ** 0.5 for x, y in finals]
    # <answer> 블록이 잘려나갔는지가 품질의 핵심 지표다. 응답 텍스트는 그대로
    # prefill 입력이 되므로, 답이 잘리면 action expert 가 받는 컨텍스트가 망가진다.
    complete = [('<answer>' in text and '</answer>' in text) for text in responses]
    return {
        'answer_complete': complete,
        'answer_complete_rate': sum(complete) / len(complete),
        'config': config,
        'total_sec': totals,
        'chat_sec': chats,
        'prefill_sec': prefills,
        'response_tokens': tokens,
        'final_dxdy': finals,
        'final_distance_m': distances,
        'responses': responses,
        'total_mean': statistics.fmean(totals),
        'chat_mean': statistics.fmean(chats),
        'prefill_mean': statistics.fmean(prefills),
        'tokens_mean': statistics.fmean(tokens),
        'distance_mean': statistics.fmean(distances),
        'distance_stdev': statistics.stdev(distances) if len(distances) > 1 else 0.0,
    }


def print_report(results: list[dict[str, Any]], baseline_label: str) -> None:
    """측정 결과를 표로 출력한다.

    Args:
        results: measure() 결과 목록 (label 키 포함).
        baseline_label: 기준으로 삼을 설정 이름.
    """
    baseline = next(r for r in results if r['label'] == baseline_label)
    base_total = baseline['total_mean']

    print()
    print('=' * 78)
    print('생성 설정별 지연시간')
    print('=' * 78)
    print(f'{"설정":<22}{"전체":>9}{"chat":>9}{"prefill":>9}'
          f'{"응답토큰":>9}{"현행대비":>11}')
    print('-' * 78)
    for row in results:
        speedup = base_total / row['total_mean']
        print(f'{row["label"]:<22}'
              f'{row["total_mean"]:>8.2f}s'
              f'{row["chat_mean"]:>8.2f}s'
              f'{row["prefill_mean"]:>8.2f}s'
              f'{row["tokens_mean"]:>9.0f}'
              f'{speedup:>10.2f}x')

    print()
    print('=' * 78)
    print('waypoint 최종 변위 (텍스트가 prefill 에 들어가므로 출력도 변한다)')
    print('=' * 78)
    for row in results:
        pairs = ', '.join(f'{d:.3f}' for d in row['final_distance_m'])
        answer = f'{row["answer_complete_rate"] * 100:.0f}%'
        print(f'{row["label"]:<22} 평균 {row["distance_mean"]:.3f}m  '
              f'표준편차 {row["distance_stdev"]:.3f}m  '
              f'<answer>완성 {answer:>4}  [{pairs}]')

    print()
    print('=' * 78)
    print('응답 텍스트 (각 설정 첫 번째 실행, 앞 160자)')
    print('=' * 78)
    for row in results:
        text = row['responses'][0].replace('\n', ' ')
        print(f'[{row["label"].strip()}] {text[:160]}')
        print(f'{"":>4}... 끝부분: ...{text[-70:]}')
        print()


def parse_configs(spec: str) -> list[dict[str, Any]]:
    """"토큰수:모드" 형식 문자열을 생성 설정 목록으로 바꾼다.

    예: '200:sample,64:greedy' -> 두 가지 설정.

    Args:
        spec: 콤마로 구분된 설정 문자열.

    Returns:
        label 과 config 를 담은 목록.

    Raises:
        ValueError: 형식이 잘못되었거나 모드 이름이 sample/greedy 가 아닌 경우.
    """
    entries: list[dict[str, Any]] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'형식이 잘못됐다 (토큰수:모드 이어야 한다): {chunk}')
        tokens_text, mode = (part.strip() for part in chunk.split(':', 1))
        if mode not in ('sample', 'greedy'):
            raise ValueError(f'모드는 sample 또는 greedy 여야 한다: {mode}')
        tokens = int(tokens_text)
        config: dict[str, Any] = {'max_new_tokens': tokens,
                                  'do_sample': mode == 'sample'}
        if mode == 'sample':
            config['temperature'] = 0.7
        entries.append({'label': f'{tokens:>3} / {mode}', 'config': config})
    if not entries:
        raise ValueError('설정이 하나도 없다')
    return entries


def parse_args(argv: list[str]) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: 인자 리스트.

    Returns:
        파싱된 Namespace.
    """
    parser = argparse.ArgumentParser(
        description='VLM 생성 설정별 지연시간 측정',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-path', default=str(ROOT / 'models/InternVL3-1B'))
    parser.add_argument('--ckpt-path',
                        default=str(ROOT / 'checkpoints/TIC-VLA-model.ckpt'))
    parser.add_argument('--image-dir', default=str(ROOT / 'logs/test_seq'))
    parser.add_argument('--num-delayed', type=int, default=4)
    parser.add_argument('--instruction',
                        default='Move forward safely and efficiently.')
    parser.add_argument('--repeat', type=int, default=3,
                        help='설정별 반복 횟수')
    parser.add_argument('--configs', default=None,
                        help='측정할 설정. "토큰수:sample|greedy" 를 콤마로 나열. '
                             '생략하면 기본 4종을 쓴다')
    parser.add_argument('--output-json',
                        default=str(ROOT / 'logs/vlm_token_bench.json'))
    return parser.parse_args(argv)


def main() -> int:
    """측정을 실행한다.

    Returns:
        종료 코드.
    """
    cfg = parse_args(sys.argv[1:])
    configs = parse_configs(cfg.configs) if cfg.configs else CONFIGS

    print('모델 로딩 중...')
    model = TICVLA(model_path=cfg.model_path, action_horizon_steps=30)
    load_checkpoint_into_model(model, Path(cfg.ckpt_path))
    model.to('cuda')
    model.eval()
    probe = Instrumented(model)

    delayed, current = select_frames(Path(cfg.image_dir), cfg.num_delayed)
    print(f'프레임: delayed {len(delayed)}장 + current 1장')

    # warmup: 첫 실행은 커널 컴파일 등으로 느리므로 통계에서 제외한다.
    print('warmup 1회...')
    probe.override = configs[0]['config']
    with torch.inference_mode():
        model.predict(delayed_image_paths=delayed, current_image_path=current,
                      instruction=cfg.instruction,
                      robot_state=torch.zeros(5, dtype=torch.bfloat16,
                                              device=model.device),
                      time_delay=0.0)

    results: list[dict[str, Any]] = []
    for entry in configs:
        print(f'측정 중: {entry["label"].strip()} x{cfg.repeat} ...', flush=True)
        row = measure(model, probe, delayed, current, cfg.instruction,
                      entry['config'], cfg.repeat)
        row['label'] = entry['label']
        results.append(row)

    print_report(results, configs[0]['label'])

    out = Path(cfg.output_json)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print(f'JSON 저장: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
