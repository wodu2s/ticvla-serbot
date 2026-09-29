#!/usr/bin/env python3
"""TIC-VLA 모델을 Jetson Orin NX 에 올려 메모리와 단일 추론 결과를 점검하는 진단 스크립트.

이 스크립트는 학습을 하지 않는다. 데이터셋(TICVLADataset)도 사용하지 않고,
이미지 디렉터리에서 프레임 몇 장을 골라 `TICVLA.predict()` 를 한 번(또는 --repeat 회)
돌려보며 다음을 확인한다.

* 단계(STAGE)별 프로세스 RSS / 시스템 여유 메모리 / CUDA 메모리
* 체크포인트 로딩 시 VLM·action expert 의 missing / unexpected 키 개수
* 생성 텍스트, 프롬프트 전문, 30스텝 waypoint 전체 값

체크포인트 로딩 로직은 `ticvla/training/evaluate.py` 의 `TICVLATester.__init__` 를
그대로 재현한 것이다. TICVLATester 자체는 데이터셋을 요구하므로 사용하지 않는다.

실행 예:
    /home/soda/TIC-VLA/venvs/ticvla/bin/python \\
        /home/soda/TIC-VLA/scripts/ticvla_smoke_test.py --repeat 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import torch

# ----------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------
#: 바이트 → 메비바이트 변환 계수.
BYTES_PER_MB: float = 1024.0 * 1024.0

#: waypoint 시간 간격(초). 액션 헤드는 10Hz 로 학습되었다.
STEP_DT_SEC: float = 0.1

#: 보고서에서 강조할 스텝 인덱스(≈3초 시점).
HIGHLIGHT_STEP: int = 29

#: Jetson 전용 PyTorch 빌드임을 식별하는 문자열.
EXPECTED_TORCH_TAG: str = 'nv24.05'

#: 이미지 디렉터리에서 프레임으로 인정할 확장자.
IMAGE_SUFFIXES: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')

#: 표 구분선 폭.
RULE_WIDTH: int = 78

T = TypeVar('T')

logger = logging.getLogger('ticvla_smoke_test')


# ======================================================================
# 메모리 측정
# ======================================================================
@dataclass
class MemorySnapshot:
    """특정 시점의 메모리 사용량 스냅샷 (모두 MB 단위)."""

    rss_mb: float
    sys_total_mb: float
    sys_available_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    cuda_max_allocated_mb: float


def read_process_rss_mb() -> float:
    """현재 프로세스의 RSS 를 MB 로 돌려준다.

    psutil 이 있으면 psutil 을 쓰고, 없으면 /proc/self/status 의 VmRSS 를 직접 읽는다.

    Returns:
        프로세스 RSS(MB). 측정에 실패하면 -1.0.
    """
    try:
        import psutil  # noqa: PLC0415 (psutil 은 선택적 의존성이라 지연 import 한다)

        return psutil.Process().memory_info().rss / BYTES_PER_MB
    except ImportError:
        pass
    except Exception as exc:
        logger.warning('psutil RSS 측정 실패, /proc 폴백을 사용한다: %s', exc)

    # 폴백: /proc/self/status 의 VmRSS 는 kB 단위로 기록된다.
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.0
    except Exception as exc:
        logger.warning('/proc/self/status 읽기 실패: %s', exc)
    return -1.0


def read_system_memory_mb() -> tuple[float, float]:
    """/proc/meminfo 에서 시스템 전체/여유 메모리를 읽는다.

    Returns:
        (MemTotal MB, MemAvailable MB). 읽기에 실패하면 (-1.0, -1.0).
    """
    total_mb = -1.0
    available_mb = -1.0
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                total_mb = float(line.split()[1]) / 1024.0
            elif line.startswith('MemAvailable:'):
                available_mb = float(line.split()[1]) / 1024.0
    except Exception as exc:
        logger.warning('/proc/meminfo 읽기 실패: %s', exc)
    return total_mb, available_mb


def take_memory_snapshot() -> MemorySnapshot:
    """프로세스/시스템/CUDA 메모리를 한 번에 측정한다.

    Returns:
        측정된 MemorySnapshot. CUDA 를 못 쓰면 CUDA 항목은 0.0 으로 채운다.
    """
    total_mb, available_mb = read_system_memory_mb()

    allocated = reserved = max_allocated = 0.0
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / BYTES_PER_MB
            reserved = torch.cuda.memory_reserved() / BYTES_PER_MB
            max_allocated = torch.cuda.max_memory_allocated() / BYTES_PER_MB
        except Exception as exc:
            logger.warning('CUDA 메모리 조회 실패: %s', exc)

    return MemorySnapshot(
        rss_mb=read_process_rss_mb(),
        sys_total_mb=total_mb,
        sys_available_mb=available_mb,
        cuda_allocated_mb=allocated,
        cuda_reserved_mb=reserved,
        cuda_max_allocated_mb=max_allocated,
    )


# ======================================================================
# 스테이지 실행 관리
# ======================================================================
@dataclass
class StageResult:
    """스테이지 하나의 실행 결과와 전후 메모리."""

    name: str
    ok: bool
    duration_sec: float
    before: MemorySnapshot
    after: MemorySnapshot
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    notes: list[str] = field(default_factory=list)


class StageTracker:
    """스테이지를 순서대로 실행하며 예외/시간/메모리를 기록한다.

    한 스테이지가 실패하면 `aborted` 가 True 가 되고, 이후 스테이지는
    실행하지 않고 건너뛴다.
    """

    def __init__(self) -> None:
        """빈 결과 목록으로 초기화한다."""
        self.results: list[StageResult] = []
        self.aborted: bool = False
        self.failed_stage: Optional[str] = None

    def run(self, name: str, func: Callable[[], T]) -> tuple[bool, Optional[T]]:
        """스테이지 하나를 실행하고 결과를 기록한다.

        Args:
            name: 스테이지 이름 (예: "STAGE 1: 모델 구조 생성").
            func: 실행할 무인자 콜러블.

        Returns:
            (성공 여부, 반환값). 실패하거나 건너뛴 경우 반환값은 None.
        """
        if self.aborted:
            logger.warning('%s 건너뜀 (이전 스테이지 실패: %s)', name, self.failed_stage)
            return False, None

        logger.info('─' * 60)
        logger.info('%s 시작', name)
        before = take_memory_snapshot()
        started = time.perf_counter()

        try:
            value = func()
        except torch.cuda.OutOfMemoryError as exc:
            elapsed = time.perf_counter() - started
            self._record_failure(name, before, elapsed, exc)
            _dump_oom_state(name)
            return False, None
        except ImportError as exc:
            elapsed = time.perf_counter() - started
            self._record_failure(name, before, elapsed, exc)
            _explain_import_error(exc)
            return False, None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._record_failure(name, before, elapsed, exc)
            logger.error('예외 상세:\n%s', traceback.format_exc())
            return False, None

        elapsed = time.perf_counter() - started
        after = take_memory_snapshot()
        self.results.append(
            StageResult(name=name, ok=True, duration_sec=elapsed,
                        before=before, after=after)
        )
        logger.info(
            '%s 완료 (%.2fs, RSS %.0f→%.0f MB, CUDA alloc %.0f→%.0f MB)',
            name, elapsed, before.rss_mb, after.rss_mb,
            before.cuda_allocated_mb, after.cuda_allocated_mb,
        )
        return True, value

    def _record_failure(
        self,
        name: str,
        before: MemorySnapshot,
        elapsed: float,
        exc: BaseException,
    ) -> None:
        """실패한 스테이지를 기록하고 이후 스테이지를 중단시킨다.

        Args:
            name: 스테이지 이름.
            before: 스테이지 시작 시점 메모리.
            elapsed: 실패까지 걸린 시간(초).
            exc: 발생한 예외.
        """
        self.aborted = True
        self.failed_stage = name
        self.results.append(
            StageResult(
                name=name,
                ok=False,
                duration_sec=elapsed,
                before=before,
                after=take_memory_snapshot(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )
        logger.error('%s 실패 (%.2fs): %s: %s', name, elapsed, type(exc).__name__, exc)


def _dump_oom_state(stage_name: str) -> None:
    """CUDA OOM 발생 시점의 메모리 상태를 덤프한다.

    Args:
        stage_name: OOM 이 발생한 스테이지 이름.
    """
    logger.error('=' * 60)
    logger.error('CUDA OUT OF MEMORY — %s', stage_name)
    snap = take_memory_snapshot()
    logger.error('프로세스 RSS          : %.1f MB', snap.rss_mb)
    logger.error('시스템 여유 메모리     : %.1f MB / %.1f MB',
                 snap.sys_available_mb, snap.sys_total_mb)
    logger.error('CUDA allocated        : %.1f MB', snap.cuda_allocated_mb)
    logger.error('CUDA reserved         : %.1f MB', snap.cuda_reserved_mb)
    logger.error('CUDA max allocated    : %.1f MB', snap.cuda_max_allocated_mb)
    if torch.cuda.is_available():
        try:
            logger.error('torch.cuda.memory_summary():\n%s', torch.cuda.memory_summary())
        except Exception as exc:
            logger.error('memory_summary() 호출 실패: %s', exc)
    logger.error('Orin NX 는 CPU/GPU 가 통합 메모리를 공유하므로, 다른 프로세스'
                 '(카메라 노드·웹 대시보드 등)를 내리고 다시 시도할 것.')
    logger.error('=' * 60)


def _explain_import_error(exc: ImportError) -> None:
    """ImportError 원인을 설명한다. flash_attn 인 경우 대처법을 안내한다.

    Args:
        exc: 발생한 ImportError.
    """
    message = str(exc)
    if 'flash_attn' in message.lower():
        logger.error('=' * 60)
        logger.error('flash_attn 이 없어서 모델 로딩에 실패했다.')
        logger.error('Jetson(aarch64)에는 flash-attn 휠이 없으므로 설치하지 말 것.')
        logger.error('InternVL3-1B 의 config.json 에서 아래 항목을 확인/수정한다:')
        logger.error('    "use_flash_attn": false')
        logger.error('그리고 remote code 쪽 attn_implementation 이 "eager" 인지 확인한다.')
        logger.error('(TICVLA 는 AutoModel.from_pretrained(use_flash_attn=False) 로 '
                     '호출하지만, config.json 값이 우선하는 경우가 있다.)')
        logger.error('=' * 60)
    else:
        logger.error('필요한 모듈을 찾지 못했다: %s', message)
        logger.error('가상환경(/home/soda/TIC-VLA/venvs/ticvla)으로 실행했는지 확인할 것.')


# ======================================================================
# STAGE 0: 환경 정보
# ======================================================================
def apply_torch_distributed_shim() -> dict[str, Any]:
    """torch.distributed 가 비활성화된 빌드에서 InternVL 원격 코드가 죽지 않게 보완한다.

    Jetson 용 PyTorch 는 USE_DISTRIBUTED=0 으로 빌드되어 있어
    `torch.distributed.is_initialized` / `get_rank` 자체가 존재하지 않는다.
    반면 InternVL3 의 modeling_internvl_chat.py 는 디버그 출력을 위해
    `torch.distributed.is_initialized()` 를 조건 없이 호출하므로 AttributeError 가 난다.

    분산 학습을 하지 않는 단일 프로세스 추론에서는 "초기화되지 않음"이 정답이므로,
    누락된 함수만 프로세스 안에서 채워 넣는다. torch 를 재설치하거나 버전을
    바꾸지 않으며, 이 프로세스 밖에는 아무 영향이 없다.

    Returns:
        패치 여부와 이유를 담은 딕셔너리.
    """
    import torch.distributed as dist

    info: dict[str, Any] = {
        'distributed_available': bool(dist.is_available()),
        'patched': False,
        'patched_attrs': [],
    }
    if dist.is_available():
        return info

    for name, func in (('is_initialized', lambda: False), ('get_rank', lambda: 0)):
        if not hasattr(dist, name):
            setattr(dist, name, func)
            info['patched_attrs'].append(name)

    if info['patched_attrs']:
        info['patched'] = True
        logger.warning(
            'torch.distributed 가 비활성화된 빌드다(USE_DISTRIBUTED=0). '
            'InternVL 원격 코드가 호출하는 %s 를 이 프로세스에 한해 보완했다.',
            ', '.join(info['patched_attrs']),
        )
        logger.warning(
            '영구 수정을 원하면 %s 의 modeling_internvl_chat.py 에서 '
            '`torch.distributed.is_initialized()` 앞에 '
            '`torch.distributed.is_available() and` 를 넣으면 된다 '
            '(해당 줄은 디버그 print 조건일 뿐이다).',
            'InternVL3-1B',
        )
    return info


def collect_environment_info() -> dict[str, Any]:
    """torch/CUDA/GPU/메모리 등 실행 환경 정보를 모은다.

    Returns:
        환경 정보 딕셔너리.
    """
    import torchvision

    total_mb, available_mb = read_system_memory_mb()
    cuda_available = torch.cuda.is_available()

    info: dict[str, Any] = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': cuda_available,
        'gpu_name': None,
        'device_count': 0,
        'sys_total_mb': total_mb,
        'sys_available_mb': available_mb,
        'jetson_torch_build': EXPECTED_TORCH_TAG in torch.__version__,
        # InternVL 원격 코드가 torch.distributed 를 무조건 호출하므로 먼저 보완한다.
        'torch_distributed': apply_torch_distributed_shim(),
    }

    if cuda_available:
        try:
            info['device_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
        except Exception as exc:
            logger.warning('GPU 정보 조회 실패: %s', exc)

    # Jetson 전용 빌드가 아니면 torch 를 갈아끼운 것이므로 반드시 경고한다.
    if not info['jetson_torch_build']:
        logger.warning('=' * 60)
        logger.warning('경고: torch.__version__("%s")에 "%s"가 없다.',
                       torch.__version__, EXPECTED_TORCH_TAG)
        logger.warning('Jetson 전용 PyTorch 빌드가 아닐 수 있다. '
                       'CUDA 가속이 동작하지 않거나 런타임 오류가 날 수 있으니 '
                       '가상환경(/home/soda/TIC-VLA/venvs/ticvla)을 확인할 것.')
        logger.warning('=' * 60)

    if not cuda_available:
        logger.error('torch.cuda.is_available() == False. '
                     'GPU 를 쓸 수 없으므로 STAGE 3 이후가 실패한다. '
                     '컨테이너/샌드박스 밖의 일반 셸에서 실행했는지 확인할 것.')

    return info


# ======================================================================
# STAGE 2: 체크포인트 로딩
# ======================================================================
def load_checkpoint_into_model(model: Any, ckpt_path: Path) -> dict[str, Any]:
    """체크포인트에서 VLM / action expert 가중치를 불러온다.

    `TICVLATester.__init__` 의 로딩 로직을 그대로 재현한다. 즉 Lightning 이 저장한
    "model.vlm." / "model.action_expert." 접두사를 제거한 뒤 각각
    strict=False 로 load_state_dict 를 호출한다.

    Args:
        model: TICVLA 인스턴스.
        ckpt_path: 체크포인트 파일 경로.

    Returns:
        키 개수와 missing/unexpected 통계를 담은 딕셔너리.

    Raises:
        FileNotFoundError: 체크포인트 파일이 없는 경우.
        KeyError: 체크포인트에 state_dict 가 없는 경우.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'체크포인트를 찾을 수 없다: {ckpt_path}')

    logger.info('체크포인트 로딩 중 (CPU 로 map): %s', ckpt_path)
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise KeyError(
            f'체크포인트에 "state_dict" 키가 없다. 최상위 키: {list(checkpoint)[:10]}')

    state_dict = checkpoint['state_dict']
    logger.info('체크포인트 전체 키 개수: %d', len(state_dict))

    # ── VLM: "model.vlm." 접두사 제거 후 model.vlm 에 로드 ──
    vlm_state_dict = {
        key[len('model.vlm.'):]: value
        for key, value in state_dict.items()
        if key.startswith('model.vlm.')
    }
    # ── Action expert: "model.action_expert." 접두사 제거 ──
    action_state_dict = {
        key[len('model.action_expert.'):]: value
        for key, value in state_dict.items()
        if key.startswith('model.action_expert.')
    }

    report: dict[str, Any] = {
        'ckpt_path': str(ckpt_path),
        'total_keys': len(state_dict),
        'vlm_keys_found': len(vlm_state_dict),
        'action_keys_found': len(action_state_dict),
    }

    if not vlm_state_dict:
        logger.warning('체크포인트에서 "model.vlm.*" 키를 하나도 찾지 못했다.')
    missing, unexpected = model.vlm.load_state_dict(vlm_state_dict, strict=False)
    report['vlm_missing'] = len(missing)
    report['vlm_unexpected'] = len(unexpected)
    report['vlm_missing_sample'] = [str(k) for k in list(missing)[:5]]
    report['vlm_unexpected_sample'] = [str(k) for k in list(unexpected)[:5]]
    logger.info('VLM 로드: found=%d, missing=%d, unexpected=%d',
                len(vlm_state_dict), len(missing), len(unexpected))
    if missing:
        logger.warning('  VLM missing 예시: %s', report['vlm_missing_sample'])
    if unexpected:
        logger.warning('  VLM unexpected 예시: %s', report['vlm_unexpected_sample'])

    if not action_state_dict:
        logger.warning('체크포인트에서 "model.action_expert.*" 키를 하나도 찾지 못했다.')
    missing, unexpected = model.action_expert.load_state_dict(
        action_state_dict, strict=False)
    report['action_missing'] = len(missing)
    report['action_unexpected'] = len(unexpected)
    report['action_missing_sample'] = [str(k) for k in list(missing)[:5]]
    report['action_unexpected_sample'] = [str(k) for k in list(unexpected)[:5]]
    logger.info('Action expert 로드: found=%d, missing=%d, unexpected=%d',
                len(action_state_dict), len(missing), len(unexpected))
    if missing:
        logger.warning('  Action missing 예시: %s', report['action_missing_sample'])
    if unexpected:
        logger.warning('  Action unexpected 예시: %s', report['action_unexpected_sample'])

    # 체크포인트 원본은 큰 메모리를 잡고 있으므로 즉시 참조를 끊는다.
    del state_dict, checkpoint, vlm_state_dict, action_state_dict
    return report


def audit_model_dtypes(model: Any) -> dict[str, Any]:
    """VLM 과 action expert 의 파라미터 dtype/device 를 확인한다.

    VLM 이 bfloat16 이 아니면 float32 로딩이 일어난 것이므로 경고한다.

    Args:
        model: TICVLA 인스턴스.

    Returns:
        dtype/device 정보 딕셔너리.
    """
    vlm_param = next(model.vlm.parameters())
    action_param = next(model.action_expert.parameters())

    info = {
        'vlm_dtype': str(vlm_param.dtype),
        'vlm_device': str(vlm_param.device),
        'action_expert_dtype': str(action_param.dtype),
        'action_expert_device': str(action_param.device),
        'vlm_param_count': sum(p.numel() for p in model.vlm.parameters()),
        'action_expert_param_count': sum(
            p.numel() for p in model.action_expert.parameters()),
    }

    if vlm_param.dtype != torch.bfloat16:
        logger.error('VLM 파라미터 dtype 이 %s 다. bfloat16 이어야 한다. '
                     'TICVLA_VLM 이 torch_dtype=bfloat16 으로 로드하는지 확인할 것.',
                     vlm_param.dtype)
    else:
        logger.info('VLM dtype 확인: bfloat16 (파라미터 %s개)',
                    f"{info['vlm_param_count']:,}")
    # action expert 는 저장소 기본 구현대로 float32 로 생성된다(작아서 영향 미미).
    logger.info('Action expert dtype: %s (파라미터 %s개)',
                info['action_expert_dtype'], f"{info['action_expert_param_count']:,}")
    return info


# ======================================================================
# STAGE 4: 추론 및 waypoint 분석
# ======================================================================
def select_frames(image_dir: Path, num_delayed: int) -> tuple[list[str], str]:
    """이미지 디렉터리에서 delayed 프레임들과 current 프레임을 고른다.

    파일명을 시간순(사전순)으로 정렬한 뒤 마지막 num_delayed+1 장을 사용한다.
    앞 num_delayed 장이 delayed, 마지막 1장이 current 다.

    Args:
        image_dir: 프레임이 들어 있는 디렉터리.
        num_delayed: 과거 프레임 장수.

    Returns:
        (delayed 이미지 경로 리스트, current 이미지 경로).

    Raises:
        NotADirectoryError: 디렉터리가 없는 경우.
        ValueError: 필요한 장수만큼 이미지가 없는 경우.
    """
    if not image_dir.is_dir():
        raise NotADirectoryError(f'이미지 디렉터리가 없다: {image_dir}')
    if num_delayed < 1:
        raise ValueError(f'--num-delayed 는 1 이상이어야 한다: {num_delayed}')

    frames = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    needed = num_delayed + 1
    if len(frames) < needed:
        raise ValueError(
            f'이미지가 부족하다: {image_dir} 에 {len(frames)}장 있는데 {needed}장 필요'
            f'(--num-delayed {num_delayed} → delayed {num_delayed} + current 1)'
        )

    selected = frames[-needed:]
    delayed = [str(p) for p in selected[:-1]]
    current = str(selected[-1])
    logger.info('delayed 프레임 %d장: %s', len(delayed), [Path(p).name for p in delayed])
    logger.info('current 프레임: %s', Path(current).name)
    return delayed, current


def analyze_waypoints(waypoints: torch.Tensor) -> dict[str, Any]:
    """waypoint 텐서를 float32 로 변환해 통계와 스텝별 값을 계산한다.

    액션 헤드 출력 (B, T, 2) 의 각 스텝은 현재 프레임 기준 상대 변위 [dx, dy]다.
    x 는 전진(+), y 는 좌측(+)이며 스텝 간격은 0.1초(10Hz)다.

    Args:
        waypoints: 모델이 낸 (B, T, 2) 텐서.

    Returns:
        shape/dtype/device, 스텝별 값, 통계, NaN/Inf 여부를 담은 딕셔너리.

    Raises:
        ValueError: 텐서 차원이 (B, T, 2) 가 아닌 경우.
    """
    if waypoints.dim() != 3 or waypoints.shape[-1] != 2:
        raise ValueError(f'waypoints 형태가 (B, T, 2) 가 아니다: {tuple(waypoints.shape)}')

    raw_shape = tuple(int(v) for v in waypoints.shape)
    raw_dtype = str(waypoints.dtype)
    raw_device = str(waypoints.device)

    # bfloat16 은 numpy 로 바로 못 가므로 float32 로 올린 뒤 CPU 로 내린다.
    values = waypoints.detach().to(torch.float32).cpu()
    has_nan = bool(torch.isnan(values).any().item())
    has_inf = bool(torch.isinf(values).any().item())

    batch0 = values[0]  # (T, 2)
    num_steps = int(batch0.shape[0])

    steps: list[dict[str, float]] = []
    cumulative_distance = 0.0
    prev_x, prev_y = 0.0, 0.0
    for index in range(num_steps):
        dx = float(batch0[index, 0].item())
        dy = float(batch0[index, 1].item())
        # 누적거리는 원점(현재 위치)에서 출발해 각 waypoint 를 잇는 경로 길이다.
        cumulative_distance += math.hypot(dx - prev_x, dy - prev_y)
        prev_x, prev_y = dx, dy
        steps.append({
            'step': index,
            't_sec': round(index * STEP_DT_SEC, 2),
            'dx': dx,
            'dy': dy,
            'cumulative_distance_m': cumulative_distance,
        })

    dx_col = batch0[:, 0]
    dy_col = batch0[:, 1]
    final_dx = float(batch0[-1, 0].item())
    final_dy = float(batch0[-1, 1].item())

    return {
        'shape': raw_shape,
        'dtype': raw_dtype,
        'device': raw_device,
        'num_steps': num_steps,
        'step_dt_sec': STEP_DT_SEC,
        'steps': steps,
        'stats': {
            'dx_min': float(dx_col.min().item()),
            'dx_max': float(dx_col.max().item()),
            'dx_mean': float(dx_col.mean().item()),
            'dy_min': float(dy_col.min().item()),
            'dy_max': float(dy_col.max().item()),
            'dy_mean': float(dy_col.mean().item()),
        },
        'final': {
            'step': num_steps - 1,
            't_sec': round((num_steps - 1) * STEP_DT_SEC, 2),
            'dx': final_dx,
            'dy': final_dy,
            # 누적 변위: 원점에서 마지막 waypoint 까지의 직선 거리와 방향.
            'displacement_m': math.hypot(final_dx, final_dy),
            'heading_deg': math.degrees(math.atan2(final_dy, final_dx)),
            'path_length_m': cumulative_distance,
        },
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


def run_inference_loop(
    model: Any,
    delayed_paths: list[str],
    current_path: str,
    instruction: str,
    repeat: int,
) -> dict[str, Any]:
    """추론을 repeat 회 반복하고 각 회차의 결과와 소요 시간을 모은다.

    첫 회차는 CUDA 커널 초기화 등이 섞이므로 warmup 으로 따로 표기하고
    평균/최소/최대 통계에서 제외한다.

    Args:
        model: cuda 로 올려둔 TICVLA 인스턴스.
        delayed_paths: 과거 프레임 경로 리스트.
        current_path: 현재 프레임 경로.
        instruction: 자연어 지시문.
        repeat: 추론 반복 횟수.

    Returns:
        회차별 결과, 타이밍 통계, 마지막 회차의 waypoint 분석을 담은 딕셔너리.

    Raises:
        ValueError: repeat 이 1 미만인 경우.
    """
    if repeat < 1:
        raise ValueError(f'--repeat 는 1 이상이어야 한다: {repeat}')

    # robot_state = [vx, vy, yaw_speed, dx, dy]. 정지 상태를 가정해 0 으로 둔다.
    # 모델 파라미터와 dtype/device 를 맞춰서 넣는다.
    robot_state = torch.zeros(5, dtype=torch.bfloat16, device=model.device)
    logger.info('robot_state: %s (dtype=%s, device=%s)',
                robot_state.tolist(), robot_state.dtype, robot_state.device)

    iterations: list[dict[str, Any]] = []
    last_response = ''
    last_prompt = ''
    last_analysis: dict[str, Any] = {}

    for index in range(repeat):
        is_warmup = index == 0
        label = 'warmup' if is_warmup else f'run {index}'
        logger.info('추론 %d/%d (%s) 시작', index + 1, repeat, label)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()

        # predict() 내부에도 inference_mode 가 걸려 있지만, 전처리까지 포함해
        # 확실히 grad 를 만들지 않도록 바깥에서도 감싼다.
        with torch.inference_mode():
            response, waypoints, prompt = model.predict(
                delayed_image_paths=delayed_paths,
                current_image_path=current_path,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=0.0,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        analysis = analyze_waypoints(waypoints)
        last_response, last_prompt, last_analysis = response, prompt, analysis

        iterations.append({
            'index': index,
            'is_warmup': is_warmup,
            'duration_sec': elapsed,
            'response': response,
            'final_displacement_m': analysis['final']['displacement_m'],
            'final_heading_deg': analysis['final']['heading_deg'],
            'has_nan': analysis['has_nan'],
            'has_inf': analysis['has_inf'],
            'memory_after': asdict(take_memory_snapshot()),
        })
        logger.info('추론 %d/%d (%s) 완료: %.3fs', index + 1, repeat, label, elapsed)

    measured = [it['duration_sec'] for it in iterations if not it['is_warmup']]
    timing: dict[str, Any] = {
        'repeat': repeat,
        'warmup_sec': iterations[0]['duration_sec'],
        'measured_count': len(measured),
        'mean_sec': (sum(measured) / len(measured)) if measured else None,
        'min_sec': min(measured) if measured else None,
        'max_sec': max(measured) if measured else None,
    }
    if not measured:
        logger.warning('--repeat 가 1이라 warmup 을 제외한 통계가 없다. '
                       '평균/최소/최대를 보려면 --repeat 2 이상으로 실행할 것.')

    return {
        'iterations': iterations,
        'timing': timing,
        'response': last_response,
        'prompt': last_prompt,
        'waypoints': last_analysis,
    }


# ======================================================================
# 결과 출력 (print 는 이 영역의 표/보고서에서만 사용한다)
# ======================================================================
def _rule(char: str = '=') -> str:
    """구분선 문자열을 만든다.

    Args:
        char: 반복할 문자.

    Returns:
        RULE_WIDTH 길이의 구분선.
    """
    return char * RULE_WIDTH


def _describe_distributed(info: dict[str, Any]) -> str:
    """torch.distributed 상태와 shim 적용 여부를 한 줄 문자열로 만든다.

    Args:
        info: apply_torch_distributed_shim() 결과.

    Returns:
        표에 넣을 설명 문자열.
    """
    if info['distributed_available']:
        return '사용 가능 (패치 불필요)'
    if info['patched']:
        return ('비활성 빌드 → ' + ', '.join(info['patched_attrs']) + ' 보완 적용')
    return '비활성 빌드 (보완할 항목 없음)'


def print_environment_table(env: dict[str, Any]) -> None:
    """STAGE 0 환경 정보를 표로 출력한다.

    Args:
        env: collect_environment_info() 결과.
    """
    print()
    print(_rule())
    print('STAGE 0: 환경 정보')
    print(_rule())
    rows = [
        ('Python', env['python_version']),
        ('Platform', f"{env['platform']} ({env['machine']})"),
        ('torch', env['torch_version']),
        ('torchvision', env['torchvision_version']),
        ('torch CUDA', str(env['torch_cuda_version'])),
        ('CUDA available', str(env['cuda_available'])),
        ('GPU', str(env['gpu_name'])),
        ('통합 메모리 총량', f"{env['sys_total_mb']:.0f} MB"),
        ('통합 메모리 여유', f"{env['sys_available_mb']:.0f} MB"),
        ('Jetson torch 빌드', 'OK' if env['jetson_torch_build']
         else f'경고: "{EXPECTED_TORCH_TAG}" 없음'),
        ('torch.distributed', _describe_distributed(env['torch_distributed'])),
    ]
    for label, value in rows:
        print(f'  {label:<20} : {value}')


def print_stage_memory_table(results: list[StageResult]) -> None:
    """스테이지별 메모리/시간 측정 결과를 표로 출력한다.

    Args:
        results: StageTracker 가 모은 결과 목록.
    """
    print()
    print(_rule())
    print('스테이지별 메모리 / 시간 측정 (MB, 초)')
    print(_rule())
    header = (f"{'STAGE':<34}{'상태':<6}{'시간':>7}"
              f"{'RSS전':>9}{'RSS후':>9}{'여유후':>9}"
              f"{'CUDA alloc':>12}{'CUDA resv':>11}{'CUDA peak':>11}")
    print(header)
    print('-' * len(header))
    for result in results:
        state = 'OK' if result.ok else 'FAIL'
        print(
            f'{result.name:<34}{state:<6}{result.duration_sec:>7.2f}'
            f'{result.before.rss_mb:>9.0f}{result.after.rss_mb:>9.0f}'
            f'{result.after.sys_available_mb:>9.0f}'
            f'{result.after.cuda_allocated_mb:>12.1f}'
            f'{result.after.cuda_reserved_mb:>11.1f}'
            f'{result.after.cuda_max_allocated_mb:>11.1f}'
        )
        if not result.ok:
            print(f'    └─ {result.error_type}: {result.error_message}')


def print_checkpoint_table(report: dict[str, Any]) -> None:
    """체크포인트 로딩 결과(키 개수)를 표로 출력한다.

    Args:
        report: load_checkpoint_into_model() 결과.
    """
    print()
    print(_rule())
    print('STAGE 2: checkpoint 키 매칭 결과')
    print(_rule())
    print(f"  체크포인트 전체 키      : {report['total_keys']}")
    print(f"  VLM 키 발견             : {report['vlm_keys_found']}")
    print(f"    missing / unexpected  : {report['vlm_missing']} / "
          f"{report['vlm_unexpected']}")
    print(f"  Action expert 키 발견   : {report['action_keys_found']}")
    print(f"    missing / unexpected  : {report['action_missing']} / "
          f"{report['action_unexpected']}")


def print_dtype_table(dtypes: dict[str, Any]) -> None:
    """모델 dtype/device 확인 결과를 출력한다.

    Args:
        dtypes: audit_model_dtypes() 결과.
    """
    print()
    print(_rule())
    print('STAGE 3: 모델 dtype / device 확인')
    print(_rule())
    print(f"  VLM            : {dtypes['vlm_dtype']} @ {dtypes['vlm_device']} "
          f"({dtypes['vlm_param_count']:,} params)")
    print(f"  Action expert  : {dtypes['action_expert_dtype']} @ "
          f"{dtypes['action_expert_device']} "
          f"({dtypes['action_expert_param_count']:,} params)")


def print_timing_table(timing: dict[str, Any]) -> None:
    """추론 시간 통계를 출력한다.

    Args:
        timing: run_inference_loop() 의 timing 딕셔너리.
    """
    print()
    print(_rule())
    print('STAGE 4: 추론 시간 (초)')
    print(_rule())
    print(f"  warmup (1회차, 제외)   : {timing['warmup_sec']:.3f}")
    if timing['mean_sec'] is None:
        print('  측정 회차 없음 (--repeat 2 이상 필요)')
        return
    print(f"  측정 회차              : {timing['measured_count']}회")
    print(f"  평균                   : {timing['mean_sec']:.3f}")
    print(f"  최소 / 최대            : {timing['min_sec']:.3f} / "
          f"{timing['max_sec']:.3f}")


def print_inference_report(inference: dict[str, Any], instruction: str) -> None:
    """VLM 응답, 프롬프트, waypoint 전체 표를 사람이 읽기 좋게 출력한다.

    Args:
        inference: run_inference_loop() 결과.
        instruction: 사용한 자연어 지시문.
    """
    analysis = inference['waypoints']

    print()
    print(_rule())
    print('추론 결과 1/4 — VLM 생성 텍스트 (response)')
    print(_rule())
    print(inference['response'] or '(빈 응답)')

    print()
    print(_rule())
    print('추론 결과 2/4 — 모델 입력 프롬프트 (prompt)')
    print(_rule())
    print(inference['prompt'] or '(빈 프롬프트)')

    print()
    print(_rule())
    print('추론 결과 3/4 — waypoints 상세')
    print(_rule())
    print(f"  instruction : {instruction}")
    print(f"  shape       : {analysis['shape']}")
    print(f"  dtype       : {analysis['dtype']}")
    print(f"  device      : {analysis['device']}")
    print('  좌표계      : 현재 프레임 기준 상대 변위, x 전진(+), y 좌측(+), 단위 m')
    print(f"  시간 간격   : {analysis['step_dt_sec']}s (10Hz), "
          f"총 {analysis['num_steps']}스텝")
    print()

    header = f"  {'step':>4} {'t(s)':>6} {'dx(m)':>10} {'dy(m)':>10} {'누적거리(m)':>13}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for row in analysis['steps']:
        marker = ' ←' if row['step'] == HIGHLIGHT_STEP else ''
        print(f"  {row['step']:>4} {row['t_sec']:>6.1f} {row['dx']:>10.4f} "
              f"{row['dy']:>10.4f} {row['cumulative_distance_m']:>13.4f}{marker}")

    stats = analysis['stats']
    print()
    print(f"  dx  min / max / mean : {stats['dx_min']:.4f} / "
          f"{stats['dx_max']:.4f} / {stats['dx_mean']:.4f}")
    print(f"  dy  min / max / mean : {stats['dy_min']:.4f} / "
          f"{stats['dy_max']:.4f} / {stats['dy_mean']:.4f}")

    final = analysis['final']
    print()
    print(f"  최종 스텝 (step {final['step']}, t={final['t_sec']:.1f}s)")
    print(f"    누적 변위 (원점→마지막) : {final['displacement_m']:.4f} m")
    print(f"    방향각                  : {final['heading_deg']:.2f}° "
          f"(0°=전진, +=좌측)")
    print(f"    경로 길이 (누적거리)    : {final['path_length_m']:.4f} m")
    print()
    nan_inf = ('있음 — 결과를 신뢰할 수 없다'
               if (analysis['has_nan'] or analysis['has_inf']) else '없음')
    print(f"  NaN / Inf : {nan_inf} "
          f"(NaN={analysis['has_nan']}, Inf={analysis['has_inf']})")

    # 3초 시점 강조
    print()
    print(_rule())
    print(f'추론 결과 4/4 — 3초 시점 (step {HIGHLIGHT_STEP}) 강조')
    print(_rule())
    steps = analysis['steps']
    if HIGHLIGHT_STEP < len(steps):
        row = steps[HIGHLIGHT_STEP]
        distance = math.hypot(row['dx'], row['dy'])
        heading = math.degrees(math.atan2(row['dy'], row['dx']))
        print(f"  ★ step {row['step']} (t = {row['t_sec']:.1f}s)")
        print(f"      dx = {row['dx']:+.4f} m  (전진 방향)")
        print(f"      dy = {row['dy']:+.4f} m  (좌측 방향)")
        print(f"      현재 위치로부터의 거리 = {distance:.4f} m")
        print(f"      방향각 = {heading:+.2f}°")
        print(f"      누적거리 = {row['cumulative_distance_m']:.4f} m")
    else:
        print(f'  step {HIGHLIGHT_STEP} 가 없다 '
              f"(총 {analysis['num_steps']}스텝). action_horizon_steps 확인 필요.")


# ======================================================================
# CLI 및 메인
# ======================================================================
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: 인자 리스트. None 이면 sys.argv 를 사용한다.

    Returns:
        파싱된 Namespace.
    """
    parser = argparse.ArgumentParser(
        description='TIC-VLA Jetson Orin NX 메모리/추론 스모크 테스트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-path', type=Path,
                        default=Path('/home/soda/TIC-VLA/models/InternVL3-1B'),
                        help='InternVL3-1B 베이스 모델 경로')
    parser.add_argument('--ckpt-path', type=Path,
                        default=Path('/home/soda/TIC-VLA/checkpoints/'
                                     'TIC-VLA-model.ckpt'),
                        help='TIC-VLA 체크포인트 경로')
    parser.add_argument('--image-dir', type=Path,
                        default=Path('/home/soda/TIC-VLA/logs/test_seq'),
                        help='추론에 사용할 프레임 디렉터리')
    parser.add_argument('--num-delayed', type=int, default=4,
                        help='과거(delayed) 프레임 장수')
    parser.add_argument('--instruction', type=str,
                        default='Move forward safely and efficiently.',
                        help='자연어 지시문')
    parser.add_argument('--repeat', type=int, default=3,
                        help='추론 반복 횟수 (1회차는 warmup 으로 별도 표기)')
    parser.add_argument('--output-json', type=Path,
                        default=Path('/home/soda/TIC-VLA/logs/'
                                     'smoke_test_result.json'),
                        help='결과 JSON 저장 경로')
    parser.add_argument('--skip-infer', action='store_true',
                        help='추론 없이 로딩과 메모리 측정만 수행')
    return parser.parse_args(argv)


def to_jsonable(value: Any) -> Any:
    """numpy/torch 타입이 섞인 값을 JSON 직렬화 가능한 형태로 바꾼다.

    Args:
        value: 변환할 값.

    Returns:
        float/int/str/list/dict 로만 이루어진 값.
    """
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().to(torch.float32).cpu().tolist())
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # NaN/Inf 는 표준 JSON 이 아니므로 문자열로 남겨 정보 손실을 막는다.
        return value if math.isfinite(value) else str(value)
    # numpy 스칼라 등: item() 을 가진 객체는 파이썬 스칼라로 내린다.
    if hasattr(value, 'item'):
        try:
            return to_jsonable(value.item())
        except Exception:
            return str(value)
    return str(value)


def save_result_json(payload: dict[str, Any], output_path: Path) -> None:
    """결과 전체를 JSON 파일로 저장한다.

    Args:
        payload: 저장할 결과 딕셔너리.
        output_path: 저장 경로. 상위 디렉터리는 자동 생성한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
    logger.info('결과 JSON 저장: %s', output_path)


def main(argv: Optional[list[str]] = None) -> int:
    """스모크 테스트를 실행한다.

    Args:
        argv: 명령행 인자. None 이면 sys.argv 를 사용한다.

    Returns:
        프로세스 종료 코드. 모든 스테이지가 성공하면 0, 실패가 있으면 1.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    args = parse_args(argv)

    tracker = StageTracker()
    payload: dict[str, Any] = {
        'args': {
            'model_path': str(args.model_path),
            'ckpt_path': str(args.ckpt_path),
            'image_dir': str(args.image_dir),
            'num_delayed': args.num_delayed,
            'instruction': args.instruction,
            'repeat': args.repeat,
            'skip_infer': args.skip_infer,
        },
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # ── STAGE 0: 환경 정보 ─────────────────────────────────────────────
    ok, env = tracker.run('STAGE 0: 환경 정보', collect_environment_info)
    if ok and env is not None:
        payload['environment'] = env
        print_environment_table(env)

    # 경로 검증은 무거운 로딩 전에 끝낸다.
    def _validate_paths() -> None:
        """모델/체크포인트/이미지 경로가 실제로 존재하는지 검증한다."""
        if not args.model_path.is_dir():
            raise NotADirectoryError(f'모델 경로가 없다: {args.model_path}')
        if not args.ckpt_path.is_file():
            raise FileNotFoundError(f'체크포인트가 없다: {args.ckpt_path}')
        if not args.skip_infer and not args.image_dir.is_dir():
            raise NotADirectoryError(f'이미지 디렉터리가 없다: {args.image_dir}')
        logger.info('경로 검증 통과')

    tracker.run('경로 검증', _validate_paths)

    # ── STAGE 1: 모델 구조 생성 ────────────────────────────────────────
    def _build_model() -> Any:
        """TICVLA 를 인스턴스화한다 (내부에서 bfloat16 으로 VLM 로드)."""
        from ticvla.models.ticvla import TICVLA  # 지연 import: 실패를 이 스테이지에 귀속

        return TICVLA(model_path=str(args.model_path), action_horizon_steps=30)

    ok, model = tracker.run('STAGE 1: 모델 구조 생성', _build_model)

    # ── STAGE 2: 체크포인트 로딩 ───────────────────────────────────────
    ckpt_report: Optional[dict[str, Any]] = None
    if model is not None:
        ok, ckpt_report = tracker.run(
            'STAGE 2: checkpoint 로딩',
            lambda: load_checkpoint_into_model(model, args.ckpt_path),
        )
        if ok and ckpt_report is not None:
            payload['checkpoint'] = ckpt_report
            print_checkpoint_table(ckpt_report)

    # ── STAGE 3: cuda 이동 + eval ──────────────────────────────────────
    def _move_to_cuda() -> dict[str, Any]:
        """모델을 CUDA 로 옮기고 평가 모드로 전환한 뒤 dtype 을 확인한다."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA 를 사용할 수 없다. Jetson 에서 nvidia 런타임이 보이는 '
                '일반 셸에서 실행했는지, venv 가 Jetson 전용 torch 인지 확인할 것.'
            )
        model.to('cuda')
        model.eval()
        torch.cuda.synchronize()
        return audit_model_dtypes(model)

    dtypes: Optional[dict[str, Any]] = None
    if model is not None:
        ok, dtypes = tracker.run('STAGE 3: .to(cuda) + .eval()', _move_to_cuda)
        if ok and dtypes is not None:
            payload['model_dtypes'] = dtypes
            print_dtype_table(dtypes)

    # ── STAGE 4: 추론 ──────────────────────────────────────────────────
    if args.skip_infer:
        logger.info('--skip-infer 지정: STAGE 4(추론)를 수행하지 않는다.')
        payload['inference'] = None
    elif model is not None:
        def _infer() -> dict[str, Any]:
            """프레임을 고르고 추론을 repeat 회 수행한다."""
            delayed_paths, current_path = select_frames(
                args.image_dir, args.num_delayed)
            result = run_inference_loop(
                model=model,
                delayed_paths=delayed_paths,
                current_path=current_path,
                instruction=args.instruction,
                repeat=args.repeat,
            )
            result['delayed_image_paths'] = delayed_paths
            result['current_image_path'] = current_path
            return result

        ok, inference = tracker.run(f'STAGE 4: 추론 x{args.repeat}', _infer)
        if ok and inference is not None:
            payload['inference'] = inference
            print_timing_table(inference['timing'])
            print_inference_report(inference, args.instruction)

    # ── 요약 및 저장 ───────────────────────────────────────────────────
    payload['stages'] = [asdict(result) for result in tracker.results]
    payload['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    payload['all_ok'] = all(result.ok for result in tracker.results)

    print_stage_memory_table(tracker.results)

    try:
        save_result_json(payload, args.output_json)
    except Exception as exc:
        logger.error('결과 JSON 저장 실패: %s', exc)

    print()
    print(_rule())
    if payload['all_ok']:
        print('결과: 모든 스테이지 성공')
    else:
        print(f'결과: 실패한 스테이지가 있다 → {tracker.failed_stage}')
    print(f'JSON: {args.output_json}')
    print(_rule())

    return 0 if payload['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
