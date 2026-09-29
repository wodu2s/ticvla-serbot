#!/usr/bin/env python3
"""합본 TIC-VLA 체크포인트 진단 + action head 로더.

왜 필요한가 (TRAIN_PROMPTS_v2.md §1-B):
    레포 ``ticvla/training/train.py`` 의 stage 2(``TICVLAActionLightningModule``)는
    ``self.model.load_vlm_checkpoint(vlm_checkpoint)`` 만 호출한다(train.py:163).
    그리고 ``TICVLA.load_vlm_checkpoint`` 는 ``model.vlm.*`` / ``model.vlm_model.vlm.*``
    키만 골라 ``self.vlm_model`` 에 넣는다(ticvla.py:268-328).
    ``action_expert`` 는 아무도 로드하지 않으므로 **매 학습마다 랜덤 초기화**된다.
    즉 레포 train.py 를 그대로 쓰면 파인튜닝이 아니라 scratch 학습이 된다.

    이 파일은 그 구멍을 우리 쪽에서 막는다. third_party 레포는 수정하지 않는다.

역할 2개:
    1. 진단 CLI (``--inspect``): 합본 체크포인트의 실제 키 구조를 눈으로 확인하고,
       "이 파일을 vlm_checkpoint 인자에 그대로 넣어도 되는가"를 판정한다(§1-C).
    2. 로더 함수 (``load_action_expert``): 다른 스크립트(train_wrapper.py)가 import 해서
       사전학습된 action head 가중치를 이어받는다.

사용 예:
    python3 train/01_load_full_ckpt.py --inspect
    python3 train/01_load_full_ckpt.py --inspect ~/TIC-VLA/checkpoints/TIC-VLA-model.ckpt
    python3 train/01_load_full_ckpt.py --inspect <ckpt> --action-expert-only

import 예 (파일명이 숫자로 시작해 일반 import 가 안 되므로 importlib 를 쓴다):

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "load_full_ckpt", "/home/soda/TIC-VLA/train/01_load_full_ckpt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.load_action_expert(model, ckpt_path, strict=True)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
# 경로·프리픽스 상수
# ──────────────────────────────────────────────────────────────────────

#: ~/TIC-VLA
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 읽기 전용으로 취급하는 업스트림 레포.
THIRD_PARTY_REPO = PROJECT_ROOT / "third_party" / "TIC-VLA"

DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "TIC-VLA-model.ckpt"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "InternVL3-1B"

#: stage1(VLM 단독) 산출물 프리픽스.
VLM_DIRECT_PREFIX = "model.vlm."
#: stage2(TICVLA 전체) 산출물의 VLM 프리픽스.
VLM_NESTED_PREFIX = "model.vlm_model.vlm."
#: 우리가 이어받아야 하는 action head 프리픽스.
ACTION_PREFIX = "model.action_expert."


# ──────────────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """섹션 제목을 찍는다."""
    print()
    print("═" * 72)
    print(f" {title}")
    print("═" * 72)


def _fmt_shape(t: Any) -> str:
    """텐서 shape 을 문자열로. 텐서가 아니면 타입명."""
    if isinstance(t, torch.Tensor):
        return "×".join(str(d) for d in t.shape) or "scalar"
    return type(t).__name__


# ──────────────────────────────────────────────────────────────────────
# 체크포인트 읽기
# ──────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str | Path) -> dict[str, Any]:
    """체크포인트를 CPU 로 읽는다.

    Args:
        ckpt_path: .ckpt 경로.

    Returns:
        torch.load 결과 dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"체크포인트가 없다: {path}")

    # Lightning ckpt 는 hyper_parameters 등 텐서 아닌 객체를 담으므로 weights_only=False.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # torch < 2.0 호환.
        return torch.load(path, map_location="cpu")


def extract_state_dict(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """체크포인트에서 state_dict 를 꺼낸다.

    레포의 ``load_vlm_checkpoint`` 는 ``checkpoint.get("state_dict", {})`` 를 쓴다.
    최상위에 state_dict 키가 없으면 레포는 빈 dict 를 받아 ValueError 를 낸다.
    진단에서 그 사실을 그대로 드러내야 하므로 존재 여부를 함께 돌려준다.

    Args:
        checkpoint: load_checkpoint() 결과.

    Returns:
        (state_dict, "state_dict" 키가 실제로 있었는지).
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], True
    # 폴백: 최상위가 곧 state_dict 인 형태(레포는 이 경우를 지원하지 않는다).
    if isinstance(checkpoint, dict):
        tensors = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        return tensors, False
    return {}, False


def classify_keys(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """키를 프리픽스별로 분류한다.

    Args:
        state_dict: 체크포인트 state_dict.

    Returns:
        버킷 이름 → 키 목록. ``other`` 는 조용히 넘기지 않고 전부 담는다.
    """
    buckets: dict[str, list[str]] = {
        "vlm_direct": [],       # model.vlm.*
        "vlm_nested": [],       # model.vlm_model.vlm.*
        "vlm_model_other": [],  # model.vlm_model.* 인데 .vlm. 이 아닌 것
        "action_expert": [],    # model.action_expert.*
        "other": [],
    }
    for key in state_dict:
        if key.startswith(VLM_DIRECT_PREFIX):
            buckets["vlm_direct"].append(key)
        elif key.startswith(VLM_NESTED_PREFIX):
            buckets["vlm_nested"].append(key)
        elif key.startswith("model.vlm_model."):
            buckets["vlm_model_other"].append(key)
        elif key.startswith(ACTION_PREFIX):
            buckets["action_expert"].append(key)
        else:
            buckets["other"].append(key)
    return buckets


def strip_prefix(
    state_dict: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """주어진 프리픽스를 가진 키만 골라 프리픽스를 벗긴다.

    Args:
        state_dict: 원본 state_dict.
        prefix: 벗길 프리픽스.

    Returns:
        프리픽스가 제거된 새 dict.
    """
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


# ──────────────────────────────────────────────────────────────────────
# 파라미터 요약 (로드 전후 비교용)
# ──────────────────────────────────────────────────────────────────────

def summarize_params(module: nn.Module) -> dict[str, float]:
    """모듈 전체 파라미터의 (mean, std, norm) 요약.

    로드 전후로 이 값이 바뀌지 않으면 가중치가 실제로 안 들어간 것이다.

    Args:
        module: 대상 모듈.

    Returns:
        {"num_params", "mean", "std", "norm"}.
    """
    flats = [
        p.detach().reshape(-1).float()
        for p in module.parameters()
        if p is not None and p.numel() > 0
    ]
    if not flats:
        return {"num_params": 0, "mean": 0.0, "std": 0.0, "norm": 0.0}
    all_params = torch.cat(flats)
    return {
        "num_params": int(all_params.numel()),
        "mean": float(all_params.mean()),
        "std": float(all_params.std()),
        "norm": float(all_params.norm()),
    }


def _print_summary(label: str, summary: dict[str, float]) -> None:
    """파라미터 요약 한 줄 출력."""
    print(
        f"  {label:<12} n={summary['num_params']:,}  "
        f"mean={summary['mean']:+.6f}  "
        f"std={summary['std']:.6f}  "
        f"norm={summary['norm']:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────
# ★ 핵심: action head 로더
# ──────────────────────────────────────────────────────────────────────

def _resolve_action_expert(model: nn.Module) -> nn.Module:
    """TICVLA 인스턴스든 ActionExpert 든 action head 모듈을 돌려준다.

    Args:
        model: TICVLA 또는 ActionExpert.

    Returns:
        action head 모듈.
    """
    action_expert = getattr(model, "action_expert", None)
    return action_expert if action_expert is not None else model


def load_action_expert(
    model: nn.Module,
    ckpt_path: str | Path,
    strict: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """합본 체크포인트에서 action head 가중치를 로드한다.

    레포 ``load_vlm_checkpoint`` 가 손대지 않는 ``model.action_expert.*`` 를
    직접 골라 넣는다(§1-B).

    Args:
        model: TICVLA 인스턴스(또는 ActionExpert 모듈).
        ckpt_path: 합본 체크포인트 경로.
        strict: True 면 missing/unexpected 가 하나라도 있으면 예외.
        verbose: 로드 전후 파라미터 요약을 출력.

    Returns:
        {"missing", "unexpected", "loaded", "before", "after", "changed"}.

    Raises:
        ValueError: 체크포인트에 action_expert 키가 없을 때.
        RuntimeError: strict=True 인데 missing/unexpected 가 있을 때.
    """
    action_expert = _resolve_action_expert(model)

    checkpoint = load_checkpoint(ckpt_path)
    state_dict, _ = extract_state_dict(checkpoint)
    action_state = strip_prefix(state_dict, ACTION_PREFIX)

    if not action_state:
        raise ValueError(
            f"체크포인트에 '{ACTION_PREFIX}*' 키가 없다: {ckpt_path}\n"
            f"  --inspect 로 실제 프리픽스를 먼저 확인하라."
        )

    # shape 이 어긋나면 load_state_dict 가 strict=False 여도 예외를 낸다.
    # 원인이 대개 action_horizon_steps 불일치라서 먼저 잡아 알려준다.
    mismatches = [
        (key, _fmt_shape(action_state[key]), _fmt_shape(tensor))
        for key, tensor in action_expert.state_dict().items()
        if key in action_state
        and _fmt_shape(action_state[key]) != _fmt_shape(tensor)
    ]
    if mismatches:
        detail = "\n".join(
            f"    · {key}: ckpt={ckpt_shape} vs model={model_shape}"
            for key, ckpt_shape, model_shape in mismatches
        )
        raise RuntimeError(
            "action head shape 불일치 — 사전학습 head 를 이어받을 수 없다.\n"
            f"{detail}\n"
            "  action_chunk_embed.weight 가 걸렸다면 action_horizon_steps 가\n"
            "  사전학습(30)과 다른 것이다. 다른 지평으로 학습하려면 head 를\n"
            "  이어받지 못한다는 사실을 인지하고 진행하라 (§1-B)."
        )

    before = summarize_params(action_expert)
    # 전체 통계만 보면 랜덤 초기화와 사전학습 가중치의 mean/std 가 비슷해 구분이 안 된다.
    # 텐서별로 값이 실제로 바뀌었는지 세는 편이 확실하다.
    before_tensors = {
        name: tensor.detach().clone()
        for name, tensor in action_expert.state_dict().items()
    }

    result = action_expert.load_state_dict(action_state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    after = summarize_params(action_expert)

    num_changed = 0
    max_delta = 0.0
    unchanged_names: list[str] = []
    for name, tensor in action_expert.state_dict().items():
        old = before_tensors.get(name)
        if old is None or old.shape != tensor.shape:
            continue
        delta = float((tensor.detach().float() - old.float()).abs().max())
        max_delta = max(max_delta, delta)
        if delta > 0.0:
            num_changed += 1
        else:
            unchanged_names.append(name)
    num_tensors = len(before_tensors)
    changed = num_changed > 0

    if verbose:
        print(f"[action head] {ckpt_path}")
        print(f"  체크포인트 키 {len(action_state)}개 → load_state_dict(strict=False)")
        _print_summary("before", before)
        _print_summary("after", after)
        print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
        print(f"  값이 바뀐 텐서 {num_changed}/{num_tensors}개  "
              f"최대 절대변화={max_delta:.6f}")
        if unchanged_names:
            # 공개 체크포인트의 action head LayerNorm gamma 는 전부 정확히 1.0 이다
            # (bf16 에서 1.0 근처 미세 업데이트가 반올림으로 사라진 결과로 보인다).
            # 랜덤 초기화도 1.0 이라 "안 바뀜"으로 나오는 게 정상이다.
            print(f"  변화 없는 텐서 {len(unchanged_names)}개: "
                  f"{', '.join(unchanged_names[:8])}"
                  f"{' …' if len(unchanged_names) > 8 else ''}")
        for key in missing[:10]:
            print(f"    - missing: {key}")
        for key in unexpected[:10]:
            print(f"    - unexpected: {key}")
        if not changed:
            print("  ★ 경고: 가중치가 하나도 바뀌지 않았다. 로드가 무효다.")

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"action head 로드 실패 (strict=True): "
            f"missing={len(missing)}, unexpected={len(unexpected)}\n"
            f"  missing={missing[:10]}\n  unexpected={unexpected[:10]}"
        )

    return {
        "missing": missing,
        "unexpected": unexpected,
        "loaded": len(action_state) - len(unexpected),
        "before": before,
        "after": after,
        "changed": changed,
        "num_changed_tensors": num_changed,
        "num_tensors": num_tensors,
        "max_abs_delta": max_delta,
        "unchanged": unchanged_names,
    }


# ──────────────────────────────────────────────────────────────────────
# 진단용 참조 모델 만들기
# ──────────────────────────────────────────────────────────────────────

def _ensure_repo_importable() -> None:
    """third_party/TIC-VLA 를 sys.path 에 넣는다 (pip -e 안 된 환경 대비)."""
    if not THIRD_PARTY_REPO.exists():
        raise FileNotFoundError(f"업스트림 레포가 없다: {THIRD_PARTY_REPO}")
    path = str(THIRD_PARTY_REPO)
    if path not in sys.path:
        sys.path.insert(0, path)


def build_reference_model(
    model_path: str | Path,
    action_horizon_steps: int,
    action_num_layers: int,
    action_expert_only: bool,
) -> tuple[nn.Module, str]:
    """키 대조에 쓸 참조 모델을 만든다.

    기본은 스펙대로 ``TICVLA(model_path=...)`` 를 통째로 인스턴스화한다.
    VLM 1B 로딩이 무거워 실패하면(메모리·의존성) ActionExpert 단독으로 폴백한다.
    action head 키/shape 대조에는 둘 다 동등하다.

    Args:
        model_path: InternVL3-1B 경로.
        action_horizon_steps: ActionExpert num_chunks.
        action_num_layers: cross-attention 블록 수.
        action_expert_only: True 면 VLM 로딩을 건너뛴다.

    Returns:
        (모델, 어떤 경로로 만들었는지 설명 문자열).
    """
    _ensure_repo_importable()

    if not action_expert_only:
        try:
            from ticvla.models.ticvla import TICVLA

            model = TICVLA(
                model_path=str(model_path),
                action_horizon_steps=action_horizon_steps,
                action_num_layers=action_num_layers,
                train_vlm=False,
            )
            return model, f"TICVLA(model_path={model_path})"
        except Exception as exc:  # noqa: BLE001
            print("[경고] TICVLA 전체 인스턴스화 실패 → ActionExpert 단독으로 폴백")
            print(f"       사유: {type(exc).__name__}: {exc}")

    # 폴백: VLM 없이 ActionExpert 만. input_dim 은 config.json 에서 읽는다.
    from ticvla.models.ticvla import ActionExpert

    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json 이 없어 hidden_size 를 알 수 없다: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["llm_config"]["hidden_size"])

    # ticvla.py:250-252 와 동일한 고정값 (InternVL3-1B 전용).
    kv_cache_feat_dim = 2 * 64

    model = ActionExpert(
        input_dim=hidden_size,
        hidden_dim=512,
        action_dim=2,
        num_layers=action_num_layers,
        num_chunks=action_horizon_steps,
        kv_cache_feat_dim=kv_cache_feat_dim,
    )
    return model, (
        f"ActionExpert(input_dim={hidden_size}, num_chunks={action_horizon_steps}, "
        f"num_layers={action_num_layers})"
    )


def compare_action_keys(
    ckpt_action_state: dict[str, Any],
    model_action_state: dict[str, Any],
) -> dict[str, list[tuple[str, str, str]]]:
    """체크포인트 ↔ 모델의 action head 키를 4분류한다.

    Args:
        ckpt_action_state: 프리픽스 벗긴 체크포인트 action 키.
        model_action_state: 모델 action_expert.state_dict().

    Returns:
        {"matched", "shape_mismatch", "missing", "unexpected"} →
        (key, ckpt shape, model shape) 목록.
    """
    out: dict[str, list[tuple[str, str, str]]] = {
        "matched": [],
        "shape_mismatch": [],
        "missing": [],     # 모델엔 있는데 체크포인트에 없음
        "unexpected": [],  # 체크포인트엔 있는데 모델에 없음
    }
    for key, model_tensor in model_action_state.items():
        if key not in ckpt_action_state:
            out["missing"].append((key, "-", _fmt_shape(model_tensor)))
            continue
        ckpt_tensor = ckpt_action_state[key]
        ckpt_shape = _fmt_shape(ckpt_tensor)
        model_shape = _fmt_shape(model_tensor)
        if ckpt_shape == model_shape:
            out["matched"].append((key, ckpt_shape, model_shape))
        else:
            out["shape_mismatch"].append((key, ckpt_shape, model_shape))
    for key, ckpt_tensor in ckpt_action_state.items():
        if key not in model_action_state:
            out["unexpected"].append((key, _fmt_shape(ckpt_tensor), "-"))
    return out


# ──────────────────────────────────────────────────────────────────────
# 진단 CLI
# ──────────────────────────────────────────────────────────────────────

def inspect_checkpoint(
    ckpt_path: str | Path,
    model_path: str | Path,
    action_horizon_steps: int,
    action_num_layers: int,
    instantiate: bool,
    action_expert_only: bool,
) -> int:
    """합본 체크포인트를 진단한다.

    Args:
        ckpt_path: 합본 체크포인트.
        model_path: InternVL3-1B 경로.
        action_horizon_steps: 대조용 ActionExpert num_chunks.
        action_num_layers: 대조용 블록 수.
        instantiate: False 면 모델 대조를 건너뛴다.
        action_expert_only: VLM 로딩 없이 ActionExpert 만 만든다.

    Returns:
        프로세스 종료 코드. 합본을 그대로 쓸 수 있으면 0.
    """
    ckpt_path = Path(ckpt_path).expanduser()

    _section(f"1. 체크포인트 최상위 구조  —  {ckpt_path}")
    size_mb = ckpt_path.stat().st_size / (1024 ** 2)
    print(f"파일 크기: {size_mb:,.1f} MiB")
    checkpoint = load_checkpoint(ckpt_path)

    if isinstance(checkpoint, dict):
        print(f"최상위 키 {len(checkpoint)}개:")
        for key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, dict):
                detail = f"dict({len(value)} entries)"
            elif isinstance(value, torch.Tensor):
                detail = f"Tensor({_fmt_shape(value)})"
            else:
                detail = repr(value)
                if len(detail) > 90:
                    detail = detail[:87] + "..."
            print(f"  - {key}: {detail}")
    else:
        print(f"최상위가 dict 가 아니다: {type(checkpoint).__name__}")

    # 사전학습 하이퍼파라미터를 알면 우리 yaml 을 여기에 맞출 수 있다.
    # 특히 action_horizon_steps 는 head 를 이어받으려면 반드시 같아야 한다.
    hparams = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    if isinstance(hparams, dict) and hparams:
        print()
        print("hyper_parameters (사전학습 설정):")
        for key, value in hparams.items():
            print(f"  - {key}: {value!r}")

    state_dict, has_state_dict_key = extract_state_dict(checkpoint)
    print()
    print(f"'state_dict' 키 존재: {has_state_dict_key}")
    print(f"state_dict 총 키 수: {len(state_dict):,}")
    if not has_state_dict_key:
        print("★ 레포 load_vlm_checkpoint 는 checkpoint['state_dict'] 만 본다"
              " → 이 파일은 그대로 쓸 수 없다.")

    _section("2. 프리픽스별 집계")
    buckets = classify_keys(state_dict)
    labels = {
        "vlm_direct": f"{VLM_DIRECT_PREFIX}*            (stage1 산출 형식)",
        "vlm_nested": f"{VLM_NESTED_PREFIX}* (stage2 산출 형식)",
        "vlm_model_other": "model.vlm_model.* (그 외)",
        "action_expert": f"{ACTION_PREFIX}*   ★ 우리가 이어받을 것",
        "other": "그 외",
    }
    for name, label in labels.items():
        print(f"  {len(buckets[name]):>4}개  {label}")

    if buckets["other"]:
        print()
        print(f"★ '그 외' {len(buckets['other'])}개 — 조용히 넘기지 않고 전부 나열한다:")
        for key in buckets["other"]:
            print(f"    · {key}  [{_fmt_shape(state_dict[key])}]")
    if buckets["vlm_model_other"]:
        print()
        print("★ model.vlm_model.* 인데 '.vlm.' 이 아닌 키 (레포 리맵이 건너뛴다):")
        for key in buckets["vlm_model_other"]:
            print(f"    · {key}  [{_fmt_shape(state_dict[key])}]")

    # 레포 load_vlm_checkpoint 의 분기 판정을 그대로 재현.
    has_direct = any(k.startswith(VLM_DIRECT_PREFIX) for k in state_dict)
    has_nested = any(k.startswith("model.vlm_model.") for k in state_dict)
    if has_direct:
        vlm_branch = "direct (model.vlm.* → vlm.*)"
        vlm_keys_used = len(buckets["vlm_direct"])
    elif has_nested:
        vlm_branch = "nested (model.vlm_model.vlm.* → vlm.*)"
        vlm_keys_used = len(buckets["vlm_nested"])
    else:
        vlm_branch = "없음 → ValueError"
        vlm_keys_used = 0
    print()
    print(f"레포가 타게 될 분기: {vlm_branch}")
    print(f"그 분기에서 실제로 로드될 VLM 키 수: {vlm_keys_used:,}")

    action_state = strip_prefix(state_dict, ACTION_PREFIX)

    _section("3. 모델 action_expert 와 키 대조")
    comparison: Optional[dict[str, list[tuple[str, str, str]]]] = None
    if not instantiate:
        print("--no-instantiate → 대조 생략")
    elif not action_state:
        print(f"체크포인트에 {ACTION_PREFIX}* 키가 없어 대조할 것이 없다.")
    else:
        model, how = build_reference_model(
            model_path=model_path,
            action_horizon_steps=action_horizon_steps,
            action_num_layers=action_num_layers,
            action_expert_only=action_expert_only,
        )
        print(f"참조 모델: {how}")
        action_expert = _resolve_action_expert(model)
        model_action_state = action_expert.state_dict()
        print(f"모델 action_expert 키 {len(model_action_state)}개 / "
              f"체크포인트 {len(action_state)}개")
        print()
        print("모델 action_expert 키 목록:")
        for key, tensor in model_action_state.items():
            ckpt_mark = "✔" if key in action_state else "✖"
            print(f"  {ckpt_mark} {key:<52} [{_fmt_shape(tensor)}]")

        comparison = compare_action_keys(action_state, model_action_state)
        print()
        print("대조표:")
        print(f"  이름·shape 일치 : {len(comparison['matched'])}")
        print(f"  shape 불일치    : {len(comparison['shape_mismatch'])}")
        print(f"  누락(모델에만)  : {len(comparison['missing'])}")
        print(f"  잉여(ckpt에만)  : {len(comparison['unexpected'])}")
        for name in ("shape_mismatch", "missing", "unexpected"):
            for key, ckpt_shape, model_shape in comparison[name]:
                print(f"    · [{name}] {key}  ckpt={ckpt_shape}  model={model_shape}")

        # 실제 로드까지 해봐서 값이 바뀌는지 확인.
        print()
        print("실제 로드 테스트 (strict=False):")
        load_action_expert(model, ckpt_path, strict=False, verbose=True)

    _section("4. 결론")
    vlm_ok = has_direct or has_nested
    action_ok = bool(action_state)
    if comparison is not None:
        action_ok = action_ok and not comparison["shape_mismatch"] and not comparison["missing"]

    verdict = "예" if (has_state_dict_key and vlm_ok) else "아니오"
    print(f"vlm_checkpoint 인자에 이 파일을 그대로 넣어도 되는가: {verdict}")
    print(f"  · state_dict 키 존재     : {has_state_dict_key}")
    print(f"  · VLM 키 분기            : {vlm_branch}")
    print(f"  · action_expert 키 존재  : {bool(action_state)} ({len(action_state)}개)")
    print(f"  · action head 이어받기   : "
          f"{'가능' if action_ok else '확인 필요'} "
          f"→ load_action_expert() 로 별도 로드해야 한다 (§1-B)")
    print()
    if not action_state:
        print("★ action_expert 키가 없으면 train_wrapper 의 action head 로드가 불가능하다.")
        print("  사전학습 action head 없이 학습하면 사실상 scratch 다.")
    return 0 if (has_state_dict_key and vlm_ok and action_ok) else 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: sys.argv[1:] 형태.

    Returns:
        Namespace.
    """
    parser = argparse.ArgumentParser(
        description="합본 TIC-VLA 체크포인트 진단 + action head 로더",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inspect", nargs="?", const=str(DEFAULT_CKPT), default=None,
        metavar="CKPT", help="합본 체크포인트를 진단한다 (경로 생략 시 기본값)")
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH),
        help="InternVL3-1B 로컬 경로 (참조 모델 생성용)")
    parser.add_argument(
        "--action-horizon-steps", type=int, default=30,
        help="대조용 ActionExpert num_chunks")
    parser.add_argument(
        "--action-num-layers", type=int, default=3,
        help="대조용 cross-attention 블록 수")
    parser.add_argument(
        "--no-instantiate", action="store_true",
        help="모델 인스턴스화·키 대조를 건너뛴다 (체크포인트 구조만 본다)")
    parser.add_argument(
        "--action-expert-only", action="store_true",
        help="VLM 1B 로딩 없이 ActionExpert 만 만들어 대조한다 (빠름)")
    return parser.parse_args(argv)


def main() -> int:
    """진입점.

    Returns:
        종료 코드.
    """
    args = parse_args()
    if args.inspect is None:
        print("할 일이 없다. --inspect [CKPT] 를 주라.", file=sys.stderr)
        return 2
    return inspect_checkpoint(
        ckpt_path=args.inspect,
        model_path=args.model_path,
        action_horizon_steps=args.action_horizon_steps,
        action_num_layers=args.action_num_layers,
        instantiate=not args.no_instantiate,
        action_expert_only=args.action_expert_only,
    )


if __name__ == "__main__":
    sys.exit(main())
