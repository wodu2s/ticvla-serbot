#!/usr/bin/env python3
"""레포 stage 2 학습을 감싸는 얇은 층. third_party 는 수정하지 않는다.

레포 ``ticvla/training/train.py`` 를 그대로 쓰면 안 되는 이유가 다섯 개 있다.
이 파일이 그 다섯 개를 한 곳에서 처리한다.

1. action head 랜덤 초기화 (§1-B)
   stage 2 는 ``load_vlm_checkpoint()`` 만 부르므로 사전학습 action head 50키가
   버려진다. ``01_load_full_ckpt.load_action_expert()`` 로 이어받는다.

2. VLM 동결 유지 (F9)
   ``train_vlm=False`` 는 ``__init__`` 에서 ``vlm.eval()`` 을 한 번 부를 뿐이다.
   Lightning 이 epoch·검증 전환마다 ``module.train()`` 을 부르면 VLM dropout 이
   다시 켜져 동결 컨텍스트가 스텝마다 흔들린다. epoch/배치 시작 때 재고정한다.

3. 정답 누수 스위치 (§1-E) ★ 이 프로젝트의 핵심 실험 변수
   ``_build_messages`` 는 assistant 롤에 3/6/9초 정답을
   ``<answer>(x, y, theta), ...</answer>`` 문자열로 넣는다(vlm_data.py:281-286).
   그 프롬프트의 ``past_key_values`` 가 action head 의 주 입력이므로, 학습 때는
   정답을 보고 예측하고 실차에서는 못 본다. ``--teacher_answer drop`` 은 assistant
   메시지를 빼서 이 경로를 끊는다.

   확인한 사실: ``guidance_waypoint`` 는 배치에 담기지만 ``TICVLA.forward`` 가
   읽지 않는다. 정답이 모델에 닿는 경로는 프롬프트 텍스트 하나뿐이므로 assistant
   메시지 제거로 누수가 완전히 끊긴다.

4. Trainer 인자 노출 (F7, F8)
   ``TrainingConfig`` 에 limit_*/val_check_interval/resume 필드가 없고, 레포
   ``_create_trainer`` 는 ``devices=-1, strategy="ddp"`` 하드코딩이다.
   A5000 1장 기준으로 우리가 직접 Trainer 를 만든다.

5. 손실 beta 노출 (§1-D)
   waypoint 는 미터 원본이라 우리 0.10 m/s 데이터는 값이 작다. 기본 beta=1.0 이면
   0.1~1.0 m 가 전부 2차항 영역이어서 작은 오차의 기울기가 매우 작다.
   yaml 에서 beta 를 받고, 붕괴 조기 감지를 위해 성분별 통계를 로깅한다.

사용 예:
    python3 train/train_wrapper.py --config train/configs/smoke.yaml
    python3 train/train_wrapper.py --config train/configs/noleak.yaml \\
        --teacher_answer drop --resume_from outputs/.../last.ckpt
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import yaml

# ──────────────────────────────────────────────────────────────────────
# 경로 준비 (pip -e 안 된 환경에서도 업스트림을 import 할 수 있게)
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_REPO = PROJECT_ROOT / "third_party" / "TIC-VLA"
P1_SCRIPT = Path(__file__).resolve().parent / "01_load_full_ckpt.py"

if str(THIRD_PARTY_REPO) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_REPO))

import pytorch_lightning as pl  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ticvla.data.policy_data import TICVLACollator, TICVLADataModule  # noqa: E402
from ticvla.training.config import TrainingConfig  # noqa: E402
from ticvla.training.train import (  # noqa: E402
    TICVLAActionLightningModule,
    _expand_env_value,
    _optional_path,
    create_callbacks,
    create_loggers,
    load_config,
)

LOGGER = logging.getLogger("train_wrapper")

STAGE = "action"
TEACHER_ANSWER_MODES = ("keep", "drop")


# ──────────────────────────────────────────────────────────────────────
# P1 로더 재사용
# ──────────────────────────────────────────────────────────────────────

def load_p1_module():
    """01_load_full_ckpt.py 를 import 한다.

    파일명이 숫자로 시작해 일반 import 가 안 되므로 importlib 를 쓴다
    (P1 독스트링에 적힌 방식).

    Returns:
        로드된 모듈 객체.

    Raises:
        FileNotFoundError: P1 스크립트가 없을 때.
    """
    if not P1_SCRIPT.exists():
        raise FileNotFoundError(
            f"P1 스크립트가 없다: {P1_SCRIPT}\n"
            f"  01_load_full_ckpt.py 가 있어야 action head 를 이어받을 수 있다."
        )
    spec = importlib.util.spec_from_file_location("load_full_ckpt", P1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses/typing 이 sys.modules[cls.__module__] 를 찾으므로 먼저 등록한다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ──────────────────────────────────────────────────────────────────────
# 우리 전용 설정 (TrainingConfig 에 없는 필드)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WrapperConfig:
    """레포 TrainingConfig 가 모르는 필드들.

    load_config 는 미지의 yaml 키를 경고만 하고 버리므로, 같은 yaml 을 우리가
    한 번 더 읽어 이 필드들을 채운다.
    """

    #: 사전학습 action head 를 담은 ckpt. 미지정이면 vlm_checkpoint 와 동일 (§1-B).
    action_checkpoint: Optional[str] = None
    #: keep = 레포 원본(정답 포함) / drop = assistant 제거 (§1-E).
    teacher_answer: str = "keep"
    #: smooth_l1_loss 의 beta (§1-D).
    smooth_l1_beta: float = 1.0
    #: action head 로드를 strict 로 강제할지. False 면 경고만.
    action_checkpoint_strict: bool = True

    # ── Trainer 인자 (F7) ──
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    val_check_interval: Optional[Union[int, float]] = None
    log_every_n_steps: int = 10
    num_sanity_val_steps: int = 0


def load_repo_config(config_path: str | Path) -> tuple[TrainingConfig, List[str]]:
    """레포 load_config 를 쓰되 경고를 정리한다.

    ``load_config`` 는 TrainingConfig 에 없는 키를 모두 "Unknown configuration key"
    로 경고한다. 우리 필드(teacher_answer 등)까지 경고에 섞이면 설정이 무시된 것으로
    오해하기 쉽다. 우리가 처리하는 키는 조용히 넘기고, 정말 아무도 모르는 키만
    우리가 경고한다.

    Args:
        config_path: yaml 경로.

    Returns:
        (TrainingConfig, 아무도 처리하지 않는 키 목록).
    """
    repo_logger = logging.getLogger("ticvla.training.train")
    previous_level = repo_logger.level
    repo_logger.setLevel(logging.ERROR)
    try:
        config = load_config(str(config_path))
    finally:
        repo_logger.setLevel(previous_level)

    with Path(config_path).expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(TrainingConfig)} | {
        f.name for f in fields(WrapperConfig)}
    unknown = [key for key in raw if key not in known]
    return config, unknown


def load_wrapper_config(config_path: str | Path) -> WrapperConfig:
    """yaml 에서 우리 전용 필드만 골라 읽는다.

    Args:
        config_path: yaml/json 경로.

    Returns:
        WrapperConfig.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: teacher_answer 값이 잘못됐을 때.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없다: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    known = {f.name for f in fields(WrapperConfig)}
    cfg = WrapperConfig()
    for key, value in raw.items():
        if key in known:
            setattr(cfg, key, _expand_env_value(value))

    if cfg.teacher_answer not in TEACHER_ANSWER_MODES:
        raise ValueError(
            f"teacher_answer 는 {TEACHER_ANSWER_MODES} 중 하나여야 한다: "
            f"{cfg.teacher_answer!r}")
    return cfg


# ──────────────────────────────────────────────────────────────────────
# ★ 정답 누수 스위치 — collator
# ──────────────────────────────────────────────────────────────────────

def strip_assistant_messages(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool]:
    """messages 에서 assistant 롤을 제거한다.

    assistant 롤에는 3/6/9초 정답이 ``<answer>...</answer>`` 로 들어 있다
    (gt_parse 가 있으면 ``<think>`` 도 함께). 추론 때는 이 자리에 VLM 이 생성한
    텍스트가 들어가므로, 정답 문자열은 학습에만 존재하는 정보다.

    Args:
        messages: _build_messages() 결과.

    Returns:
        (assistant 를 뺀 messages, 실제로 뺀 것이 있었는지).
    """
    kept = [m for m in messages if m.get("role") != "assistant"]
    return kept, len(kept) != len(messages)


@dataclass
class TeacherAnswerCollator(TICVLACollator):
    """TICVLACollator 에 정답 누수 스위치를 더한 것.

    ``keep`` 은 부모 동작 그대로다. ``drop`` 은 assistant 메시지를 빼고
    ``apply_chat_template(..., add_generation_prompt=True)`` 를 태운다.
    """

    teacher_answer: str = "keep"

    def transform_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """모드에 따라 messages 를 가공한다.

        Args:
            messages: 원본 messages.

        Returns:
            가공된 messages. keep 이면 원본 그대로.
        """
        if self.teacher_answer == "keep":
            return messages
        kept, _ = strip_assistant_messages(messages)
        return kept

    def render_prompt(self, sample: Dict[str, Any]) -> str:
        """샘플 하나의 프롬프트 문자열을 만든다 (이미지 로딩 없이).

        ``__call__`` 과 같은 변환·같은 apply_chat_template 호출을 쓰므로 학습에
        실제로 들어가는 문자열과 같다. 차이는 ``<image>`` 자리표시자가 아직
        IMG_CONTEXT 토큰으로 펼쳐지지 않았다는 점뿐이다.

        Args:
            sample: 데이터셋 __getitem__ 결과.

        Returns:
            프롬프트 문자열.
        """
        messages = self.transform_messages(sample["messages"])
        rendered = self.processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, list):
            return rendered[0] if rendered else ""
        return rendered

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치를 만든다.

        Args:
            samples: 데이터셋 샘플 목록.

        Returns:
            부모 collator 의 배치 dict.
        """
        if self.teacher_answer != "keep":
            # 원본 dict 를 건드리지 않는다 (dataset 캐시·재사용 대비).
            samples = [
                {**s, "messages": self.transform_messages(s["messages"])}
                for s in samples
            ]
        return super().__call__(samples)


class TeacherAnswerDataModule(TICVLADataModule):
    """collator 를 교체할 수 있게 한 DataModule.

    레포 ``TICVLADataModule._dataloader`` 는 ``TICVLACollator`` 를 하드코딩하므로
    (policy_data.py:628) 이 메서드를 덮어써야 스위치가 먹는다.
    """

    def __init__(self, *args: Any, teacher_answer: str = "keep", **kwargs: Any) -> None:
        """
        Args:
            *args: TICVLADataModule 인자.
            teacher_answer: keep / drop.
            **kwargs: TICVLADataModule 인자.
        """
        super().__init__(*args, **kwargs)
        self.teacher_answer = teacher_answer

    def make_collator(self) -> TeacherAnswerCollator:
        """현재 모드의 collator 를 만든다.

        Returns:
            TeacherAnswerCollator.
        """
        return TeacherAnswerCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            teacher_answer=self.teacher_answer,
        )

    def _dataloader(self, ds: Any, shuffle: bool) -> DataLoader:
        """부모와 같되 collator 만 우리 것으로 바꾼다.

        Args:
            ds: 데이터셋.
            shuffle: 셔플 여부.

        Returns:
            DataLoader.
        """
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.make_collator(),
        )


def log_prompt_comparison(
    dataset: Any,
    processor: Any,
    tokenizer: Any,
    active_mode: str,
    sample_index: int = 0,
) -> None:
    """keep / drop 두 모드의 프롬프트를 통째로 로그에 찍는다.

    ★ 이 로그가 없으면 §1-E 실험이 성립하지 않는다. 어느 모드로 돌리든 두 모드를
    모두 찍어서, 무엇이 빠졌는지 눈으로 확인할 수 있게 한다.

    DataLoader worker 안에서 찍으면 num_workers 개만큼 중복되고 순서도 뒤섞이므로,
    학습 시작 전 메인 프로세스에서 한 번만 찍는다.

    Args:
        dataset: 학습 데이터셋.
        processor: AutoProcessor.
        tokenizer: AutoTokenizer.
        active_mode: 이번 학습에 실제로 쓰는 모드.
        sample_index: 어떤 샘플을 찍을지.
    """
    if len(dataset) == 0:
        LOGGER.warning("데이터셋이 비어 프롬프트를 찍을 수 없다")
        return

    sample = dataset[sample_index % len(dataset)]
    rendered: Dict[str, str] = {}

    for mode in TEACHER_ANSWER_MODES:
        collator = TeacherAnswerCollator(
            processor=processor, tokenizer=tokenizer, teacher_answer=mode)
        # sample 은 모드별로 재사용해도 안전하다 (collator 가 복사해서 쓴다).
        rendered[mode] = collator.render_prompt(copy.deepcopy(sample))

    num_image_placeholders = rendered["keep"].count("<image>")
    for mode in TEACHER_ANSWER_MODES:
        text = rendered[mode]
        token_len = len(tokenizer(text)["input_ids"])
        marker = " ← 이번 학습에 사용" if mode == active_mode else ""
        LOGGER.info("")
        LOGGER.info("=" * 78)
        LOGGER.info(f" PROMPT (teacher_answer={mode}){marker}")
        LOGGER.info(f" 문자 {len(text):,} / 토큰 {token_len:,} "
                    f"(<image> 자리표시자 {text.count('<image>')}개는 아직 "
                    f"IMG_CONTEXT 로 펼쳐지지 않은 상태)")
        LOGGER.info(f" <answer> 포함: {'<answer>' in text}   "
                    f"<think> 포함: {'<think>' in text}")
        LOGGER.info("=" * 78)
        for line in text.split("\n"):
            LOGGER.info(line)
        LOGGER.info("=" * 78)
        LOGGER.info(f" END PROMPT (teacher_answer={mode})")
        LOGGER.info("=" * 78)

    delta = len(rendered["keep"]) - len(rendered["drop"])
    LOGGER.info("")
    LOGGER.info(f"[누수 스위치] keep − drop = {delta:,} 문자")
    if delta == 0:
        LOGGER.warning(
            "★ 두 모드의 프롬프트가 같다. assistant 메시지가 없거나 스위치가 "
            "먹지 않았다 — §1-E 실험이 성립하지 않으므로 확인이 필요하다.")
    if "<answer>" in rendered["drop"]:
        LOGGER.warning(
            "★ drop 모드인데도 <answer> 가 남아 있다. 정답 누수가 끊기지 않았다.")
    if num_image_placeholders == 0:
        LOGGER.warning(
            "<image> 자리표시자가 0개다. collator 가 이미지 토큰을 앞에 "
            "덧붙이는 경로로 빠진다 (policy_data.py:520). 의도한 것인지 확인하라.")


# ──────────────────────────────────────────────────────────────────────
# ★ Lightning 모듈 — action head 로드 + 동결 유지 + beta
# ──────────────────────────────────────────────────────────────────────

class ActionModuleWithPretrainedHead(TICVLAActionLightningModule):
    """stage 2 모듈에 사전학습 action head 로드와 동결 유지를 더한 것."""

    def __init__(
        self,
        *,
        action_checkpoint: str,
        smooth_l1_beta: float = 1.0,
        action_checkpoint_strict: bool = True,
        teacher_answer: str = "keep",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            action_checkpoint: 사전학습 action head 가 든 ckpt.
            smooth_l1_beta: smooth_l1_loss beta (§1-D).
            action_checkpoint_strict: 로드 strict 여부.
            teacher_answer: 기록용 (손실 계산에는 쓰지 않는다).
            **kwargs: 부모 TICVLAActionLightningModule 인자.
        """
        super().__init__(**kwargs)

        self.smooth_l1_beta = float(smooth_l1_beta)
        self.action_checkpoint = action_checkpoint
        self.teacher_answer = teacher_answer

        # 재현을 위해 우리 필드도 ckpt hparams 에 남긴다.
        self.hparams.update({
            "action_checkpoint": action_checkpoint,
            "smooth_l1_beta": self.smooth_l1_beta,
            "teacher_answer": teacher_answer,
        })

        # ── §1-B: 사전학습 action head 이어받기 ──
        p1 = load_p1_module()
        LOGGER.info("=" * 78)
        LOGGER.info(" action head 로드 (§1-B)")
        LOGGER.info("=" * 78)
        report = p1.load_action_expert(
            self.model,
            action_checkpoint,
            strict=action_checkpoint_strict,
            verbose=True,
        )
        LOGGER.info(
            f"action head 로드 완료: {report['loaded']}키, "
            f"값이 바뀐 텐서 {report['num_changed_tensors']}/{report['num_tensors']}")
        if not report["changed"]:
            raise RuntimeError(
                "action head 가중치가 하나도 바뀌지 않았다. 로드가 무효다 — "
                "랜덤 초기화로 학습하는 것과 같으므로 중단한다 (§1-B).")

    # ── F9: VLM 동결 유지 ─────────────────────────────────────────────

    def _refreeze_vlm(self) -> None:
        """VLM 을 eval 로 되돌린다 (dropout 재활성 방지)."""
        if self.model.vlm.training:
            self.model.vlm.eval()

    def on_train_epoch_start(self) -> None:
        """epoch 시작 시 재고정."""
        self._refreeze_vlm()

    def on_validation_epoch_start(self) -> None:
        """검증 시작 시 재고정."""
        self._refreeze_vlm()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """배치마다 재고정.

        val_check_interval < 1.0 이면 epoch 중간에 검증이 끼어들고, 그때
        Lightning 이 train 모드를 복원하며 VLM dropout 이 다시 켜진다.
        epoch 훅만으로는 그 구간을 못 막는다. eval() 호출은 forward 대비 무시할
        수준이라 배치마다 확인해도 문제없다.

        Args:
            batch: 배치.
            batch_idx: 배치 인덱스.
        """
        self._refreeze_vlm()

    def on_fit_start(self) -> None:
        """학습 시작 시 학습 대상 파라미터를 보고한다."""
        trainable: Dict[str, int] = {}
        total_trainable = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            prefix = ".".join(name.split(".")[:1]) or name
            trainable[prefix] = trainable.get(prefix, 0) + param.numel()
            total_trainable += param.numel()

        total = sum(p.numel() for p in self.model.parameters())
        LOGGER.info("=" * 78)
        LOGGER.info(" 학습 대상 파라미터 (F9 확인)")
        LOGGER.info("=" * 78)
        LOGGER.info(f"전체 {total:,} / 학습 대상 {total_trainable:,} "
                    f"({100.0 * total_trainable / max(total, 1):.4f}%)")
        for prefix, count in sorted(trainable.items()):
            LOGGER.info(f"  requires_grad=True  {prefix}.*  {count:,}")
        unexpected = [p for p in trainable if p != "action_expert"]
        if unexpected:
            LOGGER.warning(
                f"★ action_expert 외에 학습 대상이 있다: {unexpected} — "
                f"VLM 동결이 깨졌을 수 있다 (F9).")
        LOGGER.info(f"VLM training 모드: {self.model.vlm.training} (False 여야 한다)")
        LOGGER.info(f"teacher_answer={self.teacher_answer}  "
                    f"smooth_l1_beta={self.smooth_l1_beta}")

    # ── §1-D: beta 노출 + 붕괴 감지 로깅 ─────────────────────────────

    def _log_component_stats(
        self,
        prefix: str,
        predicted: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
    ) -> None:
        """x·y 성분별 평균 절대값·표준편차를 기록한다.

        전부 0 근처로 수렴하는 붕괴를 조기에 잡기 위한 것이다. x 는 스케일,
        y 는 회피 성능에 대응하므로 섞지 않고 따로 본다.

        Args:
            prefix: train / val / test.
            predicted: (B, T, 2) 예측.
            target: (B, T, 2) 정답.
            batch_size: 로깅용 배치 크기.
        """
        with torch.no_grad():
            pred = predicted.detach().float()
            tgt = target.detach().float()
            stats = {
                f"{prefix}_pred_x_absmean": pred[..., 0].abs().mean(),
                f"{prefix}_pred_y_absmean": pred[..., 1].abs().mean(),
                f"{prefix}_pred_x_std": pred[..., 0].std(),
                f"{prefix}_pred_y_std": pred[..., 1].std(),
                f"{prefix}_target_x_absmean": tgt[..., 0].abs().mean(),
                f"{prefix}_target_y_absmean": tgt[..., 1].abs().mean(),
                f"{prefix}_target_x_std": tgt[..., 0].std(),
                f"{prefix}_target_y_std": tgt[..., 1].std(),
            }
        for name, value in stats.items():
            self.log(name, value, sync_dist=True, batch_size=batch_size)

    def _shared_step(self, batch: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        """부모와 같되 beta 를 적용하고 성분별 통계를 더한다.

        Args:
            batch: 배치.
            prefix: train / val / test.

        Returns:
            총 손실.
        """
        outputs = self(batch)
        predicted_waypoints = outputs["waypoints"]
        target_waypoints = batch["waypoints"]

        action_loss = F.smooth_l1_loss(
            predicted_waypoints, target_waypoints, beta=self.smooth_l1_beta)
        total_loss = action_loss

        with torch.no_grad():
            metrics = self._trajectory_metrics(
                predicted_waypoints.detach(), target_waypoints.detach())

        batch_size = predicted_waypoints.size(0)
        self.log(f"{prefix}_total_loss", total_loss.detach(), sync_dist=True,
                 prog_bar=True, batch_size=batch_size)
        self.log(f"{prefix}_action_loss", action_loss.detach(), sync_dist=True,
                 prog_bar=(prefix == "train"), batch_size=batch_size)
        self.log(f"{prefix}_ade", metrics["ade"], sync_dist=True,
                 prog_bar=(prefix != "train"), batch_size=batch_size)
        self.log(f"{prefix}_fde", metrics["fde"], sync_dist=True,
                 prog_bar=(prefix != "train"), batch_size=batch_size)

        self._log_component_stats(
            prefix, predicted_waypoints, target_waypoints, batch_size)

        if prefix == "train":
            optimizer = self.optimizers()
            if optimizer is not None:
                self.log("train_lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return total_loss


# ──────────────────────────────────────────────────────────────────────
# Trainer (F7, F8)
# ──────────────────────────────────────────────────────────────────────

def build_trainer(
    config: TrainingConfig,
    wrapper: WrapperConfig,
    run_version: Optional[str] = None,
) -> pl.Trainer:
    """A5000 1장용 Trainer 를 만든다.

    레포 ``_create_trainer`` 는 ``devices=-1, strategy="ddp"`` 하드코딩이라
    1장 환경에서 쓰면 안 된다(F8). 콜백·로거는 레포 것을 그대로 재사용한다.

    Args:
        config: 레포 TrainingConfig.
        wrapper: 우리 전용 설정.
        run_version: 로그 버전 문자열. None 이면 타임스탬프.

    Returns:
        pl.Trainer.
    """
    if run_version is None:
        run_version = datetime.now().strftime("%Y%m%d-%H%M%S")

    kwargs: Dict[str, Any] = {
        "max_epochs": config.max_epochs,
        "precision": config.precision,
        "gradient_clip_algorithm": "norm",
        "gradient_clip_val": config.gradient_clip_val,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "callbacks": create_callbacks(
            save_dir=os.path.join(config.save_dir, STAGE), stage=STAGE),
        "logger": create_loggers(
            experiment_name=config.experiment_name,
            save_dir=os.path.join(config.log_dir, STAGE),
            version=run_version,
        ),
        # A5000 1장 고정 (F8).
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "deterministic": False,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "log_every_n_steps": wrapper.log_every_n_steps,
        "num_sanity_val_steps": wrapper.num_sanity_val_steps,
    }
    for name in ("limit_train_batches", "limit_val_batches", "val_check_interval"):
        value = getattr(wrapper, name)
        if value is not None:
            kwargs[name] = value

    return pl.Trainer(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: sys.argv[1:] 형태.

    Returns:
        Namespace.
    """
    parser = argparse.ArgumentParser(
        description="TIC-VLA stage 2 학습 래퍼 (action head 로드 + 정답 누수 스위치)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="yaml 설정 경로")
    parser.add_argument(
        "--action_checkpoint", default=None,
        help="사전학습 action head ckpt. 미지정이면 yaml → vlm_checkpoint 순")
    parser.add_argument(
        "--teacher_answer", choices=list(TEACHER_ANSWER_MODES), default=None,
        help="정답 누수 스위치. CLI 가 yaml 보다 우선한다 (§1-E)")
    parser.add_argument(
        "--resume_from", default=None, help="이어서 학습할 ckpt")
    return parser.parse_args(argv)


def resolve_settings(
    args: argparse.Namespace,
) -> tuple[TrainingConfig, WrapperConfig, str]:
    """설정을 읽고 CLI 우선순위를 적용한다.

    Args:
        args: parse_args 결과.

    Returns:
        (TrainingConfig, WrapperConfig, vlm_checkpoint 경로).

    Raises:
        ValueError: vlm_checkpoint 가 없을 때.
        FileNotFoundError: 체크포인트 파일이 없을 때.
    """
    config, unknown_keys = load_repo_config(args.config)
    wrapper = load_wrapper_config(args.config)
    for key in unknown_keys:
        LOGGER.warning(f"설정 파일에 아무도 쓰지 않는 키가 있다: {key}")

    # CLI 가 yaml 보다 우선.
    if args.teacher_answer is not None:
        wrapper.teacher_answer = args.teacher_answer
    if args.action_checkpoint is not None:
        wrapper.action_checkpoint = args.action_checkpoint

    vlm_checkpoint = _optional_path(config.vlm_checkpoint)
    if not vlm_checkpoint:
        raise ValueError(
            "vlm_checkpoint 가 필요하다. yaml 에 합본 ckpt 경로를 넣어라 (§1-C).")
    if not os.path.exists(vlm_checkpoint):
        raise FileNotFoundError(f"vlm_checkpoint 가 없다: {vlm_checkpoint}")

    # §1-B: action head 는 같은 합본 파일에서 온다.
    action_checkpoint = _optional_path(wrapper.action_checkpoint) or vlm_checkpoint
    if not os.path.exists(action_checkpoint):
        raise FileNotFoundError(f"action_checkpoint 가 없다: {action_checkpoint}")
    wrapper.action_checkpoint = action_checkpoint

    # F4: val_data_dir 를 비우면 레포가 샘플 단위 random_split 을 해 누수가 난다.
    if not config.val_data_dir:
        raise ValueError(
            "val_data_dir 가 비어 있다. 비우면 레포가 샘플 단위 random_split 을 해서 "
            "인접 프레임이 train/val 에 섞인다 (F4). 03_make_splits.py 산출 경로를 "
            "명시하라.")

    return config, wrapper, vlm_checkpoint


def log_settings(
    config: TrainingConfig,
    wrapper: WrapperConfig,
    vlm_checkpoint: str,
    resume_from: Optional[str],
) -> None:
    """확정된 설정을 로그에 남긴다.

    Args:
        config: TrainingConfig.
        wrapper: WrapperConfig.
        vlm_checkpoint: VLM ckpt 경로.
        resume_from: 재개 ckpt 경로.
    """
    LOGGER.info("=" * 78)
    LOGGER.info(" 확정 설정")
    LOGGER.info("=" * 78)
    LOGGER.info(f"model_path          : {config.model_path}")
    LOGGER.info(f"vlm_checkpoint      : {vlm_checkpoint}")
    LOGGER.info(f"action_checkpoint   : {wrapper.action_checkpoint}")
    LOGGER.info(f"train_data_dir      : {config.train_data_dir}")
    LOGGER.info(f"val_data_dir        : {config.val_data_dir}")
    LOGGER.info(f"teacher_answer      : {wrapper.teacher_answer}  ★ §1-E")
    LOGGER.info(f"smooth_l1_beta      : {wrapper.smooth_l1_beta}")
    LOGGER.info(f"action_horizon_steps: {config.action_horizon_steps}")
    LOGGER.info(f"max_sequence_length : {config.max_sequence_length}")
    LOGGER.info(f"batch_size          : {config.batch_size} "
                f"× accumulate {config.accumulate_grad_batches} "
                f"= 유효 {config.batch_size * config.accumulate_grad_batches}")
    LOGGER.info(f"learning_rate       : {config.learning_rate}")
    LOGGER.info(f"max_epochs          : {config.max_epochs}  "
                f"warmup_steps={config.warmup_steps}")
    LOGGER.info(f"precision           : {config.precision}")
    LOGGER.info(f"limit_train/val     : {wrapper.limit_train_batches} / "
                f"{wrapper.limit_val_batches}")
    LOGGER.info(f"val_check_interval  : {wrapper.val_check_interval}")
    LOGGER.info(f"resume_from         : {resume_from}")


def main(argv: Optional[List[str]] = None) -> int:
    """진입점.

    Args:
        argv: sys.argv[1:] 형태.

    Returns:
        종료 코드.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    config, wrapper, vlm_checkpoint = resolve_settings(args)
    log_settings(config, wrapper, vlm_checkpoint, args.resume_from)

    module = ActionModuleWithPretrainedHead(
        model_path=config.model_path,
        vlm_checkpoint=vlm_checkpoint,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        warmup_steps=config.warmup_steps,
        action_horizon_steps=config.action_horizon_steps,
        action_num_layers=config.action_num_layers,
        action_checkpoint=wrapper.action_checkpoint,
        smooth_l1_beta=wrapper.smooth_l1_beta,
        action_checkpoint_strict=wrapper.action_checkpoint_strict,
        teacher_answer=wrapper.teacher_answer,
    )

    data_module = TeacherAnswerDataModule(
        train_data_dir=config.train_data_dir,
        val_data_dir=config.val_data_dir,
        test_data_dir=config.test_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_sequence_length=config.max_sequence_length,
        action_horizon_steps=config.action_horizon_steps,
        processor=module.model.processor,
        tokenizer=module.model.tokenizer,
        teacher_answer=wrapper.teacher_answer,
    )

    # ★ 학습 시작 전에 두 모드의 프롬프트를 통째로 남긴다 (§1-E).
    data_module.setup("fit")
    log_prompt_comparison(
        dataset=data_module.train_dataset,
        processor=module.model.processor,
        tokenizer=module.model.tokenizer,
        active_mode=wrapper.teacher_answer,
    )

    trainer = build_trainer(config, wrapper)
    trainer.fit(module, datamodule=data_module, ckpt_path=args.resume_from)
    if config.test_data_dir:
        trainer.test(module, datamodule=data_module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
