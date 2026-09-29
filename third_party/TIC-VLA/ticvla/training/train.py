#!/usr/bin/env python3
"""Two-stage TIC-VLA supervised training.

Stage 1 fine-tunes the VLM on CoT/waypoint language targets.
Stage 2 freezes that VLM and trains the action head from detached model state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ticvla.models.ticvla import TICVLA
from ticvla.models.vlm import TICVLA_VLM
from ticvla.training.config import TrainingConfig
from ticvla.data.policy_data import TICVLADataModule
from ticvla.data.vlm_data import TICVLADataModule_VLM

torch.set_float32_matmul_precision("high")
LOGGER = logging.getLogger(__name__)


def _expand_env_value(value: Any) -> Any:
    """Expand shell-style environment variables in nested config values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value


def _optional_path(path: Optional[str]) -> Optional[str]:
    """Treat empty placeholder paths as unset."""
    if not path:
        return None
    expanded = os.path.expanduser(os.path.expandvars(path))
    if expanded in {"", "None", "none", "null"}:
        return None
    if expanded.startswith("/path/to/"):
        return None
    return expanded


def get_warmup_cosine_schedule_lambda(
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.01,
):
    """Create a warmup + cosine-decay LR schedule."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return lr_lambda


class TICVLAVLMLightningModule(pl.LightningModule):
    """Stage 1: fine-tune the TIC-VLA VLM on CoT/waypoint text."""

    def __init__(
        self,
        model_path: str,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        warmup_steps: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TICVLA_VLM(model_path=model_path)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _shared_step(self, batch: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        loss = self(batch)
        log_loss = loss.detach()
        batch_size = batch["input_ids"].size(0) if "input_ids" in batch else None
        self.log(f"{prefix}_language_loss", log_loss, sync_dist=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{prefix}_total_loss", log_loss, sync_dist=True, prog_bar=(prefix != "train"), batch_size=batch_size)
        if prefix == "train":
            optimizer = self.optimizers()
            if optimizer is not None:
                self.log("train_lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable VLM parameters found.")
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=get_warmup_cosine_schedule_lambda(
                warmup_steps=self.warmup_steps,
                max_steps=_estimate_total_steps(self, self.max_epochs),
                min_lr_ratio=0.01,
            ),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def setup(self, stage: str) -> None:
        _update_scheduler_steps(self, self.warmup_steps)


class TICVLAActionLightningModule(pl.LightningModule):
    """Stage 2: train the action head from a frozen VLM model state."""

    def __init__(
        self,
        model_path: str,
        vlm_checkpoint: str,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        warmup_steps: int,
        action_horizon_steps: int,
        action_num_layers: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = TICVLA(
            model_path=model_path,
            action_horizon_steps=action_horizon_steps,
            action_num_layers=action_num_layers,
            train_vlm=False,
        )
        self.model.load_vlm_checkpoint(vlm_checkpoint)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    @staticmethod
    def _trajectory_metrics(
        predicted_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l2_distances = torch.norm(predicted_waypoints - target_waypoints, dim=-1)
        return {
            "ade": l2_distances.mean(),
            "fde": l2_distances[:, -1].mean(),
        }

    def _shared_step(self, batch: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        outputs = self(batch)
        predicted_waypoints = outputs["waypoints"]
        target_waypoints = batch["waypoints"]
        action_loss = F.smooth_l1_loss(predicted_waypoints, target_waypoints)
        total_loss = action_loss
        with torch.no_grad():
            metrics = self._trajectory_metrics(predicted_waypoints.detach(), target_waypoints.detach())

        batch_size = predicted_waypoints.size(0)
        self.log(f"{prefix}_total_loss", total_loss.detach(), sync_dist=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{prefix}_action_loss", action_loss.detach(), sync_dist=True, prog_bar=(prefix == "train"), batch_size=batch_size)
        self.log(f"{prefix}_ade", metrics["ade"], sync_dist=True, prog_bar=(prefix != "train"), batch_size=batch_size)
        self.log(f"{prefix}_fde", metrics["fde"], sync_dist=True, prog_bar=(prefix != "train"), batch_size=batch_size)
        if prefix == "train":
            optimizer = self.optimizers()
            if optimizer is not None:
                self.log("train_lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [p for p in self.model.action_expert.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable action-head parameters found.")
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=get_warmup_cosine_schedule_lambda(
                warmup_steps=self.warmup_steps,
                max_steps=_estimate_total_steps(self, self.max_epochs),
                min_lr_ratio=0.01,
            ),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def setup(self, stage: str) -> None:
        _update_scheduler_steps(self, self.warmup_steps)


def _estimate_total_steps(module: pl.LightningModule, max_epochs: int) -> int:
    trainer = getattr(module, "trainer", None)
    if trainer is not None and hasattr(trainer, "estimated_stepping_batches"):
        try:
            estimated_steps = int(trainer.estimated_stepping_batches)
            if estimated_steps > 0:
                return estimated_steps
        except Exception:
            pass
    LOGGER.warning("Could not estimate scheduler steps from the dataloader; falling back to max_epochs=%d.", max_epochs)
    return max(1, max_epochs)


def _update_scheduler_steps(module: pl.LightningModule, warmup_steps: int) -> None:
    if not hasattr(module, "trainer") or not hasattr(module.trainer, "estimated_stepping_batches"):
        return
    try:
        scheduler = module.lr_schedulers()
        optimizer = module.optimizers()
    except Exception:
        return
    if scheduler is None:
        return
    lr_lambda = get_warmup_cosine_schedule_lambda(
        warmup_steps=warmup_steps,
        max_steps=_estimate_total_steps(module, getattr(module, "max_epochs", 1)),
        min_lr_ratio=0.01,
    )
    scheduler.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)


def create_callbacks(save_dir: str, stage: str) -> list:
    monitor = "val_language_loss" if stage == "vlm" else "val_total_loss"
    return [
        ModelCheckpoint(
            dirpath=save_dir,
            filename=f"ticvla-{stage}-{{epoch:02d}}-{{{monitor}:.4f}}",
            monitor=monitor,
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def create_loggers(experiment_name: str, save_dir: str, version: Optional[str] = None) -> list:
    if version is None:
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
        TensorBoardLogger(save_dir=save_dir, name=experiment_name, version=version, log_graph=False),
        CSVLogger(save_dir=save_dir, name=experiment_name, version=version),
    ]


def load_config(config_path: str) -> TrainingConfig:
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path_obj}")

    with open(config_path_obj, "r") as f:
        if config_path_obj.suffix.lower() in {".yaml", ".yml"}:
            config_dict = yaml.safe_load(f) or {}
        elif config_path_obj.suffix.lower() == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path_obj.suffix}")

    config = TrainingConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, _expand_env_value(value))
        else:
            LOGGER.warning("Unknown configuration key: %s", key)
    return config


def _create_vlm_datamodule(config: TrainingConfig, model: TICVLA_VLM) -> TICVLADataModule_VLM:
    return TICVLADataModule_VLM(
        train_data_dir=config.train_data_dir,
        val_data_dir=config.val_data_dir,
        test_data_dir=config.test_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_sequence_length=config.max_sequence_length,
        processor=model.processor,
        tokenizer=model.tokenizer,
    )


def _create_action_datamodule(config: TrainingConfig, model: TICVLA) -> TICVLADataModule:
    return TICVLADataModule(
        train_data_dir=config.train_data_dir,
        val_data_dir=config.val_data_dir,
        test_data_dir=config.test_data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_sequence_length=config.max_sequence_length,
        action_horizon_steps=config.action_horizon_steps,
        processor=model.processor,
        tokenizer=model.tokenizer,
    )


def _create_trainer(config: TrainingConfig, stage: str) -> pl.Trainer:
    run_version = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage_save_dir = os.path.join(config.save_dir, stage)
    stage_log_dir = os.path.join(config.log_dir, stage)
    return pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=create_callbacks(save_dir=stage_save_dir, stage=stage),
        logger=create_loggers(
            experiment_name=config.experiment_name,
            save_dir=stage_log_dir,
            version=run_version,
        ),
        devices=-1,
        accelerator="auto",
        strategy="ddp",
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TIC-VLA in two supervised stages")
    parser.add_argument("--stage", choices=["vlm", "action"], required=True, help="Training stage to run")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")
    parser.add_argument("--vlm_checkpoint", type=str, default=None, help="Override VLM checkpoint for action-stage training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config_path = args.config or f"configs/train_{args.stage}.yaml"
    config = load_config(config_path)
    LOGGER.info("Loaded configuration from: %s", config_path)

    if args.stage == "vlm":
        module = TICVLAVLMLightningModule(
            model_path=config.model_path,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=config.max_epochs,
            warmup_steps=config.warmup_steps,
        )
        data_module = _create_vlm_datamodule(config, module.model)
    else:
        vlm_checkpoint = _optional_path(args.vlm_checkpoint or config.vlm_checkpoint)
        if not vlm_checkpoint:
            raise ValueError("A VLM checkpoint is required for --stage action. Set vlm_checkpoint in the config or pass --vlm_checkpoint.")
        if not os.path.exists(vlm_checkpoint):
            raise FileNotFoundError(f"VLM checkpoint file not found: {vlm_checkpoint}")
        module = TICVLAActionLightningModule(
            model_path=config.model_path,
            vlm_checkpoint=vlm_checkpoint,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=config.max_epochs,
            warmup_steps=config.warmup_steps,
            action_horizon_steps=config.action_horizon_steps,
            action_num_layers=config.action_num_layers,
        )
        data_module = _create_action_datamodule(config, module.model)

    trainer = _create_trainer(config, args.stage)
    trainer.fit(module, datamodule=data_module)
    if config.test_data_dir:
        trainer.test(module, datamodule=data_module)


if __name__ == "__main__":
    main()
