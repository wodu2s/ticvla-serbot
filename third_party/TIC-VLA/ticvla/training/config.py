#!/usr/bin/env python3
"""
Configuration module for TIC-VLA training.

This module provides configuration classes and functions for different training scenarios.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """Configuration class for TIC-VLA training."""
    
    # Data paths
    train_data_dir: str | list[str] | None = None
    val_data_dir: str | list[str] | None = None
    test_data_dir: Optional[str] = None
    
    # Model configuration
    model_path: str = "OpenGVLab/InternVL3-1B"
    vlm_checkpoint: Optional[str] = None
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    # Data processing
    num_workers: int = 4
    max_sequence_length: int = 10
    # Number of future steps at 10 Hz for action-head targets (and ActionExpert num_chunks); e.g. 30 -> 3.0s
    action_horizon_steps: int = 30
    # Number of Transformer cross-attention blocks in the action head.
    action_num_layers: int = 3
    image_size: Tuple[int, int] = (448, 448)
    
    # Training setup
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Logging and saving
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    experiment_name: str = "ticvla"
    project_name: str = "tic-vla"

    def __post_init__(self) -> None:
        if self.train_data_dir is None:
            self.train_data_dir = [
                "${TICVLA_DATA_ROOT}/DynaNav/DynaNav_json",
                "${TICVLA_DATA_ROOT}/GND/GND_json",
                "${TICVLA_DATA_ROOT}/SCAND/SCAND_json",
            ]


def get_config(config_type: str = "standard") -> TrainingConfig:
    """
    Get a predefined configuration.
    
    Args:
        config_type: Type of configuration ("quick", "standard", "long")
        
    Returns:
        TrainingConfig instance
    """
    if config_type == "quick":
        return TrainingConfig(
            batch_size=2,
            max_epochs=5,
            image_size=(512, 512),
            max_sequence_length=5,
            warmup_steps=100,
            experiment_name="ticvla_quick"
        )
    elif config_type == "standard":
        return TrainingConfig()
    elif config_type == "long":
        return TrainingConfig(
            batch_size=8,
            max_epochs=200,
            learning_rate=5e-5,
            warmup_steps=2000,
            image_size=(1536, 1536),
            experiment_name="ticvla_long"
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def update_config(config: TrainingConfig, **kwargs) -> TrainingConfig:
    """
    Update a configuration with new values.
    
    Args:
        config: Base configuration to update
        **kwargs: New values to set
        
    Returns:
        Updated TrainingConfig instance
    """
    new_config = TrainingConfig()
    
    # Copy all attributes from the base config
    for field in TrainingConfig.__dataclass_fields__:
        setattr(new_config, field, getattr(config, field))
    
    # Update with new values
    for key, value in kwargs.items():
        if hasattr(new_config, key):
            setattr(new_config, key, value)
        else:
            raise ValueError(f"Unknown config field: {key}")
    
    return new_config
