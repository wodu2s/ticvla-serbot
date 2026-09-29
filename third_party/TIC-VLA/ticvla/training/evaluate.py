#!/usr/bin/env python3
"""
Test script for TIC-VLA model.
Loads data, tests model output, and provides visualizations.
"""

import os
import json
import random
import logging
import time
import textwrap
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from transformers import AutoProcessor
from ticvla.models.ticvla import TICVLA

from ticvla.data.policy_data import TICVLADataset
from ticvla.training.config import TrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_test_data_dir() -> str:
    """Default to the Hugging Face DynaNav JSON dataset."""
    return os.getenv(
        "TICVLA_TEST_DATA_DIR",
        os.path.join(
            os.getenv("TICVLA_DATA_ROOT", "data/hugging_face"),
            "DynaNav",
            "DynaNav_json",
        ),
    )


@dataclass
class TestConfig:
    """Configuration for testing."""
    # Model settings
    model_path: str = os.getenv(
        "TICVLA_BASE_MODEL_PATH",
        os.getenv("TICVLA_MODEL_PATH", "OpenGVLab/InternVL3-1B"),
    )
    ckpt_path: str = os.getenv("TICVLA_CHECKPOINT_PATH", "checkpoints/ticvla.ckpt")
    
    # Data settings
    data_dir: str = _default_test_data_dir()
    max_sequence_length: int = 90
    action_horizon_steps: int = 30
    num_test_samples: int = 10
    
    # Visualization settings
    save_plots: bool = True
    plot_dir: str = os.getenv("TICVLA_TEST_OUTPUT_DIR", "outputs/test_plots")
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True


class TICVLATester:
    """Test class for TIC-VLA model."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        logger.info(f"Loading TIC-VLA model from {config.model_path}")
        self.ticvla = TICVLA(
            model_path=config.model_path,
            action_horizon_steps=config.action_horizon_steps,
        ).to(self.device)
        
        # Load both VLM and action expert from the same checkpoint
        if config.ckpt_path and os.path.exists(config.ckpt_path):
            logger.info(f"Loading checkpoint (VLM + action expert): {config.ckpt_path}")
            checkpoint = torch.load(config.ckpt_path, map_location='cpu')
            state_dict = checkpoint.get("state_dict", {})
            
            # Debug: Show available keys
            all_keys = list(state_dict.keys())
            logger.info(f"Total keys in checkpoint: {len(all_keys)}")
            logger.info(f"Sample keys (first 20): {all_keys[:20]}")
            
            # Extract VLM weights
            # PyTorch Lightning saves as "model.vlm.*" (since model is stored as self.model)
            vlm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.vlm."):
                    new_key = k[len("model.vlm."):]
                    vlm_state_dict[new_key] = v
                elif k.startswith("vlm."):
                    # Fallback for non-Lightning checkpoints
                    new_key = k[len("vlm."):]
                    vlm_state_dict[new_key] = v
            
            # Load VLM weights
            if vlm_state_dict:
                logger.info(f"Found {len(vlm_state_dict)} VLM keys in checkpoint")
                try:
                    missing_keys, unexpected_keys = self.ticvla.vlm.load_state_dict(vlm_state_dict, strict=False)
                    if missing_keys or unexpected_keys:
                        logger.warning(f"VLM key mismatch: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
                        if len(missing_keys) <= 5:
                            logger.warning(f"  Missing VLM keys: {list(missing_keys)[:5]}")
                        if len(unexpected_keys) <= 5:
                            logger.warning(f"  Unexpected VLM keys: {list(unexpected_keys)[:5]}")
                    else:
                        logger.info("✅ VLM weights loaded successfully from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load VLM weights directly: {e}")
                    # Fallback: try using load_vlm_checkpoint if it exists
                    if hasattr(self.ticvla, 'load_vlm_checkpoint'):
                        try:
                            self.ticvla.load_vlm_checkpoint(config.ckpt_path)
                            logger.info("✅ VLM weights loaded via load_vlm_checkpoint method")
                        except Exception as e2:
                            logger.error(f"Failed to load VLM checkpoint: {e2}")
            else:
                logger.warning("No VLM keys found in checkpoint, trying load_vlm_checkpoint method")
                # Try using load_vlm_checkpoint as fallback
                if hasattr(self.ticvla, 'load_vlm_checkpoint'):
                    try:
                        self.ticvla.load_vlm_checkpoint(config.ckpt_path)
                        logger.info("✅ VLM weights loaded via load_vlm_checkpoint method (fallback)")
                    except Exception as e:
                        logger.warning(f"Could not load VLM via load_vlm_checkpoint: {e}")
            
            # Extract action_expert weights
            # PyTorch Lightning saves as "model.action_expert.*" (since model is stored as self.model)
            action_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.action_expert."):
                    new_key = k[len("model.action_expert."):]
                    action_state_dict[new_key] = v
                elif k.startswith("action_expert."):
                    # Fallback for non-Lightning checkpoints
                    new_key = k[len("action_expert."):]
                    action_state_dict[new_key] = v
            
            # Load action expert weights
            if action_state_dict:
                logger.info(f"Found {len(action_state_dict)} action expert keys in checkpoint")
                missing_keys, unexpected_keys = self.ticvla.action_expert.load_state_dict(action_state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    logger.warning(f"Action expert key mismatch: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
                    if len(missing_keys) <= 10:
                        logger.warning(f"  Missing action expert keys: {list(missing_keys)[:10]}")
                    if len(unexpected_keys) <= 10:
                        logger.warning(f"  Unexpected action expert keys: {list(unexpected_keys)[:10]}")
                else:
                    logger.info("✅ Action expert weights loaded successfully from checkpoint")
            else:
                logger.warning("No action_expert keys found in checkpoint")
                logger.warning(f"  Available keys (first 30): {all_keys[:30]}")
        else:
            if config.ckpt_path:
                logger.error(f"❌ Checkpoint not found: {config.ckpt_path}")
            else:
                logger.warning("⚠️  No checkpoint path provided. Using base pretrained VLM and randomly initialized action expert.")
        
        self.ticvla.eval()
        
        # Load dataset
        logger.info(f"Loading dataset from {config.data_dir}")
        self.dataset = TICVLADataset(
            data_dir=config.data_dir,
            max_sequence_length=config.max_sequence_length,
            action_horizon_steps=config.action_horizon_steps,
        )
        
        # Create plot directory
        if config.save_plots:
            os.makedirs(config.plot_dir, exist_ok=True)
    
    def _remap_image_path(self, img_path: str, sample_path: Path) -> str:
        """Delegate image-path resolution to the dataset loader."""
        return self.dataset._remap_image_path(img_path, sample_path)
    
    def test_model_inference(self, sample_idx: int = 0) -> Dict[str, Any]:
        """Test model inference on a single sample."""
        logger.info(f"Testing model inference on sample {sample_idx}")
        
        # Get sample and sample path for remapping
        sample = self.dataset[sample_idx]
        sample_path = self.dataset.samples[sample_idx]
        
        # Load raw JSON data to extract history and timestamp for previous waypoints
        with open(sample_path, 'r') as f:
            raw_data = json.load(f)
        history = raw_data.get('history', [])
        current_timestamp = raw_data.get('timestamp', 0.0)
        
        # Extract data from dataset output
        # The dataset returns: messages, delayed_images, current_image, robot_state, waypoints
        messages = sample.get("messages", [])
        delayed_image_paths = sample.get("delayed_images", [])  # Images for VLM context
        current_image_path = sample.get("current_image", "")  # Current frame image for action expert
        robot_state = sample.get("robot_state", None)
        
        # Remap delayed image paths for VLM context
        delayed_images = []
        for img_path in delayed_image_paths:
            if img_path:
                remapped_path = self._remap_image_path(img_path, sample_path)
                if os.path.exists(remapped_path):
                    delayed_images.append(remapped_path)
                else:
                    logger.warning(f"Delayed image path does not exist: {remapped_path} (original: {img_path})")
        
        # Remap current image path for action expert
        if current_image_path:
            current_image_path = self._remap_image_path(current_image_path, sample_path)
            if not os.path.exists(current_image_path):
                logger.warning(f"Current image path does not exist: {current_image_path}")
                current_image_path = ""
        
        # Extract instruction and past trajectory from messages - REQUIRED, no defaults
        # NOTE: During testing, model should GENERATE COT (not use GT COT)
        instruction = None
        past_trajectory_text = None
        full_user_text = None
        
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        full_user_text = text_content
                        # Look for "The navigation instruction is: ..." pattern
                        if "The navigation instruction is:" in text_content:
                            # Extract instruction
                            parts = text_content.split("The navigation instruction is:", 1)
                            if len(parts) > 1:
                                rest = parts[1]
                                # Instruction is the first line after "The navigation instruction is:"
                                instruction = rest.split("\n")[0].strip()
                                # Past trajectory is everything after the instruction line, but BEFORE waypoint instructions
                                remaining_lines = rest.split("\n", 1)
                                if len(remaining_lines) > 1:
                                    past_trajectory_text = remaining_lines[1].strip()
                                    # Remove waypoint instruction text if it's included (to avoid duplication)
                                    waypoint_markers = [
                                        "\nUse reasoning to predict future target waypoints",
                                        "\nReturn the future target waypoints",
                                        "Use reasoning to predict future target waypoints",
                                        "Return the future target waypoints"
                                    ]
                                    for marker in waypoint_markers:
                                        if marker in past_trajectory_text:
                                            past_trajectory_text = past_trajectory_text.split(marker)[0].strip()
                                            break
                                    # Also check for the pattern "waypoints for next 5s and 10s"
                                    if "waypoints for next 5s and 10s" in past_trajectory_text:
                                        past_trajectory_text = past_trajectory_text.split("waypoints for next 5s and 10s")[0].strip()
                                        # Clean up any trailing instruction text
                                        for suffix in ["Return the future target", "Use reasoning to predict future target"]:
                                            if past_trajectory_text.endswith(suffix):
                                                past_trajectory_text = past_trajectory_text[:-len(suffix)].strip()
                                                break
                            break
                        elif "instruction" in text_content.lower():
                            # Try to extract instruction from text
                            lines = text_content.split("\n")
                            for line in lines:
                                if "instruction" in line.lower() and ":" in line:
                                    instruction = line.split(":", 1)[-1].strip()
                                    break
                            if not instruction:
                                instruction = text_content.split("\n")[0].strip()
                            break
                if instruction:
                    break
        
        # Instruction is REQUIRED - raise error if not found
        if not instruction:
            raise ValueError(f"No instruction found in sample {sample_idx}. messages: {messages}")

        # Check if we have valid images
        if not delayed_images:
            logger.error(f"No valid delayed images found in sample {sample_idx}")
            logger.error(f"Sample path: {sample_path}")
            logger.error(f"Delayed image paths from dataset: {delayed_image_paths}")
            raise ValueError(f"No valid delayed images found in sample {sample_idx}")
        
        if not current_image_path:
            logger.error(f"No valid current image found in sample {sample_idx}")
            raise ValueError(f"No valid current image found in sample {sample_idx}")

        # Extract time_delay from sample if available
        time_delay = float(sample.get("time_delay", 0.0)) if torch.is_tensor(sample.get("time_delay")) else sample.get("time_delay", 0.0)
        
        # Model inference with timing
        # predict method signature: predict(delayed_image_paths, current_image_path, instruction, robot_state, history, current_timestamp, time_delay)
        start_time = time.time()
        predict_result = self.ticvla.predict(
            delayed_image_paths=delayed_images,
            current_image_path=current_image_path,
            instruction=instruction,
            robot_state=robot_state,
            history=history,  # Extract from raw JSON for previous_waypoints_text
            current_timestamp=current_timestamp,  # Extract from raw JSON
            time_delay=time_delay,
        )
        inference_time = time.time() - start_time
        
        # Handle return value (should be 3 items: response, waypoints, prompt)
        if len(predict_result) == 3:
            response, waypoints, prompt_text = predict_result
            # prompt_text already includes system + user (with past trajectory)
        else:
            # Fallback if predict doesn't return prompt
            response, waypoints = predict_result
            system_text = "You are a physical mobile robot assigned to perform navigation tasks.\n" + \
                          "You are provided with a video consisting of visual observations, including historical and current frames.\n"
            user_text = f"The navigation instruction is: {instruction}\n"
            if past_trajectory_text:
                user_text += f"{past_trajectory_text}\n"
            user_text += "Use reasoning to predict future target waypoints for next 5s and 10s in format: (x, y). Each waypoint represents the cumulative offset from the current position (total displacement over 5s or 10s), where x is positive for forward, y is positive for left."
            prompt_text = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"
        
        logger.info(f"model state inference time for sample {sample_idx}: {inference_time:.3f} seconds")
        
        # Print instruction, prompt (includes past trajectory), and response
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx} - Instruction, Prompt, and Response")
        print(f"{'='*80}")
        print(f"\n📋 INSTRUCTION:")
        print(f"{instruction}")
        print(f"\n💬 FULL PROMPT (sent to model - includes past trajectory):")
        print(f"{prompt_text}")
        print(f"\n🤖 MODEL RESPONSE (generated COT + waypoints):")
        print(f"{response}")
        print(f"{'='*80}\n")
    
        waypoints_np = waypoints.detach().float().cpu().numpy()
        if waypoints_np.ndim == 3:   # (B,T,2)
            waypoints_np = waypoints_np[0]

        return {
            "response": response,
            "instruction": instruction,
            "prompt": prompt_text,
            "past_trajectory": past_trajectory_text,  # Past trajectory text if available
            "gt_waypoints": sample.get("waypoints", torch.zeros(30, 2)),  # (T, 2) where 2 = [dx, dy]
            "pred_waypoints": waypoints_np,
            "messages": messages,  # Use messages from dataset
            "sample_idx": sample_idx,
            "sample_path": sample_path,
            "inference_time": inference_time,
            "delayed_image_paths": delayed_images,  # Store delayed image paths for visualization
            "current_image_path": current_image_path,  # Store current image path
            "robot_state": robot_state,
            "time_delay": time_delay,
            "current_timestamp": current_timestamp,
        }

    def visualize_trajectory(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize the trajectory comparison."""
        gt_waypoints = result["gt_waypoints"]
        if torch.is_tensor(gt_waypoints):
            gt_waypoints = gt_waypoints.float().cpu().numpy()
        pred_waypoints = result["pred_waypoints"]
        print(f"pred_waypoints: {pred_waypoints}")
        
        # Create subplots: 1. Trajectory, 2. X vs time, 3. Y vs time
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Trajectory Plot (Y vs X)
        # Plot Ground Truth trajectory
        if len(gt_waypoints) > 0:
            ax1.plot(gt_waypoints[:, 0], gt_waypoints[:, 1], 'b-o', label='Ground Truth', linewidth=2, markersize=4, alpha=0.7)
            # Mark start
            ax1.plot(gt_waypoints[0, 0], gt_waypoints[0, 1], 'bx', markersize=10, markeredgewidth=2)
        
        # Plot Predicted trajectory
        if len(pred_waypoints) > 0:
            ax1.plot(pred_waypoints[:, 0], pred_waypoints[:, 1], 'r-o', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
            # Mark start
            ax1.plot(pred_waypoints[0, 0], pred_waypoints[0, 1], 'rx', markersize=10, markeredgewidth=2)
        
        ax1.set_xlabel('X Position (m) - Forward')
        ax1.set_ylabel('Y Position (m) - Left')
        ax1.set_title('Trajectory Comparison (y vs x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Time steps for plotting
        t_gt = np.arange(len(gt_waypoints))
        t_pred = np.arange(len(pred_waypoints))
        
        # 2. X vs Time
        if len(gt_waypoints) > 0:
            ax2.plot(t_gt, gt_waypoints[:, 0], 'b-o', label='GT X', linewidth=2, markersize=4, alpha=0.7)
        if len(pred_waypoints) > 0:
            ax2.plot(t_pred, pred_waypoints[:, 0], 'r-o', label='Pred X', linewidth=2, markersize=4, alpha=0.7)
            
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('X Position (m)')
        ax2.set_title('X (Forward) Cumulative vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Y vs Time
        if len(gt_waypoints) > 0:
            ax3.plot(t_gt, gt_waypoints[:, 1], 'b-o', label='GT Y', linewidth=2, markersize=4, alpha=0.7)
        if len(pred_waypoints) > 0:
            ax3.plot(t_pred, pred_waypoints[:, 1], 'r-o', label='Pred Y', linewidth=2, markersize=4, alpha=0.7)
            
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('Y (Left) Cumulative vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    logger.info(f"✅ Saved trajectory plot to {save_path} ({file_size} bytes)")
                else:
                    logger.error(f"❌ Failed to save trajectory: file was not created at {save_path}")
            except Exception as e:
                logger.error(f"❌ Error saving trajectory to {save_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            plt.show()
        plt.close()

    @staticmethod
    def _format_tensor_like(value: Any, precision: int = 3) -> str:
        """Format tensors/lists/scalars compactly for plot annotations."""
        if value is None:
            return "None"
        if torch.is_tensor(value):
            values = value.detach().float().cpu().view(-1).tolist()
        elif isinstance(value, np.ndarray):
            values = value.astype(float).reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            try:
                return f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                return str(value)

        formatted = []
        for item in values:
            try:
                formatted.append(f"{float(item):.{precision}f}")
            except (TypeError, ValueError):
                formatted.append(str(item))
        return "[" + ", ".join(formatted) + "]"

    @staticmethod
    def _wrap_and_truncate(text: str, width: int = 115, max_chars: int = 2200) -> str:
        """Keep long prompts readable inside matplotlib text boxes."""
        text = str(text or "")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        wrapped_lines = []
        for line in text.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=width) or [""])
        return "\n".join(wrapped_lines)

    def visualize_model_inputs(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Plot the exact sample inputs and generated response together."""
        delayed_images = result.get("delayed_image_paths", [])
        current_image = result.get("current_image_path", "")
        image_items = [(f"VLM reasoning input {i + 1}", p) for i, p in enumerate(delayed_images)]
        if current_image:
            image_items.append(("Action vision encoder input", current_image))

        n_images = max(1, len(image_items))
        n_cols = min(3, n_images)
        n_rows = int(np.ceil(n_images / n_cols))
        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 13))
        gs = fig.add_gridspec(
            n_rows + 3,
            n_cols,
            height_ratios=[*[1.5] * n_rows, 0.9, 1.25, 1.0],
            hspace=0.35,
            wspace=0.25,
        )

        for i in range(n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            if i >= len(image_items):
                continue

            title, img_path = image_items[i]
            try:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{title}\n{os.path.basename(img_path)}", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"{title}\nFailed to load:\n{img_path}\n{e}", ha="center", va="center", fontsize=8)

        robot_state = self._format_tensor_like(result.get("robot_state"))
        time_delay = self._format_tensor_like(result.get("time_delay"))
        metadata = "\n".join([
            f"Sample index: {result.get('sample_idx', 'N/A')}",
            f"Sample path: {result.get('sample_path', 'N/A')}",
            f"Current timestamp: {self._format_tensor_like(result.get('current_timestamp'))} s",
            f"Time delay: {time_delay} s",
            "Robot state [vx, vy, yaw_rate, delayed_to_current_dx, delayed_to_current_dy]:",
            robot_state,
            f"VLM reasoning frames processed: {len(delayed_images)}",
            f"Action vision encoder frame: {current_image or 'N/A'}",
        ])

        ax_meta = fig.add_subplot(gs[n_rows, :])
        ax_meta.axis("off")
        ax_meta.text(
            0.02,
            0.98,
            metadata,
            transform=ax_meta.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
        )

        prompt_text = (
            f"[INSTRUCTION]\n{result.get('instruction', 'N/A')}\n\n"
            f"[FULL PROMPT SENT TO MODEL]\n{result.get('prompt', 'N/A')}"
        )
        ax_prompt = fig.add_subplot(gs[n_rows + 1, :])
        ax_prompt.axis("off")
        ax_prompt.text(
            0.02,
            0.98,
            self._wrap_and_truncate(prompt_text),
            transform=ax_prompt.transAxes,
            fontsize=7,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.9),
        )

        response_text = (
            f"[MODEL RESPONSE / REASONING]\n{result.get('response', 'N/A')}\n\n"
            f"[PREDICTED FINAL WAYPOINT]\n"
            f"{result.get('pred_waypoints', np.array([]))[-1] if len(result.get('pred_waypoints', [])) > 0 else 'N/A'}"
        )
        ax_response = fig.add_subplot(gs[n_rows + 2, :])
        ax_response.axis("off")
        ax_response.text(
            0.02,
            0.98,
            self._wrap_and_truncate(response_text, max_chars=1800),
            transform=ax_response.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew", alpha=0.9),
        )

        plt.suptitle(f"Model Inputs and Response - Sample {result.get('sample_idx', 'N/A')}", fontsize=14, y=0.995)

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved model input/response plot to {save_path}")
            except Exception as e:
                logger.error(f"Error saving model input plot to {save_path}: {e}")
        else:
            plt.show()
        plt.close()
    
    def visualize_image_sequence(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize the image sequence with instruction, prompt, and response."""
        # Get delayed images and current image
        delayed_images = result.get("delayed_image_paths", [])
        current_image = result.get("current_image_path", "")
        images = delayed_images.copy()
        if current_image:
            images.append(current_image)  # Add current image at the end
        instruction = result.get("instruction", "N/A")
        prompt = result.get("prompt", "N/A")
        response = result.get("response", "N/A")
        
        if not images:
            # Try to extract from messages as fallback
            messages = result.get("messages", [])
            sample_path = result.get("sample_path")
            for message in messages:
                if message.get("role") == "user":
                    for content in message.get("content", []):
                        if content.get("type") == "image":
                            img_path = content.get("image", "")
                            if sample_path:
                                img_path = self._remap_image_path(img_path, sample_path)
                            if img_path and os.path.exists(img_path):
                                images.append(img_path)
        
        if not images:
            logger.warning("No images found in the sample")
            return
        
        # Create figure with images on top and text below
        n_images = min(len(images), 6)  # Limit to 6 images
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Top row: Images
        for i, img_path in enumerate(images[-n_images:]):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[0, col])
            try:
                from PIL import Image
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'Frame {i+1}', fontsize=10)
                ax.axis('off')
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                ax.text(0.5, 0.5, f'Image {i+1}\n(Load Failed)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')
        
        # Hide unused image subplots
        for i in range(n_images, 3):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')
        
        # Middle row: Instruction and Prompt (prompt includes past trajectory)
        ax_instruction = fig.add_subplot(gs[1, :])
        ax_instruction.axis('off')
        instruction_text = f"[INSTRUCTION]:\n{instruction}\n"
        instruction_text += f"\n[FULL PROMPT (includes past trajectory)]:\n{prompt}"
        ax_instruction.text(0.05, 0.95, instruction_text, transform=ax_instruction.transAxes,
                           fontsize=8, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                           wrap=True)
        
        # Bottom row: Model Response
        ax_response = fig.add_subplot(gs[2, :])
        ax_response.axis('off')
        response_text = f"[MODEL RESPONSE]:\n{response}"
        ax_response.text(0.05, 0.95, response_text, transform=ax_response.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                        wrap=True)
        
        plt.suptitle(f'Sample {result.get("sample_idx", "N/A")} - Images, Instruction, Prompt, and Response', 
                    fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved image sequence plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def create_response_analysis(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Create a text analysis of the model response."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        # Prepare text content
        inference_time = result.get('inference_time', 0.0)
        gt_waypoints = result['gt_waypoints']
        if torch.is_tensor(gt_waypoints):
            gt_waypoints = gt_waypoints.float().cpu().numpy()
        
        text_content = f"""
            Model Response Analysis - Sample {result['sample_idx']}

            GROUND TRUTH WAYPOINT (final):
            {gt_waypoints[-1] if len(gt_waypoints) > 0 else 'N/A'}

            MODEL RESPONSE:
            {result['response']}

            PREDICTED WAYPOINT (final):
            {result['pred_waypoints'][-1] if len(result['pred_waypoints']) > 0 else 'N/A'}

            ANALYSIS:
            - Response Length: {len(result['response'])} characters
            - Response Quality: {'Good' if len(result['response']) > 50 else 'Short'}
            - TIC-VLA Inference Time: {inference_time:.3f} seconds
        """
        
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.title('Model Response Analysis', fontsize=16, pad=20)
        
        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    logger.info(f"✅ Saved response analysis to {save_path} ({file_size} bytes)")
                else:
                    logger.error(f"❌ Failed to save analysis: file was not created at {save_path}")
            except Exception as e:
                logger.error(f"❌ Error saving analysis to {save_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            plt.show()
        plt.close()
    
    def run_comprehensive_test(self):
        """Run comprehensive testing with visualizations."""
        logger.info("Starting comprehensive model testing")
        
        results = []
        
        # Test multiple samples
        for i in range(min(self.config.num_test_samples, len(self.dataset))):
            logger.info(f"Testing sample {i+1}/{self.config.num_test_samples}")
            id = random.randint(0, len(self.dataset) - 1)
            
            try:
                result = self.test_model_inference(id)
                results.append(result)
                
                # Create visualizations
                if self.config.save_plots:
                    base_path = f"{self.config.plot_dir}/sample_{id}"
                    
                    # Trajectory visualization
                    self.visualize_trajectory(
                        result,
                        save_path=f"{base_path}_trajectory.png"
                    )

                    # Model input and response visualization
                    self.visualize_model_inputs(
                        result,
                        save_path=f"{base_path}_inputs_response.png",
                    )
                    
                    # Image sequence visualization (DISABLED per user request)
                    # self.visualize_image_sequence(
                    #     result, 
                    #     save_path=f"{base_path}_images.png"
                    # )
                    
            except Exception as e:
                logger.error(f"Error testing sample {id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results: List[Dict[str, Any]]):
        """Create a summary report of all test results."""
        if not results:
            logger.warning("No results to summarize")
            return
        
        # Calculate statistics
        response_lengths = []
        inference_times = []
        ade_list = []
        fde_list = []
        
        for result in results:
            response_lengths.append(len(result["response"]))
            if "inference_time" in result:
                inference_times.append(result["inference_time"])
        
        # Compute ADE/FDE across trajectories
        for result in results:
            try:
                # Cast to float32 before converting to NumPy to avoid bfloat16 issues
                gt_waypoints = result["gt_waypoints"]
                if torch.is_tensor(gt_waypoints):
                    gt_traj = gt_waypoints.float().cpu().numpy()
                else:
                    gt_traj = np.array(gt_waypoints)
                
                pred_traj = result["pred_waypoints"]
                gt_xy = gt_traj[:, :2]
                pred_xy = pred_traj[:, :2]
                min_len = min(len(gt_xy), len(pred_xy))
                if min_len == 0:
                    continue
                dists = np.linalg.norm(pred_xy[:min_len] - gt_xy[:min_len], axis=1)
                ade_list.append(float(np.mean(dists)))
                fde_list.append(float(np.linalg.norm(pred_xy[min_len - 1] - gt_xy[min_len - 1])))
            except Exception as e:
                logger.warning(f"Failed to compute ADE/FDE for a result: {e}")
                continue
        
        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Response length distribution
        ax1.hist(response_lengths, bins=10, alpha=0.7, color='blue')
        ax1.set_title('Response Length Distribution')
        ax1.set_xlabel('Response Length (characters)')
        ax1.set_ylabel('Frequency')
        
        # Plot 2: Inference time distribution
        if inference_times:
            ax2.hist(inference_times, bins=10, alpha=0.7, color='green')
            ax2.set_title('TIC-VLA Inference Time Distribution')
            ax2.set_xlabel('Inference Time (seconds)')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('TIC-VLA Inference Time Distribution')
        
        # Plot 3: ADE/FDE distribution
        if ade_list:
            ax3.hist(ade_list, bins=10, alpha=0.7, color='orange', label='ADE')
            if fde_list:
                ax3.hist(fde_list, bins=10, alpha=0.7, color='red', label='FDE')
            ax3.set_title('Trajectory Error Distribution')
            ax3.set_xlabel('Error (meters)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No error data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Trajectory Error Distribution')
        
        # Plot 4: Summary text
        ax4.axis('off')
        # Prepare ADE/FDE summary strings
        ade_summary = f"{np.mean(ade_list):.3f}" if ade_list else "N/A"
        fde_summary = f"{np.mean(fde_list):.3f}" if fde_list else "N/A"
        
        # Prepare timing summary strings
        if inference_times:
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            total_time = np.sum(inference_times)
            timing_summary = f"""
            Average TIC-VLA Inference Time: {avg_time:.3f} seconds
            Min Inference Time: {min_time:.3f} seconds
            Max Inference Time: {max_time:.3f} seconds
            Total Inference Time: {total_time:.2f} seconds"""
        else:
            timing_summary = "\n            TIC-VLA Inference Time: N/A"
        
        summary_text = f"""
            Test Summary Report - TIC-VLA

            Total Samples Tested: {len(results)}
            Successful Predictions: {len(results)}

            Average Response Length: {np.mean(response_lengths):.1f} chars
            Min Response Length: {min(response_lengths)}
            Max Response Length: {max(response_lengths)}
            {timing_summary}

            ADE (average over samples): {ade_summary}
            FDE (average over samples): {fde_summary}

            Model: {self.config.model_path}
            Device: {self.device}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('TIC-VLA Model Test Summary', fontsize=16)
        plt.tight_layout()
        
        if self.config.save_plots:
            summary_path = f"{self.config.plot_dir}/test_summary.png"
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                if os.path.exists(summary_path):
                    file_size = os.path.getsize(summary_path)
                    logger.info(f"✅ Saved summary report to {summary_path} ({file_size} bytes)")
                else:
                    logger.error(f"❌ Failed to save summary: file was not created at {summary_path}")
            except Exception as e:
                logger.error(f"❌ Error saving summary to {summary_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            plt.show()
        plt.close()

def main():
    """Main function to run the test."""
    # fix random seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    # Configuration
    config = TestConfig(
        data_dir=_default_test_data_dir(),
        num_test_samples=10,
        save_plots=True
    )
    
    # Check if data directory exists
    if not os.path.exists(config.data_dir):
        logger.error(f"Data directory {config.data_dir} does not exist!")
        logger.info("Please update the data_dir in TestConfig to point to your dataset location.")
        return
    
    # Create tester and run tests
    tester = TICVLATester(config)
    results = tester.run_comprehensive_test()
    
    logger.info(f"Testing completed! Generated {len(results)} test results.")
    if config.save_plots:
        logger.info(f"Plots saved to {config.plot_dir}/")

if __name__ == "__main__":
    main()

