import math
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import re
from typing import Optional, List, Dict, Any, Tuple

from ticvla.utils.vision import (
    build_transform,
    dynamic_preprocess,
    load_image,
    CrossAttentionTransformer,
)
from ticvla.models.vlm import TICVLA_VLM


class ActionExpert(nn.Module):
    """Produces a short-horizon chunked waypoint plan from model-state context."""

    def __init__(
        self,
        input_dim: int,  # Dimension of image embeddings (hidden_size from VLM)
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 3,
        num_chunks: int = 5,
        max_state_tokens: int = 16,
        kv_cache_feat_dim: int = 128,  # num_heads * head_dim (typically 2 * 64 = 128 for InternVL3-1B)
    ) -> None:
        super().__init__()

        self.num_chunks = num_chunks
        self.action_dim = action_dim
        self.max_state_tokens = max_state_tokens
        self.hidden_dim = hidden_dim

        # Down-project visual tokens (they are usually larger than 512)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        
        # Process model state: project directly from model state dimension to hidden_dim
        # model state from VLM: (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        # Initialize kv_cache_proj here (not dynamically) for easier model loading
        self.kv_cache_proj = nn.Linear(kv_cache_feat_dim, hidden_dim)
        # Token dropout rate for model state (0.1 = drop 10% of tokens)
        self.kv_cache_token_dropout_rate = 0.1

        # discrete chunk embedding table 
        self.action_chunk_embed = nn.Embedding(num_chunks, hidden_dim)

        # Transformer cross-attention stack
        self.attention_layers = nn.ModuleList(
            [
                CrossAttentionTransformer(
                    input_dim=hidden_dim,
                    hidden_dim=4 * hidden_dim,
                    num_heads=8,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

        # State encoder for robot low-dimensional states
        self.state_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Learnable positional embeddings for scalar state sequence
        self.state_pos_embedding = nn.Embedding(max_state_tokens, hidden_dim)
        self.state_pos_dropout_rate = 0.5
        self.state_pos_dropout = nn.Dropout(self.state_pos_dropout_rate)

        # MLP head → predict each atomic action component
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _process_kv_cache(self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        """
        Process model state from VLM into a sequence of embeddings.

        Single path, assuming standard InternVL layout:
        values are shaped (B, num_heads, seq_len, head_dim).
        """
        if not past_key_values:
            # Return empty tensor with hidden_dim (will be set when kv_cache_proj is created)
            return torch.zeros(
                1,
                0,
                self.hidden_dim,
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )

        # Take values from the last layer
        _, last_layer_value = past_key_values[-1]
        if last_layer_value.dim() != 4:
            raise ValueError(
                f"Expected model state values of shape (B, num_heads, seq_len, head_dim), "
                f"got {last_layer_value.shape}. "
                f"If this triggers, print the shape once and adjust _process_kv_cache."
            )

        B, num_heads, seq_len, head_dim = last_layer_value.shape

        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        value_reshaped = last_layer_value.permute(0, 2, 1, 3).contiguous()
        value_reshaped = value_reshaped.view(B, seq_len, num_heads * head_dim)
        value_reshaped = value_reshaped.to(
            device=self.kv_cache_proj.weight.device,
            dtype=self.kv_cache_proj.weight.dtype,
        )

        # Verify dimension matches (kv_cache_proj is now initialized in __init__)
        feat_dim = value_reshaped.size(-1)  # num_heads * head_dim
        if feat_dim != self.kv_cache_proj.in_features:
            raise ValueError(
                f"model state feature dimension mismatch: expected {self.kv_cache_proj.in_features}, "
                f"got {feat_dim} (num_heads={num_heads}, head_dim={head_dim}). "
                f"Please update kv_cache_feat_dim in ActionExpert.__init__ to {feat_dim}."
            )
        
        # Project directly to hidden_dim
        value_reshaped = self.kv_cache_proj(value_reshaped)

        # Token-level dropout on model state
        if self.training and self.kv_cache_token_dropout_rate > 0:
            keep_prob = 1.0 - self.kv_cache_token_dropout_rate
            token_mask = torch.rand(
                B, seq_len, 1, device=value_reshaped.device, dtype=value_reshaped.dtype
            )
            token_mask = (token_mask < keep_prob).to(value_reshaped.dtype)
            value_reshaped = value_reshaped * token_mask

        return value_reshaped  # (B, seq_len, input_dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        state: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        """
        Return (B, T, 2) tensor - predicted waypoints relative to current frame.
        
        Predicts waypoints (dx, dy) relative to current frame.
        
        Args:
            image_embeds: (B, L_img, H) image embeddings from current frame
            state: (B, S, 1) robot state tokens including velocity, displacement, and time delay
            kv_cache: Optional model state from VLM processing delayed frame context
        
        Returns:
            waypoints: (B, T, 2) tensor of waypoints [dx, dy] relative to current frame
        """
        B = image_embeds.size(0)
        action_dtype = self.down_proj.weight.dtype
        action_device = self.down_proj.weight.device
        image_embeds = image_embeds.to(device=action_device, dtype=action_dtype)
        state = state.to(device=action_device, dtype=action_dtype)

        # 1) Token preparation ------------------------------------------------
        image_embeds = self.down_proj(image_embeds)  # (B, L_img, hidden_dim)
        
        # Process model state if provided
        kv_embeds = None
        if kv_cache is not None:
            # _process_kv_cache already projects directly to hidden_dim (128 -> 512)
            kv_embeds = self._process_kv_cache(kv_cache)  # (B, L_kv, hidden_dim)
        
        state_embeds = self.state_encoder(state)  # (B, S, hidden_dim)
        state_len = state_embeds.size(1)
        if state_len > 0:
            pos_ids = torch.arange(state_len, device=state_embeds.device)
            pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
            pos_emb = self.state_pos_embedding(pos_ids).to(dtype=state_embeds.dtype)
            state_embeds = self.state_pos_dropout(state_embeds + pos_emb)
        
        # Concatenate: model state (if available) + current image + state embeddings
        if kv_embeds is not None:
            hidden_states = torch.cat([kv_embeds, image_embeds, state_embeds], dim=1)  # (B, L_kv + L_img + S, hidden_dim)
        else:
            hidden_states = torch.cat([image_embeds, state_embeds], dim=1)  # (B, L_img + S, hidden_dim)

        # 2) Build action queries -------------------------------------------
        chunk_ids = torch.arange(self.num_chunks, device=hidden_states.device)
        chunk_emb = self.action_chunk_embed(chunk_ids)[None, ...].expand(B, -1, -1)  # (B, T, H)
        waypoint_queries = chunk_emb  # (B, T, H)

        # 3) Cross-attention stack ------------------------------------------
        for layer in self.attention_layers:
            waypoint_queries = layer(waypoint_queries, hidden_states)

        # 4) Directly predict waypoints relative to current frame ------------
        waypoints = self.mlp(waypoint_queries)  # (B, T, 2) where 2 = [dx, dy] relative to current frame

        return waypoints
        

class TICVLA(nn.Module):
    def __init__(
        self,
        model_path: str = 'InternVL3-1B',
        action_horizon_steps: int = 30,
        action_num_layers: int = 3,
        train_vlm: bool = True,
    ) -> None:
        """
        Initialize TIC-VLA model.
        
        Args:
            model_path: Path to pretrained VLM model
            action_horizon_steps: Future (dx, dy) steps at 10 Hz for the action head (num_chunks)
            action_num_layers: Number of Transformer cross-attention blocks in the action head
            train_vlm: If True, fine-tune the language model on CoT/waypoint text.
        """
        super().__init__()

        # Use VLM-only model for VLM component (imported from ticvla.models.vlm)
        self.vlm_model = TICVLA_VLM(model_path=model_path)
        # Access VLM, tokenizer, and processor from the VLM model
        self.vlm = self.vlm_model.vlm
        self.tokenizer = self.vlm_model.tokenizer
        self.processor = self.vlm_model.processor
        
        self.train_vlm = train_vlm
        if not train_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
            self.vlm.eval()

        # Action expert: num_chunks = future steps at 10 Hz (0.1s per step)
        if action_horizon_steps < 1:
            raise ValueError(f"action_horizon_steps must be >= 1, got {action_horizon_steps}")
        self.action_horizon_steps = action_horizon_steps
        # The actual model state has shape (B, num_heads=2, seq_len, head_dim=64)
        # So feat_dim = 2 * 64 = 128
        kv_num_heads = 2  # Fixed value for InternVL3-1B model state (NOT from VLM config)
        kv_head_dim = 64  # Fixed value for InternVL3-1B model state
        kv_cache_feat_dim = kv_num_heads * kv_head_dim  # 2 * 64 = 128
        logging.info(f"Initializing kv_cache_proj with feat_dim={kv_cache_feat_dim} (kv_num_heads={kv_num_heads}, kv_head_dim={kv_head_dim})")
        
        self.action_expert = ActionExpert(
            input_dim=self.vlm.config.llm_config.hidden_size,
            hidden_dim=512,
            action_dim=2,  # Offset (dx, dy) relative to current frame
            num_layers=action_num_layers,
            num_chunks=action_horizon_steps,
            kv_cache_feat_dim=kv_cache_feat_dim,  # Initialize kv_cache_proj in __init__
        )

    @property
    def device(self) -> torch.device:
        return next(self.action_expert.parameters()).device

    def load_vlm_checkpoint(self, checkpoint_path: str):
        """
        Load VLM weights from a TIC-VLA checkpoint.
        
        Args:
            checkpoint_path: Path to a checkpoint file containing VLM weights
        """
        import logging
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", {})
        
        has_direct_vlm_keys = any(k.startswith("model.vlm.") for k in state_dict.keys())
        has_nested_vlm_keys = any(k.startswith("model.vlm_model.") for k in state_dict.keys())
        
        if not has_direct_vlm_keys and not has_nested_vlm_keys:
            raise ValueError(f"Checkpoint does not contain expected VLM keys. Found keys: {list(state_dict.keys())[:5]}...")
        
        if has_direct_vlm_keys:
            logging.info(f"Loading direct VLM checkpoint: {checkpoint_path}")
            logging.info("Remapping model.vlm.* checkpoint keys to VLM structure...")
            
            vlm_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model.vlm."):
                    # Remove "model." prefix to get "vlm.*"
                    new_key = key.replace("model.vlm.", "vlm.", 1)
                    vlm_state_dict[new_key] = value
            
            # Load VLM weights into vlm_model
            missing_keys, unexpected_keys = self.vlm_model.load_state_dict(vlm_state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading VLM weights: {len(missing_keys)} keys")
                if len(missing_keys) <= 10:
                    logging.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading VLM weights: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:
                    logging.warning(f"Unexpected keys: {unexpected_keys}")
            
            logging.info("Successfully loaded direct VLM weights")
        elif has_nested_vlm_keys:
            logging.info(f"Loading nested VLM checkpoint: {checkpoint_path}")
            
            vlm_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model.vlm_model.vlm."):
                    # Remove "model.vlm_model." prefix to get "vlm.*"
                    new_key = key.replace("model.vlm_model.vlm.", "vlm.", 1)
                    vlm_state_dict[new_key] = value
            
            missing_keys, unexpected_keys = self.vlm_model.load_state_dict(vlm_state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading VLM weights: {len(missing_keys)} keys")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading VLM weights: {len(unexpected_keys)} keys")
            
            logging.info("Successfully loaded nested VLM weights")

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Training forward pass with state extraction from delayed frame context.

        Expects batch keys (from `ticvla.data.policy_data.TICVLACollator`):
        - delayed_input_ids: (B, N) input IDs for delayed frame context (past images, COT, instruction)
        - delayed_attention_mask: (B, N)
        - delayed_pixel_values: (sum_tiles, C, H, W) images from delayed frame
        - delayed_image_flags: (sum_tiles, 1)
        - delayed_num_tiles_per_sample: (B,) number of tiles per sample for delayed frame
        - current_image_paths: List[List[str]] paths to current frame's single closest image
        - robot_state: (B, 5) compact state tokens [vx, vy, yaw_speed, dx, dy]
                      where dx, dy is displacement from delayed time to current time
        - time_delay: (B,) scalar delay in seconds
        - current_image_paths: List[List[str]] paths to current frame's single closest image (optional)

        Returns:
        - waypoints: (B, T, 2) where 2 = [dx, dy]; T = action_horizon_steps (10 Hz)
        - language_loss: scalar CoT/waypoint language-modeling loss
        """
        # Ensure IMG_CONTEXT token id is set for InternVL forward() token replacement
        if getattr(self.vlm, 'img_context_token_id', None) is None:
            try:
                self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            except Exception:
                pass

        # Get delayed frame inputs for VLM (from collator: delayed_input_ids, delayed_pixel_values, etc.)
        delayed_input_ids = batch.get('delayed_input_ids', batch.get('input_ids')).to(self.device)
        delayed_attention_mask = batch.get('delayed_attention_mask', batch.get('attention_mask')).to(self.device)
        delayed_pixel_values = batch.get('delayed_pixel_values', batch.get('pixel_values', None))
        delayed_image_flags = batch.get('delayed_image_flags', batch.get('image_flags', None))
        delayed_labels = batch.get('delayed_labels', batch.get('labels', None))
        delayed_num_tiles_per_sample = batch.get('delayed_num_tiles_per_sample', batch.get('num_tiles_per_sample', None))
        robot_state = batch.get('robot_state', None)
        current_image_paths = batch.get('current_image_paths', [])  # List of lists for current frame images
        
        if delayed_pixel_values is None:
            raise RuntimeError(
                "delayed_pixel_values is None. Check the action collator: delayed frame images "
                "should produce pixel values for VLM model-state extraction."
            )
        delayed_pixel_values = delayed_pixel_values.to(self.device).to(torch.bfloat16)
        if delayed_pixel_values.shape[0] == 0:
            raise RuntimeError(f"delayed_pixel_values is empty: shape={delayed_pixel_values.shape}")

        # delayed_image_flags MUST exist if delayed_pixel_values exists (created by collator)
        if delayed_image_flags is None:
            raise RuntimeError(
                "delayed_image_flags is None but delayed_pixel_values exists. "
                "This is a collator bug - image_flags should always be created when pixel_values exist."
            )
        delayed_image_flags = delayed_image_flags.to(self.device)

        if self.train_vlm and delayed_labels is None:
            raise RuntimeError("delayed_labels is required for VLM language training.")
        if delayed_labels is not None:
            delayed_labels = delayed_labels.to(self.device)

        # Forward pass through VLM to get model state. Stage 1 also computes language loss.
        vlm_kwargs = {
            'input_ids': delayed_input_ids,
            'attention_mask': delayed_attention_mask,
            'pixel_values': delayed_pixel_values,
            'image_flags': delayed_image_flags,
            'return_dict': True,
            'use_cache': True,
        }
        if self.train_vlm:
            vlm_kwargs['labels'] = delayed_labels
        if self.train_vlm:
            vlm_outputs = self.vlm(**vlm_kwargs)
        else:
            with torch.no_grad():
                vlm_outputs = self.vlm(**vlm_kwargs)
        language_loss = getattr(vlm_outputs, 'loss', None)
        if language_loss is None:
            language_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        
        # Extract past_key_values (model state) from VLM output
        past_key_values = getattr(vlm_outputs, 'past_key_values', None)
        
        # Convert DynamicCache to tuple format (single path, no fallbacks)
        if past_key_values is not None:
            cache_type = type(past_key_values).__name__

            if cache_type == 'DynamicCache':
                # DynamicCache: extract from layers attribute
                past_key_values = tuple(
                    (layer.keys, layer.values)
                    for layer in past_key_values.layers
                )
            elif not isinstance(past_key_values, tuple):
                # Unknown format - raise error instead of fallback
                raise ValueError(
                    f"Unsupported model state type: {cache_type}. "
                    f"Expected DynamicCache or tuple, got {type(past_key_values)}"
                )
            # If already tuple, use as-is

            # Detach the model state before action decoding. The VLM is trained only
            # through language_loss; action_loss should update the action expert.
            past_key_values = tuple(
                (key.detach(), value.detach())
                for key, value in past_key_values
            )
        
        # Visual features → per-sample token sequences (current frame image)
        # Extract visual features from current frame images (for action expert)
        if current_image_paths and len(current_image_paths) > 0:
            # Extract embeddings for each sample's current image
            current_image_embeds_list = []
            for paths in current_image_paths:
                if paths and len(paths) > 0:
                    # Load and extract features for the first (closest) image
                    pixel_values_current = load_image(paths[0], input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
                    with torch.no_grad():
                        img_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
                        img_embeds = img_embeds.reshape(-1, img_embeds.shape[-1])  # (total_tokens, H)
                    current_image_embeds_list.append(img_embeds)
                else:
                    # Empty embeddings if no image
                    current_image_embeds_list.append(torch.zeros(0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16))
            
            # Pad to same length
            max_tokens = max(e.shape[0] for e in current_image_embeds_list) if current_image_embeds_list else 0
            if max_tokens > 0:
                padded_embeds = []
                for e in current_image_embeds_list:
                    if e.shape[0] < max_tokens:
                        pad = torch.zeros(max_tokens - e.shape[0], e.shape[1], device=e.device, dtype=e.dtype)
                        e = torch.cat([pad, e], dim=0) # pad to the front
                    padded_embeds.append(e)
                image_embeds = torch.stack(padded_embeds, dim=0)  # (B, L_img, H)
            else:
                batch_size = delayed_input_ids.shape[0]
                image_embeds = torch.zeros(batch_size, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)
        else:
            # No images available; create empty visual tokens
            batch_size = delayed_input_ids.size(0)
            image_embeds = torch.zeros(batch_size, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)

        # State features: robot_state (5) + time_delay (1) = (6,)
        # robot_state: (B, 5) = [vx, vy, yaw_speed, dx, dy] from dataset
        robot_state = robot_state.to(self.device).to(torch.bfloat16)  # (B, 5)

        # time delay (robot input)
        time_delay = (batch["time_delay"].to(self.device).to(torch.bfloat16)).unsqueeze(1)  # (B, 1)

        # concat robot state and time delay → (B, 6)
        state = torch.cat([robot_state, time_delay], dim=1).unsqueeze(-1)  # (B, 6, 1)

        # Decode action waypoints (relative to current frame)
        # Uses model state (from delayed frame) + current image + state
        predicted_waypoints = self.action_expert(image_embeds, state, kv_cache=past_key_values)  # (B, T, 2) where 2 = [dx, dy] relative to current frame

        return {
            "waypoints": predicted_waypoints,
            "language_loss": language_loss,
        }

    @torch.inference_mode()
    def load_images(self, image_paths: list[str], input_size: int = 448, max_num: int = 1) -> torch.Tensor:
        pixel_values_list = []
        num_patches_list = []
        for p in image_paths:
            try:
                pv = load_image(p, input_size=input_size, max_num=max_num).to(torch.bfloat16).to(self.device)
                if pv is not None and pv.numel() > 0:
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.shape[0])
                # Skip empty tensors
                pass
            except Exception as e:
                logging.warning("Failed to load image %s: %s", p, e)

        return pixel_values_list, num_patches_list

    @torch.inference_mode()
    def predict(
        self,
        delayed_image_paths: list[str],
        current_image_path: str,
        instruction: str | None = None,
        robot_state: torch.Tensor | None = None,
        history: Optional[List[Dict[str, Any]]] = None,
        current_timestamp: Optional[float] = None,
        time_delay: float = 0.0,
    ) -> tuple[str, torch.Tensor, str]:
        """
        Inference pipeline producing VLM model state and decoded waypoints.
        
        Args:
            delayed_image_paths: List of image paths from delayed/historical frames (for VLM context)
            current_image_path: Single image path for current frame (for action expert)
            instruction: Navigation instruction text
            robot_state: Current robot state tensor [vx, vy, yaw_speed, dx, dy]
                        where dx, dy is displacement from delayed time to current time
            history: Optional history (not used in simplified inference)
            current_timestamp: Optional timestamp (not used in simplified inference)
            time_delay: Time delay in seconds (default 0.0)
        
        Returns:
            response: Generated assistant response (dummy, for compatibility)
            waypoints: Predicted waypoints (B, T, 2) relative to current frame, where 2 = [dx, dy], T = action_horizon_steps (10 Hz)
            prompt: Full prompt string sent to model (for visualization)
        """
        instruction = instruction or "Move forward safely and efficiently."

        # 1) Load delayed images for VLM context ------------------------------
        delayed_pixel_values_list, delayed_num_patches_list = self.load_images(
            image_paths=delayed_image_paths,
            input_size=448,
            max_num=1,
        )

        # 2) Query VLM with delayed images to get model state --------------------
        # Build prompt for VLM (same format as training - matching ticvla.data.vlm_data)
        system_text = "You are a physical mobile robot assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
        self.vlm.system_message = system_text
        
        user_text = f"The navigation instruction is: {instruction}"
        
        # Include previous waypoints if history is available (matching training format)
        previous_waypoints_text = ""
        if history is not None and current_timestamp is not None:
            # Extract previous waypoints from history (same logic as dataset)
            from ticvla.data.vlm_data import TICVLADataset_VLM
            temp_dataset = TICVLADataset_VLM([], max_sequence_length=90)
            previous_waypoints = temp_dataset._extract_previous_waypoints_from_history(history, current_timestamp)
            previous_waypoints_text = temp_dataset._format_previous_waypoints_text(previous_waypoints, current_timestamp)
        
        if previous_waypoints_text:
            user_text += f"\n{previous_waypoints_text}"
        
        # Ask the VLM to expose its reasoning before returning waypoint targets.
        user_text += (
            "\nUse reasoning to predict the future target waypoints. "
            "First describe the relevant visual/navigation evidence, then return the future target waypoints "
            "for the next 3s, 6s, and 9s in format: (x, y, theta). "
            "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s),"
            "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
        )
        
        generation_prompt = ''.join([f'Frame {i}: <image>\n' for i in range(len(delayed_num_patches_list))]) + user_text
        full_prompt_text = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"
        
        # Generate response (for compatibility, but we only need model state)
        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.7)
        with torch.no_grad():
            generated_response = self.vlm.chat(
                self.tokenizer, 
                torch.cat(delayed_pixel_values_list, dim=0), 
                generation_prompt,
                generation_config,
                history=None, 
                return_history=False, 
                num_patches_list=delayed_num_patches_list
            )
        
        # Build full conversation to extract model state
        messages = [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': [{'type': 'image', 'image': p} for p in delayed_image_paths] + [{'type': 'text', 'text': user_text}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': generated_response}]}
        ]
        
        text_batch = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        if isinstance(text_batch, list):
            text_batch = text_batch[0] if len(text_batch) > 0 else ""
        
        # Build queries with image tokens
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        num_image_token = 256  # Default for InternVL
        
        query = text_batch
        if '<image>' in query:
            for tiles_for_image in delayed_num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN
                    query = query.replace('<image>', image_tokens, 1)
        else:
            prepend = []
            for tiles_for_image in delayed_num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN)
            if len(prepend) > 0:
                query = '\n'.join(prepend) + '\n' + query

        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer([query], return_tensors='pt', padding=True)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        pixel_values = torch.cat(delayed_pixel_values_list, dim=0)
        total_patches = pixel_values.shape[0]
        image_flags = torch.ones(total_patches, 1, dtype=torch.long, device=self.device)

        # Forward pass through VLM to get model state
        with torch.no_grad():
            vlm_outputs = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                return_dict=True,
                use_cache=True,
            )
        
        past_key_values = vlm_outputs.past_key_values
        
        # Convert DynamicCache to tuple format
        if past_key_values is not None and hasattr(past_key_values, 'layers'):
            past_key_values = tuple(
                (layer.keys, layer.values) 
                for layer in past_key_values.layers
            )
        elif past_key_values is None:
            past_key_values = tuple()

        # 3) Extract visual embeddings from CURRENT image for action expert ---
        current_pixel_values = load_image(current_image_path, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            image_embeds = self.vlm.extract_feature(current_pixel_values)  # (num_tiles, num_img_tokens, H)
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1]).unsqueeze(0)  # (1, L_img, H)

        # 4) Prepare robot state tokens --------------------------------------
        if robot_state is None:
            robot_state_tensor = torch.zeros(5, device=self.device, dtype=torch.bfloat16)
        else:
            if not torch.is_tensor(robot_state):
                robot_state = torch.tensor(robot_state)
            robot_state_tensor = robot_state.to(self.device, dtype=torch.bfloat16).view(-1)  # (5,)

        time_delay_tensor = torch.tensor([time_delay], device=self.device, dtype=torch.bfloat16)
        
        # Concat robot state and time delay → (6,)
        state = torch.cat([robot_state_tensor, time_delay_tensor], dim=0).unsqueeze(0).unsqueeze(-1)  # (1, 6, 1)

        # 5) Decode trajectory (relative to current frame) -------------------
        waypoints = self.action_expert(image_embeds, state, kv_cache=past_key_values)  # (1, T, 2) where 2 = [dx, dy] relative to current frame
        
        # Return waypoints directly (relative to current frame) - no cumsum transformation
        return generated_response, waypoints, full_prompt_text
