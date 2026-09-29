"""
TIC-VLA VLM Model for training just the Vision-Language Model component.

This module provides a simplified version of TIC-VLA that only includes
the VLM component, without the action expert. Used for training the VLM
to predict future waypoints from visual observations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from typing import Optional

from ticvla.data.vlm_data import TICVLADataset_VLM, TICVLACollator_VLM, TICVLADataModule_VLM


class TICVLA_VLM(nn.Module):
    """
    Vision-Language Model only version of TICVLA.
    
    This model only includes the VLM component for language modeling and
    waypoint prediction. No action expert is included.
    """
    
    def __init__(self, model_path: str = 'InternVL3-1B') -> None:
        """
        Initialize the VLM-only model.
        
        Args:
            model_path: Path to pretrained VLM model (e.g., InternVL3-1B)
        """
        super().__init__()

        # VLM and tokenizer
        self.vlm = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )

        for param in self.vlm.vision_model.parameters():
            param.requires_grad = False
        self.vlm.vision_model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.vlm.parameters()).device

    def train(self, mode: bool = True):
        """Keep the frozen vision backbone in eval mode while fine-tuning language weights."""
        super().train(mode)
        if hasattr(self, "vlm") and hasattr(self.vlm, "vision_model"):
            self.vlm.vision_model.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Training forward pass for VLM-only model.
        
        Expects batch keys (from `ticvla.data.vlm_data.TICVLACollator_VLM`):
        - input_ids: (B, N) tokenized input text
        - attention_mask: (B, N) attention mask
        - labels: (B, N) labels for language modeling (-100 for tokens to ignore)
        - pixel_values: (sum_tiles, C, H, W) image pixel values
        - image_flags: (sum_tiles, 1) flags indicating which tokens are images
        - num_tiles_per_sample: (B,) number of tiles per sample
        
        Returns:
        - language_loss: scalar tensor for language modeling loss
        """
        # Ensure IMG_CONTEXT token id is set for InternVL forward() token replacement
        if getattr(self.vlm, 'img_context_token_id', None) is None:
            try:
                self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            except Exception:
                pass

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        pixel_values = batch.get('pixel_values', None)
        image_flags = batch.get('image_flags', None)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        if image_flags is not None:
            image_flags = image_flags.to(self.device)

        # Language modeling loss
        vlm_outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=labels,
            return_dict=True,
        )
        language_loss = vlm_outputs.loss

        return language_loss

    def generate(
        self, 
        pixel_values: torch.Tensor,
        prompt: str,
        generation_config: dict,
        num_patches_list: Optional[list[int]] = None,
    ) -> str:
        """
        Generate text response from VLM.
        
        Args:
            pixel_values: (N, C, H, W) tensor of image pixel values
            prompt: Text prompt string
            generation_config: Generation configuration dict
            num_patches_list: List of number of patches per image (optional)
        
        Returns:
            Generated response string
        """
        # Set system message
        system_text = "You are a physical mobile robot assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
        self.vlm.system_message = system_text
        
        # Generate response
        response = self.vlm.chat(
            self.tokenizer, 
            pixel_values, 
            prompt, 
            generation_config,
            history=None, 
            return_history=False, 
            num_patches_list=num_patches_list
        )
        
        return response

    def _extract_waypoint_from_response(self, response: str) -> Optional[torch.Tensor]:
        """
        Parse three (x, y, theta) triplets from the VLM response text (for 3s, 6s, and 9s).
        Matches exact training format from dataset: <answer>(x1, y1, theta1), (x2, y2, theta2), (x3, y3, theta3)</answer>
        
        Args:
            response: VLM generated response string
            
        Returns:
            Flattened tensor of length 9 (matching guidance_waypoint format) or None when parsing fails.
            Format: [x_3s, y_3s, theta_3s, x_6s, y_6s, theta_6s, x_9s, y_9s, theta_9s]
        """
        import re
        
        try:
            # Extract content between <answer> and </answer> tags (exact match to dataset format)
            if "<answer>" not in response:
                return None
            
            excerpt = response.split("<answer>", 1)[1]
            if "</answer>" not in excerpt:
                return None
            excerpt = excerpt.split("</answer>", 1)[0]
            
            # Extract exactly 9 numbers matching the format: (x1, y1, theta1), (x2, y2, theta2), (x3, y3, theta3)
            numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', excerpt)
            
            # Must have exactly 9 values (3 waypoints × 3 values each: x, y, theta)
            # No fallback - return None if not exactly 9
            if len(numbers) != 9:
                return None
            
            values = [float(n) for n in numbers[:9]]  # Extract exactly 9 values
            return torch.tensor(values, device=self.device, dtype=torch.bfloat16)
        except Exception:
            return None


if __name__ == "__main__":
    # Test the model
    path = 'InternVL3-1B'
    model = TICVLA_VLM(model_path=path)
    
    # Create dummy batch for testing
    batch = {
        'input_ids': torch.randint(0, 1000, (1, 10)),
        'attention_mask': torch.ones(1, 10),
        'labels': torch.randint(0, 1000, (1, 10)),
        'pixel_values': None,  # Would contain actual images in real usage
        'image_flags': None,
    }
    
    print("Model created successfully!")
    print(f"Model device: {model.device}")
    print(f"VLM config: {model.vlm.config}")

