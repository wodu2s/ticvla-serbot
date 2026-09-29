import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
import pytorch_lightning as pl

# Import VLM data components to reuse
from ticvla.data.vlm_data import (
    TICVLADataset_VLM,
    TICVLACollator_VLM,
)


def _build_transform(image_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[tuple[int, int]],
                               width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448,
                       use_thumbnail: bool = True) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(orig_height, 1)

    # enumerate possible tilings
    target_ratios = sorted({
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }, key=lambda x: x[0] * x[1])

    tiles = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * tiles[0]
    target_height = image_size * tiles[1]
    blocks = tiles[0] * tiles[1]

    resized = image.resize((target_width, target_height))
    images: List[Image.Image] = []
    grid_w = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size,
        )
        images.append(resized.crop(box))

    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def load_image_to_tensor(path: str, image_size: int = 448, max_num: int = 12) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    transform = _build_transform(image_size)
    tiles = dynamic_preprocess(img, image_size=image_size, max_num=max_num, use_thumbnail=True)
    tensors = [transform(t) for t in tiles]
    return torch.stack(tensors)


class TICVLADataset(Dataset):
    """Dataset for model state training - extends VLM dataset with action expert data.
    
    Uses TICVLADataset_VLM for VLM data extraction (delayed frame images, messages)
    and adds action expert data (current frame image, waypoints, robot_state).
    """

    def __init__(
        self,
        data_dir: str | List[str],
        max_sequence_length: int,
        action_horizon_steps: int,
    ) -> None:
        # Use VLM dataset for VLM-related data extraction
        self.vlm_dataset = TICVLADataset_VLM(data_dir, max_sequence_length)
        self.max_sequence_length = max_sequence_length
        if action_horizon_steps < 1:
            raise ValueError(f"action_horizon_steps must be >= 1, got {action_horizon_steps}")
        self.action_horizon_steps = action_horizon_steps
        self.samples = self.vlm_dataset.samples

    def __len__(self) -> int:
        return len(self.samples)

    # Delegate VLM-related methods to vlm_dataset
    def _extract_previous_waypoints_from_history(
        self,
        history: List[Dict[str, Any]],
        current_timestamp: float,
    ) -> List[tuple[float, List[float]]]:
        return self.vlm_dataset._extract_previous_waypoints_from_history(history, current_timestamp)
    
    def _format_previous_waypoints_text(self, previous_waypoints: List[tuple[float, List[float]]], elapsed_time: float) -> str:
        return self.vlm_dataset._format_previous_waypoints_text(previous_waypoints, elapsed_time)
    
    def _build_messages(
        self,
        image_paths: List[str],
        annotation: Dict[str, Any],
        waypoints: torch.Tensor,
        previous_waypoints_text: str = "",
        robot_type: str = "",
    ) -> List[Dict[str, Any]]:
        return self.vlm_dataset._build_messages(image_paths, annotation, waypoints, previous_waypoints_text, robot_type)
    
    def _load_annotation(self, data: Dict[str, Any], sample: Path) -> Dict[str, Any]:
        return self.vlm_dataset._load_annotation(data, sample)
    
    def _detect_dataset_info(self, sample: Path) -> tuple[str | None, str | None]:
        return self.vlm_dataset._detect_dataset_info(sample)
    
    def _infer_robot_type(self, sample: Path) -> str:
        return self.vlm_dataset._infer_robot_type(sample)
    
    def _remap_image_path(self, img_path: str, sample: Path) -> str:
        return self.vlm_dataset._remap_image_path(img_path, sample)
    
    def _remap_text_file_path(self, file_path: str, sample: Path) -> str:
        return self.vlm_dataset._remap_text_file_path(file_path, sample)
    
    # Keep action expert specific methods (not in VLM dataset)
    def _quaternion_to_yaw(self, quat: List[float]) -> float:
        """
        Extract yaw (heading angle) from quaternion [x, y, z, w].
        Returns yaw in radians in range [-pi, pi].
        """
        if len(quat) < 4:
            return 0.0
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        # Convert quaternion to yaw using standard formula
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # Keep action expert specific methods (not in VLM dataset)
    def _extract_waypoints_xy(
        self,
        future: List[Dict[str, Any]],
        max_waypoints: int = 30,
    ) -> torch.Tensor:
        """f
        Extract waypoints as (dx, dy) relative to current frame.
        
        Args:
            future: List of future waypoint dicts with 'offset' [x, y, z]
            max_waypoints: Maximum number of waypoints to extract
        
        Returns:
            waypoints: (max_waypoints, 2) tensor of [dx, dy]
                      where dx, dy are position offsets relative to current frame
        """
        waypoints = torch.zeros(max_waypoints, 2, dtype=torch.float32)
        
        # Extract waypoints from future
        for i in range(max_waypoints):
            if i < len(future) and isinstance(future[i], dict):
                w = future[i]
                # Extract position offsets (dx, dy) - already relative to current frame
                offset = w.get('offset', [0.0, 0.0, 0.0])
                if len(offset) >= 2:
                    waypoints[i, 0] = float(offset[0])  # dx
                    waypoints[i, 1] = float(offset[1])  # dy
        
        return waypoints

    def _compute_robot_state_from_future(self, future: List[Dict[str, Any]], current_orientation: List[float]) -> torch.Tensor:
        """
        Build a compact robot state vector from the most recent future waypoint.
        Uses the first future waypoint (at t+0.1s) to estimate current velocity.
        Yaw rate (speed of change of yaw) is computed from orientation quaternions.
        
        Args:
            future: List of future waypoint dicts with 'offset' [x, y, z] and 'orientation' [x, y, z, w]
            current_orientation: Current frame's orientation quaternion [x, y, z, w] (defaults to [0,0,0,1] if missing)
        
        Returns a 1D tensor: [vx, vy, yaw_rate]
        """
        dt = 0.1  # time interval between frames (seconds)
        
        # Default values if no future data
        if not future or len(future) == 0 or not isinstance(future[0], dict):
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
        first_waypoint = future[0]
        offset = first_waypoint.get('offset', [0.0, 0.0, 0.0])
        
        # Velocity = displacement / time
        # The first future waypoint represents movement from current (t) to t+0.1s
        vx = float(offset[0]) / dt if len(offset) >= 2 else 0.0
        vy = float(offset[1]) / dt if len(offset) >= 2 else 0.0
        
        # Yaw rate from orientation quaternions
        if not current_orientation or len(current_orientation) < 4:
            current_orientation = [0.0, 0.0, 0.0, 1.0]
        current_yaw = self._quaternion_to_yaw(current_orientation)
        
        future_orientation = first_waypoint.get('orientation', [0.0, 0.0, 0.0, 1.0])
        if len(future_orientation) >= 4:
            future_yaw = self._quaternion_to_yaw(future_orientation)
            # Yaw change from current (t) to future (t+0.1s)
            dtheta = future_yaw - current_yaw
            # Wrap to [-pi, pi]
            dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
            # Yaw rate = angular change / time
            yaw_rate = dtheta / dt
        else:
            yaw_rate = 0.0
        
        return torch.tensor([vx, vy, yaw_rate], dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        data = json.load(open(sample, 'r'))

        # Get current frame image and orientation (for action expert)
        current_data = data.get('current', {})
        current_img = current_data.get('img', '')
        current_image_path = self._remap_image_path(current_img, sample) if current_img else ''
        
        # Extract current frame's orientation for yaw rate computation and waypoint extraction
        current_orientation = current_data.get('orientation')
        if not current_orientation or len(current_orientation) < 4:
            # Try to get from history
            history = data.get('history', [])
            if history and len(history) > 0:
                last_history = history[-1]
                if isinstance(last_history, dict):
                    current_orientation = last_history.get('orientation')
        
        # Extract waypoints as (dx, dy) relative to current frame
        future = data.get('future', [])
        waypoints = self._extract_waypoints_xy(future, max_waypoints=self.action_horizon_steps)
        
        # Compute robot state from most recent future waypoint (at t+0.1s)
        # Uses future waypoints to estimate current velocity, using orientation quaternions for yaw rate (matching waypoint version)
        robot_state = self._compute_robot_state_from_future(future, current_orientation)
        
        # Images will be collected from delayed frame's history (for VLM model-state context)
        # This is done after we load the delayed_sample below
        image_paths: List[str] = []  # Will be populated from delayed frame's history (for VLM context)

        # guidance_waypoint will be extracted from delayed frame's future (in delayed frame coordinates)
        # This is the ground truth for VLM training - VLM predicts future waypoints from delayed frame's perspective
        # 3 waypoints at 3s, 6s, and 9s → 3 × 3 values
        guidance_waypoint = torch.ones(9, dtype=torch.float32) * -100  # Will be set from delayed frame

        # Extract relative time from current sample (prefer timestamp field for relative time)
        current_time = data.get('timestamp')  # Relative time
        
        # Sample a latency delay uniformly from 0-10s, capped by available history.
        max_delay = min(10.0, current_time) if current_time > 0 else 0.0
        time_delay = max_delay * np.random.rand()

        # elapsed_time is the relative time (time since start)
        elapsed_time = float(current_time)
        
        # Find delayed sample using index order within the same directory
        # Files in each directory are sampled at 0.1s intervals (10Hz)
        # So to go back time_delay seconds, we need to go back time_delay / 0.1 = time_delay * 10 files
        delayed_sample = None
        
        # Get all files in the same directory as current sample, sorted by numeric filename
        sample_dir = sample.parent
        if sample_dir.exists():
            def get_numeric_key(path: Path) -> float:
                stem = path.stem
                if '_' in stem:
                    numeric_str = stem.split('_', 1)[1]
                    # Handle both integer and decimal numbers (e.g., timestamps)
                    try:
                        return float(numeric_str)
                    except ValueError:
                        raise ValueError(f"Filename {path} has non-numeric suffix: {numeric_str}")
                raise ValueError(f"Filename {path} doesn't have expected format (prefix_number)")
            
            # Get all JSON files in the directory and sort them
            all_files_in_dir = sorted(sample_dir.glob('*.json'), key=get_numeric_key)
            
            # Find current file's index within this directory
            current_idx = all_files_in_dir.index(sample)
            
            files_back = int(time_delay / 0.1)  # = time_delay * 10
            delayed_idx = current_idx - files_back
            delayed_sample = all_files_in_dir[delayed_idx]

        previous_waypoints_text = ""
        displacement_from_delayed = torch.zeros(2, dtype=torch.float32)  # [dx, dy] displacement from delayed to current
        
        if delayed_sample and os.path.exists(delayed_sample):
            last_state = json.load(open(delayed_sample, 'r'))
            
            # Collect images from delayed frame's history (for VLM model-state context)
            # Specifically collect images at 3s, 6s, 9s intervals before delayed frame (if they exist)
            # Matching ticvla.training.datasets.vlm format
            delayed_history = last_state.get('history', [])
            delayed_timestamp = last_state.get('timestamp', 0.0)
            
            # Use last N frames from delayed frame's history
            for t, h in enumerate(delayed_history[-self.max_sequence_length:]):
                # Sample every 30 frames and collect image paths
                if t % 30 == 0 and isinstance(h, dict) and 'img' in h:
                    img_path = h['img']
                    # Remap old absolute paths to correct dataset paths
                    image_paths.append(self._remap_image_path(img_path, delayed_sample))
            
            # Add delayed frame's current image (the frame at delayed time)
            delayed_current_img = last_state.get('current', {}).get('img', '')
            if delayed_current_img:
                image_paths.append(self._remap_image_path(delayed_current_img, delayed_sample))
            
            # Extract previous waypoints from delayed file's history
            # At the delayed time, there will only be waypoints up to that point
            previous_waypoints = self._extract_previous_waypoints_from_history(delayed_history, delayed_timestamp)
            previous_waypoints_text = self._format_previous_waypoints_text(previous_waypoints, delayed_timestamp)
            
            # Extract displacement from delayed time to current time from delayed frame's future
            # time_delay is in seconds, at 10Hz: index = time_delay / 0.1 = time_delay * 10
            future_list = last_state.get('future', [])
            delay_index = int(round(float(time_delay) * 10.0))  # Matching waypoint version
            
            if delay_index < len(future_list) and isinstance(future_list[delay_index], dict):
                offset = future_list[delay_index].get('offset', [0.0, 0.0, 0.0])
                if len(offset) >= 2:
                    # This offset represents cumulative displacement from delayed frame to current frame
                    displacement_from_delayed = torch.tensor([
                        float(offset[0]), float(offset[1])
                    ], dtype=torch.float32)

            # Extract guidance waypoint from delayed frame's future (ground truth for VLM training)
            # VLM predicts future waypoints from delayed frame's perspective, in delayed frame coordinates
            # At 10Hz: 3s = index 29, 6s = index 59, 9s = index 89
            future_list = last_state.get('future', [])
            waypoint_extraction_length = max(90, self.max_sequence_length)  # Need at least 90 for 9s waypoint
            future_offsets = [
                w['offset'] for w in future_list[:waypoint_extraction_length]
                if isinstance(w, dict) and 'offset' in w
            ]
            
            # Extract waypoints at 3s (index 29), 6s (index 59) and 9s (index 89)
            if len(future_offsets) > 89:
                # Extract waypoints at 3s, 6s and 9s from delayed frame's future
                # Convert from [x, y, z] to [x, y, theta] where theta = atan2(y, x) (matching ticvla.training.datasets.vlm)
                guidance_waypoint_list = []
                for idx in [29, 59, 89]:  # 3s, 6s, 9s
                    offset = future_offsets[idx]
                    x, y = float(offset[0]), float(offset[1])
                    theta = math.atan2(y, x + 1e-3)  # Compute theta from (x, y)
                    guidance_waypoint_list.extend([x, y, theta])
                guidance_waypoint = torch.tensor(guidance_waypoint_list, dtype=torch.float32)
        
        time_delay = torch.tensor(time_delay, dtype=torch.float32)
        
        # Track statistics
        if not hasattr(self, '_waypoint_stats'):
            self._waypoint_stats = {'valid': 0, 'invalid': 0, 'no_delayed_sample': 0, 'insufficient_future': 0}
        
        # Update statistics
        if (guidance_waypoint == -100).all():
            self._waypoint_stats['invalid'] += 1
            if delayed_sample is None or not os.path.exists(delayed_sample):
                self._waypoint_stats['no_delayed_sample'] += 1
            else:
                self._waypoint_stats['insufficient_future'] += 1
        else:
            self._waypoint_stats['valid'] += 1
        
        # current_image_path already extracted above
        
        # Build messages with delayed frame images only (for VLM model-state context)
        annotation = self._load_annotation(data, sample)
        robot_type = self._infer_robot_type(sample)
        messages = self._build_messages(image_paths, annotation, guidance_waypoint, previous_waypoints_text, robot_type)
        
        # Concatenate displacement from delayed time to current time to robot_state
        # robot_state: [vx, vy, yaw_speed] + displacement: [dx, dy] = [vx, vy, yaw_speed, dx, dy]
        robot_state_with_displacement = torch.cat([robot_state, displacement_from_delayed], dim=0)
    
        return {
            'messages': messages,
            'delayed_images': image_paths,  # delayed frame images (from delayed sample's history)
            'current_image': current_image_path,  # current frame image (from current sample)
            'waypoints': waypoints,  # action_horizon_steps × (dx, dy) at 10 Hz from cumulative offsets
            'robot_state': robot_state_with_displacement,  # [vx, vy, yaw_speed, dx, dy]
            'time_delay': time_delay,
            'guidance_waypoint': guidance_waypoint, #GT (x, y, z) into the future of DELAYED frame (in delayed frame coordinates, ground truth for VLM training).
            'robot_type': robot_type,
        }


@dataclass
class TICVLACollator:
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    image_size: int = 448
    max_tiles_per_image: int = 1

    # cache for computed tokens-per-tile (InternVL num_image_token)
    _num_image_token: Optional[int] = None

    def _compute_num_image_token(self) -> int:
        if self._num_image_token is not None:
            return self._num_image_token
        # Try to read from model config colocated with tokenizer
        try:
            name_or_path = getattr(self.tokenizer, 'name_or_path', None)
            if name_or_path and os.path.isdir(name_or_path):
                cfg_path = os.path.join(name_or_path, 'config.json')
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                    vision_cfg = cfg.get('vision_config', {})
                    image_size = int(cfg.get('force_image_size') or vision_cfg.get('image_size', self.image_size))
                    patch_size = int(vision_cfg.get('patch_size', 14))
                    downsample_ratio = float(cfg.get('downsample_ratio', 0.5))
                    self._num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
                else:
                    # fallback constant used by InternVL-448/14 with 0.5 downsample
                    self._num_image_token = 256
            else:
                self._num_image_token = 256
        except Exception:
            self._num_image_token = 256
        return self._num_image_token

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages = [s['messages'] for s in samples]
        robot_state = [s.get('robot_state') for s in samples]
        waypoints = [s['waypoints'] for s in samples]
        time_delays = [s['time_delay'] for s in samples]
        guidance_waypoints = [s['guidance_waypoint'] for s in samples]
        robot_types = [s.get('robot_type', '') for s in samples]

        # Get delayed and current images from separate fields
        delayed_image_paths: List[List[str]] = []  # List of lists: each sample has list of delayed frame images
        current_image_paths: List[List[str]] = []  # List of lists: each sample has list with single current frame image
        
        for s in samples:
            # Delayed frame images (from delayed sample's history)
            delayed_imgs = s.get('delayed_images', [])
            delayed_image_paths.append(delayed_imgs if isinstance(delayed_imgs, list) else [])
            
            # Current frame image (from current sample)
            current_img = s.get('current_image', '')
            current_image_paths.append([current_img] if current_img else [])

        # Prepare chat text for delayed frame context (past images + COT + instruction)
        # Messages already contain the delayed frame context
        text_batch: List[str] = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Gather delayed frame image tiles
        delayed_pixel_values_chunks: List[torch.Tensor] = []
        delayed_per_sample_image_tile_counts: List[List[int]] = []
        delayed_per_sample_total_tiles: List[int] = []
        for delayed_imgs in delayed_image_paths:
            image_tile_counts: List[int] = []
            total_tiles_this_sample = 0
            for p in delayed_imgs:
                tiles = load_image_to_tensor(p, image_size=self.image_size, max_num=self.max_tiles_per_image)
                delayed_pixel_values_chunks.append(tiles)
                image_tile_counts.append(int(tiles.shape[0]))
                total_tiles_this_sample += int(tiles.shape[0])
            delayed_per_sample_image_tile_counts.append(image_tile_counts)
            delayed_per_sample_total_tiles.append(total_tiles_this_sample)

        delayed_pixel_values = torch.cat(delayed_pixel_values_chunks, dim=0).to(torch.bfloat16) if len(delayed_pixel_values_chunks) > 0 else None

        # Build queries for delayed frame by replacing each <image> with its own token block
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        num_image_token = self._compute_num_image_token()
        delayed_queries: List[str] = []
        for q, per_image_counts in zip(text_batch, delayed_per_sample_image_tile_counts):
            if '<image>' in q:
                # Replace one-by-one
                for tiles_for_image in per_image_counts:
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * tiles_for_image)) + IMG_END_TOKEN
                    q = q.replace('<image>', image_tokens, 1)
            else:
                # Prepend all images when placeholder is missing
                prepend = []
                for tiles_for_image in per_image_counts:
                    prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * tiles_for_image)) + IMG_END_TOKEN)
                q = ('\n'.join(prepend) + '\n' + q) if len(prepend) > 0 else q
            delayed_queries.append(q)

        # Tokenize delayed frame queries
        self.tokenizer.padding_side = 'left'
        delayed_tokenized = self.tokenizer(delayed_queries, return_tensors='pt', padding=True)
        delayed_input_ids = delayed_tokenized['input_ids']
        delayed_labels = delayed_input_ids.clone()
        assistant_id = self.tokenizer.convert_tokens_to_ids('assistant')
        for i in range(delayed_labels.shape[0]):
            try:
                sep_positions = (delayed_labels[i] == assistant_id).nonzero(as_tuple=False).squeeze(-1)
                start_idx = int(sep_positions[0].item()) if sep_positions.numel() > 0 else delayed_labels.shape[1] - 1
            except Exception:
                start_idx = delayed_labels.shape[1] - 1
            delayed_labels[i, :start_idx] = -100

        batch: Dict[str, Any] = {
            # Delayed frame inputs (for VLM model state)
            'delayed_input_ids': delayed_input_ids,
            'delayed_attention_mask': delayed_tokenized['attention_mask'],
            'delayed_labels': delayed_labels,
            'current_image_paths': current_image_paths,  # List of lists for current frame images
            'robot_type': robot_types,
            'delayed_image_paths': delayed_image_paths,
        }

        if delayed_pixel_values is not None:
            batch['delayed_pixel_values'] = delayed_pixel_values
            # Build flat image_flags aligned with total number of delayed frame tiles
            delayed_flags: List[int] = []
            for total in delayed_per_sample_total_tiles:
                delayed_flags.extend([1] * int(total))
            batch['delayed_image_flags'] = torch.tensor(delayed_flags, dtype=torch.long).view(-1, 1)
            batch['delayed_num_tiles_per_sample'] = torch.tensor(delayed_per_sample_total_tiles, dtype=torch.long)

        # Other inputs
        batch['waypoints'] = torch.stack(waypoints, dim=0)
        batch['time_delay'] = torch.stack(time_delays, dim=0)
        batch['guidance_waypoint'] = torch.stack(guidance_waypoints, dim=0)
        batch['robot_state'] = torch.stack(robot_state, dim=0)

        return batch


class TICVLADataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str | List[str],
        val_data_dir: Optional[str],
        test_data_dir: Optional[str],
        batch_size: int,
        num_workers: int,
        max_sequence_length: int,
        action_horizon_steps: int,
        processor: Optional[AutoProcessor] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__()
        if isinstance(train_data_dir, (list, tuple)):
            self.train_data_dirs = list(train_data_dir)
        else:
            self.train_data_dirs = [str(train_data_dir)]
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.action_horizon_steps = action_horizon_steps
        self.processor = processor
        self.tokenizer = tokenizer
        # Ensure Lightning expects the attribute in distributed setups
        self.prepare_data_per_node = False

    def prepare_data(self) -> None:
        # Data already exists on disk; nothing to download/prepare
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_dataset = TICVLADataset(
                self.train_data_dirs, self.max_sequence_length, self.action_horizon_steps,
            )
            if self.val_data_dir:
                self.val_dataset = TICVLADataset(
                    [self.val_data_dir], self.max_sequence_length, self.action_horizon_steps,
                )
            else:
                n = len(self.train_dataset)
                val_size = max(1, n // 20)
                train_size = n - val_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size])
        if stage in (None, 'test') and self.test_data_dir:
            self.test_dataset = TICVLADataset(
                [self.test_data_dir], self.max_sequence_length, self.action_horizon_steps,
            )

    def _dataloader(self, ds, shuffle: bool):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=TICVLACollator(processor=self.processor, tokenizer=self.tokenizer),
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return self._dataloader(self.test_dataset, shuffle=False)
        return None

    def on_exception(self, exception: BaseException) -> None:
        # Keep as no-op to satisfy Lightning's hook access during interruptions
        pass

