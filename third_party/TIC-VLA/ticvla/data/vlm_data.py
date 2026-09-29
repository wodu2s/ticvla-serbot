"""
Dataset for TIC-VLA VLM training.

This dataset provides only the data needed for VLM training:
- Images from the current sample's history
- Text messages (system, user, assistant) with waypoint predictions
- No action expert related data (robot_state, delayed_waypoint, waypoints)
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor
import pytorch_lightning as pl

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def _build_transform(image_size: int):
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


class TICVLADataset_VLM(Dataset):
    """Dataset producing chat messages and images for VLM-only training.
    
    Simplified version that only provides data for VLM training:
    - Images from the current sample's history
    - Text messages with waypoint predictions
    - No action expert data
    """

    def __init__(self, data_dir: str | List[str], max_sequence_length: int = 90) -> None:
        self.data_dir = [Path(d) for d in data_dir] if isinstance(data_dir, list) else [Path(data_dir)]
        self.max_sequence_length = max_sequence_length
        self.samples = self._find_samples()
        # Track consecutive missing files per folder for error detection
        self._last_folder: Path | None = None
        self._consecutive_missing_count = 0

    def _find_samples(self) -> List[Path]:
        """
        Find all JSON sample files in the data directories.
        Filters to keep every 5th frame by sorting and indexing.
        """
        all_samples: List[Path] = []
        
        # Collect all JSON files
        for data_dir in self.data_dir:
            for p in data_dir.rglob('*.json'):
                all_samples.append(p)
        
        # Group by directory and sort each group to get every 5th frame
        from collections import defaultdict
        samples_by_dir = defaultdict(list)
        for p in all_samples:
            samples_by_dir[p.parent].append(p)
        
        # Sort each directory's files and take every 5th one
        filtered_samples: List[Path] = []
        for dir_path, files in samples_by_dir.items():
            # Sort files by timestamp in filename
            def get_timestamp_from_name(path: Path) -> float:
                stem = path.stem
                if '_' in stem:
                    try:
                        timestamp_str = stem.split('_', 1)[1]
                        return float(timestamp_str)
                    except (ValueError, IndexError):
                        return 0.0
                return 0.0
            
            sorted_files = sorted(files, key=get_timestamp_from_name)
            
            # Take every 5th frame (index 0, 5, 10, 15, ...)
            for idx in range(0, len(sorted_files), 5):
                filtered_samples.append(sorted_files[idx])
        
        return filtered_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_previous_waypoints_from_history(
        self,
        history: List[Dict[str, Any]],
        current_timestamp: float,
    ) -> List[tuple[float, List[float]]]:
        """
        Extract previous waypoints from history at 1Hz, returning *1-second displacements* as (x, y, z).
        Same as waypoint dataset.
        """
        def _wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        previous_waypoints: List[tuple[float, List[float]]] = []
        if not history:
            return previous_waypoints

        dt = 0.1
        steps_per_sec = int(round(1.0 / dt))  # 10

        n = len(history)
        max_end_step = min(n - 1, int(round(float(current_timestamp) / dt)))
        max_k = max_end_step // steps_per_sec
        if max_k <= 0:
            return previous_waypoints

        dx: List[float] = [0.0] * n
        dy: List[float] = [0.0] * n
        dz: List[float] = [0.0] * n
        for i, h in enumerate(history):
            if not isinstance(h, dict):
                continue
            traj = h.get('trajectory') or h.get('offset', [])
            if not (isinstance(traj, list) and len(traj) >= 2):
                continue
            dx[i] = float(traj[0])
            dy[i] = float(traj[1])
            dz[i] = float(traj[2]) if len(traj) >= 3 else 0.0

        # Aggregate 10 steps per second
        for k in range(1, max_k + 1):
            end_step = k * steps_per_sec
            start_step = end_step - steps_per_sec

            x_rel = 0.0
            y_rel = 0.0
            z_rel = 0.0
            yaw_rel = 0.0
            for i in range(start_step + 1, end_step + 1):
                c = math.cos(yaw_rel)
                s = math.sin(yaw_rel)
                x_rel += c * float(dx[i]) - s * float(dy[i])
                y_rel += s * float(dx[i]) + c * float(dy[i])
                z_rel += float(dz[i])

                dtheta = 2.0 * float(math.atan2(float(dy[i]), float(dx[i]) + 1e-6))
                yaw_rel = _wrap_pi(yaw_rel + dtheta)

            previous_waypoints.append((float(k), [x_rel, y_rel, z_rel]))

        return previous_waypoints

    def _format_previous_waypoints_text(self, previous_waypoints: List[tuple[float, List[float]]], elapsed_time: float) -> str:
        """Format previous waypoints into text describing execution times and points."""
        lines = []
        
        if previous_waypoints:
            waypoint_strs = []
            for relative_time, waypoint in previous_waypoints:
                x, y, z = waypoint
                waypoint_strs.append(f"({x:.2f}, {y:.2f}, {z:.2f})")
            waypoint_list = ", ".join(waypoint_strs)
            
            lines.append(f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. (a list of waypoints 1s in between): {waypoint_list}")
            lines.append("Each waypoint (x, y, z) is the displacement over the previous 1.0s. x is forward, y is left, z is up.")
        else:
            lines.append(f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. No waypoints available.")
        
        return "\n".join(lines)

    def _build_messages(
        self,
        image_paths: List[str],
        annotation: Dict[str, Any],
        waypoints: torch.Tensor,
        previous_waypoints_text: str = "",
        robot_type: str = "",
    ) -> List[Dict[str, Any]]:
        """Build messages for VLM training - same format as waypoint dataset."""
        system_text = f"You are a {robot_type} assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
                      
        if annotation.get('gt_parse') is not None:
            instruction = annotation['gt_parse'].get('Instruction (15s)', '')
            think = (
                f"<think>\n"
                f"{annotation['gt_parse'].get('Chain of Thought Reasoning', '')}\n"
                f"</think>\n"
            )

            user_text = f"The navigation instruction is: {instruction}"
            user_text += f"\n{previous_waypoints_text}"
            user_text += (
                "\nUse reasoning to predict future target waypoints for the next 3s, 6s, and 9s in format: (x, y, theta). "
                "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s),"
                "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
            )

            return [
                {
                    'role': 'system',
                    'content': system_text,
                },
                {
                    'role': 'user',
                    'content': [
                        *[{ 'type': 'image', 'image': str(p) } for p in image_paths],
                        { 'type': 'text', 'text': user_text },
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        { 'type': 'text', 'text': think },
                        { 'type': 'text', 'text': f"<answer>({waypoints[0]:.2f}, {waypoints[1]:.2f}, {waypoints[2]:.2f}), "
                                                     f"({waypoints[3]:.2f}, {waypoints[4]:.2f}, {waypoints[5]:.2f}), "
                                                     f"({waypoints[6]:.2f}, {waypoints[7]:.2f}, {waypoints[8]:.2f})</answer>" },
                    ],
                },
            ]
        else:
            instruction = annotation.get('Instruction', '')
            user_text = f"The navigation instruction is: {instruction}"
            user_text += f"\n{previous_waypoints_text}"
            user_text += (
                "\nReturn the future target waypoints for the next 3s, 6s, and 9s in format: (x, y, theta). "
                "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s),"
                "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
            )
            
            return [
                {
                    'role': 'system',
                    'content': system_text,
                },
                {
                    'role': 'user',
                    'content': [
                        *[{ 'type': 'image', 'image': str(p) } for p in image_paths],
                        { 'type': 'text', 'text': user_text },
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [{ 'type': 'text', 'text': f"<answer>({waypoints[0]:.2f}, {waypoints[1]:.2f}, {waypoints[2]:.2f}), "
                                                           f"({waypoints[3]:.2f}, {waypoints[4]:.2f}, {waypoints[5]:.2f}), "
                                                           f"({waypoints[6]:.2f}, {waypoints[7]:.2f}, {waypoints[8]:.2f})</answer>" }],
                },
            ]

    def _load_annotation(self, data: Dict[str, Any], sample: Path) -> Dict[str, Any]:
        """Load annotation from instruction and CoT files - same as waypoint dataset."""
        instruction: str = ''
        cot_text: str = ''
        current_folder = sample.parent
        
        if self._last_folder != current_folder:
            self._last_folder = current_folder
            self._consecutive_missing_count = 0
        
        instruction_file = data.get('instruction_file')
        if not instruction_file or not isinstance(instruction_file, str):
            sample_dir = sample.parent
            instruction_files = list(sample_dir.glob('instruction_*.txt'))
            if instruction_files:
                instruction_file = str(instruction_files[0])
            else:
                instruction_file = None
        
        if instruction_file:
            instruction_file = self._remap_text_file_path(instruction_file, sample)
            try:
                instruction = open(instruction_file, 'r').read().strip()
                if not instruction:
                    print(f"Warning: Instruction file is empty: {instruction_file}")
            except FileNotFoundError:
                print(f"Warning: Instruction file not found: {instruction_file}")
                sample_dir = sample.parent
                instruction_filename = Path(instruction_file).name
                potential_path = sample_dir / instruction_filename
                if potential_path.exists():
                    try:
                        instruction = open(potential_path, 'r').read().strip()
                        print(f"Found instruction file at: {potential_path}")
                    except Exception:
                        pass
        
        cot_file = data.get('cot')
        if cot_file and isinstance(cot_file, str):
            cot_file = self._remap_text_file_path(cot_file, sample)
            try:
                cot_text = open(cot_file, 'r').read().strip()
                self._consecutive_missing_count = 0
            except FileNotFoundError:
                self._consecutive_missing_count += 1
                if self._consecutive_missing_count >= 11:
                    raise FileNotFoundError(
                        f"11 consecutive cot files missing in folder {current_folder}. "
                        f"Last attempted file: {cot_file}"
                    )
                cot_text = ''
        
        if cot_text:
            return {
                'gt_parse': {
                    'Instruction (15s)': instruction,
                    'Chain of Thought Reasoning': cot_text,
                }
            }
        return {'Instruction': instruction}

    _JSON_DIRS = frozenset({'GND_json', 'SCAND_json', 'DynaNav_json'})

    def _detect_dataset_info(self, sample: Path) -> tuple[str | None, str | None]:
        """Detect dataset type and image-data directory from sample path."""
        sample_str = str(sample).replace('\\', '/')
        if '/GND/GND_json' in sample_str or '/GND_json/' in sample_str:
            return 'GND', 'GND_data'
        if '/SCAND/SCAND_json' in sample_str or '/SCAND_json/' in sample_str:
            return 'SCAND', 'SCAND_data'
        if '/DynaNav/DynaNav_json' in sample_str or '/DynaNav_json/' in sample_str:
            return 'DynaNav', 'DynaNav_data'
        return None, None

    def _find_dataset_root(self, sample: Path) -> Path:
        """Return the dataset folder that contains JSON and image subdirectories."""
        for parent in sample.parents:
            if parent.name in self._JSON_DIRS:
                return parent.parent
        env_root = os.environ.get('TICVLA_DATA_ROOT')
        if env_root:
            return Path(env_root).expanduser()
        return sample.parents[2] if len(sample.parents) > 2 else sample.parent

    def _find_json_dataset_dir(self, sample: Path) -> Path:
        """Return the JSON dataset directory that owns the sample."""
        for parent in sample.parents:
            if parent.name in self._JSON_DIRS:
                return parent
        return sample.parent

    def _candidate_image_paths(
        self,
        img_path_normalized: str,
        sample: Path,
        data_dir: str,
    ) -> List[Path]:
        """Build candidate absolute paths for an image reference."""
        candidates: List[Path] = []
        if not os.path.isabs(img_path_normalized):
            candidates.append((sample.parent / img_path_normalized).resolve())

        data_root = self._find_dataset_root(sample)
        marker = f'/{data_dir}/'
        if marker in img_path_normalized:
            relative_path = img_path_normalized.split(marker, 1)[1].lstrip('/')
            candidates.append(data_root / data_dir / relative_path)
        elif not os.path.isabs(img_path_normalized):
            candidates.append(data_root / data_dir / img_path_normalized.lstrip('/'))

        return candidates

    def _infer_robot_type(self, sample: Path) -> str:
        """Infer robot type string from sample path."""
        dataset, _ = self._detect_dataset_info(sample)
        if dataset == 'GND':
            return 'wheeled robot'
        if dataset in ('SCAND', 'DynaNav'):
            folder_name = sample.parent.name.lower()
            return 'legged robot' if 'spot' in folder_name else 'wheeled robot'
        raise ValueError(f"Unknown dataset: {dataset}")

    def _remap_image_path(self, img_path: str, sample: Path) -> str:
        """Resolve image paths for the Hugging Face dataset layout."""
        if not img_path:
            return img_path

        img_path_normalized = img_path.replace('\\', '/')
        if os.path.isabs(img_path_normalized) and os.path.exists(img_path_normalized):
            return img_path_normalized

        dataset, data_dir = self._detect_dataset_info(sample)
        if dataset and data_dir:
            for candidate in self._candidate_image_paths(
                img_path_normalized, sample, data_dir,
            ):
                if candidate.exists():
                    return str(candidate)

        return img_path

    def _remap_text_file_path(self, file_path: str, sample: Path) -> str:
        """Remap old absolute paths for text files to correct dataset paths."""
        if not file_path:
            return file_path

        file_path_normalized = file_path.replace('\\', '/')
        if os.path.exists(file_path_normalized):
            return file_path_normalized

        sample_dir = sample.parent
        local_by_name = sample_dir / Path(file_path_normalized).name
        if local_by_name.exists():
            return str(local_by_name)

        dataset, _ = self._detect_dataset_info(sample)
        if not dataset:
            return file_path

        json_dataset_dir = self._find_json_dataset_dir(sample)
        marker = f'/{json_dataset_dir.name}/'
        if marker in file_path_normalized:
            relative_path = file_path_normalized.split(marker, 1)[1].lstrip('/')
            return str(json_dataset_dir / relative_path)

        if not os.path.isabs(file_path_normalized):
            return str(sample_dir / file_path_normalized)

        return file_path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample for VLM training - only images and messages."""
        sample = self.samples[index]
        data = json.load(open(sample, 'r'))

        # Stage-1 VLM training has no latency sampling. Use the current sample's
        # own history/current image and current future waypoints.
        image_paths: List[str] = []
        previous_waypoints_text = ""

        history = data.get('history', [])
        timestamp = data.get('timestamp', 0.0)

        for t, h in enumerate(history[-self.max_sequence_length:]):
            if t % 30 == 0 and isinstance(h, dict) and 'img' in h:
                image_paths.append(self._remap_image_path(h['img'], sample))

        current_img = data.get('current', {}).get('img', '')
        if current_img:
            image_paths.append(self._remap_image_path(current_img, sample))

        previous_waypoints = self._extract_previous_waypoints_from_history(history, timestamp)
        previous_waypoints_text = self._format_previous_waypoints_text(previous_waypoints, timestamp)

        future_list = data.get('future', [])
        waypoint_extraction_length = max(90, self.max_sequence_length)
        future_offsets = [
            w['offset'] for w in future_list[:waypoint_extraction_length]
            if isinstance(w, dict) and 'offset' in w
        ]

        guidance_waypoint = torch.ones(9, dtype=torch.float32) * -100
        if len(future_offsets) > 89:
            guidance_waypoint_list = []
            for idx in [29, 59, 89]:  # 3s, 6s, 9s at 10 Hz
                offset = future_offsets[idx]
                x, y = float(offset[0]), float(offset[1])
                theta = math.atan2(y, x + 1e-3)
                guidance_waypoint_list.extend([x, y, theta])
            guidance_waypoint = torch.tensor(guidance_waypoint_list, dtype=torch.float32)

        # annotations
        annotation = self._load_annotation(data, sample)
        robot_type = self._infer_robot_type(sample)
        messages = self._build_messages(image_paths, annotation, guidance_waypoint, previous_waypoints_text, robot_type)

        return {
            'messages': messages,
            'images': image_paths,
            'robot_type': robot_type,
        }


@dataclass
class TICVLACollator_VLM:
    """Collator for VLM-only training - simplified version."""
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    image_size: int = 448
    max_tiles_per_image: int = 1

    # cache for computed tokens-per-tile (InternVL num_image_token)
    _num_image_token: Optional[int] = None

    def _compute_num_image_token(self) -> int:
        if self._num_image_token is not None:
            return self._num_image_token
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
                    self._num_image_token = 256
            else:
                self._num_image_token = 256
        except Exception:
            self._num_image_token = 256
        return self._num_image_token

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate samples for VLM training."""
        messages = [s['messages'] for s in samples]
        robot_types = [s.get('robot_type', '') for s in samples]
        image_paths = [s.get('images', []) for s in samples]

        # Prepare chat text with placeholders
        text_batch: List[str] = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_prompt_text = text_batch

        # Gather image tiles
        pixel_values_chunks: List[torch.Tensor] = []
        per_sample_image_tile_counts: List[List[int]] = []
        per_sample_total_tiles: List[int] = []
        for s in samples:
            img_paths: List[str] = s['images']
            image_tile_counts: List[int] = []
            total_tiles_this_sample = 0
            for p in img_paths:
                tiles = load_image_to_tensor(p, image_size=self.image_size, max_num=self.max_tiles_per_image)
                pixel_values_chunks.append(tiles)
                image_tile_counts.append(int(tiles.shape[0]))
                total_tiles_this_sample += int(tiles.shape[0])
            per_sample_image_tile_counts.append(image_tile_counts)
            per_sample_total_tiles.append(total_tiles_this_sample)

        pixel_values = torch.cat(pixel_values_chunks, dim=0).to(torch.bfloat16) if len(pixel_values_chunks) > 0 else None

        # Build queries by replacing each <image> with its own token block
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        num_image_token = self._compute_num_image_token()
        queries: List[str] = []
        for q, per_image_counts in zip(text_batch, per_sample_image_tile_counts):
            if '<image>' in q:
                for tiles_for_image in per_image_counts:
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * tiles_for_image)) + IMG_END_TOKEN
                    q = q.replace('<image>', image_tokens, 1)
            else:
                prepend = []
                for tiles_for_image in per_image_counts:
                    prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * tiles_for_image)) + IMG_END_TOKEN)
                q = ('\n'.join(prepend) + '\n' + q) if len(prepend) > 0 else q
            queries.append(q)

        # Left padding for generation compatibility
        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer(queries, return_tensors='pt', padding=True)

        # labels: only compute loss on assistant part; mask others by -100
        input_ids = tokenized['input_ids']
        labels = input_ids.clone()
        sep_token = 'assistant'
        sep_id = self.tokenizer.convert_tokens_to_ids(sep_token)
        for i in range(labels.shape[0]):
            try:
                sep_positions = (labels[i] == sep_id).nonzero(as_tuple=False).squeeze(-1)
                start_idx = int(sep_positions[0].item()) if sep_positions.numel() > 0 else labels.shape[1] - 1
            except Exception:
                start_idx = labels.shape[1] - 1
            
            labels[i, :start_idx] = -100

        batch: Dict[str, Any] = {
            'input_ids': input_ids,
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'robot_type': robot_types,
            'image_paths': image_paths,
            'raw_prompt_text': raw_prompt_text,
        }

        if pixel_values is not None:
            batch['pixel_values'] = pixel_values
            flags: List[int] = []
            for total in per_sample_total_tiles:
                flags.extend([1] * int(total))
            batch['image_flags'] = torch.tensor(flags, dtype=torch.long).view(-1, 1)
            batch['num_tiles_per_sample'] = torch.tensor(per_sample_total_tiles, dtype=torch.long)

        return batch


class TICVLADataModule_VLM(pl.LightningDataModule):
    """Data module for VLM-only training."""
    
    def __init__(
        self,
        train_data_dir: str | List[str],
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        max_sequence_length: int = 90,
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
        self.processor = processor
        self.tokenizer = tokenizer
        self.prepare_data_per_node = False

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_dataset = TICVLADataset_VLM(self.train_data_dirs, self.max_sequence_length)
            if self.val_data_dir:
                self.val_dataset = TICVLADataset_VLM([self.val_data_dir], self.max_sequence_length)
            else:
                n = len(self.train_dataset)
                val_size = max(1, n // 20)
                train_size = n - val_size
                split_generator = torch.Generator().manual_seed(42)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset,
                    [train_size, val_size],
                    generator=split_generator,
                )
        if stage in (None, 'test') and self.test_data_dir:
            self.test_dataset = TICVLADataset_VLM([self.test_data_dir], self.max_sequence_length)

    def _dataloader(self, ds, shuffle: bool):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=TICVLACollator_VLM(processor=self.processor, tokenizer=self.tokenizer),
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
        pass

