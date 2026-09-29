# TIC-VLA

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://ucla-mobility.github.io/TIC-VLA/)
[![paper](https://img.shields.io/badge/Paper-ICML2026-red.svg)](https://arxiv.org/abs/2602.02459)
[![dataset](https://img.shields.io/badge/Dataset-HuggingFace-F9D371.svg)](https://huggingface.co/datasets/handsomeYun/TIC-VLA)

**[ICML 2026] TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments**

TIC-VLA is a latency-aware vision-language-action model for robot navigation in dynamic, human-centric environments. This repository contains the released model code, supervised training entrypoints, DynaNav benchmark assets, and IsaacLab RL fine-tuning scripts.

![overview](docs/img/framework.png)

## Repository Layout

- `ticvla/`: model, dataset, training, config, and utility code.
- `data/`: batch scripts for generating JSON, instructions, and CoT annotations from raw navigation trajectories.
- `configs/train_vlm.yaml`, `configs/train_action.yaml`: supervised TIC-VLA stage configs.
- `DynaNav/`: Isaac Sim benchmark runner, configs, behavior scripts, and local scene assets.
- `rl/fine_tuning/`: PPO-based IsaacLab RL fine-tuning scripts.
- `requirements-train.txt`, `requirements-test.txt`: dependency groups for conda training/RL and DynaNav.

## Setup

TIC-VLA uses one conda environment named `tic-vla` for model training. Simulation workflows use Isaac Sim's Python interpreter instead of a separate conda env.

### 1. Install Isaac Sim

Install Isaac Sim 5.0.0 outside conda. Download the Linux workstation build from the [official Isaac Sim download page](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html), then follow NVIDIA's [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_workstation.html).

```bash
# Download Linux (x86_64):
# https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip

mkdir -p ~/isaacsim
unzip "isaac-sim-standalone-5.0.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
```

Point the repository to that installation:

```bash
export ISAAC_SIM_ROOT=~/isaacsim
export ISAAC_SIM_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
test -x "${ISAAC_SIM_PYTHON}"
"${ISAAC_SIM_PYTHON}" -c "from isaacsim import SimulationApp; print('Isaac Sim 5.0.0 ready')"
```

### 2. Create The Training Env

```bash
conda env create -f tic-vla.yaml
conda activate tic-vla
pip install -e .
```

### 3. Download Data And Base Model

Download the TIC-VLA reasoning datasets from [Hugging Face](https://huggingface.co/datasets/handsomeYun/TIC-VLA) into a local data root:

```bash
export TICVLA_DATA_ROOT=/path/to/ticvla/dataset
mkdir -p "${TICVLA_DATA_ROOT}"

python - <<'PY'
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="handsomeYun/TIC-VLA",
    repo_type="dataset",
    local_dir=os.environ["TICVLA_DATA_ROOT"],
    local_dir_use_symlinks=False,
)
PY
```

Download the InternVL3-1B base model from Hugging Face:

```bash
export TICVLA_BASE_MODEL_PATH=/path/to/InternVL3-1B
mkdir -p "${TICVLA_BASE_MODEL_PATH}"

python - <<'PY'
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="OpenGVLab/InternVL3-1B",
    repo_type="model",
    local_dir=os.environ["TICVLA_BASE_MODEL_PATH"],
    local_dir_use_symlinks=False,
)
PY
```

Add `TICVLA_DATA_ROOT` and `TICVLA_BASE_MODEL_PATH` to `.env.training`; `TICVLA_BASE_MODEL_PATH` should be the downloaded InternVL3-1B base model directory.

### 4. Install DynaNav Extras

DynaNav benchmarking runs with Isaac Sim Python. Install the DynaNav testing dependencies there:

```bash
"${ISAAC_SIM_PYTHON}" -m pip install -e .
"${ISAAC_SIM_PYTHON}" -m pip install -r requirements-test.txt
```

### 5. Configure Local Paths

Edit the local environment files for your machine:

```bash
$EDITOR .env.training
$EDITOR .env.testing
```

## Raw Data Annotation

If you are starting from raw navigation trajectories instead of the released dataset, use the scripts in `data/` to build a supervised training dataset in three steps.

Each scene folder should contain:

- `trajectory.csv` with columns `time,x,y,z,qx,qy,qz,qw`
- an `rgb/` image directory (indexed to match trajectory timestamps)
- `instruction.txt` (placeholder; step 2 writes per-window instruction files)

```bash
export OPENAI_API_KEY="your_api_key"

# Example paths (edit for your dataset layout)
export RAW_DATA_DIR=/path/to/raw_trajectories
export ANNOTATED_JSON_DIR=/path/to/annotated_json
```

### 1. Generate JSON Windows

Convert raw scene folders into per-window JSON files:

```bash
python s01_batch_json_generation.py \
  --input_dir "${RAW_DATA_DIR}" \
  --output_dir "${ANNOTATED_JSON_DIR}"
```

`--input_dir` can be a single scene directory or a parent directory of scene folders.

### 2. Generate Instructions

Preview which scene folders still need instruction generation:

```bash
python s02_batch_instruction.py \
  --json_folder "${ANNOTATED_JSON_DIR}" \
  --preview_only
```

Run instruction generation on all folders:

```bash
python s02_batch_instruction.py \
  --json_folder "${ANNOTATED_JSON_DIR}" \
  --model gpt-5
```

### 3. Generate CoT Annotations

Preview which JSON files still need CoT annotations:

```bash
python s03_batch_annotate.py \
  --json_folder "${ANNOTATED_JSON_DIR}" \
  --preview_only
```

Run CoT generation:

```bash
python s03_batch_annotate.py \
  --json_folder "${ANNOTATED_JSON_DIR}" \
  --num_workers 32 \
  --model gpt-5 \
  --call_gpt true
```

After annotation, point `TICVLA_DATA_ROOT` in `.env.training` at the processed dataset directory used for supervised training.

## Supervised Training

Supervised training runs in two stages. First fine-tune the VLM on CoT/waypoint text:

```bash
conda activate tic-vla
source .env.training
python -m ticvla.training.train --stage vlm --config configs/train_vlm.yaml
```

Then train the action head from the VLM checkpoint configured in `configs/train_action.yaml`. The action stage freezes the VLM and detaches VL hidden features before action decoding:

```bash
conda activate tic-vla
source .env.training
python -m ticvla.training.train --stage action --config configs/train_action.yaml
```

Open-loop evaluation uses the same conda environment and dataset layout. Point `TICVLA_CHECKPOINT_PATH` at the trained checkpoint you want to evaluate:

```bash
conda activate tic-vla
source .env.training
export TICVLA_CHECKPOINT_PATH=/path/to/ticvla.ckpt
export TICVLA_TEST_DATA_DIR="${TICVLA_DATA_ROOT}/DynaNav/DynaNav_json"
export TICVLA_TEST_OUTPUT_DIR="${TICVLA_OUTPUT_DIR}/open_loop_eval"
python -m ticvla.training.evaluate
```

## DynaNav Testing

DynaNav runs through Isaac Sim Python via `ISAAC_SIM_PYTHON`:

```bash
source .env.testing
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_example.yaml
# Full benchmark set:
DynaNav/run_benchmark.sh DynaNav/configs/benchmark_full.yaml
```

## RL Fine-Tuning

RL fine-tuning is optional. If you want to run it, install IsaacLab `v2.2.1` after setting up Isaac Sim.

The PPO scripts live in `rl/fine_tuning/`. See `rl/README.md` for Python version requirements, PyTorch pinning, environment registration, training, and evaluation commands.

## Paths And Checkpoints

Local paths are configured through environment variables:

- `ISAAC_SIM_ROOT`: Isaac Sim installation directory.
- `ISAAC_SIM_PYTHON`: Isaac Sim Python launcher, usually `${ISAAC_SIM_ROOT}/python.sh`.
- `TICVLA_DATA_ROOT`: root directory for supervised training/evaluation data.
- `TICVLA_BASE_MODEL_PATH`: base VLM path or Hugging Face model id.
- `TICVLA_CHECKPOINT_PATH`: TIC-VLA checkpoint for DynaNav testing.
- `TICVLA_OUTPUT_DIR`: output root for logs, checkpoints, and benchmark results.

## Pretrained Checkpoint

We release the pretrained **TIC-VLA** checkpoint trained on the released TIC-VLA reasoning dataset (SCAND, GND, and DynaNav). It can be downloaded from the [TIC-VLA Hugging Face dataset repository](https://huggingface.co/datasets/handsomeYun/TIC-VLA/blob/main/TIC-VLA-model.ckpt).


## Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{huang2026ticvla,
  title={TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments},
  author={Zhiyu Huang and Yun Zhang and Johnson Liu and Rui Song and Chen Tang and Jiaqi Ma},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```
