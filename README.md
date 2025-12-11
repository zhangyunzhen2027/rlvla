# RLVLA: Reinforcement Learning for Visuomotor Learning and Adaptation

A complete framework for improving pretrained Pi0 visuomotor policies on the LIBERO benchmark using chunk-level offline reinforcement learning.

## Overview

This project implements a chunk-level offline RL framework that:

1. **Improves Pi0 Policies**: Uses offline RL to enhance pretrained Pi0 visuomotor policies
2. **Chunk-level Abstraction**: Groups consecutive actions into coherent action chunks for high-level decision making
3. **LIBERO Benchmark**: Evaluates on the LIBERO manipulation benchmark
4. **Simulation**: Runs LIBERO simulations using the openpi framework

## Project Structure

```
rlvla/
├── data/                    # Dataset pipeline
│   ├── datasets/           # Raw LIBERO datasets (HDF5)
│   ├── lerobot/            # Converted datasets (LeRobot format)
│   └── LIBERO/             # LIBERO codebase
├── model/                   # Chunk-level offline RL framework
│   ├── data/               # Dataset loading
│   ├── models/              # Neural networks (Critic, Value, Policy)
│   ├── training/            # Training loop and losses
│   └── scripts/             # Training and checkpoint scripts
├── pi0_interface/          # Pi0 simulation interface
│   └── openpi/             # openpi framework for running simulations
└── checkpoints/            # Model checkpoints
    └── pi0/                # Pi0 and RLVLA checkpoints
```

## 1. Hardware Requirements

### Minimum Requirements (Inference)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 8GB VRAM | RTX 4090 (24GB) or A100 (80GB) |
| **VRAM** | > 8 GB | 16GB+ |
| **System RAM** | 16 GB | 32 GB |
| **Storage** | 50 GB free space | 100 GB+ |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 |

### Training Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 16GB VRAM | A100 80GB |
| **VRAM** | > 16 GB | 80 GB |
| **System RAM** | 32 GB | 64 GB+ |
| **Storage** | 100 GB free space | 200 GB+ |

### Software Requirements

- **Python**: >= 3.10 (>= 3.11 for openpi)
- **CUDA**: 11.3+ (for GPU)
- **Poetry**: For dependency management
- **Docker**: Optional but recommended for simulation
- **NVIDIA Container Toolkit**: Required for Docker GPU support

## 2. Data Preparation and Acquisition

### Step 1: Download LIBERO Dataset

Download the LIBERO-Spatial dataset (10 tasks, 50 demonstrations each):

```bash
cd data/LIBERO
poetry run python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --download-dir ../datasets \
  --use-huggingface
```

This downloads 10 HDF5 files (~5 GB total) to `data/datasets/libero_spatial/`.

**Alternative**: Download from HuggingFace directly:
```bash
cd data/LIBERO
poetry run python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --download-dir ../datasets \
  --use-huggingface
```

### Step 2: Convert to LeRobot Format

Convert HDF5 datasets to LeRobot format (required for training):

```bash
poetry run python pi0_interface/openpi/examples/libero/convert_libero_hdf5_to_lerobot.py \
  --data-dir data/datasets/libero_spatial \
  --output-dir data/lerobot
```

**What this does**:
- Extracts images, states, actions, and language instructions
- Resizes images from 128×128 to 256×256
- Combines state vectors (ee_pos + ee_ori + gripper)
- Saves as LeRobot format in `data/lerobot/libero_spatial/`

**Output**: 
- 500 episodes (10 tasks × 50 demos)
- ~62,250 total frames
- Images stored in parquet files

### Step 3: Add Reward Field

Add reward field to LeRobot dataset (required for offline RL):

```bash
poetry run python data/add_rewards_to_lerobot.py \
  --dataset-dir data/lerobot/libero_spatial
```

This initializes a `reward` field (set to 0.0) in all parquet files.

### Step 4: Annotate Rewards (Optional)

Annotate rewards by viewing images and marking important frames:

```bash
poetry run python data/view_lerobot_parquet.py \
  data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet
```

**Controls**:
- `1`: Mark current frame reward = 1 (important milestone)
- `0`: Mark current frame reward = 0
- `←/→` or `a/d`: Navigate frames
- `s`: Save changes
- `q`: Quit

**Annotation Strategy**: Mark frames for successful grasps, alignments, placements, and task milestones.

### Dataset Format

The final dataset in `data/lerobot/libero_spatial/` contains:
- `data/chunk-*/episode_*.parquet`: Episode data files
- `meta/info.json`: Dataset metadata
- Each episode includes:
  - `observation/image`: RGB images (256×256×3)
  - `observation/wrist_image`: Wrist camera images (256×256×3)
  - `observation/state`: State vectors (8D)
  - `action`: Robot actions (7D)
  - `reward`: Reward signal (for offline RL)
  - `task`: Language instruction

## 3. Running LIBERO Simulation with Pi0

### Installation

#### Step 1: Install Project Dependencies

```bash
# Install main project dependencies
poetry install
```

#### Step 2: Install openpi Dependencies

```bash
cd pi0_interface/openpi

# Install dependencies using uv (recommended)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Initialize git submodules (required for LIBERO)
git submodule update --init --recursive
```

#### Step 3: Download Checkpoints

**Option A: Download Pi0 Checkpoint from Google Cloud Storage**

```bash
# From project root
poetry run python -m model.scripts.download_pi0_checkpoints \
    --checkpoints pi05_libero
```

**Option B: Download RLVLA Checkpoint from HuggingFace**

```bash
# Download RLVLA checkpoint (renamed from pi05_libero)
poetry run python -m model.scripts.download_from_huggingface \
    --repo-id yunzhenzhang/rlvla \
    --download-dir checkpoints/pi0/rlvla
```

Checkpoints will be saved to `checkpoints/pi0/` by default.

### Running Simulation with Docker (Recommended)

#### Step 1: Grant X11 Access

```bash
sudo xhost +local:docker
```

#### Step 2: Run Simulation with Default Checkpoint

```bash
cd pi0_interface/openpi

# Run with default checkpoint (pi05_libero) and libero_spatial task suite
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# If you encounter EGL errors, use glx instead:
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

#### Step 3: Run Simulation with RLVLA Checkpoint

```bash
cd pi0_interface/openpi

# Set checkpoint path (relative to openpi directory)
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ../../checkpoints/pi0/rlvla"

# Run simulation
docker compose -f examples/libero/compose.yml up --build
```

#### Step 4: Customize Task Suite

To run different LIBERO task suites:

```bash
# Run libero_10 task suite
export CLIENT_ARGS="--args.task-suite-name libero_10"
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# Available task suites:
# - libero_spatial (default)
# - libero_object
# - libero_goal
# - libero_10
# - libero_90
```

### Running Simulation without Docker

#### Terminal 1: Start Policy Server

```bash
cd pi0_interface/openpi

# Create virtual environment
uv venv --python 3.11 examples/libero/.venv
source examples/libero/.venv/bin/activate

# Install dependencies
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the policy server (with default checkpoint)
uv run scripts/serve_policy.py --env LIBERO

# Or with RLVLA checkpoint
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir ../../checkpoints/pi0/rlvla
```

#### Terminal 2: Run Simulation Client

```bash
cd pi0_interface/openpi
source examples/libero/.venv/bin/activate

# Run simulation
python examples/libero/main.py

# If you encounter EGL errors:
MUJOCO_GL=glx python examples/libero/main.py
```

### Simulation Output

- **Videos**: Saved to `data/libero/videos/` by default
- **Success Rate**: Printed to console after each task
- **Logs**: Policy server logs show inference details

## 4. Training (Optional)

To train the chunk-level offline RL framework:

### Step 1: Install Training Dependencies

```bash
# Dependencies are already installed with poetry install
# Verify installation:
poetry run python -m model.scripts.check_dependencies
```

### Step 2: Train Model

```bash
# Basic training
poetry run python -m model.scripts.train \
    --dataset-dir data/lerobot/libero_spatial \
    --action-horizon 10 \
    --action-dim 7 \
    --num-epochs 100 \
    --device cuda

# Training with Pi0-generated action chunks (recommended)
poetry run python -m model.scripts.train \
    --dataset-dir data/lerobot/libero_spatial \
    --use-pi0-actions \
    --pi0-config pi05_libero \
    --pi0-checkpoints-dir checkpoints/pi0 \
    --action-horizon 10 \
    --action-dim 7 \
    --num-epochs 100 \
    --device cuda
```

### Step 3: Training on A100 80GB

For optimal performance on A100 80GB:

```bash
poetry run python -m model.scripts.train \
    --dataset-dir data/lerobot/libero_spatial \
    --use-pi0-actions \
    --pi0-config pi05_libero \
    --pi0-checkpoints-dir checkpoints/pi0 \
    --action-horizon 10 \
    --action-dim 7 \
    --batch-size 64 \
    --num-epochs 100 \
    --lr 3e-4 \
    --device cuda
```

See [`model/README.md`](model/README.md) for complete training documentation and hyperparameters.

## Quick Start Guide

### Complete Workflow

```bash
# 1. Install dependencies
poetry install

# 2. Download and prepare data
cd data/LIBERO
poetry run python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial --download-dir ../datasets --use-huggingface
cd ../..
poetry run python pi0_interface/openpi/examples/libero/convert_libero_hdf5_to_lerobot.py \
  --data-dir data/datasets/libero_spatial --output-dir data/lerobot
poetry run python data/add_rewards_to_lerobot.py --dataset-dir data/lerobot/libero_spatial

# 3. Download checkpoints
poetry run python -m model.scripts.download_from_huggingface \
    --repo-id yunzhenzhang/rlvla --download-dir checkpoints/pi0/rlvla

# 4. Install openpi dependencies
cd pi0_interface/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
git submodule update --init --recursive
cd ../..

# 5. Run simulation
cd pi0_interface/openpi
sudo xhost +local:docker
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ../../checkpoints/pi0/rlvla"
docker compose -f examples/libero/compose.yml up --build
```

## Documentation

- **[Model Training](model/README.md)**: Complete guide to the chunk-level offline RL framework
- **[Data Pipeline](data/README.md)**: Dataset preparation and annotation tools
- **[Pi0 Interface](pi0_interface/README.md)**: LIBERO simulation guide

## Troubleshooting

### Common Issues

1. **EGL Errors**: Use `MUJOCO_GL=glx` instead of default EGL
2. **X11 Errors**: Run `sudo xhost +local:docker` before Docker
3. **Checkpoint Loading**: Verify path and config name (`pi05_libero`)
4. **Port Conflicts**: Change port in `main.py` if 8000 is occupied
5. **CUDA Errors**: Ensure NVIDIA drivers and CUDA are properly installed
6. **Import Errors**: Check that all dependencies are installed with `poetry install`

### Getting Help

- Check individual README files in `model/`, `data/`, and `pi0_interface/`
- Review openpi issues: https://github.com/Physical-Intelligence/openpi/issues
- Check LIBERO documentation: https://libero-project.github.io/

## References

- **Chunk-level Offline RL**: Methodology for improving visuomotor policies
- **Pi0 Model**: [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0)
- **LIBERO Benchmark**: [LIBERO Project](https://libero-project.github.io/)
- **openpi Framework**: [Physical Intelligence GitHub](https://github.com/Physical-Intelligence/openpi)

## License

See LICENSE file in project root.

