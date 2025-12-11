# Chunk-Level Offline Reinforcement Learning Framework

A complete offline reinforcement learning framework for improving pretrained Pi0 visuomotor policies on the LIBERO benchmark using chunk-level action abstractions.

## Overview

This framework implements chunk-level offline RL to improve Pi0 policies by:

1. **Chunk-level Abstraction**: Groups consecutive actions into coherent action chunks, treating each chunk as a single high-level decision unit
2. **Offline Dataset Construction**: Converts LIBERO demonstrations into chunk-level transitions with aggregated rewards
3. **Offline RL Optimization**: Trains critic and value networks, updates policy using advantage-weighted maximum-likelihood objective

### Key Concepts

- **Action Chunks**: Sequences of $H$ consecutive actions $(a_t, a_{t+1}, ..., a_{t+H-1})$ treated as a single decision unit
- **Chunk-level Rewards**: Binary indicators for sub-goal completion within each chunk
- **Advantage-weighted Learning**: Policy updates weighted by advantage estimates from the critic network

## Project Structure

```
model/
├── __init__.py              # Module entry point
├── config/                   # Configuration module
│   ├── defaults.py          # Default hyperparameters
│   └── config.json          # JSON config file
├── data/                     # Data module
│   └── dataset.py           # ChunkDataset class
├── models/                   # Model architectures
│   ├── encoders.py          # Encoders (image, observation, action)
│   ├── critic.py            # Q-function network
│   ├── value.py             # V-function network
│   └── policy.py            # Policy network
├── training/                 # Training module
│   ├── trainer.py           # Main trainer class
│   └── losses.py            # Loss functions
├── integration/              # Integration module
│   └── pi0.py               # Pi0 integration
├── utils/                    # Utility functions
│   └── helpers.py           # Helper functions
└── scripts/                  # Scripts
    ├── train.py             # Main training script
    ├── download_pi0_checkpoints.py  # Download Pi0 checkpoints
    └── download_from_huggingface.py # Download from HuggingFace
```

## Installation

### 1. Install Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install all dependencies
poetry install

# Or if you prefer pip
pip install -r requirements.txt
```

**Core dependencies**:
- `torch>=2.0.0`: PyTorch for neural networks
- `numpy>=1.24.0,<2.0`: Numerical computations
- `pandas>=2.0.0`: Data manipulation
- `pyarrow>=10.0.0`: Parquet file reading
- `tqdm>=4.65.0`: Progress bars
- `pillow>=9.5.0`: Image processing
- `fsspec[gcs]>=2024.6.0`: For downloading from Google Cloud Storage
- `filelock>=3.16.1`: For concurrent download safety
- `huggingface-hub>=0.20.0`: For downloading from HuggingFace Hub

### 2. Download Checkpoints

#### Option A: Download Pi0 Checkpoints from Google Cloud Storage

Download the original Pi0 checkpoints (e.g., `pi0_libero`) from Google Cloud Storage:

```bash
# Download pi0_libero checkpoint
poetry run python -m model.scripts.download_pi0_checkpoints \
    --checkpoints pi0_libero

# Or download all available checkpoints
poetry run python -m model.scripts.download_pi0_checkpoints \
    --checkpoints all
```

Checkpoints will be downloaded to `checkpoints/pi0/` by default.

**Available Pi0 checkpoints**:
- `pi0_libero`: Pi0 model fine-tuned on LIBERO benchmark
- `pi0_base`: Base Pi0 model

#### Option B: Download RLVLA Checkpoint from HuggingFace

Download the RLVLA checkpoint (renamed from `pi05_libero`) from HuggingFace Hub:

```bash
# Download RLVLA checkpoint
poetry run python -m model.scripts.download_from_huggingface \
    --repo-id yunzhenzhang/rlvla \
    --download-dir checkpoints/pi0/rlvla
```

The checkpoint will be downloaded to `checkpoints/pi0/rlvla/` by default.


### 3. Verify Installation

Check if all dependencies are installed:

```bash
poetry run python -m model.scripts.check_dependencies
```

## Training

### Quick Start

Train the model on LIBERO dataset:

```bash
poetry run python -m model.scripts.train \
    --dataset-dir data/lerobot/libero_spatial \
    --action-horizon 10 \
    --action-dim 7 \
    --num-epochs 100 \
    --device cuda
```

### Training with Pi0-generated Action Chunks (Recommended)

Use Pi0 to generate action chunks instead of using demonstration actions:

```bash
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

### Training on A100 80GB

For training on NVIDIA A100 80GB GPUs, use the following recommended settings:

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

**A100 80GB Optimizations**:
- Use `--batch-size 64` or larger (A100 can handle larger batches)
- Enable mixed precision training (automatically handled by PyTorch)
- Monitor GPU memory usage with `nvidia-smi`

### Command Line Arguments

```bash
poetry run python -m model.scripts.train \
    --dataset-dir <path>              # LeRobot dataset directory (required)
    --action-horizon 10               # Action chunk length H (default: 10)
    --action-dim 7                    # Action dimension (default: 7)
    --gamma 0.99                      # Discount factor (default: 0.99)
    --beta 1.0                        # Advantage weighting temperature (default: 1.0)
    --lambda-v 1.0                    # Value loss weight (default: 1.0)
    --lambda-pi 1.0                   # Policy loss weight (default: 1.0)
    --lr 3e-4                         # Learning rate (default: 3e-4)
    --batch-size 32                   # Batch size (default: 32, use 64+ for A100)
    --num-epochs 100                  # Number of epochs (default: 100)
    --device cuda                     # Device: 'cpu' or 'cuda' (default: auto)
    --use-pi0-actions                 # Use Pi0 to generate action chunks
    --pi0-config pi05_libero          # Pi0 configuration name
    --pi0-checkpoints-dir <path>      # Local Pi0 checkpoints directory (default: checkpoints/pi0)
    --save-interval 10                # Checkpoint save interval (epochs, default: 10)
    --log-interval 100                # Logging interval (steps, default: 100)
    --resume <path>                   # Resume from checkpoint
```

### Hyperparameters

All hyperparameters are documented in [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md). Key hyperparameters include:

- **action_horizon (H)**: `10` - Length of action chunks
- **action_dim**: `7` - Action space dimension (LIBERO uses 7D actions)
- **gamma (γ)**: `0.99` - Discount factor
- **beta (β)**: `1.0` - Advantage weighting temperature
- **lambda_v (λ_V)**: `1.0` - Value loss weight
- **lambda_pi (λ_π)**: `1.0` - Policy loss weight
- **lr**: `3e-4` - Learning rate
- **batch_size**: `32` (default), `64+` recommended for A100 80GB

See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for complete documentation.

## Core Formulations

### Chunk-level Transition

Each transition contains:
- $x_t$: observation at chunk start (image, state, prompt)
- $A_t$: action chunk $(a_t, a_{t+1}, ..., a_{t+H-1}) \in \mathbb{R}^{H \times d_a}$
- $r_t$: chunk-level reward
- $x_{t+H}$: observation after executing the chunk
- $\text{done}$: task completion flag

### Chunk-level Reward

$$r^{\text{chunk}}_t = \mathbf{1}\left(\max_{k \in [0,H-1]} b_{t+k} = 1\right)$$

A chunk is considered successful if *any* frame within it corresponds to a sub-goal completion.

### Loss Functions

1. **Critic Loss** ($L_Q$):
   $$L_Q = \mathbb{E}_\mathcal{D}\left[\left(Q_\theta(x_t, A_t) - y_t\right)^2\right]$$
   where $y_t = r_t + \gamma^H V_\psi(x_{t+H})$

2. **Value Loss** ($L_V$):
   $$L_V = \mathbb{E}_{\mathcal{D}}\left[\left(V_\psi(x_t) - \mathbb{E}_{A_t\sim\pi_\phi}\left[Q_\theta(x_t,A_t) - \log \pi_\phi(A_t\mid x_t)\right]\right)^2\right]$$

3. **Policy Loss** ($L_\pi$):
   $$L_\pi = - \mathbb{E}_{\mathcal{D}}\left[w_t \log \pi_\phi(A_t \mid x_t)\right]$$
   where $w_t = \exp\left(\beta (Q_\theta(x_t, A_t) - V_\psi(x_t))\right)$

4. **Total Loss**:
   $$L = L_Q + \lambda_V L_V + \lambda_\pi L_\pi$$

## Dataset Requirements

The dataset should be in LeRobot format with:
- `data/chunk-*/episode_*.parquet`: Episode data files
- `meta/info.json`: Dataset metadata
- Each episode should contain:
  - `observation/image`: RGB images (256×256×3)
  - `observation/state`: State vectors
  - `action`: Action sequences
  - `reward`: Frame-level rewards
  - `subgoal`: Binary sub-goal completion indicators

## Usage as a Python Module

```python
from model import ChunkLevelOfflineRL
from model.data import ChunkDataset
from model.models import Critic, ValueNetwork, PolicyNetwork

# Create trainer
trainer = ChunkLevelOfflineRL(
    dataset_dir="data/lerobot/libero_spatial",
    action_horizon=10,
    action_dim=7,
    use_pi0_actions=True,
    pi0_config="pi05_libero",
    pi0_checkpoints_dir="checkpoints/pi0",
)

# Train
trainer.train(num_epochs=100)
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space (for dataset and checkpoints)

### Recommended for Training
- **GPU**: NVIDIA A100 80GB (recommended) or RTX 4090 (24GB)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space

### A100 80GB Specific Notes
- Can handle batch sizes of 64-128
- Supports longer action horizons (H=10-20)
- Enables faster training with larger models
- Use `--batch-size 64` or larger for optimal performance

## Troubleshooting

### Import Errors

If you encounter import errors:
1. **Missing packages**: Install missing packages from `requirements.txt`
2. **Pi0 imports**: Ensure `pi0_interface/openpi` path is correct (optional, only needed for Pi0 integration)
3. **Version conflicts**: Check Python version (requires >=3.10)

### Download Errors

If checkpoint downloads fail:
1. **Network issues**: Check internet connection and firewall settings
2. **GCS access**: Ensure you can access Google Cloud Storage (for Pi0 checkpoints)
3. **HuggingFace access**: Ensure you're logged in for private repositories
4. **Disk space**: Ensure sufficient disk space (each checkpoint is ~5-10 GB)

### CUDA/GPU Issues

- **PyTorch CUDA**: Install PyTorch with CUDA support if using GPU
- **CUDA version**: Match PyTorch CUDA version with your system CUDA version
- **Out of memory**: Reduce batch size or use gradient checkpointing

### Training Issues

- **Slow training**: Increase batch size (if GPU memory allows), use A100 80GB
- **Convergence issues**: Adjust learning rate, check hyperparameters in `HYPERPARAMETERS.md`
- **Checkpoint loading**: Ensure checkpoint paths are correct

## References

- Chunk-level offline RL methodology
- Pi0 visuomotor policy: [Physical Intelligence](https://www.physicalintelligence.company/blog/pi0)
- LIBERO benchmark: [LIBERO Project](https://libero-project.github.io/)

## License

See project root LICENSE file.

