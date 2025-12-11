# RLVLA Dataset Pipeline

This directory contains tools and datasets for creating and managing datasets for the RLVLA (Reinforcement Learning for Visuomotor Learning and Adaptation) project. The pipeline supports downloading LIBERO datasets, converting them to LeRobot format, and annotating rewards for offline reinforcement learning.

## Overview

The `data/` directory provides a complete workflow for:
1. **Downloading** LIBERO demonstration datasets (HDF5 format)
2. **Converting** HDF5 datasets to LeRobot format for training with Pi0
3. **Annotating** reward signals for offline reinforcement learning
4. **Viewing and managing** datasets in both formats

## Directory Structure

```
data/
├── datasets/                    # Raw datasets in HDF5 format
│   └── libero_spatial/         # LIBERO-Spatial dataset (10 tasks, 50 demos each)
├── lerobot/                     # Converted datasets in LeRobot format
│   └── libero_spatial/         # LeRobot-formatted dataset ready for training
├── LIBERO/                      # LIBERO codebase (for dataset downloading)
├── view_hdf5.py                 # Tool to view and inspect HDF5 files
├── annotate_rewards.py          # Tool to annotate rewards in HDF5 files
├── add_rewards_to_lerobot.py    # Tool to add reward field to LeRobot datasets
└── view_lerobot_parquet.py      # Tool to view and annotate rewards in LeRobot datasets
```

## Dataset Formats

### 1. HDF5 Format (`datasets/libero_spatial/`)

**Purpose**: Raw demonstration data from LIBERO benchmark.

**Structure**: Each HDF5 file contains multiple demonstrations (typically 50 demos per file):
- `data/demo_0/`: First demonstration
  - `obs/`: Observations (images, states)
    - `agentview_rgb`: Main camera view (128×128×3)
    - `eye_in_hand_rgb`: Wrist camera view (128×128×3)
    - `ee_pos`, `ee_ori`: End-effector pose
    - `gripper_states`: Gripper state
  - `actions`: Robot actions (7D: position control)
  - `rewards`: Sparse rewards (0 or 1)
  - `dones`: Episode termination flags

**Usage**: 
- View with: `poetry run python data/view_hdf5.py <file.hdf5>`
- Annotate rewards with: `poetry run python data/annotate_rewards.py <file.hdf5>`

### 2. LeRobot Format (`lerobot/libero_spatial/`)

**Purpose**: Standardized format for training visuomotor policies (e.g., Pi0).

**Structure**:
- `data/chunk-000/`: Parquet files containing episode data
  - `episode_*.parquet`: Each file contains one episode
- `images/`: Image files (256×256×3, resized from original)
  - `image/`: Main camera images
  - `wrist_image/`: Wrist camera images
- `meta/`: Dataset metadata
  - `info.json`: Dataset configuration and feature definitions
  - `episodes.jsonl`: Episode-level metadata
  - `tasks.jsonl`: Task descriptions

**Features**:
- `image`: Main camera view (256×256×3)
- `wrist_image`: Wrist camera view (256×256×3)
- `state`: Proprioceptive state (8D: ee_pos + ee_ori + gripper)
- `actions`: Robot actions (7D)
- `reward`: Reward signal (1D, float32) - **for offline RL**
- `task`: Language instruction

**Usage**:
- View and annotate: `poetry run python data/view_lerobot_parquet.py <episode.parquet>`

## Workflow

### Step 1: Download LIBERO Dataset

Download the LIBERO-Spatial dataset using the official download script:

```bash
cd data/LIBERO
poetry run python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --download-dir ../datasets \
  --use-huggingface
```

This will download 10 HDF5 files (one per task) to `data/datasets/libero_spatial/`, each containing 50 demonstrations (~500 MB per file, ~5 GB total).

### Step 2: Convert to LeRobot Format

Convert HDF5 datasets to LeRobot format for training:

```bash
poetry run python pi0_interface/openpi/examples/libero/convert_libero_hdf5_to_lerobot.py \
  --data-dir data/datasets/libero_spatial \
  --output-dir data/lerobot
```

**What this does**:
- Reads all HDF5 files from the input directory
- Extracts images, states, actions, and language instructions
- Resizes images from 128×128 to 256×256
- Combines state vectors (ee_pos + ee_ori + gripper)
- Saves as LeRobot format in `data/lerobot/libero_spatial/`

**Output**: 
- 500 episodes (10 tasks × 50 demos)
- ~62,250 total frames
- Images stored as PNG bytes in parquet files
- Metadata in JSON format

### Step 3: Add Reward Field

Add reward field to LeRobot dataset (initialized to 0.0):

```bash
poetry run python data/add_rewards_to_lerobot.py \
  --dataset-dir data/lerobot/libero_spatial
```

This adds a `reward` field to all parquet files, which is required for offline reinforcement learning.

### Step 4: Annotate Rewards

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
- `q`: Quit (prompts to save if unsaved)

**Annotation Strategy**:
Mark frames corresponding to:
- Successful object grasp
- Object alignment with target
- Stable placement
- Task-specific milestones

## Dataset Descriptions

### LIBERO-Spatial Dataset

**Purpose**: Spatial reasoning benchmark for manipulation tasks.

**Tasks** (10 total):
1. Pick up the black bowl between the plate and the ramekin and place it on the plate
2. Pick up the black bowl from table center and place it on the plate
3. Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
4. Pick up the black bowl next to the cookie box and place it on the plate
5. Pick up the black bowl next to the plate and place it on the plate
6. Pick up the black bowl next to the ramekin and place it on the plate
7. Pick up the black bowl on the cookie box and place it on the plate
8. Pick up the black bowl on the ramekin and place it on the plate
9. Pick up the black bowl on the stove and place it on the plate
10. Pick up the black bowl on the wooden cabinet and place it on the plate

**Characteristics**:
- **Focus**: Spatial relationships and object localization
- **Difficulty**: Medium (requires understanding spatial relationships)
- **Demos per task**: 50
- **Total demonstrations**: 500
- **Average episode length**: ~100-120 frames

**Use Cases**:
- Training spatial reasoning policies
- Testing generalization to different object locations
- Offline RL with reward upsampling

## Tools Reference

### `view_hdf5.py`
View and inspect HDF5 files:
```bash
poetry run python data/view_hdf5.py <file.hdf5> [--inspect] [--demo-idx N]
```

### `annotate_rewards.py`
Annotate rewards in HDF5 files (legacy, use `view_lerobot_parquet.py` for LeRobot format):
```bash
poetry run python data/annotate_rewards.py <file.hdf5>
```

### `add_rewards_to_lerobot.py`
Add reward field to LeRobot dataset:
```bash
poetry run python data/add_rewards_to_lerobot.py \
  --dataset-dir data/lerobot/libero_spatial \
  --initial-reward 0.0
```

### `view_lerobot_parquet.py`
View images and annotate rewards in LeRobot parquet files:
```bash
# View and annotate
poetry run python data/view_lerobot_parquet.py <episode.parquet>
# Example:

poetry run python data/view_lerobot_parquet.py \ data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet

# Inspect file structure only
poetry run python data/view_lerobot_parquet.py <episode.parquet> --inspect

# Start from specific frame
poetry run python data/view_lerobot_parquet.py <episode.parquet> --start-frame 50
```

## Using the Dataset for Training

### For Imitation Learning (IL)

The LeRobot dataset can be used directly with Pi0 for imitation learning:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("libero_spatial")
# Use image, wrist_image, state, actions, task fields
```

### For Offline Reinforcement Learning

After annotating rewards, use the dataset for offline RL:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("libero_spatial")
# Access reward field for value function training
for episode in dataset:
    for frame in episode:
        reward = frame["reward"]  # Use for RL training
```

**Note**: Set `HF_LEROBOT_HOME` environment variable to point to `data/lerobot/`:
```bash
export HF_LEROBOT_HOME=/path/to/rlvla/data/lerobot
```

## File Sizes

- **HDF5 files**: ~500 MB each (10 files, ~5 GB total)
- **LeRobot dataset**: ~4-5 GB (includes resized images)
- **Parquet files**: ~8-10 MB per episode (500 episodes)

## Dependencies

Key dependencies (installed via Poetry):
- `h5py>=3.8.0`: HDF5 file handling
- `pyarrow`: Parquet file handling
- `opencv-python>=4.8.0`: Image display
- `lerobot`: LeRobot dataset format
- `pillow>=9.5.0`: Image processing

## Notes

- All tools automatically create backup files (`.backup` extension) before modifying data
- Reward annotation is frame-level (each frame can have reward 0 or 1)
- Images are automatically resized from 128×128 to 256×256 during conversion
- The dataset is compatible with Pi0 training pipeline
- Reward field is optional for IL but required for offline RL

## References

- [LIBERO Repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/)
- [Pi0 Training Guide](../pi0_interface/openpi/README.md)
