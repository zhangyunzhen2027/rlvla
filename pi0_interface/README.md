# Pi0 Interface - LIBERO Simulation Guide

This directory contains the Pi0 interface for running LIBERO simulations using the openpi framework. This guide covers minimum inference requirements, running LIBERO simulations, and using the RLVLA checkpoint from HuggingFace.

## 1. Minimum Inference Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 8GB VRAM | RTX 4090 (24GB) or A100 (80GB) |
| **VRAM** | > 8 GB | 16GB+ |
| **System RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB free space | 50 GB+ |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 |

### Software Requirements

- **Python**: >= 3.11
- **CUDA**: 11.3+ (for GPU inference)
- **Docker**: Optional but recommended
- **NVIDIA Container Toolkit**: Required for Docker GPU support

### Dependencies

The openpi framework manages its own dependencies. See installation instructions below.

## 2. Running LIBERO Simulation

### Installation

#### Step 1: Install openpi Dependencies

```bash
cd pi0_interface/openpi

# Install dependencies using uv (recommended)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Initialize git submodules (required for LIBERO)
git submodule update --init --recursive
```

#### Step 2: Download LIBERO Checkpoint

The default checkpoint (`pi05_libero`) will be automatically downloaded from Google Cloud Storage when first used. Alternatively, you can download it manually:

```bash
# The checkpoint will be cached in ~/.cache/openpi by default
# Or set custom location:
export OPENPI_DATA_HOME=/path/to/checkpoints
```

### Running with Docker (Recommended)

#### Step 1: Grant X11 Access

```bash
sudo xhost +local:docker
```

#### Step 2: Run Simulation

```bash
cd pi0_interface/openpi

# Run with default checkpoint (pi05_libero) and libero_spatial task suite
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# If you encounter EGL errors, use glx instead:
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

#### Step 3: Customize Task Suite

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

### Running without Docker

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

# Run the policy server
uv run scripts/serve_policy.py --env LIBERO
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

### Understanding the Simulation

The simulation runs in two parts:

1. **Policy Server** (`scripts/serve_policy.py`): 
   - Loads the Pi0 model checkpoint
   - Serves policy inference via WebSocket
   - Handles observation processing and action generation

2. **Simulation Client** (`examples/libero/main.py`):
   - Runs LIBERO simulation environment
   - Sends observations to policy server
   - Executes actions in simulation
   - Records results and videos

### Output

- **Videos**: Saved to `data/libero/videos/` by default
- **Success Rate**: Printed to console after each task
- **Logs**: Policy server logs show inference details

## 3. Using RLVLA Checkpoint from HuggingFace

The RLVLA checkpoint is a renamed version of `pi05_libero` available on HuggingFace Hub. Follow these steps to use it:

### Step 1: Download RLVLA Checkpoint

From the project root directory:

```bash
# Download RLVLA checkpoint from HuggingFace
poetry run python -m model.scripts.download_from_huggingface \
    --repo-id yunzhenzhang/rlvla \
    --download-dir checkpoints/pi0/rlvla
```

The checkpoint will be downloaded to `checkpoints/pi0/rlvla/` (relative to project root).

### Step 2: Run Simulation with RLVLA Checkpoint

#### Option A: Using Docker

```bash
cd pi0_interface/openpi

# Set checkpoint path (relative to openpi directory)
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ../../checkpoints/pi0/rlvla"

# Run simulation
docker compose -f examples/libero/compose.yml up --build
```

**Note**: The checkpoint directory path should be relative to the `openpi` directory, or use an absolute path.

#### Option B: Without Docker

**Terminal 1: Start Policy Server with RLVLA Checkpoint**

```bash
cd pi0_interface/openpi
source examples/libero/.venv/bin/activate

# Run server with custom checkpoint
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir ../../checkpoints/pi0/rlvla
```

**Terminal 2: Run Simulation Client**

```bash
cd pi0_interface/openpi
source examples/libero/.venv/bin/activate

python examples/libero/main.py
```

### Step 3: Verify Checkpoint Loading

The policy server will print checkpoint loading information:

```
Loading checkpoint from: /path/to/checkpoints/pi0/rlvla
Using config: pi05_libero
Checkpoint loaded successfully
```

### Important Notes

1. **Config Name**: Use `pi05_libero` as the config name (RLVLA checkpoint is compatible with pi05_libero config)
2. **Checkpoint Structure**: The RLVLA checkpoint should have the same structure as the original Pi0 checkpoints:
   - `params/`: Model parameters
   - `assets/`: Normalization statistics and other assets
3. **Path Resolution**: 
   - Docker: Use paths relative to `openpi` directory or mount volumes
   - Without Docker: Use absolute paths or paths relative to current working directory

### Troubleshooting

#### Checkpoint Not Found

If you see "Checkpoint not found" errors:

1. **Verify checkpoint path**: Ensure the path is correct
   ```bash
   ls -la checkpoints/pi0/rlvla/
   # Should show: assets/ and params/ directories
   ```

2. **Use absolute path**: Try using absolute path instead
   ```bash
   export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir /absolute/path/to/checkpoints/pi0/rlvla"
   ```

3. **Check Docker volumes**: If using Docker, ensure the checkpoint directory is mounted
   ```yaml
   # In compose.yml, add to volumes:
   - ../../checkpoints:/checkpoints
   ```

#### Config Mismatch

If you see config-related errors:

1. **Use correct config**: RLVLA uses `pi05_libero` config (same as original pi05_libero checkpoint)
2. **Check config exists**: Verify the config exists in `src/openpi/training/config.py`

#### Performance Issues

- **GPU Memory**: If running out of memory, reduce batch size or use CPU (slower)
- **Network Latency**: WebSocket communication should be fast on localhost
- **Simulation Speed**: LIBERO simulation speed depends on CPU and GPU

## Framework Details

This interface is built on the [openpi](https://github.com/Physical-Intelligence/openpi) framework, which provides:

- **Model Loading**: Automatic checkpoint downloading and loading
- **Policy Inference**: Efficient action generation from observations
- **WebSocket Server**: Real-time policy serving for simulation
- **LIBERO Integration**: Seamless integration with LIBERO benchmark

## Additional Resources

- **openpi Documentation**: See `pi0_interface/openpi/README.md`
- **LIBERO Benchmark**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **openpi Examples**: `pi0_interface/openpi/examples/libero/README.md`
- **RLVLA Model**: https://huggingface.co/yunzhenzhang/rlvla

## Example: Complete Workflow

```bash
# 1. Download RLVLA checkpoint
cd /path/to/rlvla
poetry run python -m model.scripts.download_from_huggingface \
    --repo-id yunzhenzhang/rlvla \
    --download-dir checkpoints/pi0/rlvla

# 2. Navigate to openpi
cd pi0_interface/openpi

# 3. Install dependencies (if not already done)
GIT_LFS_SKIP_SMUDGE=1 uv sync
git submodule update --init --recursive

# 4. Grant X11 access
sudo xhost +local:docker

# 5. Run simulation with RLVLA checkpoint
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ../../checkpoints/pi0/rlvla"
docker compose -f examples/libero/compose.yml up --build

# 6. View results
# Videos saved to: data/libero/videos/
# Success rates printed to console
```

## Troubleshooting

### Common Issues

1. **EGL Errors**: Use `MUJOCO_GL=glx` instead of default EGL
2. **X11 Errors**: Run `sudo xhost +local:docker` before Docker
3. **Checkpoint Loading**: Verify path and config name
4. **Port Conflicts**: Change port in `main.py` if 8000 is occupied
5. **CUDA Errors**: Ensure NVIDIA drivers and CUDA are properly installed

### Getting Help

- Check openpi issues: https://github.com/Physical-Intelligence/openpi/issues
- Check LIBERO documentation: https://libero-project.github.io/
- Review logs from both server and client terminals

