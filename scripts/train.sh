#!/usr/bin/env bash
set -euo pipefail

# Fresh-instance launcher for an arbitrary LLM-Light training config.
#
# Required:
#   CONFIG_PATH=configs/python_moe_full.yaml bash scripts/train.sh
#
# Common overrides:
#   REPO_URL=https://github.com/BertilBraun/LLM-Light.git
#   REPO_DIR=LLM-Light
#   BRANCH=master
#   GPU_IDS=0,1
#   NPROC_PER_NODE=2
#   PREPARE_DATA=1
#   RUN_EVALUATION=1

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/LLM-Light.git}"
REPO_DIR="${REPO_DIR:-LLM-Light}"
BRANCH="${BRANCH:-master}"
GPU_IDS="${GPU_IDS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"

if [ -z "${CONFIG_PATH:-}" ]; then
  echo "CONFIG_PATH is required."
  echo "Example: CONFIG_PATH=configs/python_moe_full.yaml NPROC_PER_NODE=2 GPU_IDS=0,1 bash scripts/train.sh"
  exit 1
fi

echo "Checking basic system tools..."
command -v git >/dev/null 2>&1 || { echo "git is required but not installed."; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required but not installed."; exit 1; }

echo "Cloning or updating repository..."
if [ -d ".git" ] && [ -f "pyproject.toml" ]; then
  echo "Using current repository checkout."
else
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

echo "Repository revision:"
git log --oneline -1

echo "Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv install completed, but uv is still not on PATH."
  echo "Run: export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
}
uv --version

echo "Syncing Python dependencies..."
uv sync

echo "Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not on PATH; continuing to the PyTorch CUDA check."
fi

NPROC_PER_NODE="$NPROC_PER_NODE" uv run python - <<'PY'
import os

import torch

expected_device_count = int(os.environ["NPROC_PER_NODE"])
actual_device_count = torch.cuda.device_count()
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={actual_device_count}")
for device_index in range(actual_device_count):
    print(f"cuda_device_{device_index}={torch.cuda.get_device_name(device_index)}")
if expected_device_count > 1:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")
    if actual_device_count < expected_device_count:
        raise SystemExit(
            f"This run expects at least {expected_device_count} visible CUDA devices."
        )
PY

if [ "$PREPARE_DATA" = "1" ]; then
  echo "Preparing dataset, tokenizer, and packed sequences..."
  uv run python -m llm_lite.scripts.run_pipeline \
    --config "$CONFIG_PATH" \
    --to packed_dataset
fi

echo "Starting pretraining..."
if [ "$NPROC_PER_NODE" -gt 1 ]; then
  if [ -n "$GPU_IDS" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
      -m llm_lite.scripts.run_pipeline \
      --config "$CONFIG_PATH" \
      --from pretraining \
      --to pretraining
  else
    uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
      -m llm_lite.scripts.run_pipeline \
      --config "$CONFIG_PATH" \
      --from pretraining \
      --to pretraining
  fi
else
  uv run python -m llm_lite.scripts.run_pipeline \
    --config "$CONFIG_PATH" \
    --from pretraining \
    --to pretraining
fi

if [ "$RUN_EVALUATION" = "1" ]; then
  echo "Running post-training and final evaluation stages..."
  uv run python -m llm_lite.scripts.run_pipeline \
    --config "$CONFIG_PATH" \
    --from post_training
fi
