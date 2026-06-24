#!/usr/bin/env bash
set -euo pipefail

# Paste this whole file into a fresh Linux GPU instance shell.
# It clones the repository, installs uv if needed, syncs dependencies,
# verifies CUDA, reviews the pipeline, then starts the 2-GPU TinyStories run.

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/LLM-Light.git}"
REPO_DIR="${REPO_DIR:-LLM-Light}"
BRANCH="${BRANCH:-master}"
CONFIG_PATH="${CONFIG_PATH:-configs/tinystories_moe_full.yaml}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_COUNT="${GPU_COUNT:-2}"

echo "Checking basic system tools..."
command -v git >/dev/null 2>&1 || { echo "git is required but not installed."; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required but not installed."; exit 1; }

echo "Cloning or updating repository..."
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "Repository revision:"
git log --oneline -1

echo "Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not on PATH; continuing to the PyTorch CUDA check."
fi

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

echo "Checking PyTorch CUDA access..."
GPU_COUNT="$GPU_COUNT" uv run python - <<'PY'
import os

import torch

expected_device_count = int(os.environ["GPU_COUNT"])
actual_device_count = torch.cuda.device_count()
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={actual_device_count}")
for device_index in range(actual_device_count):
    print(f"cuda_device_{device_index}={torch.cuda.get_device_name(device_index)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")
if actual_device_count < expected_device_count:
    raise SystemExit(
        f"This run expects at least {expected_device_count} visible CUDA devices."
    )
PY

echo "Reviewing data-preparation pipeline stages..."
uv run python -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH" \
  --to packed_dataset \
  --dry-run

echo "Preparing dataset, tokenizer, and packed sequences once..."
uv run python -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH" \
  --to packed_dataset

echo "Reviewing distributed pretraining stage..."
uv run python -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH" \
  --from pretraining \
  --to pretraining \
  --dry-run

echo "Starting TinyStories MoE distributed training..."
CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  -m llm_lite.scripts.run_pipeline \
  --from pretraining \
  --to pretraining \
  --config "$CONFIG_PATH"

echo "Running post-training and final evaluation once..."
uv run python -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH" \
  --from post_training
