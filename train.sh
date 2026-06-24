#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/tinystories_moe_full.yaml}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_COUNT="${GPU_COUNT:-2}"

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv install completed, but uv is still not on PATH." >&2
  echo "Try: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

echo "Syncing Python dependencies..."
uv sync

echo "Checking CUDA visibility..."
GPU_COUNT="$GPU_COUNT" uv run python - <<'PY'
import os

import torch

expected_device_count = int(os.environ["GPU_COUNT"])
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")
if torch.cuda.device_count() < expected_device_count:
    raise SystemExit(
        f"This run expects at least {expected_device_count} visible CUDA devices."
    )
PY

echo "Reviewing pipeline stages..."
uv run python -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH" \
  --dry-run

echo "Starting TinyStories MoE distributed training..."
CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run torchrun --standalone --nproc_per_node="$GPU_COUNT" \
  -m llm_lite.scripts.run_pipeline \
  --config "$CONFIG_PATH"
