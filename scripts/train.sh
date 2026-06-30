#!/usr/bin/env bash
set -euo pipefail

# Fresh-instance launcher for the TinyPython model sweep.
#
# Common usage:
#   bash scripts/train.sh
#   SWEEP_MODE=full bash scripts/train.sh
#   SWEEP_MODE=pilot_then_full bash scripts/train.sh
#   SWEEP_GENERATOR_SCRIPT=./scripts/generate_python_model_sweep_two.py SWEEP_OUTPUT_DIRECTORY=configs/generated/python_model_sweep_two bash scripts/train.sh
#
# Common overrides:
#   REPO_URL=https://github.com/BertilBraun/LLM-Light.git
#   REPO_DIR=LLM-Light
#   BRANCH=master
#   GPU_IDS=0,1,2,3
#   MAX_PARALLEL_JOBS=2
#   SWEEP_MODE=pilot
#   SWEEP_GENERATOR_SCRIPT=./scripts/generate_python_model_sweep.py
#   SYNC_EXTRAS=dev

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/LLM-Light.git}"
REPO_DIR="${REPO_DIR:-LLM-Light}"
BRANCH="${BRANCH:-master}"
GPU_IDS="${GPU_IDS-0,1,2,3}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-2}"
SWEEP_MODE="${SWEEP_MODE:-pilot}"
SWEEP_GENERATOR_SCRIPT="${SWEEP_GENERATOR_SCRIPT:-./scripts/generate_python_model_sweep.py}"
BASE_CONFIGURATION_PATH="${BASE_CONFIGURATION_PATH:-configs/python_moe_full.yaml}"
SWEEP_OUTPUT_DIRECTORY="${SWEEP_OUTPUT_DIRECTORY:-configs/generated/python_model_sweep}"
SYNC_EXTRAS="${SYNC_EXTRAS-dev}"

python_gpu_count_from_list() {
  local gpu_list="$1"
  if [ -z "$gpu_list" ]; then
    echo "0"
    return
  fi
  local comma_count
  comma_count="$(printf "%s" "$gpu_list" | tr -cd "," | wc -c)"
  echo "$((comma_count + 1))"
}

run_sweep_mode() {
  local mode="$1"
  local mode_arguments=()

  if uv run python "$SWEEP_GENERATOR_SCRIPT" --help 2>&1 | grep -q -- "--mode"; then
    mode_arguments=(--mode "$mode")
  elif [ "$SWEEP_MODE" = "pilot_then_full" ]; then
    echo "SWEEP_GENERATOR_SCRIPT does not accept --mode; pilot_then_full would run it twice."
    echo "Use SWEEP_MODE=pilot or SWEEP_MODE=full for this generator."
    exit 1
  fi

  echo "Generating Python model sweep: $mode with $SWEEP_GENERATOR_SCRIPT"
  uv run python "$SWEEP_GENERATOR_SCRIPT" \
    "${mode_arguments[@]}" \
    --base-configuration-path "$BASE_CONFIGURATION_PATH" \
    --output-directory "$SWEEP_OUTPUT_DIRECTORY"

  echo "Running Python model sweep plan: $mode"
  if [ -n "$GPU_IDS" ]; then
    uv run python -m llm_lite.scripts.run_plan \
      --config "$SWEEP_OUTPUT_DIRECTORY"/*.yaml \
      --max-parallel-jobs "$MAX_PARALLEL_JOBS" \
      --gpus "$GPU_IDS"
  else
    uv run python -m llm_lite.scripts.run_plan \
      --config "$SWEEP_OUTPUT_DIRECTORY"/*.yaml \
      --max-parallel-jobs "$MAX_PARALLEL_JOBS"
  fi
}

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
if [ -n "$SYNC_EXTRAS" ]; then
  uv sync --extra "$SYNC_EXTRAS"
else
  uv sync
fi

echo "Checking GPU visibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not on PATH; continuing to the PyTorch CUDA check."
fi

REQUIRED_CUDA_DEVICES="$(python_gpu_count_from_list "$GPU_IDS")"
export REQUIRED_CUDA_DEVICES
uv run python -c "
import os
import torch

required_device_count = int(os.environ['REQUIRED_CUDA_DEVICES'])
actual_device_count = torch.cuda.device_count()
print(f'torch={torch.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
print(f'cuda_device_count={actual_device_count}')
for device_index in range(actual_device_count):
    print(f'cuda_device_{device_index}={torch.cuda.get_device_name(device_index)}')
if required_device_count > 0:
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available to PyTorch.')
    if actual_device_count < required_device_count:
        raise SystemExit(
            f'This sweep expects at least {required_device_count} visible CUDA devices.'
        )
"

case "$SWEEP_MODE" in
  pilot)
    run_sweep_mode "pilot"
    ;;
  full)
    run_sweep_mode "full"
    ;;
  pilot_then_full)
    run_sweep_mode "pilot"
    run_sweep_mode "full"
    ;;
  *)
    echo "SWEEP_MODE must be pilot, full, or pilot_then_full."
    exit 1
    ;;
esac

echo "Export bundles:"
find runs -path "*/export/bundle.zip" -print 2>/dev/null || true
