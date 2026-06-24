#!/usr/bin/env bash
set -euo pipefail

# Paste this whole file into a fresh Linux GPU instance shell.
# It clones the repository, installs uv if needed, syncs generation dependencies,
# verifies CUDA, then runs the TinyPython teacher-generation pilot.
#
# Default behavior:
#   - 500 semantic seeds
#   - 2 samples per seed
#   - 2 teacher models
#   - one independent vLLM process per GPU
#
# Later full run:
#   RUN_FULL=1 bash generate_samples.sh

REPO_URL="${REPO_URL:-https://github.com/BertilBraun/LLM-Light.git}"
REPO_DIR="${REPO_DIR:-LLM-Light}"
BRANCH="${BRANCH:-master}"

TEACHER_A="${TEACHER_A:-Qwen/Qwen2.5-Coder-7B-Instruct}"
TEACHER_B="${TEACHER_B:-microsoft/Phi-4-mini-instruct}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

PILOT_SEEDS="${PILOT_SEEDS:-500}"
FULL_SEEDS="${FULL_SEEDS:-10000}"
SAMPLES_PER_SEED="${SAMPLES_PER_SEED:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

RUN_FULL="${RUN_FULL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/tinypython}"

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

echo "Syncing Python dependencies for generation..."
uv sync --extra generation

echo "Checking PyTorch CUDA access..."
uv run python - <<'PY'
import torch

actual_device_count = torch.cuda.device_count()
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={actual_device_count}")
for device_index in range(actual_device_count):
    print(f"cuda_device_{device_index}={torch.cuda.get_device_name(device_index)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch.")
if actual_device_count < 2:
    raise SystemExit("This generation plan expects at least 2 visible CUDA devices.")
PY

mkdir -p "$OUTPUT_DIR"

if [ "$RUN_FULL" = "1" ]; then
  SEED_COUNT="$FULL_SEEDS"
  RUN_NAME="full"
else
  SEED_COUNT="$PILOT_SEEDS"
  RUN_NAME="pilot"
fi

OUTPUT_A="$OUTPUT_DIR/${RUN_NAME}_teacher_a.jsonl"
OUTPUT_B="$OUTPUT_DIR/${RUN_NAME}_teacher_b.jsonl"
LOG_A="$OUTPUT_DIR/${RUN_NAME}_teacher_a.log"
LOG_B="$OUTPUT_DIR/${RUN_NAME}_teacher_b.log"

echo "Starting TinyPython $RUN_NAME generation..."
echo "teacher_a=$TEACHER_A gpu=$GPU_A output=$OUTPUT_A"
echo "teacher_b=$TEACHER_B gpu=$GPU_B output=$OUTPUT_B"

CUDA_VISIBLE_DEVICES="$GPU_A" uv run python -m llm_lite.scripts.generate_tinypython \
  --model "$TEACHER_A" \
  --num-seeds "$SEED_COUNT" \
  --samples-per-seed "$SAMPLES_PER_SEED" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT_A" \
  2>&1 | tee "$LOG_A" &
PID_A="$!"

CUDA_VISIBLE_DEVICES="$GPU_B" uv run python -m llm_lite.scripts.generate_tinypython \
  --model "$TEACHER_B" \
  --num-seeds "$SEED_COUNT" \
  --samples-per-seed "$SAMPLES_PER_SEED" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT_B" \
  2>&1 | tee "$LOG_B" &
PID_B="$!"

wait "$PID_A"
wait "$PID_B"

echo "Generation complete."
wc -l "$OUTPUT_A" "$OUTPUT_B" || true
ls -lh "$OUTPUT_DIR"/*.jsonl "$OUTPUT_DIR"/*.log 2>/dev/null || true

if [ "$RUN_FULL" != "1" ]; then
  cat <<'EOF'

Pilot run finished. Inspect:
  data/tinypython/pilot_teacher_a.jsonl
  data/tinypython/pilot_teacher_a.invalid.jsonl
  data/tinypython/pilot_teacher_b.jsonl
  data/tinypython/pilot_teacher_b.invalid.jsonl

To run the full 10,000-seed generation later, paste:

  cd LLM-Light
  RUN_FULL=1 bash generate_samples.sh

Or override teachers/output:

  cd LLM-Light
  RUN_FULL=1 \
  TEACHER_A=Qwen/Qwen2.5-Coder-7B-Instruct \
  TEACHER_B=microsoft/Phi-4-mini-instruct \
  OUTPUT_DIR=data/tinypython \
  bash generate_samples.sh

EOF
fi
