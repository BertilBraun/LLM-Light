# LLM-Light

LLM-Light is a PyTorch-native LLM training system built as a research project
with larger training-system concepts kept in view. The actual model runs are
budget-sized, but the repository covers synthetic corpus generation,
dataset ingestion, parallel preprocessing, tokenizer training, packed training
shards, GPT/MoE pretraining, distributed data parallelism, checkpoint resume,
evaluation, inference, throughput logging, TensorBoard observability, and
artifact export.

This is a systems and experiment-validation project. The current models are
small research artifacts, not broadly capable assistants or production code
models.

## Documentation Map

- [docs/TRAINING.md](docs/TRAINING.md): setup, tests, smoke runs, Python MoE
  reproduction commands, generation, evaluation, and bundle export.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): implemented pipeline stages,
  artifacts, configuration surface, datasets, models, evaluation, distributed
  behavior, and known implementation limits.
- [docs/RESULTS.md](docs/RESULTS.md): validated experiment summaries, including
  the TinyPython MoE run.
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md): future work and open project
  gaps only.
- [docs/ORCHESTRATION.md](docs/ORCHESTRATION.md): artifact-store execution,
  local subprocess jobs, and async evaluation architecture.
- [docs/EXTENDING.md](docs/EXTENDING.md): short guides for adding evaluators,
  dataset sources, and model architectures.
- [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md): local coding and testing
  standards.

## Implemented Surface

The ordered pipeline currently supports:

```text
raw_dataset
processed_dataset
tokenizer
packed_dataset
pretraining
post_training
evaluation
```

Implemented capabilities include:

- Inline, local text, and Hugging Face dataset ingestion.
- Synthetic TinyPython corpus generation with local vLLM teacher models,
  semantic task seeds, parsing, Python validation, resumable JSONL output, and
  invalid-sample tracking.
- Ordered preprocessing with Unicode normalization, line-ending normalization,
  length filters, exact deduplication, split assignment, and Python function
  extraction.
- Parallel preprocessing over workers and split-sharded text artifacts.
- Character tokenizer, Python byte-level BPE, and Rust-backed byte-level BPE.
- Packed fixed-length autoregressive datasets backed by shard files and indexes.
- Dense GPT and top-k MoE GPT models.
- Causal language-model pretraining with checkpoint resume.
- Single-process training and working distributed data-parallel training through
  `torchrun`.
- Full checkpoints and rank-local sharded checkpoints for distributed runs.
- Greedy and sampled generation through naive or KV-cache inference engines.
- Exact reproduction, perplexity, fixed-prompt generation, and Python completion
  evaluation with parse, execution, and check-pass metrics.
- Training-time evaluation, throughput metrics, structured artifacts, and
  TensorBoard traces for inspecting runs.
- Direct preference optimization utilities and generated Python DPO data flow.
- Compact run-bundle export for moving completed experiments without committing
  large run directories.

Fully sharded data parallel training and model-parallel variants were part of
the intended scale-out direction, but they are not yet validated project results.

## Quick Start

Install dependencies with `uv`:

```bash
uv sync --extra dev
```

Run tests:

```bash
uv run python -m pytest
```

Run the one-sentence smoke pipeline:

```bash
uv run python -m llm_lite.scripts.run_plan \
  --config configs/verify_one_sentence.yaml
```

Generate from a completed run:

```bash
uv run python -m llm_lite.scripts.generate \
  --config configs/verify_one_sentence.yaml \
  --prompt "" \
  --maximum-new-tokens 20
```

For a fresh GPU instance, use the training helper to generate and run the
TinyPython model sweep pilot:

```bash
bash scripts/train.sh
```

To run the pilot and then the full sweep, reusing completed pilot artifacts:

```bash
SWEEP_MODE=pilot_then_full bash scripts/train.sh
```

See [docs/TRAINING.md](docs/TRAINING.md) for full commands, including the
TinyPython MoE pipeline.

## Results

Two meaningful validation tracks are documented in
[docs/RESULTS.md](docs/RESULTS.md).

### TinyStories MoE Validation

`configs/tinystories_moe_full.yaml` validated the story-data path. It streamed
TinyStories from Hugging Face, trained a 4096-token Rust byte-level BPE
tokenizer, packed 256-token story sequences, and trained a small top-1 MoE GPT
with distributed data parallelism.

Recorded local validation notes:

- Two RTX 4090 GPUs.
- 6-layer top-1 MoE decoder, 4 experts.
- Approximately 9.0M total parameters and 3.7M active parameters.
- 50,000 training steps.
- Final evaluation perplexity over 500 validation documents: 6.8403.
- Final evaluation loss: 1.9228.
- Throughput: approximately 487k global tokens/s.

This run was quick to get going and produced recognizable TinyStories-style
continuations, which made it useful as a pipeline validation before the Python
experiment.

### TinyPython MoE Validation

The latest run uses `configs/python_moe_full.yaml` and the public
[`BertilBraun/TinyPython`](https://huggingface.co/datasets/BertilBraun/TinyPython)
dataset. TinyPython was generated by this project with
`llm_lite.scripts.generate_tinypython` and then uploaded to Hugging Face. It
contains about 2.19M synthetic task-to-function Python records.

- Model: top-1 MoE GPT, dimension 320, 6 layers, 8 heads, 4 experts, expert FFN
  dimension 1280.
- Tokenizer: 6000-token Rust byte-level BPE.
- Parameters: 24,428,160 total, 9,653,760 active per token.
- Training: distributed data parallel, world size 2, batch size 512 packed
  sequences per rank.
- Schedule: 3,750 steps at LR 0.001, then until 7,500 at LR 0.0005, then until
  10,000 at LR 0.00025.
- Final validation: perplexity 1.5516, loss 0.4393.
- Python completion evaluation: 174 tasks, 172 parsed completions, 157 executed
  completions, 786 / 1012 checks passed, pass rate 0.7767.

The compact artifact bundle for this run is published as the
[Python-Run-V1 GitHub release](https://github.com/BertilBraun/LLM-Light/releases/tag/Python-Run-V1).
It contains the resolved config, tokenizer files, latest checkpoint, training
metrics, performance logs, TensorBoard logs, evaluation reports, and generated
examples.

Qualitatively, the model learned many simple list, dictionary, numeric, and
format patterns. It remains weak on exact semantic instruction following,
literal replacement, case sensitivity, and confusing nearby task families.

Example generated completion:

```text
Prompt:
Given a list of integers, write a function that returns the sum of all odd numbers in the list.
```

```python
def sum_odd_numbers(numbers: list[int]) -> int:
    total = 0
    for number in numbers:
        if number % 2 != 0:
            total += number
    return total
```

## Artifacts

Run directories under `runs/` contain datasets, tokenizer files, packed shards,
checkpoints, metrics, TensorBoard logs, and evaluation reports. Large artifacts
and checkpoints should stay out of source control. The repository may include
small smoke-run artifacts, but the full TinyPython MoE artifacts are distributed
through the release bundle rather than committed under `runs/python_moe_full`.

Export a compact run bundle instead:

```bash
uv run python -m llm_lite.scripts.run_plan \
  --config configs/python_moe_full.yaml \
  --from export \
  --to export
```
