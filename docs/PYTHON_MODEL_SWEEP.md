# Python Model Sweep Plan

This page records the planned TinyPython model sweep. Results belong in
[PYTHON_MODEL_SWEEP_RESULTS.md](PYTHON_MODEL_SWEEP_RESULTS.md) after the rented
GPU run completes.

## Goal

Validate the artifact-store `run_plan` workflow on a rented multi-GPU node and
collect a compact dense-versus-MoE comparison under one consistent TinyPython
training setup.

The sweep should verify:

- Non-overlapping GPU allocation for parallel jobs.
- Reuse of shared raw, processed, tokenizer, and packed artifacts.
- Training-time checkpoint evaluation for distributed DDP runs.
- Checkpoint retention through `training.max_checkpoints`.
- Final export bundles through the `export` stage.
- TensorBoard and manifest output sufficient for later analysis.

## Commands

Generate the four-config pilot:

```powershell
uv run python .\scripts\generate_python_model_sweep.py
```

Run the pilot on a four-GPU node with two concurrent world-size-2 jobs:

```powershell
uv run python -m llm_lite.scripts.run_plan `
  --config configs\generated\python_model_sweep\*.yaml `
  --max-parallel-jobs 2 `
  --gpus 0,1,2,3
```

If the pilot completes and GPU allocation/artifact reuse look correct, generate
the full sweep:

```powershell
uv run python .\scripts\generate_python_model_sweep.py --mode full
```

Then rerun the same plan command. The pilot configs are included in the full
set, so completed artifacts should be reused rather than recomputed.

## Pilot Set

The pilot contains exactly four small models, each below one million active
parameters:

| Config | Purpose | Active parameters |
| --- | --- | ---: |
| `python_moe_small_deep_plain.yaml` | Small deep plain MoE baseline | 995,984 |
| `python_moe_small_wide_plain.yaml` | Small wide plain MoE ablation | 993,824 |
| `python_dense_small_deep_plain.yaml` | Small deep dense baseline | 994,576 |
| `python_dense_small_wide_plain.yaml` | Small wide dense ablation | 992,992 |

## Full Set

The full set adds five configs:

| Config | Purpose | Active parameters |
| --- | --- | ---: |
| `python_moe_small_deep_fim.yaml` | FIM augmentation ablation | 995,984 |
| `python_moe_small_deep_linear_warmup_decay.yaml` | Linear warmup/decay LR schedule | 995,984 |
| `python_moe_small_deep_cosine_warmup_decay.yaml` | Cosine warmup/decay LR schedule | 995,984 |
| `python_modern_moe_small_deep_plain.yaml` | RoPE/RMSNorm/SwiGLU MoE ablation | 924,440 |
| `python_dense_active_9m6.yaml` | Dense comparison to existing 9.65M-active MoE run | 9,646,080 |

## Shared Settings

- Dataset: `BertilBraun/TinyPython`, streamed from Hugging Face.
- Tokenizer: 6000-token Rust byte-level BPE.
- Context length: 256.
- Batch size: 512 packed sequences per rank.
- Distributed world size: 2.
- Precision: BF16.
- Checkpoint interval: 500 steps.
- `training.max_checkpoints`: 2.
- Export bundle path: `runs/<experiment>/export/bundle.zip`.

## Interpretation

This sweep is a systems and modeling comparison, not a claim that the resulting
models are general code assistants. The strongest conclusions should be about
relative behavior under the same corpus, tokenizer, context length, batch size,
and training budget.
