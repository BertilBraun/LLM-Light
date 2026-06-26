# Project Plan

This page records future work only. Implemented and validated behavior is
documented in [ARCHITECTURE.md](ARCHITECTURE.md) and [RESULTS.md](RESULTS.md).

## Project Direction

LLM-Light should remain a small, inspectable implementation of a complete LLM
training lifecycle:

- Synthetic dataset generation.
- Dataset ingestion and preprocessing.
- Tokenizer training.
- Packed autoregressive datasets.
- Dense and MoE decoder models.
- Pretraining, checkpointing, resume, and evaluation.
- Distributed execution.
- Inference.
- Post-training experiments.
- Run artifact export.

The project should avoid overclaiming model capability. The strongest current
result is a constrained TinyPython MoE baseline and a validation of the training
system, not a general-purpose programming assistant.

The project has grown beyond a tiny training script: it now includes corpus
generation, multiple evaluation paths, artifact-based pipeline design,
distributed data-parallel training, throughput logging, TensorBoard
observability, and run-bundle export.

A future orchestration rewrite is sketched in
[ORCHESTRATION.md](ORCHESTRATION.md). The intent is to keep YAML configs and
run-centric TensorBoard inspection while moving artifacts into a canonical
deduplicated artifact store managed by a local plan executor.

## Open Documentation Gaps

- Add a concise artifact schema reference if manifest formats stabilize.
- Add a short guide for adding a new evaluator.
- Add a short guide for adding a new dataset source.
- Add a result template for future experiment reports.

## Experiment Gaps

### Dense-Versus-MoE Comparison

Create a dense GPT baseline against the TinyPython MoE result with a comparable
active parameter count or compute budget.

Compare:

- Total parameters.
- Active parameters.
- Tokens per second.
- Memory usage.
- Validation loss and perplexity.
- Python completion parse and pass rates.
- Qualitative failure modes.

### Targeted TinyPython Data Expansion

The TinyPython MoE run remains brittle on exact semantics. Add or generate more
examples for:

- Literal replacement.
- Case-sensitive string operations.
- String reversal.
- Dictionary aggregation.
- Nearby task-family disambiguation.
- Signature-prefix body completion.

Evaluate whether targeted data improves completion pass rate more than simply
continuing the same pretraining run.

### Heldout Evaluation Sets

Maintain separate heldout suites for:

- In-distribution TinyPython tasks.
- Unseen combinations of known operations.
- Alternate natural-language wording.
- Harder out-of-distribution tasks.
- Signature-prefix completion tasks.

Report parse rate, execution rate, passed checks, total checks, pass rate, and
representative failures for every serious run.

## Systems Gaps

### Orchestration

The current runner is sequential and run-directory-local. Future work should
replace it with the local executor design in [ORCHESTRATION.md](ORCHESTRATION.md):

- Canonical artifact store keyed by stage fingerprints.
- Small run directories focused on TensorBoard and run manifests.
- Subprocess jobs that can run CPU stages or launch `torchrun`.
- Resource-aware local scheduling for one compute node.
- Artifact locks to prevent duplicate work across submitted runs.
- Async checkpoint evaluation.
- Parameter sweeps through generated normal configs.

### Distributed Scaling

Distributed data parallelism is implemented and has been used for the
TinyStories and TinyPython MoE runs. Rank-local sharded checkpoints are
implemented for distributed runs. Future work:

- Fully sharded data parallel training.
- More detailed distributed checkpoint validation.
- Throughput comparison across world sizes.
- Rank-local and node-local artifact diagnostics.

### Model Parallelism

The config model represents tensor, pipeline, context, and expert parallelism,
but those dimensions are not implemented. Future work:

- Implement expert parallelism for MoE layers.
- Add topology tests for nontrivial process groups.
- Report expert-parallel memory and throughput tradeoffs.

### Router Observability

Improve MoE reporting with:

- Per-layer expert utilization.
- Router entropy.
- Auxiliary loss trends.
- Expert load imbalance over time.
- Routing summaries in evaluation reports.

## Post-Training Gaps

The repository contains DPO utilities, including generated Python preference
data flow, but the latest TinyPython MoE result is a base model. Future work:

- Generate preference pairs from Python completion candidates.
- Filter pairs through syntax and execution checks.
- Run direct preference optimization.
- Compare base versus post-trained checkpoints on the same heldout suite.
- Track whether DPO improves semantic correctness or only output formatting.

## Inference Gaps

Current generation supports naive and KV-cache engines with greedy and sampled
decoding. Future work:

- Batch-generation benchmarks.
- Quantization quality and latency comparison.
- More robust stop-sequence handling for code tasks.
- Optional local inference service if interactive use becomes important.

## Reporting Checklist for Future Runs

Every future meaningful experiment should record:

- Git commit and config path.
- Hardware and distributed world size.
- Dataset source, preprocessing, tokenizer, and packed-token counts.
- Model architecture, total parameters, and active parameters.
- Training phases, learning rates, batch size, precision, and wall time.
- Checkpoint and evaluation intervals.
- Final validation loss and perplexity.
- Task-specific evaluation metrics.
- TensorBoard plots or exported scalar summaries.
- Representative successes and failures.
- Exported compact run bundle path.
- Clear statement of what the run does and does not prove.
