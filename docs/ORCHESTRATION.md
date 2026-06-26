# Orchestration Plan

This document sketches a future replacement for the current sequential
`run_pipeline` implementation. It is a design target, not implemented behavior.

The goal is a practical local orchestration layer for rented one-node compute
runs with two to four GPUs. It should improve artifact reuse, sweeps, live
TensorBoard inspection, and asynchronous evaluation without reimplementing a
cluster scheduler such as Slurm, Kubernetes, or Ray.

## Goals

- Keep YAML experiment configs as the primary user interface.
- Treat every pipeline stage as a job with dependencies, resources, and a
  content fingerprint.
- Deduplicate artifacts across runs and parameter sweeps.
- Keep `runs/<experiment>/tensorboard` as the main inspection surface.
- Support live TensorBoard inspection while long jobs are still running.
- Support asynchronous evaluation jobs triggered by checkpoints.
- Execute jobs as local subprocesses, including `torchrun` for multi-GPU
  training.
- Recover by rerunning the same plan command after interruption.
- Keep the first implementation single-node and local.

## Non-Goals

- No persistent scheduler service in the first version.
- No worker registration, REST API, web dashboard, or service database.
- No multi-node training requirement in the first version.
- No Kubernetes, Slurm, Ray, or external scheduling dependency.
- No backward compatibility requirement for the current `run_pipeline`
  internals.
- No complex workflow language for arbitrary Python pipeline programs.
- No hard resource isolation. CPU and GPU requests are a scheduling contract,
  not a sandbox.

## Core Model

The future system should separate four concepts:

```text
Experiment config -> resolved run -> artifact jobs -> run view
```

- An experiment config says what the user wants.
- A resolved run maps that config to stage fingerprints and dependencies.
- A job produces one artifact for one stage and one fingerprint.
- A run view records which artifacts were used and exposes TensorBoard logs for
  that experiment.

The current stage names remain a good starting point:

```text
raw_dataset
processed_dataset
tokenizer
packed_dataset
pretraining
post_training
evaluation
export
```

Evaluation can remain one stage in the first version. Individual evaluators can
still run inside that stage without becoming separate executor-visible jobs.

## Canonical Artifact Store

Artifacts should always be written to a global artifact store. The run directory
should not own large datasets, tokenizer files, packed shards, checkpoints, or
evaluation reports.

```text
artifact_store/
  raw_dataset/<fingerprint>/
  processed_dataset/<fingerprint>/
  tokenizer/<fingerprint>/
  packed_dataset/<fingerprint>/
  pretraining/<fingerprint>/
  post_training/<fingerprint>/
  evaluation/<fingerprint>/
```

Each artifact directory contains:

```text
manifest.json
payload files
tensorboard/
```

The manifest should record:

- stage name
- fingerprint
- status
- configuration hash
- parent artifact fingerprints
- implementation version
- produced files
- scalar metrics
- creation and completion timestamps

## Run View

The run directory remains the main human-facing view:

```text
runs/<experiment>/
  resolved_config.json
  run_manifest.json
  pipeline.jsonl
  tensorboard/
```

`run_manifest.json` should map stages to artifact fingerprints:

```json
{
  "experiment": "python_moe_small_dim192",
  "artifacts": {
    "raw_dataset": "sha256:...",
    "processed_dataset": "sha256:...",
    "tokenizer": "sha256:...",
    "packed_dataset": "sha256:...",
    "pretraining": "sha256:...",
    "evaluation": "sha256:..."
  }
}
```

The run directory can be small. It exists to answer:

- What config was submitted?
- Which artifacts did this run use?
- What happened in what order?
- What should I open in TensorBoard?

## TensorBoard

TensorBoard should be run-centric, even when artifacts are shared.

All jobs should write TensorBoard events to both:

```text
artifact_store/<stage>/<fingerprint>/tensorboard/
runs/<experiment>/tensorboard/<stage>/
```

This should be implemented as a wrapper around the TensorBoard writer rather
than duplicated stage logic.

For cache hits, the local executor should copy or materialize the existing
artifact TensorBoard logs into the run view:

```text
runs/<experiment>/tensorboard/<stage>/
```

Tags should be stage-relative and reusable across runs:

```text
pipeline/packed_dataset/non_pad_tokens
pretraining/loss
pretraining/tokens_per_second
evaluation/python_completion/pass_rate
evaluation/python_completion/by_family/<family>
moe/router_entropy/layer_00
moe/expert_usage_std/layer_00
```

Tags should not include the producing run name, because cached artifacts may be
viewed from many run directories.

## Fingerprints and Cache Hits

A stage fingerprint should use the same idea as the current per-stage
configuration hashes and parent artifact hashes. The new orchestration should
standardize that identity and use it as the artifact-store key.

A fingerprint should include:

- stage name
- relevant configuration subset
- parent artifact fingerprints
- implementation version
- schema version

Examples:

```text
raw_dataset = dataset config + ingestion implementation
processed_dataset = raw_dataset fingerprint + preprocessing config
tokenizer = processed_dataset fingerprint + tokenizer config
packed_dataset = processed_dataset fingerprint + tokenizer fingerprint + packing config
pretraining = packed_dataset fingerprint + tokenizer fingerprint + model/training/distributed config
post_training = pretraining fingerprint + post-training config
evaluation = checkpoint or model artifact fingerprint + evaluator/inference config
```

If `artifact_store/<stage>/<fingerprint>/manifest.json` exists and is complete,
the job is a cache hit. The executor should not re-run it. It should record the
cache hit in the run event log and materialize TensorBoard logs into the run
view.

If the directory exists but the manifest is incomplete, the executor should
inspect the stage lock. If the lock is fresh, another subprocess is producing
the artifact and dependent jobs wait. If the lock is stale, the artifact is
interrupted and can be resumed or retried according to stage policy.

## Local Plan Executor

The first implementation should be a single local process that owns config
resolution, job DAG construction, small-scale scheduling, subprocess launches,
and recovery. There is no separate planner service or database.

Useful command shape:

```bash
python -m llm_lite.scripts.run_plan \
  --config configs/python_moe_small.yaml \
  --max-parallel-jobs 2 \
  --gpus 0,1
```

For a sweep:

```bash
python -m llm_lite.scripts.run_plan \
  --config configs/generated/small_models/*.yaml \
  --max-parallel-jobs 2 \
  --gpus 0,1,2,3
```

The executor loop is:

1. Resolve all configs.
2. Compute stage fingerprints using existing stage hash logic and parent
   fingerprints.
3. Build the artifact job DAG.
4. Mark complete artifacts as cache hits.
5. Select ready jobs whose dependencies are complete.
6. Skip jobs blocked by a fresh lock for the same artifact fingerprint.
7. Start jobs when resource requests fit currently available resources.
8. Run each job as a subprocess.
9. Stream subprocess output to the console and a job log.
10. Update run manifests and TensorBoard views when jobs complete.
11. Repeat until all jobs are complete or a job fails.

This does not require an always-on scheduler. If the executor process dies,
rerun the same command. Existing manifests and locks determine what is complete,
what is cacheable, and what needs recovery.

## Subprocess Jobs

Each job should run through a narrow stage-job entry point:

```bash
python -m llm_lite.scripts.run_job \
  --config configs/python_moe_small.yaml \
  --stage packed_dataset \
  --fingerprint sha256:...
```

For multi-GPU training, the executor launches `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m llm_lite.scripts.run_job \
  --config configs/python_moe_small.yaml \
  --stage pretraining \
  --fingerprint sha256:...
```

The executor does not need to understand the internal training loop. It only
needs to know the stage, artifact fingerprint, dependencies, and resources.

## Resource Model

Resources should be simple and explicit:

```yaml
resources:
  cpu_workers: 8
  gpu_count: 2
  exclusive_gpus: true
```

For evaluation:

```yaml
resources:
  cpu_workers: 2
  gpu_count: 1
  exclusive_gpus: false
```

The first version can derive defaults from existing config:

- `raw_dataset`, `processed_dataset`, `tokenizer`, and `packed_dataset` use CPU.
- `pretraining` uses the configured distributed world size as `gpu_count`.
- `post_training` uses its configured training resources.
- `evaluation` uses one GPU by default when generation is required and CPU only
  for pure report aggregation.

The executor should set `CUDA_VISIBLE_DEVICES` for GPU jobs. It should not try
to enforce CPU or memory limits.

## Locks

Each artifact should have a local lock:

```text
artifact_store/<stage>/<fingerprint>/.lock
```

The lock records:

- process id
- hostname
- command
- start time
- heartbeat time

While a subprocess is running, the executor updates the heartbeat. If another
job needs the same artifact, it waits for the lock to clear and then rechecks
the manifest.

If the lock is stale, the executor can apply the stage recovery policy:

- Resume if the stage supports continuation.
- Retry after deleting incomplete payload files if restart is cheaper.
- Fail clearly if safe recovery is unknown.

This avoids duplicate work without a database or service scheduler.

## Robustness

### Executor Death

If the local executor dies, subprocesses may die with it or may continue
depending on the platform and launch mode. The supported recovery path is to
rerun the same `run_plan` command.

On restart, the executor should:

1. Recompute the plan.
2. Treat complete manifests as cache hits.
3. Detect stale locks.
4. Resume or retry incomplete artifacts according to stage policy.
5. Continue remaining jobs.

### Job Failure

If a subprocess exits nonzero, the executor should:

- mark the job failed in the run event log
- leave the artifact manifest incomplete or failed
- stop scheduling dependent jobs
- keep independent jobs configurable: either continue or stop the full plan

The first version can stop the full plan on the first job failure.

### Duplicate Work

If two submitted runs require the same missing artifact, only one subprocess
should acquire the lock. Other jobs requiring that fingerprint remain blocked.
When the producer completes, the blocked jobs become cache hits.

## Async Evaluation

The training subprocess cannot call back into the parent executor directly. It
should communicate checkpoint availability through the filesystem.

Pretraining and post-training jobs should write checkpoint manifests and a small
event record when a checkpoint is complete:

```text
artifact_store/pretraining/<fingerprint>/checkpoints/step_001000/manifest.json
artifact_store/pretraining/<fingerprint>/events/checkpoint_001000.json
```

The local executor polls the artifact directory while the training subprocess is
still running. When it sees a new complete checkpoint manifest or checkpoint
event, it can enqueue an evaluation job for that checkpoint.

The evaluation job needs an explicit checkpoint target:

```bash
python -m llm_lite.scripts.run_job \
  --config configs/python_moe_small.yaml \
  --stage evaluation \
  --fingerprint sha256:... \
  --checkpoint-artifact sha256:... \
  --checkpoint-step 1000
```

The exact CLI can change, but the important contract is that evaluation can be
run for a specific checkpoint artifact rather than only "latest".

This lets training continue while evaluation jobs run in parallel when resource
limits allow.

The same model supports post-training comparisons:

```text
pretraining/base
  -> evaluation/base
  -> post_training/dpo
    -> evaluation/dpo
```

`evaluation/base` can run while `post_training/dpo` is already running.

## Parameter Sweeps

Sweeps should generate ordinary config files. They do not need a special runtime
format.

Example workflow:

```bash
python -m llm_lite.scripts.generate_sweep_configs \
  --base-config configs/python_moe_full.yaml \
  --sweep configs/sweeps/small_models.yaml \
  --output-dir configs/generated/small_models

python -m llm_lite.scripts.run_plan \
  --config configs/generated/small_models/*.yaml \
  --max-parallel-jobs 2 \
  --gpus 0,1,2,3
```

A sweep definition can stay simple:

```yaml
name_template: "python_small_d{model.dimension}_l{model.layers}"
grid:
  model.dimension: [128, 192]
  model.layers: [6, 8]
  model.expert_feed_forward_dimension: [512, 768]
```

The generated configs should be normal experiment configs with unique
`experiment.name` and `experiment.output_dir` values. Artifact deduplication
comes from fingerprints, not from sweep-specific logic.

## Export

Export should resolve `run_manifest.json`, copy the referenced artifact store
entries, include the run TensorBoard view, and write a bundle manifest.

This means export no longer depends on large files living under
`runs/<experiment>/artifacts`.

## Migration Plan

The intended migration is a clean rewrite of orchestration, not strict backward
compatibility with the old runner internals.

Suggested phases:

1. Add this design document and keep the existing runner unchanged.
2. Add artifact-store path resolution and run-view types.
3. Add dual TensorBoard writer support.
4. Implement `run_job` for one stage and one artifact fingerprint.
5. Implement `run_plan` as a local executor with dependency scheduling.
6. Move cheap stages to artifact-store-first execution.
7. Move pretraining and evaluation to job execution.
8. Add async checkpoint evaluation.
9. Add sweep config generation.
10. Replace `run_pipeline` with `run_plan` for normal usage.

The final command for the common rented-node case should remain simple:

```bash
python -m llm_lite.scripts.run_plan --config configs/python_moe_small.yaml
```

This command plans the run, executes local subprocess jobs, materializes
run-centric TensorBoard logs, and exits when the submitted run or sweep is
complete.
