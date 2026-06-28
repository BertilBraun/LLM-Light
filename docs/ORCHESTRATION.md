# Orchestration

This document describes the local artifact-store executor used by `run_plan`
and its per-stage subprocess entry point, `run_job`.

The goal is a practical local orchestration layer for rented one-node compute
runs with two to four GPUs. It improves artifact reuse, multi-config runs, live
TensorBoard inspection, and asynchronous evaluation without reimplementing a
cluster scheduler such as Slurm, Kubernetes, or Ray.

## Goals

- Keep YAML experiment configs as the primary user interface.
- Treat every pipeline stage as a job with dependencies, resources, and a
  content fingerprint.
- Deduplicate artifacts across runs.
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
- No backward compatibility with the removed `run_pipeline` entry point.
- No complex workflow language for arbitrary Python pipeline programs.
- No hard resource isolation. CPU and GPU requests are a scheduling contract,
  not a sandbox.
- No compatibility guarantee for stage names, config fields, fingerprints, or
  artifact layouts while this rewrite is landing. The implementation may rename
  stages and reshape configs when that produces a simpler runtime.
- No stage-specific payload validation in the first version. A complete
  manifest is the completion contract.

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

`run_plan` should resolve the submitted YAML config once at startup and write
that snapshot to the run directory before launching jobs:

```text
runs/<experiment>/resolved_config.json
```

All subprocess jobs should use this resolved config snapshot, not the original
YAML path. Edits to the external config file after planning must not change an
active run.

The current stage names remain a good starting point:

```text
raw_dataset
processed_dataset
tokenizer
packed_dataset
pretraining
post_training
evaluation
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
- produced files
- scalar metrics
- creation and completion timestamps

The fingerprint is the artifact identity. The manifest is mutable execution
state for that identity. Timestamps, status, subprocess logs, TensorBoard event
file names, and other run-time details must not contribute to the fingerprint.

The completion rule is intentionally simple: jobs write the complete manifest
last. Once `manifest.json` exists with `status: complete` and the expected
fingerprint, dependent jobs may trust that the stage succeeded and that its
payload exists. The first implementation does not need a broader atomic
artifact-publish protocol.

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

For cache hits, the local executor should copy the existing artifact
TensorBoard logs into the run view:

```text
runs/<experiment>/tensorboard/<stage>/
```

Run-view TensorBoard files are disposable copies. The artifact TensorBoard files
are authoritative. If a run-view copy is missing or stale, it can be recreated
from the artifact store.

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

## Fingerprints, Status, and Cache Hits

A stage fingerprint should use the same idea as the current per-stage
configuration hashes and parent artifact hashes. The new orchestration should
standardize that identity and use it as the artifact-store key. Parent identity
must use the parent's semantic fingerprint, not a hash of the parent manifest
file.

A fingerprint should include:

- stage name
- relevant configuration subset
- parent artifact fingerprints
- stage contract version

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

The stage contract version is a small manually updated version for cases where
the meaning of a stage changes without a YAML config change. It does not need
to be tied to the package version or Git commit in the first implementation.

Manifest status is mutable execution state:

```text
pending
running
complete
failed
interrupted
```

Only `complete` can be reused. A job is a cache hit only when
`artifact_store/<stage>/<fingerprint>/manifest.json` exists, has status
`complete`, and has the expected fingerprint. The executor should not re-run
cache hits. It should record the cache hit in the run event log and copy
TensorBoard logs into the run view.

If the directory exists but the manifest is incomplete, the executor should
inspect the stage lock. If the lock is fresh, another subprocess is producing
the artifact and dependent jobs wait. If the lock is stale, the artifact is
interrupted and can be resumed or retried according to stage policy.

Failed, interrupted, or stale-running artifacts are not cache hits. They are
recovery inputs.

### Training Fingerprints

Pretraining and post-training fingerprints should include the training
configuration that changes the training trajectory, including optimizer
settings, precision, distributed configuration, batch size, objective settings,
model configuration, seed when it affects initialization or data order,
checkpoint interval when it changes observable artifacts, and requested maximum
steps. Changing learning rate or similar hyperparameters creates a new artifact
rather than mutating the old one.

Training may resume only when the recomputed fingerprint matches the incomplete
artifact. This supports ordinary interruption recovery without supporting
manual in-place hyperparameter changes.

Training longer should be an explicit config-defined continuation, not an
in-place change to the old artifact. A longer training run can use a previous
checkpoint artifact as its initialization input and produce a new pretraining
artifact with its own fingerprint.

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

For a multi-config run:

```bash
python -m llm_lite.scripts.run_plan \
  --config configs/experiment_a.yaml configs/experiment_b.yaml \
  --max-parallel-jobs 2 \
  --gpus 0,1,2,3
```

The executor loop is:

1. Resolve all configs and write each run's `resolved_config.json`.
2. Compute stage fingerprints using existing stage hash logic and parent
   fingerprints.
3. Build the artifact job DAG.
4. Mark complete artifacts as cache hits.
5. Select ready jobs whose dependencies are complete.
6. Skip jobs blocked by a fresh lock for the same artifact fingerprint.
7. Start all ready jobs whose resource requests fit currently available
   resources, up to `--max-parallel-jobs`.
8. Run each job as a subprocess.
9. Stream subprocess output to the console and a job log.
10. Poll running training jobs for new checkpoint manifests and enqueue
    checkpoint evaluation jobs when configured.
11. Update run manifests and TensorBoard views when jobs complete.
12. Repeat until all jobs are complete or a job fails.

This does not require an always-on scheduler. If the executor process dies,
rerun the same command. Existing manifests and locks determine what is complete,
what is cacheable, and what needs recovery.

The executor is still a single local parent process, but it may supervise
multiple subprocess jobs concurrently. For example, if training is running and a
checkpoint evaluation job becomes ready, the evaluation job can run in parallel
when GPU and CPU resource limits allow it.

## Subprocess Jobs

Each job should run through a narrow stage-job entry point:

```bash
python -m llm_lite.scripts.run_job \
  --resolved-config runs/python_moe_small/resolved_config.json \
  --stage packed_dataset \
  --fingerprint sha256:...
```

For multi-GPU training, the executor launches `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m llm_lite.scripts.run_job \
  --resolved-config runs/python_moe_small/resolved_config.json \
  --stage pretraining \
  --fingerprint sha256:...
```

The executor does not need to understand the internal training loop. It only
needs to know the stage, artifact fingerprint, dependencies, resources, and
checkpoint event contract.

The subprocess job contract is:

- read one resolved config snapshot
- receive the stage name and expected fingerprint
- write payload files under the artifact directory for that fingerprint
- write TensorBoard events to the artifact view and run view when applicable
- write the complete artifact manifest last
- exit nonzero on failure

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

Lock acquisition must be atomic. The first implementation can use exclusive
file creation or atomic directory creation for the lock path.

The lock records:

- artifact fingerprint
- process id
- hostname
- command
- start time
- heartbeat time

While a subprocess is running, the parent executor updates the heartbeat. The
child job does not need to write heartbeat records. If another job needs the
same artifact, it waits for the lock to clear and then rechecks the manifest.

On restart, a lock is fresh only when the recorded process is still alive on the
same host and the heartbeat is recent. If the process is gone or the heartbeat
is stale, the lock is stale.

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

Training stages should resume from their latest valid checkpoint only when the
requested fingerprint is unchanged. Changed training hyperparameters create a
new training artifact.

### Job Failure

If a subprocess exits nonzero, the executor should:

- mark the job failed in the run event log
- leave the artifact manifest incomplete or failed
- stop scheduling dependent jobs

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

Checkpoint events should be written only after the checkpoint payload and
checkpoint manifest are complete. The event should include the producing
training fingerprint, checkpoint step, checkpoint manifest path, and checkpoint
kind such as full or sharded.

The evaluation job needs an explicit checkpoint target:

```bash
python -m llm_lite.scripts.run_job \
  --resolved-config runs/python_moe_small/resolved_config.json \
  --stage evaluation \
  --fingerprint sha256:... \
  --checkpoint-artifact sha256:... \
  --checkpoint-step 1000
```

The exact CLI can change, but the important contract is that evaluation can be
run for a specific checkpoint artifact rather than only "latest".

Checkpoint evaluation fingerprints should include the producing training
fingerprint, checkpoint step, evaluator configuration, inference configuration,
and an evaluation contract version. Final evaluation and checkpoint evaluation
can share the same stage implementation while still producing distinct
artifacts.

This lets training continue while evaluation jobs run in parallel when resource
limits allow.

The same model can later support post-training comparisons:

```text
pretraining/base
  -> evaluation/base
  -> post_training/dpo
    -> evaluation/dpo
```

`evaluation/base` can run while `post_training/dpo` is already running.

## Migration Plan

The intended migration is a clean rewrite of orchestration, not strict backward
compatibility with the old runner internals.

The old `run_pipeline` entry point has been removed. `run_plan` is the normal
executor, and `run_job` is the narrow per-stage subprocess entry point.

The final command for the common rented-node case should remain simple:

```bash
python -m llm_lite.scripts.run_plan --config configs/python_moe_small.yaml
```

This command plans the run, executes local subprocess jobs, materializes
run-centric TensorBoard logs, and exits when the submitted run or config set is
complete.
