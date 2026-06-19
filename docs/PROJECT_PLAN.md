# LLM-Lite Project Plan

## 1. Project Thesis

LLM-Lite is a PyTorch-native, configuration-driven, artifact-based system for training, evaluating, post-training, and serving small autoregressive language models.

The project applies the complete LLM lifecycle at a scale that is practical on limited compute:

* data ingestion and preprocessing
* tokenizer training
* tokenization and sequence packing
* model construction
* autoregressive pretraining
* checkpointing and resumption
* evaluation and experiment comparison
* post-training
* inference with KV caching and quantization
* parallel preprocessing
* distributed training
* sharded checkpointing
* simulated multi-node process topology

The models and datasets are intentionally small enough to train within a reasonable budget. The software architecture should nevertheless reflect the structure of larger LLM training systems.

The final showcase should demonstrate:

> I can implement and reason about the complete LLM training lifecycle. The experiments run at small scale, but the pipeline, artifacts, distributed process model, checkpointing, observability, and evaluation design correspond to the same concerns found in larger training systems.

---

## 2. End Goals

The project should produce two main model applications.

### 2.1 Story-generation model

Train a small decoder-only language model on a constrained natural-language dataset such as TinyStories.

The model should demonstrate:

* grammatical text generation
* short-range semantic consistency
* simple narrative continuation
* measurable improvement throughout training
* comparison across architecture and training variants

An initial target is approximately five million parameters, subject to adjustment based on tokenizer vocabulary size and model dimensions.

### 2.2 Python code-completion model

Train a small decoder-only language model entirely on Python source code.

The model should operate as a prefix-completion model rather than an instruction-following assistant.

It should demonstrate:

* syntactically plausible Python continuation
* indentation-aware generation
* common local code patterns
* function-body completion
* KV-cached low-latency inference
* automated syntax and execution-based evaluation

An initial target is approximately five to fifteen million parameters.

### 2.3 Systems showcase

The project should evolve from a single-process verification run into a small but complete training system supporting:

* reproducible experiments
* typed configuration
* automatic artifact reuse
* automatic training resumption
* frequent intermediate checkpoints
* TensorBoard and structured logging
* multiple preprocessing workers
* multi-GPU execution
* distributed data and model-state handling
* sharded checkpointing
* logical node and process topology
* dense and mixture-of-experts models
* post-training from generated feedback
* optimized inference

---

## 3. Core Technology

The implementation is based on:

* Python
* PyTorch
* Pydantic configuration models
* TensorBoard
* JSONL metrics and manifests
* local filesystem artifact storage
* `torch.distributed`
* `torchrun`
* PyTorch distributed checkpointing where appropriate

The repository's coding standards document is authoritative for implementation style, naming, typing, testing, documentation, and module organization.

---

## 4. Repository Structure

```text
llm-lite/
  README.md
  pyproject.toml

  configs/
    verify_one_sentence.yaml
    tinystories_dense_5m.yaml
    python_dense_10m.yaml
    tinystories_distributed.yaml
    python_distributed.yaml
    tinystories_moe.yaml

  docs/
    PROJECT_PLAN.md
    CODING_STANDARDS.md

  llm_lite/
    config/
      models.py
      loading.py
      validation.py

    pipeline/
      runner.py
      stage.py
      context.py
      artifact.py
      registry.py
      hashing.py

    data/
      document.py
      sources.py
      transforms.py
      splitting.py
      tokenization.py
      packing.py
      sharding.py
      datasets.py

    tokenizer/
      protocol.py
      character.py
      byte_bpe.py

    model/
      protocol.py
      output.py
      gpt.py
      attention.py
      normalization.py
      position.py
      feed_forward.py
      moe.py
      routing.py

    training/
      trainer.py
      objectives.py
      optimizer.py
      schedules.py
      checkpoint.py
      logging.py
      distributed.py
      topology.py

    post_training/
      datasets.py
      supervised.py
      preference.py
      rejection_sampling.py
      execution_feedback.py
      reinforcement.py

    evaluation/
      evaluator.py
      perplexity.py
      generation.py
      exact_reproduction.py
      python_syntax.py
      python_execution.py
      comparison.py
      reporting.py

    inference/
      requests.py
      engine.py
      naive.py
      kv_cache.py
      sampling.py
      quantization.py
      server.py

    utilities/
      paths.py
      random.py
      device.py
      serialization.py
      logging.py

  scripts/
    run_pipeline.py
    preprocess.py
    train.py
    evaluate.py
    generate.py
    serve.py

  tests/
    config/
    pipeline/
    data/
    tokenizer/
    model/
    training/
    evaluation/
    inference/
```

The exact module decomposition may evolve during implementation, while the stage boundaries and artifact contracts remain stable.

---

## 5. Experiment Configuration

Each experiment is described by one validated Pydantic configuration.

The configuration defines:

* dataset source
* preprocessing transforms
* tokenizer parameters
* packing parameters
* model architecture
* training parameters
* post-training method
* evaluation methods
* inference behavior
* distributed topology

Execution controls such as forcing recomputation are command-line arguments rather than experiment parameters.

Example:

```yaml
experiment:
  name: tinystories_dense_5m
  seed: 1337
  output_dir: runs/tinystories_dense_5m

dataset:
  type: tinystories
  max_documents: null

preprocessing:
  transforms:
    - type: normalize_unicode
    - type: normalize_line_endings
    - type: min_length
      min_characters: 50
    - type: max_length
      max_characters: 4096
    - type: exact_deduplication

tokenizer:
  type: byte_bpe
  vocabulary_size: 8192
  special_tokens:
    bos: "<bos>"
    eos: "<eos>"
    pad: "<pad>"

packing:
  context_length: 512
  add_bos: true
  add_eos: true
  pack_documents: true
  output_shard_size_tokens: 10000000

model:
  type: dense_gpt
  dimension: 192
  layers: 6
  attention_heads: 6
  feed_forward_dimension: 768
  normalization: rms_norm
  position_encoding: rope
  feed_forward: swiglu
  dropout: 0.1
  tie_embeddings: true

training:
  objective: causal_language_modeling
  maximum_steps: 50000
  batch_size_tokens: 65536
  optimizer:
    type: adamw
    learning_rate: 0.0006
    weight_decay: 0.1
  schedule:
    type: cosine
    warmup_steps: 1000
  precision: bf16
  gradient_clip_norm: 1.0
  checkpoint_interval_steps: 1000
  validation_interval_steps: 500
  sample_interval_steps: 1000

post_training:
  type: none

evaluation:
  evaluators:
    - type: perplexity
    - type: fixed_prompt_generation
  prompts:
    - "Once upon a time"
    - "The little dog was afraid because"
    - "Lily wanted to help her friend"

inference:
  engine: kv_cache
  precision: bf16
  quantization: none
  maximum_new_tokens: 128
  temperature: 0.8
  top_k: 50

distributed:
  enabled: false
```

---

## 6. Default Pipeline Behavior

The pipeline operates conservatively by default.

Running:

```bash
python -m llm_lite.scripts.run_pipeline \
  --config configs/tinystories_dense_5m.yaml
```

should:

1. inspect all expected artifacts
2. validate their manifests and dependency hashes
3. skip compatible completed work
4. resume compatible incomplete training
5. execute missing or invalidated work
6. preserve all valid upstream artifacts

The pipeline should print a review before execution:

```text
Raw dataset          compatible, skip
Preprocessing        compatible, skip
Tokenizer            compatible, skip
Tokenization         compatible, skip
Packing              compatible, skip
Pretraining          incomplete at step 24,000, resume
Post-training        disabled
Evaluation           waiting for pretraining
Inference export     waiting for checkpoint
```

A dry-run mode should display this plan without executing it:

```bash
python -m llm_lite.scripts.run_pipeline \
  --config configs/tinystories_dense_5m.yaml \
  --dry-run
```

Explicit recomputation:

```bash
--force tokenizer
--force pretraining
```

`--force <stage>` recomputes that stage and every downstream stage.

---

## 7. Pipeline Stages

The initial pipeline is an ordered sequence of stages:

```text
Dataset acquisition
→ preprocessing
→ dataset splitting
→ tokenizer training
→ tokenization
→ sequence packing and sharding
→ model pretraining
→ optional post-training data generation
→ optional post-training
→ evaluation
→ inference artifact creation
```

Each stage:

* receives validated configuration
* consumes one or more upstream artifacts
* produces a versioned artifact
* records parent artifact identifiers
* records its effective configuration hash
* records status and metrics
* can determine whether existing output remains compatible

The ordered pipeline is sufficient because the dependencies are explicit and stable.

---

## 8. Artifact Registry

The artifact registry is a local filesystem registry with structured manifests.

Example run:

```text
runs/tinystories_dense_5m/
  resolved_config.yaml

  artifacts/
    raw_dataset/
      manifest.json
      data/

    processed_dataset/
      manifest.json
      statistics.json
      train/
      validation/
      test/

    tokenizer/
      manifest.json
      tokenizer.json
      vocabulary.json
      merges.json

    tokenized_dataset/
      manifest.json
      train/
      validation/
      test/

    packed_dataset/
      manifest.json
      train/
        shard_000000.bin
        shard_000001.bin
      validation/
        shard_000000.bin
      index.json

    pretraining/
      manifest.json
      metrics.jsonl
      tensorboard/
      samples/
      checkpoints/

    post_training_data/
      manifest.json
      data/

    post_training/
      manifest.json
      metrics.jsonl
      checkpoints/

    evaluation/
      manifest.json
      report.json
      samples/

    inference/
      manifest.json
      model_config.json
      generation_config.json
```

### 8.1 Artifact manifest

A manifest contains:

```json
{
  "artifact_type": "tokenizer",
  "artifact_version": 1,
  "status": "complete",
  "created_at": "2026-06-19T12:00:00Z",
  "configuration_hash": "sha256:...",
  "implementation_version": "git:...",
  "parents": {
    "processed_dataset": "sha256:..."
  },
  "files": {
    "tokenizer": "tokenizer.json",
    "vocabulary": "vocabulary.json",
    "merges": "merges.json"
  },
  "metrics": {
    "vocabulary_size": 8192,
    "merge_count": 7933,
    "training_documents": 1000000,
    "bytes_per_token": 3.74
  }
}
```

### 8.2 Artifact status

Useful states include:

```text
pending
running
incomplete
complete
failed
```

A stage should write completion atomically so that interrupted work is never mistaken for a valid completed artifact.

---

## 9. Compatibility and Invalidation

A stage artifact is compatible when:

* its configuration hash matches
* all required parent artifact hashes match
* its files are present and valid
* its implementation compatibility requirements are satisfied

Typical dependency behavior:

```text
Dataset change
  → recompute all downstream stages

Preprocessing change
  → keep raw dataset
  → recompute processed data and downstream stages

Tokenizer change
  → keep processed data
  → recompute tokenizer, tokenization, packing, training, and downstream stages

Packing change
  → keep tokenizer and tokenized documents
  → recompute packed data and downstream stages

Model architecture change
  → keep all data artifacts
  → restart pretraining and downstream stages

Maximum training steps increase
  → resume compatible training checkpoint

Evaluation change
  → retain model checkpoints
  → rerun evaluation

Inference change
  → retain model checkpoints
  → recreate inference artifacts
```

Compatibility rules should be explicit and tested.

---

## 10. Data Model

The fundamental preprocessing object is a document:

```python
@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, object]
```

Metadata can include:

* source
* path
* repository
* language
* license
* original byte size
* document hash
* dataset-specific attributes

Documents should preserve enough provenance to support:

* debugging
* deduplication
* split construction
* leakage checks
* dataset statistics
* later filtering experiments

---

## 11. Dataset Sources

Initial dataset sources:

### 11.1 Inline text source

Used for verification and integration tests.

```yaml
dataset:
  type: inline_text
  documents:
    - "hello world\n"
```

### 11.2 TinyStories source

Used for natural-language pretraining.

It should expose the dataset as a stream or iterable of `Document` objects rather than coupling later stages to the source format.

### 11.3 Python files source

Used for code completion.

It should support:

* local Python file collections
* dataset archives or prepared Python corpora
* repository and path metadata
* file-based split grouping

The Python train, validation, and test split should be performed at file or repository level rather than by splitting token sequences from the same file.

---

## 12. Preprocessing

Preprocessing is represented as an ordered collection of document transforms.

```python
class DocumentTransform(Protocol):
    def apply(
        self,
        documents: Iterable[Document],
    ) -> Iterable[Document]:
        ...
```

Initial transforms:

* Unicode normalization
* line-ending normalization
* minimum-length filtering
* maximum-length filtering
* exact deduplication
* deterministic split assignment

Python-specific transforms:

* Python file selection
* generated-file detection
* file-size filtering
* optional AST parsing
* optional vendored-code filtering
* preservation of whitespace and indentation

Each transform should record statistics:

```text
input documents
output documents
rejected documents
processed bytes
elapsed time
documents per second
bytes per second
transform-specific counters
```

### 12.1 Parallel preprocessing

The local implementation should support processing independent input shards with multiple worker processes.

Conceptual execution:

```text
coordinator
  ├── worker 0 → input shards 0, 4, 8, ...
  ├── worker 1 → input shards 1, 5, 9, ...
  ├── worker 2 → input shards 2, 6, 10, ...
  └── worker 3 → input shards 3, 7, 11, ...
```

Workers produce:

* output shards
* shard manifests
* local transform statistics

The coordinator produces:

* global manifest
* aggregated statistics
* deterministic shard index

The same structure should work with two local workers or hundreds of externally scheduled workers.

---

## 13. Tokenizer

The project includes two tokenizer implementations.

### 13.1 Character tokenizer

Used only for the one-sentence verification run.

Its purpose is to remove tokenizer complexity while validating the full model-training pipeline.

### 13.2 Byte-level BPE tokenizer

The byte-level BPE tokenizer is the primary tokenizer.

It should implement:

* initial byte vocabulary
* special-token allocation
* pair-frequency counting
* iterative merge training
* deterministic merge ordering
* encoding
* decoding
* serialization
* loading
* vocabulary statistics

Protocol:

```python
class Tokenizer(Protocol):
    @property
    def vocabulary_size(self) -> int:
        ...

    @property
    def bos_token_id(self) -> int:
        ...

    @property
    def eos_token_id(self) -> int:
        ...

    @property
    def pad_token_id(self) -> int | None:
        ...

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...

    def save(self, directory: Path) -> None:
        ...
```

The tokenizer is trained independently for each experiment using the configured tokenizer-training dataset.

A TinyStories tokenizer and a Python tokenizer use the same implementation but learn different vocabularies and merge rules.

The central correctness invariant is:

```python
tokenizer.decode(tokenizer.encode(text)) == text
```

This is especially important for:

* spaces
* tabs
* newlines
* indentation
* Unicode
* string escapes
* Python syntax

Tokenizer artifact metrics should include:

* vocabulary size
* merge count
* training bytes
* training document count
* bytes per token
* tokens per document
* special-token mapping

---

## 14. Tokenization, Packing, and Sharding

### 14.1 Tokenization

The tokenization stage converts processed documents into tokenized documents while preserving:

* document identifier
* metadata
* token sequence
* document boundaries

### 14.2 Packing

The packing stage creates fixed-length autoregressive training sequences.

Configuration includes:

* context length
* BOS insertion
* EOS insertion
* cross-document packing
* padding behavior
* output shard size

For TinyStories, multiple stories can be packed into one context with EOS boundaries.

For Python, file boundaries should initially be preserved. Cross-file packing can later be evaluated experimentally.

### 14.3 Sharding

Packed data is written as independently readable shards with a global index.

The packed dataset manifest records:

* total tokens
* total sequences
* shard count
* context length
* document-boundary behavior
* tokenizer artifact identifier
* split-level statistics

Sharding is required even for small runs because it enables:

* streaming
* distributed rank assignment
* restart-safe iteration
* parallel preprocessing
* deterministic data loading
* scale-independent dataset structure

---

## 15. Model Interface

Models are selected from a registry.

```python
MODEL_REGISTRY = {
    "dense_gpt": DenseGPT,
    "moe_gpt": MoEGPT,
}
```

Model output:

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    cache: object | None = None
    auxiliary: dict[str, torch.Tensor | object] = field(
        default_factory=dict
    )
```

The `auxiliary` mapping allows model-specific outputs such as routing statistics without adding MoE-specific fields to every model.

---

## 16. Dense GPT Model

The first production model is a decoder-only Transformer.

Initial architecture:

* token embeddings
* causal self-attention
* RMSNorm
* RoPE
* SwiGLU feed-forward layers
* residual connections
* tied input and output embeddings
* optional dropout
* PyTorch scaled dot-product attention where appropriate

Example TinyStories model:

```yaml
model:
  type: dense_gpt
  dimension: 192
  layers: 6
  attention_heads: 6
  feed_forward_dimension: 768
  normalization: rms_norm
  position_encoding: rope
  feed_forward: swiglu
  dropout: 0.1
  tie_embeddings: true
```

Example Python model:

```yaml
model:
  type: dense_gpt
  dimension: 256
  layers: 6
  attention_heads: 8
  feed_forward_dimension: 1024
  normalization: rms_norm
  position_encoding: rope
  feed_forward: swiglu
  dropout: 0.0
  tie_embeddings: true
```

Parameter counts should be computed and logged automatically from the resolved configuration.

---

## 17. Mixture-of-Experts Model

The MoE model reuses the GPT backbone and replaces selected dense feed-forward blocks with routed expert layers.

Initial design:

* configurable MoE layers
* configurable expert count
* top-k routing
* expert-specific SwiGLU feed-forward networks
* routing probabilities
* expert-capacity handling
* token dispatch and combination
* load-balancing loss
* router z-loss
* expert utilization metrics

Example:

```yaml
model:
  type: moe_gpt
  dimension: 256
  layers: 8
  attention_heads: 8
  feed_forward_dimension: 1024
  normalization: rms_norm
  position_encoding: rope
  feed_forward: swiglu

  moe:
    layers: [2, 4, 6]
    expert_count: 8
    top_k: 2
    expert_feed_forward_dimension: 1024
    capacity_factor: 1.25
    load_balance_loss_weight: 0.01
    router_z_loss_weight: 0.001
```

The MoE implementation should record:

* total parameters
* active parameters per token
* token count per expert
* routing entropy
* expert utilization distribution
* dropped-token count where applicable
* load-balancing loss
* router z-loss

The central experiment is a comparison between:

* dense model
* MoE model with similar active parameters
* MoE model with greater total capacity

---

## 18. Training Architecture

Training is built from:

* a centralized trainer
* a model
* a dataset
* a training objective
* optimizer and schedule configuration
* logging callbacks
* checkpoint management
* optional distributed execution

The trainer owns:

* device placement
* precision and autocast
* optimizer creation
* scheduler creation
* gradient accumulation
* gradient clipping
* forward/backward execution
* optimizer stepping
* checkpoint creation
* checkpoint loading
* training resumption
* validation scheduling
* sample-generation scheduling
* TensorBoard logging
* JSONL metric logging
* distributed metric aggregation

The objective owns:

* expected batch structure
* model invocation
* loss calculation
* objective-specific metrics

Protocol:

```python
class TrainingObjective(Protocol):
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: object,
        step: int,
    ) -> "LossOutput":
        ...
```

```python
@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float]
```

---

## 19. Training Objectives

### 19.1 Causal language modeling

Used for TinyStories and Python pretraining.

```text
inputs  = tokens[:, :-1]
targets = tokens[:, 1:]
loss    = cross_entropy(model(inputs), targets)
```

### 19.2 MoE causal language modeling

Uses the same language-modeling loss plus configured router-related loss terms exposed by the model.

```text
total loss
  = language-modeling loss
  + load-balance weight × load-balance loss
  + router-z weight × router-z loss
```

### 19.3 Masked causal language modeling

Used for supervised post-training where only completion tokens should contribute to the loss.

The dataset provides:

* input token IDs
* target token IDs
* loss mask

### 19.4 Direct preference optimization

Uses preference batches containing:

* prompt
* chosen completion
* rejected completion
* reference-model log probabilities or access to a frozen reference model

The generic trainer can still manage optimization and checkpointing while the DPO objective calculates the preference loss.

### 19.5 Reinforcement learning objective

Active reinforcement learning is a later training mode involving generation, evaluation, reward construction, and policy updates.

It may use a dedicated trainer because its execution loop differs materially from ordinary dataset-based optimization.

---

## 20. Training Checkpoints

Checkpointing is a first-class part of the initial trainer.

A checkpoint records:

* model state
* optimizer state
* scheduler state
* gradient-scaler state where applicable
* training step
* consumed token count
* epoch or data cursor
* random-number-generator states
* resolved model configuration
* resolved training configuration
* dataset artifact identifiers
* tokenizer artifact identifier
* distributed topology
* world size
* metric summary

Training writes regular checkpoint artifacts:

```text
pretraining/
  checkpoints/
    step_00001000/
    step_00002000/
    step_00003000/
    latest.json
```

The training manifest is updated throughout execution:

```json
{
  "artifact_type": "pretraining",
  "status": "incomplete",
  "current_step": 24000,
  "target_step": 50000,
  "consumed_tokens": 1572864000,
  "latest_checkpoint": "checkpoints/step_00024000",
  "configuration_hash": "sha256:..."
}
```

When a compatible run is restarted, it resumes automatically from the latest valid checkpoint.

Increasing the maximum number of steps should continue training from the previous final checkpoint.

---

## 21. Logging and Observability

The system writes:

* readable console logs
* TensorBoard events
* structured JSONL metrics
* generated samples
* artifact manifests
* evaluation reports

### 21.1 Training metrics

Initial metrics:

```text
train/loss
train/perplexity
train/learning_rate
train/gradient_norm
train/tokens_per_second
train/sequences_per_second
train/optimizer_step_time
train/data_wait_time
train/gpu_memory_allocated
train/gpu_memory_reserved
validation/loss
validation/perplexity
```

### 21.2 Distributed metrics

Later metrics:

```text
distributed/world_size
distributed/global_tokens_per_second
distributed/rank_tokens_per_second
distributed/communication_time
distributed/data_wait_time
distributed/checkpoint_time
distributed/load_imbalance
```

### 21.3 Sample logging

At configured intervals, the current checkpoint generates completions for fixed prompts.

Samples are written to:

* console
* TensorBoard text
* Markdown or JSON files associated with the training step

This provides direct qualitative visibility without turning generated-text quality into a training gate.

---

## 22. Evaluation

Evaluation is separate from unit and integration testing.

Correctness tests validate implementation behavior.

Evaluation measures trained-model behavior and system performance.

### 22.1 Shared model evaluation

* validation loss
* perplexity
* fixed-prompt generations
* generation speed
* checkpoint size
* parameter count
* active parameter count where applicable

### 22.2 Story evaluation

* validation perplexity
* grammaticality
* local coherence
* prompt adherence
* repetition
* story completion
* pairwise model comparison

A stronger external model can later perform blinded pairwise comparisons between outputs from two checkpoints or experiment variants.

Pairwise comparison is preferred over an uncalibrated absolute score.

### 22.3 Python evaluation

* validation perplexity
* AST parse success rate
* indentation and delimiter failures
* completion exact match on simple deterministic tasks
* unit-test pass rate
* pass@k
* execution result
* generation latency

Example prompts:

```python
def add(a, b):
    return
```

```python
def factorial(n):
    if n == 0:
        return 1
    return
```

```python
class Counter:
    def __init__(self):
        self.count =
```

Generated code must be evaluated in an isolated subprocess with time and resource limits.

### 22.4 Systems evaluation

Every systems milestone should be evaluated against the previous implementation.

Examples:

* single-worker vs multi-worker preprocessing
* single-GPU vs multi-GPU throughput
* DDP vs sharded execution
* full checkpoint vs sharded checkpoint duration
* naive decoding vs KV-cached decoding
* full-precision vs quantized inference
* dense vs MoE active-compute efficiency

Metrics include:

* wall-clock duration
* throughput
* peak memory
* scaling efficiency
* startup cost
* checkpoint cost
* storage size
* model-quality change

---

## 23. Inference

Inference consumes:

* tokenizer artifact
* model configuration
* checkpoint
* generation configuration

Generation API:

```python
@dataclass
class GenerateRequest:
    prompt: str
    maximum_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    stop_sequences: tuple[str, ...]
```

### 23.1 Naive inference

The initial engine reruns the model on the complete sequence for each generated token.

It provides the simplest correctness reference.

### 23.2 KV-cached inference

The optimized engine performs:

1. prompt prefill
2. key/value cache construction
3. one-token decoding using cached attention state
4. cache extension at every generation step

A correctness test must verify that naive and cached inference produce equivalent logits within numerical tolerance.

### 23.3 Quantized inference

Later inference should support configured weight quantization, initially weight-only int8 where practical.

Evaluation should compare:

* model size
* memory usage
* latency
* throughput
* perplexity or task quality

### 23.4 Serving

A local endpoint can expose:

* text generation
* Python prefix completion
* generation parameters
* health and model metadata

The initial deliverable can be a CLI. A lightweight HTTP service follows after the inference implementation is stable.

---

## 24. Post-Training

Post-training begins from a completed base-model checkpoint.

The post-training pipeline is:

```text
base checkpoint
→ post-training prompt/data source
→ candidate generation or dataset preparation
→ feedback/reward construction
→ post-training optimization
→ post-trained checkpoint
→ base-vs-post-trained evaluation
```

### 24.1 Supervised fine-tuning

The first reusable post-training mode.

The dataset contains prompt/completion pairs and a loss mask identifying supervised completion tokens.

### 24.2 Rejection-sampling distillation

For each prompt:

1. generate multiple candidates
2. evaluate each candidate
3. retain the strongest candidate
4. fine-tune the model on retained examples

Story candidates can be evaluated by pairwise model judgment.

Python candidates can be evaluated by parsing and execution.

### 24.3 Direct preference optimization

Preference records contain:

```text
prompt
chosen completion
rejected completion
```

Examples:

* coherent story preferred over incoherent story
* passing Python completion preferred over failing completion
* syntactically valid completion preferred over invalid completion

### 24.4 Execution-feedback training

For Python:

1. generate multiple solutions
2. parse generated code
3. run unit tests in isolation
4. record syntax and test rewards
5. construct supervised or preference datasets
6. continue training
7. compare against the base model

### 24.5 Active reinforcement learning

A later active loop should support:

```text
prompt sampling
→ policy generation
→ external evaluation
→ scalar or structured reward
→ policy update
→ repeated evaluation
```

Potential Python reward components:

* parse success
* compilation success
* unit tests passed
* runtime constraints
* output correctness

Potential story reward components:

* coherence
* prompt consistency
* grammaticality
* repetition penalties
* external preference score

---

## 25. Distributed Preprocessing

The preprocessing implementation should begin with a local worker pool but use scale-independent concepts.

Each worker:

* receives independent input shards
* has a worker-local temporary directory
* writes immutable output shards
* writes a local manifest
* records transform metrics
* never edits another worker's output

The coordinator:

* assigns work
* detects completed shards
* resumes missing shards
* aggregates manifests
* creates the global artifact manifest

Example layout:

```text
processed_dataset/
  manifest.json
  shard_index.json

  shards/
    shard_000000.jsonl
    shard_000001.jsonl
    shard_000002.jsonl

  shard_manifests/
    shard_000000.json
    shard_000001.json
    shard_000002.json
```

This same contract can later be executed by local processes, a job scheduler, or a distributed data-processing framework.

---

## 26. Distributed Training

Distributed training should preserve the same trainer, objective, model, and artifact interfaces used by single-process training.

### 26.1 Local multi-process execution

Use `torchrun` with one process per GPU:

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m llm_lite.scripts.run_pipeline \
  --config configs/tinystories_distributed.yaml
```

The distributed runtime manages:

* global rank
* local rank
* world size
* device assignment
* process groups
* rank-aware logging
* data shard assignment
* collective metric aggregation
* checkpoint coordination

### 26.2 Data parallelism

The first distributed training implementation uses data parallelism:

* one model replica per rank
* different training batches per rank
* gradient synchronization
* globally consistent optimizer steps
* rank-zero global reporting
* rank-local diagnostics

This establishes the distributed execution semantics and enables throughput scaling experiments.

### 26.3 Fully sharded training

The next implementation shards:

* parameters
* gradients
* optimizer states

This enables:

* lower memory per rank
* larger model configurations
* distributed checkpoint state
* comparison of replicated and sharded execution

### 26.4 Multi-dimensional parallelism

The topology configuration should represent:

* data-parallel groups
* tensor-parallel groups
* pipeline-parallel groups
* context-parallel groups
* expert-parallel groups

The first versions may only execute data parallelism and fully sharded data parallelism. The topology representation should allow later model-parallel implementations without changing experiment configuration structure.

Example:

```yaml
distributed:
  enabled: true
  world_size: 8

  simulated_nodes:
    count: 4
    processes_per_node: 2

  parallelism:
    data: 4
    tensor: 2
    pipeline: 1
    context: 1
    expert: 1
```

Logical groups:

```text
tensor-parallel:
  [0, 1]
  [2, 3]
  [4, 5]
  [6, 7]

data-parallel:
  [0, 2, 4, 6]
  [1, 3, 5, 7]
```

---

## 27. Simulated Multi-Node Topology

One physical machine with several GPUs can be divided into logical nodes.

Example:

```text
8 GPUs
4 logical nodes
2 GPUs per logical node

node 0: ranks 0, 1
node 1: ranks 2, 3
node 2: ranks 4, 5
node 3: ranks 6, 7
```

Directory layout:

```text
runs/experiment/
  artifacts/

  work/
    node_000/
      shared/
      rank_000/
      rank_001/

    node_001/
      shared/
      rank_002/
      rank_003/

    node_002/
      shared/
      rank_004/
      rank_005/

    node_003/
      shared/
      rank_006/
      rank_007/
```

This provides explicit concepts for:

* rank-local temporary state
* node-local caches
* global artifact storage
* coordinator behavior
* process-group membership
* distributed barriers
* shard ownership
* sharded checkpoint completion

The topology runs over local communication but uses the same logical organization expected from a physical multi-node execution.

---

## 28. Distributed Checkpointing

Distributed training should write sharded checkpoints.

Example:

```text
checkpoints/
  step_00010000/
    manifest.json
    metadata.json
    rank_00000/
    rank_00001/
    rank_00002/
    rank_00003/
```

The checkpoint manifest records:

* training step
* consumed tokens
* model architecture
* tokenizer artifact
* dataset artifact
* world size
* process topology
* parallelism dimensions
* shard ownership
* expected checkpoint files
* completion state

Checkpoint completion should use a coordinated protocol:

1. all ranks write temporary shard state
2. each rank marks its shard complete
3. ranks synchronize
4. coordinator validates every shard
5. coordinator writes the global manifest atomically
6. `latest` is updated only after successful validation

---

## 29. Scaling to Large Training Systems

The small-scale implementation should preserve concepts that remain useful at much larger scale:

* immutable data shards
* shard-level manifests
* independent preprocessing tasks
* explicit data mixtures
* deterministic sample accounting
* explicit process topology
* rank-local and node-local state
* distributed model state
* sharded checkpointing
* restart-safe progress tracking
* consumed-token accounting
* structured observability
* pluggable training objectives
* explicit model and inference configurations

A larger physical deployment would replace or extend:

* the local process launcher
* the local filesystem backend
* node discovery and rendezvous
* cluster scheduling
* distributed storage
* high-performance model-parallel implementations
* production monitoring and failure recovery

The experiment and artifact model should remain recognizable across those environments.

---

## 30. Testing Strategy

### 30.1 Tokenizer tests

* character roundtrip
* byte-level BPE roundtrip
* special-token handling
* deterministic tokenizer training
* serialization roundtrip
* Python whitespace preservation
* Unicode preservation

### 30.2 Model tests

* parameter shape checks
* forward output shapes
* causal masking
* deterministic inference
* loss shifting
* gradient propagation
* tied embeddings
* RoPE behavior
* MoE routing and combination
* expert utilization accounting

### 30.3 Training tests

* one-batch overfit
* checkpoint save/load
* resumed training equivalence
* maximum-step extension
* optimizer-state restoration
* scheduler restoration
* deterministic restart where feasible
* gradient accumulation equivalence

### 30.4 Pipeline tests

* compatible stage skipping
* parent-change invalidation
* force behavior
* interrupted artifact handling
* atomic artifact completion
* dry-run review output

### 30.5 Distributed tests

* rank assignment
* dataset-shard uniqueness
* global sample accounting
* metric reduction
* distributed checkpoint save/load
* logical node directory isolation
* multi-process resume

### 30.6 Inference tests

* naive generation
* greedy deterministic generation
* KV-cache equivalence
* batched generation
* quantized model loading
* stop-sequence handling

---

## 31. Milestones

### M0 — Project foundation

Deliverables:

* repository skeleton
* coding standards integration
* Pydantic configuration hierarchy
* artifact and manifest types
* artifact registry
* ordered pipeline runner
* default review/resume behavior
* console and JSONL logging

### M1 — One-sentence verification

Pipeline:

```text
inline sentence
→ character tokenizer
→ tiny GPT
→ causal language-model training
→ checkpoint
→ reload
→ greedy generation
→ exact reproduction
```

Suggested dataset:

```text
hello world\n
```

Suggested model:

* one Transformer layer
* dimension approximately 16
* one attention head
* approximately ten thousand parameters

Success conditions:

* training loss approaches zero
* checkpoint reload works
* resumed training works
* greedy completion reproduces the learned sentence
* artifact skipping works on rerun

### M2 — Byte-level BPE and TinyStories subset

Deliverables:

* byte-level BPE implementation
* TinyStories source
* document preprocessing
* tokenization
* packing and sharding
* small dense GPT
* validation perplexity
* fixed-prompt samples

### M3 — Full TinyStories model

Deliverables:

* approximately five-million-parameter model
* longer training run
* frequent checkpoints
* TensorBoard curves
* generated samples throughout training
* naive inference
* KV-cached inference
* quality and latency evaluation

### M4 — Python completion model

Deliverables:

* Python source
* Python preprocessing
* Python-trained byte-level BPE tokenizer
* five-to-fifteen-million-parameter model
* Python prefix-completion prompts
* AST evaluation
* execution-based mini benchmark
* KV-cached completion inference

### M5 — Parallel preprocessing

Deliverables:

* input sharding
* multiple local workers
* worker-local work directories
* shard manifests
* restart of incomplete shards
* global manifest aggregation
* single-worker versus multi-worker timing report

### M6 — Multi-GPU data-parallel training

Deliverables:

* `torchrun` integration
* one process per GPU
* distributed data assignment
* rank-aware logging
* synchronized optimization
* distributed metric aggregation
* throughput and scaling report

### M7 — Fully sharded training and checkpoints

Deliverables:

* sharded model-state execution
* reduced per-rank memory usage
* sharded optimizer state
* distributed checkpoint format
* distributed resume
* replicated-versus-sharded comparison

### M8 — Simulated multi-node topology

Deliverables:

* logical node configuration
* rank-local work directories
* node-local shared directories
* process-group construction
* topology-aware logs
* coordinated artifact finalization
* logical multi-node checkpoint manifests

### M9 — Post-training

Deliverables:

* masked supervised fine-tuning
* candidate generation
* rejection-sampling distillation
* execution-filtered Python fine-tuning
* direct preference optimization
* base-versus-post-trained reports

### M10 — Mixture-of-experts model

Deliverables:

* top-k router
* expert feed-forward layers
* router auxiliary losses
* routing observability
* dense-versus-MoE comparison
* active-versus-total parameter reporting
* optional expert-parallel experiment

### M11 — Optimized inference

Deliverables:

* KV-cache benchmarks
* weight quantization
* batched generation
* local inference service
* memory/latency/quality comparison

### M12 — Active reinforcement learning

Deliverables:

* online candidate generation
* reward evaluators
* execution reward for Python
* preference or quality reward for stories
* policy-update loop
* comparison against SFT and DPO approaches

---

## 32. Initial Implementation Sequence

The first implementation sequence is:

1. create Pydantic configuration models
2. implement artifact manifests and registry
3. implement ordered pipeline execution
4. implement inline text dataset
5. implement character tokenizer
6. implement basic tokenized dataset and packing
7. implement tiny dense GPT
8. implement causal language-modeling objective
9. implement trainer
10. implement checkpoint and automatic resume
11. implement console, JSONL, and TensorBoard logging
12. implement naive greedy inference
13. implement exact-reproduction evaluator
14. complete the one-sentence verification pipeline
15. implement byte-level BPE
16. add TinyStories preprocessing
17. train on a TinyStories subset
18. scale to the full small TinyStories model
19. implement KV-cached inference
20. implement the Python dataset pipeline
21. train the Python completion model
22. add parallel preprocessing
23. add distributed data-parallel training
24. add fully sharded training and distributed checkpoints
25. add simulated multi-node topology
26. implement post-training
27. implement the MoE model
28. implement optimized inference
29. implement active reinforcement learning experiments

---

## 33. Final Demonstration

The finished project should provide several reproducible demonstrations.

### Demonstration A — Complete minimal pipeline

```text
one sentence
→ tokenizer
→ model
→ training
→ checkpoint
→ evaluation
→ generation
```

### Demonstration B — Natural-language pretraining

```text
TinyStories
→ byte-level BPE
→ packed shards
→ dense GPT
→ coherent story generation
→ perplexity and sample evaluation
```

### Demonstration C — Code completion

```text
Python corpus
→ Python-specific preprocessing
→ Python-trained byte-level BPE
→ GPT pretraining
→ prefix completion
→ syntax and execution evaluation
```

### Demonstration D — Distributed systems

```text
parallel preprocessing
→ multi-GPU training
→ sharded state
→ simulated logical nodes
→ distributed checkpoints
→ scaling measurements
```

### Demonstration E — Model improvement

```text
base model
→ generated candidates
→ external evaluation
→ SFT / DPO / execution feedback / RL
→ measurable task improvement
```

### Demonstration F — Architecture comparison

```text
dense GPT
vs
MoE GPT

comparison:
- total parameters
- active parameters
- training throughput
- memory usage
- validation loss
- downstream quality
- routing behavior
```

The resulting repository should make the entire lifecycle inspectable from raw documents to a running inference endpoint, while remaining small enough that every component can be understood, tested, and executed independently.
