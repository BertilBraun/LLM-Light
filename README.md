# LLM-Light

LLM-Light is a small PyTorch-native, configuration-driven LLM training pipeline.
This repository currently implements milestones M0 and M1, with M2 local text
ingestion and TinyStories-ready preprocessing underway: typed experiment
configuration, local artifact manifests, ordered pipeline execution, inline text
and local text data, Unicode and line-ending normalization, exact
deduplication, split-sharded text artifacts, character and byte-level BPE
tokenizers, tiny dense GPT pretraining, checkpoint resume, greedy inference,
configurable naive or KV-cached greedy inference,
exact-reproduction/perplexity/generation evaluation, and optional training-time
evaluation.

Current verification stage order:

```text
raw_dataset
processed_dataset
tokenizer
packed_dataset
pretraining
evaluation
```

## Verification

Run the one-sentence verification pipeline:

```bash
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml
```

Review without execution:

```bash
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --dry-run
```

Force recomputation from a stage and all downstream stages:

```bash
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --force
python -m llm_lite.scripts.run_pipeline \
  --config configs/verify_one_sentence.yaml \
  --force pretraining
```

Run tests:

```bash
python -m pytest
```

Run the local text verification pipeline:

```bash
python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_local_text.yaml
```

Run the byte-level BPE verification pipeline:

```bash
python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_byte_bpe.yaml
```

Generate from a completed run:

```bash
python -m llm_lite.scripts.generate \
  --config configs/tinystories_hf_smoke.yaml \
  --prompt "Once upon a time" \
  --maximum-new-tokens 80
```

Generation defaults to KV-cached autoregressive decoding with `fp32` precision,
no quantization, greedy token selection, and 80 new tokens. Set
`inference.engine` to `naive` only when you want full-sequence reference
decoding.

```yaml
inference:
  decoding:
    strategy: sample
    temperature: 0.8
    top_k: 40
  maximum_new_tokens: 80
```

## Local Text Data

Use `dataset.type: local_text` to ingest UTF-8 text files from explicit paths,
glob patterns, or both. Files are resolved, deduplicated, and processed in
deterministic path order. Raw and processed text artifacts are written as
split folders containing compressed tar shards of plain `.txt` members:

```text
processed_dataset/
  corpus.json
  train/
    shard_000000.tar.gz
  validation/
    shard_000000.tar.gz
```

Inside each shard, samples are plain files named from their document id. There
is no per-document JSON metadata in the processed corpus. Split membership is
defined by the containing folder.

```yaml
dataset:
  type: local_text
  paths:
    - data/stories/example.txt
  glob_patterns:
    - data/stories/**/*.txt
```

Preprocessing remains ordered and streaming. `lower_case` is available but is
not recommended as a default for story or code corpora because it destroys
useful casing signal.

```yaml
preprocessing:
  output_shard_documents: 10000
  transforms:
    - type: normalize_unicode
      form: NFC
    - type: normalize_line_endings
    - type: min_length
      min_characters: 50
    - type: max_length
      max_characters: 4096
    - type: exact_deduplication
    - type: assign_split
      train_probability: 0.98
      validation_probability: 0.01
      test_probability: 0.01
```

Recommended preliminary TinyStories defaults are NFC Unicode normalization,
LF line endings, minimum length 50 characters, maximum length 4096 characters,
exact deduplication after normalization, no lowercasing, and deterministic
split metadata. `configs/tinystories_local_text.yaml` points at
`data/tinystories/**/*.txt` as a local prepared corpus path and uses byte-level
BPE; it does not download data.

Use `dataset.type: huggingface` to stream a text column from Hugging Face
Datasets and map source splits into pipeline split folders. Use
`skip_documents` when multiple output splits are carved from one source split:

```yaml
dataset:
  type: huggingface
  name: roneneldan/TinyStories
  text_column: text
  streaming: true
  splits:
    - source_split: train
      split: train
      max_documents: 1000
    - source_split: validation
      split: validation
      max_documents: 200
    - source_split: validation
      split: validation_small
      max_documents: 50
```

For datasets whose training text spans multiple string columns, configure
`text_template` with Python format fields instead of `text_column`. Hugging
Face configs must set exactly one of `text_column` or `text_template`:

```yaml
dataset:
  type: huggingface
  name: BertilBraun/TinyPython
  text_template: "{task_description}\n\n{code}\n"
  streaming: true
  splits:
    - source_split: train
      split: validation
      max_documents: 1000
    - source_split: train
      split: test
      skip_documents: 1000
      max_documents: 1000
    - source_split: train
      split: train
      skip_documents: 2000
```

When a source already assigns splits, do not configure `assign_split`; the
preprocessor rejects split reassignment.

Byte-level BPE starts from all 256 byte values plus configured special tokens,
then learns deterministic pair merges up to `tokenizer.vocabulary_size`.
Training uses a bounded deterministic prefix sample instead of materializing the
whole corpus; configure at least one of `max_training_documents` or
`max_training_bytes`. Encoding and decoding preserve Unicode, whitespace, tabs,
and newlines through UTF-8 bytes. For TinyStories-style smoke runs, start with:

```yaml
tokenizer:
  type: byte_bpe
  vocabulary_size: 512
  max_training_documents: 100
  max_training_bytes: 100000
  add_bos_token: true
  add_eos_token: true
  add_pad_token: true
```

Evaluation is configured by named optional blocks. Exact reproduction is kept
for tiny verification configs; story runs should generally use perplexity and
fixed-prompt generation:

```yaml
training:
  evaluation:
    interval_steps: 50
    evaluators:
      perplexity:
        split: validation_small
        maximum_documents: 50

evaluation:
  perplexity:
    split: validation
    maximum_documents: 200
  fixed_prompt_generation:
    prompts:
      - "Once upon a time"
    maximum_new_tokens: 120
```

Python completion evaluation measures raw Python continuation, not
instruction-to-code prompting. Each JSONL task supplies only Python source as
the model prompt plus expression checks that are counted after the generated
continuation parses:

```json
{"task_id":"reverse_string","prompt":"def reverse_string(text: str) -> str:\n","checks":["reverse_string('abc') == 'cba'","reverse_string('') == ''"]}
```

Configure shared stop sequences at the evaluator level:

```yaml
evaluation:
  python_completion:
    tasks_path: tests/fixtures/python_completion/tasks.jsonl
    maximum_tasks: 10
    maximum_new_tokens: 80
    execution_timeout_seconds: 2.0
    stop_sequences:
      - "\n\ndef "
      - "\nclass "
      - "\nif __name__"
```

Inspect training curves in TensorBoard:

```bash
python -m tensorboard.main --logdir runs/verify_one_sentence/artifacts/pretraining/tensorboard
```

Pipeline events are written to `runs/verify_one_sentence/pipeline.jsonl`. Training
metrics are written to `runs/verify_one_sentence/artifacts/pretraining/metrics.jsonl`
and mirrored to TensorBoard scalars.

Increasing `training.maximum_steps` on a compatible pretraining run resumes from the
latest checkpoint instead of restarting. Architecture, tokenizer, packed data, optimizer,
batch, precision, and gradient-clipping changes still invalidate pretraining.

Training dataloader behavior is configured under `training.dataloader`:

```yaml
dataloader:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  prefetch_factor: null
```

`persistent_workers` and `prefetch_factor` require `num_workers` greater than
zero.

## Packed Data Access

Packed training data is stored as uint16 shard files plus a small JSON index. The
default training dataset is a map-style `PackedSequenceDataset`, so PyTorch's
standard `DataLoader` can provide fully random shuffling and worker
parallelization. An explicit `IterablePackedSequenceDataset` is also available as
a lower-memory fallback: each epoch shuffles shard order, loads one shard at a
time, shuffles rows within that shard, and yields token rows as `torch.long`.
Multiworker iterable loading partitions shards by worker id, so workers do not
duplicate shard reads.

Compare access modes with:

```bash
python -m llm_lite.scripts.benchmark_packed_datasets
```

Temporary local benchmark on 2026-06-20:

```text
rows_per_dataset_pass: 20000
passes: 3
row_length: 128
batch_size: 64
shard_sequences: 512
num_workers: 0
map_random_seconds: 0.8451
map_random_vs_memory_ratio: 3.74x
iterable_sharded_seconds: 0.2694
iterable_sharded_vs_memory_ratio: 1.19x
in_memory_random_seconds: 0.2262
map_vs_iterable_ratio: 3.14x

num_workers: 2
map_random_seconds: 9.6713
map_random_vs_memory_ratio: 1.26x
iterable_sharded_seconds: 8.0780
iterable_sharded_vs_memory_ratio: 1.05x
in_memory_random_seconds: 7.6949
map_vs_iterable_ratio: 1.20x
```

## Achieved Results

Validated on 2026-06-20 with:

```bash
python -m pytest -q
python -m ruff check .
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --dry-run
python -m llm_lite.scripts.run_pipeline \
  --config configs/verify_one_sentence.yaml \
  --force pretraining
```

Results:

```text
26 passed
All checks passed!
pretraining final_step: 60
pretraining final_loss: 0.00016388329095207155
exact reproduction: passed
generated_text: "hello world\n"
tensorboard event files: present
pipeline event log: present
```

M2 local text validation should be run with:

```bash
python -m pytest -q
python -m ruff check .
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml
python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_local_text.yaml
python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_byte_bpe.yaml
```

Validated on 2026-06-23 with:

```text
python -m pytest -q
43 passed

python -m ruff check .
All checks passed!

python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml
pretraining complete at step 60, compatible skip

python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_local_text.yaml
raw_documents: 1
processed_documents: 1
rejected_documents: 0
total_characters: 12
total_bytes: 12
packed sequences: 1
exact reproduction: passed

python -m llm_lite.scripts.run_pipeline --config tests/configs/verify_byte_bpe.yaml
vocabulary_size: 260
merge_count: 1
training_documents: 1
training_bytes: 12
training_tokens: 11
packed sequences: 1
exact reproduction: passed
```
