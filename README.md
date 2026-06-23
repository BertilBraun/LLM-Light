# LLM-Light

LLM-Light is a small PyTorch-native, configuration-driven LLM training pipeline.
This repository currently implements milestones M0 and M1, with M2 local text
ingestion and TinyStories-ready preprocessing underway: typed experiment
configuration, local artifact manifests, ordered pipeline execution, inline text
and local text data, Unicode and line-ending normalization, exact
deduplication, deterministic split metadata, character and byte-level BPE
tokenizers, tiny dense GPT pretraining, checkpoint resume, greedy inference,
and evaluator-specific exact-reproduction evaluation.

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
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --force pretraining
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

## Local Text Data

Use `dataset.type: local_text` to ingest UTF-8 text files from explicit paths,
glob patterns, or both. Files are resolved, deduplicated, and processed in
deterministic path order. Raw documents include stable document ids plus
metadata for source type, absolute path, byte size, and content hash.

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

Byte-level BPE starts from all 256 byte values plus configured special tokens,
then learns deterministic pair merges up to `tokenizer.vocabulary_size`.
Encoding and decoding preserve Unicode, whitespace, tabs, and newlines through
UTF-8 bytes. For TinyStories-style runs, start with:

```yaml
tokenizer:
  type: byte_bpe
  vocabulary_size: 8192
  add_bos_token: true
  add_eos_token: true
  add_pad_token: true
```

Inspect training curves in TensorBoard:

```bash
python -m tensorboard.main --logdir runs/verify_one_sentence/artifacts/pretraining/tensorboard
```

Pipeline events are written to `runs/verify_one_sentence/pipeline.jsonl`. Training
metrics are written to `runs/verify_one_sentence/artifacts/pretraining/metrics.jsonl`
and mirrored to TensorBoard scalars.

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
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --force pretraining
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
39 passed

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
