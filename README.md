# LLM-Light

LLM-Light is a small PyTorch-native, configuration-driven LLM training pipeline.
This repository currently implements milestones M0 and M1 only: typed experiment
configuration, local artifact manifests, ordered pipeline execution, inline text
data, a character tokenizer, tiny dense GPT pretraining, checkpoint resume,
greedy inference, and evaluator-specific exact-reproduction evaluation.

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

Inspect training curves in TensorBoard:

```bash
python -m tensorboard.main --logdir runs/verify_one_sentence/artifacts/pretraining/tensorboard
```

Pipeline events are written to `runs/verify_one_sentence/pipeline.jsonl`. Training
metrics are written to `runs/verify_one_sentence/artifacts/pretraining/metrics.jsonl`
and mirrored to TensorBoard scalars.

## Packed Data Access

Packed training data is stored as uint16 shard files plus a small JSON index. The
training dataset is a finite `IterableDataset`: each epoch shuffles shard order,
loads one shard at a time, shuffles rows within that shard, and yields token rows
as `torch.long`. Multiworker loading partitions shards by worker id, so workers do
not duplicate shard reads.

Temporary local benchmark on 2026-06-20:

```text
rows_per_dataset_pass: 20000
passes: 3
row_length: 128
batch_size: 64
shard_sequences: 512
iterable_sharded_seconds: 1.1236
in_memory_shuffle_seconds: 0.6466
iterable_vs_memory_ratio: 1.74x
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
22 passed
All checks passed!
pretraining final_step: 60
pretraining final_loss: 0.00016388329095207155
exact reproduction: passed
generated_text: "hello world\n"
tensorboard event files: present
pipeline event log: present
```
