# LLM-Light

LLM-Light is a small PyTorch-native, configuration-driven LLM training pipeline.
This repository currently implements milestones M0 and M1 only: typed experiment
configuration, local artifact manifests, ordered pipeline execution, inline text
data, a character tokenizer, tiny dense GPT pretraining, checkpoint resume,
greedy inference, and evaluator-specific exact-reproduction evaluation.

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

## Achieved Results

Validated on 2026-06-19 with:

```bash
python -m pytest -q
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --dry-run
python -m llm_lite.scripts.run_pipeline --config configs/verify_one_sentence.yaml --force pretraining
```

Results:

```text
10 passed
pretraining final_step: 60
pretraining final_loss: 0.00016388329095207155
exact reproduction: passed
generated_text: "hello world\n"
tensorboard event files: present
pipeline event log: present
```
