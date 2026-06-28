# Python Model Sweep Results

This page is reserved for the results of the TinyPython model sweep described
in [PYTHON_MODEL_SWEEP.md](PYTHON_MODEL_SWEEP.md).

Record each completed run with:

- Config path and Git commit.
- Hardware, GPU allocation, and wall time.
- Model total and active parameters.
- Final training loss.
- Validation loss and perplexity.
- Python completion parse, execution, and pass rates.
- Throughput and memory notes.
- Export bundle path.
- Representative successes and failures.

## Results Table

| Config | Status | Active params | Final loss | Perplexity | Completion pass rate | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `python_moe_small_deep_plain.yaml` | Pending | 995,984 | | | | |
| `python_moe_small_wide_plain.yaml` | Pending | 993,824 | | | | |
| `python_dense_small_deep_plain.yaml` | Pending | 994,576 | | | | |
| `python_dense_small_wide_plain.yaml` | Pending | 992,992 | | | | |
| `python_moe_small_deep_fim.yaml` | Pending | 995,984 | | | | |
| `python_moe_small_deep_linear_warmup_decay.yaml` | Pending | 995,984 | | | | |
| `python_moe_small_deep_cosine_warmup_decay.yaml` | Pending | 995,984 | | | | |
| `python_modern_moe_small_deep_plain.yaml` | Pending | 924,440 | | | | |
| `python_dense_active_9m6.yaml` | Pending | 9,646,080 | | | | |
