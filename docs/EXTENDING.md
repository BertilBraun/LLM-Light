# Extending LLM-Light

This page covers the small, typed extension points used by the current local
pipeline. It does not cover the planned orchestration rewrite.

## Add an Evaluator

Evaluators are configured through `EvaluationConfiguration` in
`llm_lite/config/models.py` and run from `llm_lite/evaluation/runner.py`.

1. Add a frozen Pydantic configuration model in `llm_lite/config/models.py`.
2. Add an optional field for that model to `EvaluationConfiguration`.
3. Implement the evaluator in a focused module under `llm_lite/evaluation/`.
4. Return a typed result model with report fields and aggregate scalar metrics.
5. Call the evaluator from `run_configured_evaluators`.
6. Add TensorBoard scalar tags in `llm_lite/evaluation/tensorboard.py` for
   important numeric metrics.
7. Add unit tests for the evaluator, runner integration, and TensorBoard tags.

Keep evaluator reports detailed enough for debugging, but keep TensorBoard
scalars compact. Prefer aggregate counts, rates, and per-family summaries over
large raw text or plot collections.

## Add a Dataset Source

Dataset sources emit `Document` records and are materialized by the
`raw_dataset` stage.

1. Add a `DatasetType` value and a frozen Pydantic configuration model in
   `llm_lite/config/models.py`.
2. Add the model to the discriminated `ExperimentFile.dataset` union.
3. Implement an iterator in `llm_lite/data/sources.py` that yields `Document`
   objects with stable `document_id`, `text`, and optional `split`.
4. Add a `match` branch in `iter_dataset_documents`.
5. Include explicit validation for required source fields at the configuration
   boundary.
6. Add tests for config loading, document IDs, split behavior, filtering, and
   raw-stage metrics when relevant.

Source iterators should stream when possible. The `raw_dataset`,
`processed_dataset`, `tokenizer`, and `packed_dataset` stages already write
stage metrics that appear in the run-level TensorBoard text summaries.

## Add a Model Architecture

Model architectures are selected by the typed model configuration union and
built by `llm_lite/model/factory.py`.

1. Add a `ModelType` value and a frozen Pydantic configuration model in
   `llm_lite/config/models.py`.
2. Add the configuration model to the `ModelConfiguration` discriminated union.
3. Implement the `torch.nn.Module` under `llm_lite/model/`.
4. Return `ModelOutput` from `forward`, including `auxiliary_loss` only when the
   architecture needs it.
5. Add a `match` branch in `build_model`.
6. Update parameter accounting if active parameters differ from total
   parameters.
7. If the loss function differs from standard causal language modelling, define a custom `TrainingObjectiveRunner` in `llm_lite/training/objectives.py`
8. Add tests for factory selection, output shapes, loss compatibility,
   checkpoint round trips, and a tiny pipeline smoke config if the architecture
   changes training behavior.

Training expects autoregressive logits shaped as batch, sequence, vocabulary.
Architectures with routing or other auxiliary state should expose compact
training observability rather than large per-token dumps.
