# TinyPython Model and Dataset Plan

## 1. Project objective

The project is intended to demonstrate a complete small-scale language-model training pipeline, including:

* tokenizer training;
* dataset preparation;
* dense and Mixture-of-Experts model components;
* distributed training;
* checkpointing and resume;
* evaluation;
* inference;
* artifact management;
* efficient synthetic-data generation.

The final showcase model should be a small Python model rather than only a story-generation model.

The original motivation is to train a model capable of useful Python completion and simple task-to-code generation, with outputs that can be parsed and potentially executed.

---

## 2. TinyStories as pipeline validation

Before training the Python model, reproduce a small TinyStories-style training run.

The purpose is not to make story generation the final project. It is a controlled validation of:

* tokenizer behavior;
* model implementation;
* MoE routing;
* distributed execution;
* optimizer and scheduler;
* loss reduction;
* checkpointing;
* resume;
* inference;
* generation quality;
* evaluation and experiment tracking.

TinyStories is attractive because small models can visibly learn:

* grammar;
* sentence structure;
* simple narrative consistency;
* restricted vocabulary and concepts.

The exact smallest useful TinyStories configuration should be selected after reviewing the paper and estimating the cost.

The TinyStories run should be intentionally limited and inexpensive. Its role is to prove that the pipeline works before introducing the harder Python dataset.

### Validation result: 2026-06-24 TinyStories MoE run

The first full TinyStories MoE validation run completed successfully on two RTX 4090 GPUs in under one hour.

Configuration:

* `configs/tinystories_moe_full.yaml`;
* 4,096-token Rust byte-BPE tokenizer;
* 6-layer top-1 MoE decoder;
* 4 experts;
* approximately 9.0M total parameters;
* approximately 3.7M active parameters;
* 256-token context length;
* 50,000 training steps;
* data-parallel world size 2.

Observed result:

* training loss fell smoothly from roughly 15 to the low 2.x range;
* final logged training loss was 2.40625;
* validation perplexity during training at step 50,000 was 6.1347;
* final evaluation perplexity was 6.8403 over 500 validation documents;
* final evaluation loss was 1.9228;
* throughput was approximately 487k global tokens/s, or 243k tokens/s per GPU rank;
* generated samples showed clear TinyStories structure, dialogue, simple characters, and story-like continuation.

This is sufficient evidence that the current tokenizer, data pipeline, MoE training path, distributed execution, checkpointing, evaluation, and inference path work together. The visible remaining quality issues are normal for a small one-hour validation run: repetition, local grammar errors, semantic drift, and one replacement character (`�`) in a generated sample. The replacement character should be tracked as a tokenizer or decode-path issue before Python training, but it does not block moving to the TinyPython synthetic-data pilot.

---

## 3. Final Python task definition

The Python model should not be trained on unrestricted repository code or arbitrary competitive-programming scripts.

The target task is:

> Given a short, precise natural-language description, generate a complete, typed, standalone Python function.

Example:

```text
Return the number of positive integers in values.

def count_positive(values: list[int]) -> int:
    count = 0
    for value in values:
        if value > 0:
            count += 1
    return count
```

The distribution should be deliberately constrained:

* short descriptions;
* short functions;
* built-in Python types;
* meaningful identifiers;
* no stdin/stdout;
* no classes;
* no third-party libraries;
* no repository context;
* no long explanations;
* no arbitrary advanced algorithms;
* no extensive world knowledge.

This is the Python equivalent of the TinyStories idea: reduce breadth enough that a small model can learn useful semantics.

---

## 4. Training sample format

The final dataset entries should contain only:

```json
{
  "task_description": "Return the number of positive integers in values.",
  "code": "def count_positive(values: list[int]) -> int:\n    ..."
}
```

Metadata may remain in the raw generation files for analysis, but the training representation should remain simple.

The cleaned JSONL contains separate `task_description` and `code` fields.

---

## 5. Synthetic dataset generation

The main dataset will be generated synthetically using local instruction-tuned coding models.

The generation pipeline is implemented in:

[Generate TinyPython script](../llm_lite/scripts/generate_tinypython.py)

The repository root also keeps [a compatibility wrapper](../generate_tinypython.py)
for short command-line usage.

It performs:

1. semantic task-seed generation;
2. compatibility filtering;
3. rendering of a detailed shared system prompt;
4. local batched inference with vLLM;
5. two stochastic completions per seed;
6. parsing of `<task>` and `<code>`;
7. Python syntax validation;
8. validation that the output contains exactly one top-level function;
9. writing valid records to JSONL;
10. writing rejected generations to a separate invalid JSONL;
11. resumable processing based on completed seed attempts.

It intentionally does not perform:

* semantic execution tests;
* generated test cases;
* behavioral verification;
* deduplication;
* corpus-level balancing after generation.

Those responsibilities can be handled later by the existing dataset-processing pipeline.

---

## 6. Semantic task seeds

Completely unconstrained prompts such as:

```text
Generate a simple Python function.
```

would collapse into repeated tasks such as:

* count vowels;
* reverse a string;
* find the maximum;
* check a palindrome;
* sum even values.

Instead, each request contains a compact semantic seed.

Example:

```text
Input: a list of integers
Operation: count matching elements
Condition or relation: positive
Required output: an integer
Edge behavior: handle an empty input naturally
Implementation style: use an explicit loop
Additional constraint: do not mutate the input list
```

The teacher model converts this seed into:

* a concise natural-language description;
* a complete typed function;
* suitable names;
* one concrete interpretation of minor ambiguity.

The script currently defines several task families.

### Input structures

* list of integers;
* list of strings;
* string;
* dictionary from strings to integers;
* two lists of integers.

### Operations

Examples include:

* count;
* sum;
* filter;
* transform;
* search;
* first or last match;
* partition;
* frequency counting;
* deduplication;
* string normalization;
* elementwise combination;
* dictionary selection and transformation.

### Conditions

Examples include:

* positive or negative;
* even or odd;
* threshold comparison;
* equality to a target;
* divisibility;
* substring and prefix conditions;
* case and length conditions;
* value conditions in dictionaries.

### Edge behavior

Examples include:

* empty-input behavior;
* returning `None`;
* preserving order;
* resolving ties by first occurrence;
* stopping at the shorter sequence;
* retaining remaining elements when interleaving.

### Implementation styles

Examples include:

* explicit loops;
* comprehensions;
* early returns;
* accumulators;
* dictionary accumulation;
* index-based iteration;
* `zip`.

Compatibility rules remove obviously contradictory combinations.

The semantic ontology includes explicit description-style and naming-style variants. These are not intended to train broad natural language competence; they provide controlled surface variation so the model sees the same simple semantics expressed with different task wording and identifier choices.

As of the first TinyPython pilot, the compatible seed space is approximately 760k unique semantic seeds. The generator prints a warning when `--num-seeds` exceeds the current unique compatible seed count, because that means it will cycle through semantic seeds and rely on stochastic sampling for additional variants.

---

## 7. Number of tasks and completions

Initial pilot:

```text
500 semantic seeds
× 2 completions per model
× 2 teacher models
= 2,000 raw generations
```

The pilot should measure:

* valid generation percentage;
* invalid formatting percentage;
* truncation percentage;
* average output length;
* throughput;
* output diversity;
* general qualitative quality.

After the pilot:

```text
50,000 semantic seeds
× 2 completions per model
× 2 models
= 200,000 raw generations
```

This should be sufficient for the first serious Python training experiment while staying well within the current unique semantic-seed space.

The corpus can later be expanded to:

* more semantic seeds;
* broader input types;
* more composed operations;
* additional teacher models;
* more implementation variants;
* additional description variants.

Generating hundreds of thousands of samples immediately is unnecessary. The first generation run should reveal prompt weaknesses and distribution collapse before scaling.

---

## 8. Teacher-model deployment

Use two GPUs and run one independent local model instance per GPU.

The same generation script is run twice with different model arguments and different output files.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm_lite.scripts.generate_tinypython \
  --model TEACHER_A \
  --num-seeds 50000 \
  --samples-per-seed 2 \
  --batch-size 512 \
  --output data/teacher_a.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m llm_lite.scripts.generate_tinypython \
  --model TEACHER_B \
  --num-seeds 50000 \
  --samples-per-seed 2 \
  --batch-size 512 \
  --output data/teacher_b.jsonl
```

No multi-GPU tensor parallelism is necessary for a 7B-class model when one model fits on one GPU.

This setup provides:

* simple deployment;
* independent failure handling;
* model-family diversity;
* straightforward throughput scaling;
* two directly mergeable JSONL files.

---

## 9. Inference backend

Use vLLM offline inference.

Reasons:

* local model loading;
* high-throughput continuous batching;
* support for multiple samples per prompt;
* automatic prefix caching;
* efficient KV-cache management;
* no HTTP service required;
* simple Python integration.

The script uses the offline `LLM` interface rather than deploying an API server.

An HTTP service would only be useful if:

* multiple remote producers submitted requests;
* generation needed to remain permanently available;
* several machines shared one inference service;
* model workers and producers had separate lifecycles.

None of those are required for the initial dataset generation.

---

## 10. Prefix caching

Every request shares:

* the same chat template;
* the same detailed system prompt;
* the same four examples.

Only the semantic seed at the end changes.

Automatic prefix caching allows vLLM to reuse the KV state for identical prompt blocks.

Important consequences:

* all variable content should remain at the end;
* the system prompt must remain token-identical;
* task IDs or model-specific data should not be inserted before the common prefix;
* each GPU replica has its own independent cache;
* caching reduces prompt-prefill cost but not generated-token decoding cost.

A roughly 1,000-token system prompt is acceptable because:

* it defines the output contract precisely;
* it is reused across all requests;
* prefix caching amortizes much of the prefill cost;
* better adherence is more important than minimizing the prefix to a few hundred tokens.

The system prompt includes four examples, which is enough to establish the format without overly anchoring the output distribution.

---

## 11. Sampling strategy

Do not use beam search.

Beam search tends to produce:

* conservative completions;
* highly similar candidates;
* small syntactic differences;
* low diversity.

Use stochastic sampling instead.

Current defaults:

```text
temperature:          0.8
top_p:                0.95
top_k:                50
repetition penalty:   1.03
samples per seed:     2
```

Each teacher model produces two candidates for the same seed.

Across two teacher models, each task receives four generated variants.

This provides:

* surface-language variation;
* implementation variation;
* model-family variation;
* manageable generation cost.

---

## 12. Batch size

The script now defaults to a moderate submission batch.

Recommended pilot value:

```text
batch size: 512
```

Possible later values:

```text
256, 512, or 1024
```

This argument controls how many prompts are submitted to one `llm.generate` call. It does not imply that every request is decoded simultaneously.

vLLM internally schedules active sequences according to:

* KV-cache capacity;
* available GPU memory;
* prompt length;
* generation length;
* scheduler configuration.

Start with 512 for 7B-class teacher models on 24 GB GPUs when the pilot shows stable memory and good scheduler behavior. Lower it to 256 if the model runs out of KV-cache memory, or try 1024 only after confirming that end-to-end throughput improves.

---

## 13. Generation length and invalid outputs

The generation ceiling is:

```text
max tokens: 512
```

The system prompt restricts functions to 3–30 non-empty lines, so 512 output tokens should provide ample headroom.

A generation is invalid when:

* `finish_reason` is not `stop`;
* the model reaches the output-token limit;
* the tag structure cannot be parsed;
* task or code is empty;
* the code is invalid Python;
* the code contains anything other than exactly one top-level function.

Valid records are written to:

```text
teacher_a.jsonl
```

Invalid records are written to:

```text
teacher_a.invalid.jsonl
```

The invalid file contains:

* the raw generation;
* the finish reason;
* the rejection reason;
* the semantic seed;
* the sample index.

Resume logic counts both valid and invalid attempts, preventing repeated generation of the same failed sample after restarting.

---

## 14. Output schema

Valid output:

```json
{
  "model": "teacher/model",
  "seed": {
    "seed_id": 0,
    "input_kind": "a list of integers",
    "operation": "count matching elements",
    "condition": "positive",
    "output_kind": "an integer",
    "edge_behavior": "handle an empty input naturally",
    "implementation_style": "use an explicit loop",
    "extra_constraint": "do not mutate the input list"
  },
  "sample_index": 0,
  "task_description": "Return the number of positive integers in values.",
  "code": "def count_positive(values: list[int]) -> int:\n    ..."
}
```

The semantic seed is retained in the raw generated dataset because it is useful for:

* corpus analysis;
* distribution balancing;
* debugging;
* filtering;
* tracing problematic task families.

It does not need to be included in the eventual training sequence.

---

## 15. Teacher-model selection

The teacher should be:

* instruction tuned;
* strong at Python;
* reliable at structured output;
* compatible with vLLM;
* small enough to run one instance per GPU;
* able to generate without verbose reasoning.

Recommended first teacher:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Reasons:

* code-specific training;
* good Python generation;
* instruction following;
* reliable formatting;
* standard Transformer architecture;
* mature vLLM support;
* manageable 7B size.

Possible second teacher:

```text
microsoft/Phi-4-mini-instruct
```

Advantages:

* different model family;
* lower compute cost;
* different generation distribution;
* strong instruction following for its size.

Alternative second teacher:

```text
Qwen/Qwen3-8B
```

This likely provides stronger generation quality, but it is from the same broad Qwen family. It must be run in non-thinking mode to avoid reasoning output and unnecessary token generation.

The final choice should be made after a 500-seed pilot.

Compare:

* valid-output rate;
* truncation rate;
* average generation length;
* code quality;
* adherence to type annotations;
* task diversity;
* implementation diversity;
* throughput.

If Phi-4-mini performs poorly, use Qwen3-8B as the second teacher.

---

## 16. Precision and quantization

Start with BF16 when supported.

For a 7–8B model on a GPU with approximately 24 GB memory:

* BF16 weights use roughly 14–16 GB;
* the remaining memory is available for KV cache and runtime overhead;
* context and output lengths are modest.

Quantization should not be assumed to increase speed.

It may:

* reduce memory consumption;
* permit larger batches;
* increase KV-cache headroom;

but actual throughput depends on:

* GPU architecture;
* supported kernels;
* quantization format;
* model implementation;
* vLLM backend support.

A clean initial configuration is:

```text
dtype: bfloat16
quantization: auto
one model per GPU
batch size: 512
prefix caching: enabled
```

Use AWQ, GPTQ, FP8, or another format only after confirming that it is supported efficiently on the rented GPU.

---

## 17. Python model architecture

The final Python model should use a Mixture-of-Experts architecture because validating the full MoE pipeline is an explicit project objective.

The MoE is not being selected solely because it is expected to outperform a dense model at the same total parameter count.

Initial target:

```text
Total parameters:       approximately 20–30M
Active parameters:      approximately 8–12M
Experts:                4
Routing:                top-1
Context length:         256–512
Vocabulary:             approximately 4k–8k
```

Four experts are preferred initially over eight because:

* each expert receives more training tokens;
* expert collapse is less likely;
* experts remain large enough to learn useful behavior;
* router behavior is easier to inspect;
* the architecture still validates expert routing and expert-parallel components.

If eight experts are important for pipeline coverage, increase total capacity toward approximately 30–40M parameters while keeping active parameters around 10–15M.

Expert FFNs can replace the dense FFN modules in selected or all Transformer blocks.

The model should track:

* router probabilities;
* expert utilization;
* token counts per expert;
* load-balancing loss;
* router entropy;
* expert overflow or dropped tokens, if applicable.

---

## 18. Vocabulary

Use a relatively small tokenizer vocabulary.

Initial target:

```text
4,000–8,000 tokens
```

Reasons:

* smaller embedding and output matrices;
* less parameter budget spent on vocabulary;
* more capacity retained for Transformer blocks and experts;
* narrower Python and task-description domain;
* fewer extremely rare logits;
* potentially easier optimization for a small model.

The tokenizer should cover:

* Python syntax;
* indentation and whitespace behavior;
* common identifiers;
* common task-description vocabulary;
* type annotations;
* operators;
* punctuation;
* string and numeric literals.

The final vocabulary size should be selected after inspecting tokenization statistics on the generated corpus.

---

## 19. Training mixture

The initial generated dataset consists of task-description-to-function examples.

The first Python training run can use straightforward causal sequences:

```text
<task description>

<function code>
```

Later the dataset may be transformed into several modes:

* description-to-code;
* signature-plus-description to function body;
* partial-function completion;
* full-code causal modeling.

However, this is not necessary for the first generated corpus.

The immediate goal is to determine whether the small MoE can learn:

* short English task descriptions;
* type signatures;
* meaningful identifiers;
* elementary control flow;
* simple semantic transformations;
* complete function generation.

---

## 20. Evaluation

The existing approximately 50 handwritten functions can form the starting holdout set.
These exist in evaluation/python_completion.py but have to be reworked to match the new task-description-to-function format.

They should eventually be categorized into:

* in-distribution tasks;
* unseen combinations of known operations;
* alternate natural-language wording;
* harder out-of-distribution tasks.

Each evaluation task should ideally include:

* task description;
* expected function signature;
* reference behavior or hidden tests;
* one or more partial prefixes where useful.

Primary metrics:

* parse success;
* compile success;
* exact function-format adherence;
* executable correctness;
* pass rate on hidden tests;
* completion quality from partial prefixes;
* robustness to variable renaming;
* robustness to alternate task wording.

The holdout set may need adjustment after the final synthetic-data distribution is known.

---

## 21. Immediate implementation steps

### Phase 1: TinyStories validation

1. Read the TinyStories paper.
2. Identify a small configuration with visible generation quality.
3. Estimate tokens, compute, and training duration.
4. Train using the project pipeline.
5. Validate:

   * tokenizer;
   * data stages;
   * MoE training;
   * checkpoint/resume;
   * distributed execution;
   * evaluation;
   * inference.

### Phase 2: Synthetic Python pilot

1. Install the optional teacher-generation dependencies.
2. Run the integrated TinyPython generator.
3. Rent one or two suitable GPUs.
4. Run a 500-seed pilot for each teacher.
5. Compare:

   * valid percentage;
   * invalid reasons;
   * throughput;
   * token lengths;
   * task variation;
   * implementation variation.
6. Adjust:

   * system prompt;
   * task ontology;
   * compatibility rules;
   * sampling temperature;
   * teacher models.

### Phase 3: Full Python dataset generation

1. Generate approximately 50,000 semantic seeds.
2. Generate two completions per seed per teacher.
3. Produce approximately 200,000 raw generations.
4. Merge valid teacher JSONL files.
5. Run the existing dataset pipeline for:

   * deduplication;
   * filtering;
   * statistics;
   * train/validation split;
   * tokenizer preparation.
6. Inspect distribution summaries rather than manually reviewing hundreds of samples.
7. Extend underrepresented task families if necessary.

### Phase 4: Python MoE training

1. Train tokenizer on task descriptions and Python.
2. Configure approximately 20–30M total parameters.
3. Use four top-1 experts initially.
4. Train on the synthetic task-to-function corpus.
5. Track routing and expert utilization.
6. Evaluate against the handwritten holdout.
7. Analyze:

   * syntax learning;
   * semantic correctness;
   * description understanding;
   * function completion;
   * generalization to unseen combinations.

### Phase 5: Optional expansion

Depending on the first result:

* increase task composition depth;
* add more input and output types;
* add partial-function completion examples;
* add a small amount of curated human code;
* increase the number of experts;
* expand total model size;
* generate additional teacher data;
* add execution-based dataset verification;
* add preference or correction data for failed generations.

---

## 22. Final project direction

The final project is no longer:

> Train a tiny model on arbitrary Python code and hope useful completion emerges.

It is:

> Build and validate a complete small-scale MoE language-model pipeline, first on TinyStories, then train a deliberately constrained Python model on a synthetic curriculum of short task descriptions and typed standalone functions.

The core assumptions are:

* arbitrary code lacks sufficient intent;
* competitive-programming scripts are especially unsuitable;
* unrestricted English-to-code is too broad for a very small model;
* a controlled semantic task space can make useful behavior learnable;
* local 7B-class teacher models can efficiently generate the corpus;
* semantic task seeds provide controllable diversity;
* prefix caching and vLLM make generation economical;
* a small MoE is justified because MoE infrastructure is itself part of the project objective.

The immediate next artifact is a 500-seed TinyPython pilot generated with
`python -m llm_lite.scripts.generate_tinypython`.

---

## 23. Downloadable run artifacts

A complete run directory is not the right default download artifact for rented GPU instances. It contains raw dataset shards, processed documents, packed sequence shards, TensorBoard files, and every checkpoint interval. For TinyStories-scale runs this can easily become many GB, even when the only useful long-term artifact is the final model state plus the small metadata needed to interpret it.

Use the compact bundle exporter for run downloads:

```bash
python -m llm_lite.scripts.export_run_bundle \
  --run-dir runs/tinystories_moe_full \
  --output tinystories_moe_full_bundle.zip
```

The default bundle includes:

* `resolved_config.json`;
* `pipeline.jsonl`;
* `performance.jsonl`, when present;
* tokenizer manifest and tokenizer files;
* pretraining manifest;
* pretraining metrics and training-evaluation logs;
* final evaluation manifest and report;
* the latest pretraining checkpoint only;
* `bundle_manifest.json`, which lists the included files.

The default bundle excludes:

* raw dataset shards;
* processed dataset shards;
* packed training shards;
* older checkpoint intervals;
* TensorBoard event files.

Optional flags:

```bash
python -m llm_lite.scripts.export_run_bundle \
  --run-dir runs/tinystories_moe_full \
  --output tinystories_moe_full_full_checkpoints.zip \
  --include-all-checkpoints
```

```bash
python -m llm_lite.scripts.export_run_bundle \
  --run-dir runs/tinystories_moe_full \
  --output tinystories_moe_full_with_tensorboard.zip \
  --include-tensorboard
```

For normal experiment archiving, keep the default compact bundle and delete or ignore the full run directory after confirming that the bundle contains the latest checkpoint, tokenizer, config, metrics, and evaluation report.
