import json
from pathlib import Path

from llm_lite.config.models import (
    ByteBpeTokenizerConfiguration,
    LowerCaseTransformConfiguration,
    PreprocessingTransformType,
    TokenizerType,
)
from llm_lite.data.document import Document
from llm_lite.data.packing import pack_text_shards
from llm_lite.data.preprocessing import preprocess_text_shards
from llm_lite.data.text_shards import iter_text_shard_documents, write_text_shards
from llm_lite.pipeline.performance import (
    PipelinePerformanceLogger,
)
from llm_lite.tokenizer.byte_bpe import (
    _best_pair,
    bounded_training_document_references,
    train_byte_bpe_tokenizer,
    train_byte_bpe_tokenizer_from_text_shards,
)


def test_parallel_preprocessing_matches_single_worker(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    single_directory = tmp_path / "single"
    parallel_directory = tmp_path / "parallel"
    write_text_shards(
        documents=iter(
            [
                Document(document_id="a", text="HELLO", split=None),
                Document(document_id="b", text="WORLD", split=None),
                Document(document_id="c", text="AGAIN", split=None),
                Document(document_id="d", text="TEXT", split=None),
            ],
        ),
        artifact_directory=raw_directory,
        shard_document_limit=1,
    )
    transforms = (LowerCaseTransformConfiguration(type=PreprocessingTransformType.LOWER_CASE),)

    single_result = preprocess_text_shards(
        input_artifact_directory=raw_directory,
        output_artifact_directory=single_directory,
        transforms=transforms,
        output_shard_documents=2,
        workers=1,
    )
    parallel_result = preprocess_text_shards(
        input_artifact_directory=raw_directory,
        output_artifact_directory=parallel_directory,
        transforms=transforms,
        output_shard_documents=2,
        workers=2,
    )

    assert _documents(single_directory) == _documents(parallel_directory)
    assert single_result.counters.output_documents == parallel_result.counters.output_documents
    assert parallel_result.worker_count == 2


def test_parallel_packing_matches_single_worker(tmp_path: Path) -> None:
    processed_directory = tmp_path / "processed"
    tokenizer_directory = tmp_path / "tokenizer"
    single_directory = tmp_path / "single-packed"
    parallel_directory = tmp_path / "parallel-packed"
    write_text_shards(
        documents=iter(
            [
                Document(document_id="a", text="abababab", split="train"),
                Document(document_id="b", text="bcbcbcbc", split="train"),
                Document(document_id="c", text="cdcdcdcd", split="train"),
                Document(document_id="d", text="dededede", split="train"),
            ],
        ),
        artifact_directory=processed_directory,
        shard_document_limit=1,
    )
    tokenizer_result = train_byte_bpe_tokenizer_from_text_shards(
        artifact_directory=processed_directory,
        split="train",
        vocabulary_size=263,
        max_training_documents=4,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    tokenizer_result.tokenizer.save(directory=tokenizer_directory)
    tokenizer_configuration = ByteBpeTokenizerConfiguration(
        type=TokenizerType.BYTE_BPE,
        vocabulary_size=263,
        max_training_documents=4,
        max_training_bytes=None,
    )

    single_result = pack_text_shards(
        input_artifact_directory=processed_directory,
        output_artifact_directory=single_directory,
        tokenizer_directory=tokenizer_directory,
        tokenizer_configuration=tokenizer_configuration,
        split="train",
        context_length=4,
        pad_token_id=tokenizer_result.tokenizer.pad_token_id,
        add_bos=True,
        add_eos=True,
        maximum_shard_tokens=10,
        workers=1,
    )
    parallel_result = pack_text_shards(
        input_artifact_directory=processed_directory,
        output_artifact_directory=parallel_directory,
        tokenizer_directory=tokenizer_directory,
        tokenizer_configuration=tokenizer_configuration,
        split="train",
        context_length=4,
        pad_token_id=tokenizer_result.tokenizer.pad_token_id,
        add_bos=True,
        add_eos=True,
        maximum_shard_tokens=10,
        workers=2,
    )

    assert single_result.index.total_sequences == parallel_result.index.total_sequences
    assert _packed_rows(single_directory) == _packed_rows(parallel_directory)
    assert parallel_result.worker_count == 2


def test_parallel_byte_bpe_training_matches_single_worker(tmp_path: Path) -> None:
    processed_directory = tmp_path / "processed"
    write_text_shards(
        documents=iter(
            [
                Document(document_id="a", text="abababab", split="train"),
                Document(document_id="b", text="bcbcbcbc", split="train"),
                Document(document_id="c", text="ababcccc", split="train"),
                Document(document_id="d", text="ddddeeee", split="train"),
            ],
        ),
        artifact_directory=processed_directory,
        shard_document_limit=1,
    )

    single_result = train_byte_bpe_tokenizer_from_text_shards(
        artifact_directory=processed_directory,
        split="train",
        vocabulary_size=266,
        max_training_documents=4,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    parallel_result = train_byte_bpe_tokenizer_from_text_shards(
        artifact_directory=processed_directory,
        split="train",
        vocabulary_size=266,
        max_training_documents=4,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=2,
    )

    assert single_result.tokenizer.merge_rules == parallel_result.tokenizer.merge_rules
    assert single_result.tokenizer.byte_token_to_id == parallel_result.tokenizer.byte_token_to_id
    assert parallel_result.worker_count == 2


def test_iterable_byte_bpe_workers_preserve_current_behavior() -> None:
    single_result = train_byte_bpe_tokenizer(
        texts=["abababab", "bcbcbcbc"],
        vocabulary_size=264,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=1,
    )
    worker_result = train_byte_bpe_tokenizer(
        texts=["abababab", "bcbcbcbc"],
        vocabulary_size=264,
        max_training_documents=2,
        max_training_bytes=None,
        add_bos_token=True,
        add_eos_token=True,
        add_pad_token=True,
        workers=2,
    )

    assert single_result.tokenizer.merge_rules == worker_result.tokenizer.merge_rules
    assert single_result.tokenizer.byte_token_to_id == worker_result.tokenizer.byte_token_to_id


def test_pair_selection_uses_stable_tie_breaker() -> None:
    pair_counts = {
        ((2,), (1,)): 3,
        ((1,), (9,)): 3,
        ((1,), (8,)): 3,
    }

    assert _best_pair(pair_counts=pair_counts) == ((1,), (8,))


def test_bounded_bpe_selection_is_deterministic_without_duplicates(tmp_path: Path) -> None:
    processed_directory = tmp_path / "processed"
    write_text_shards(
        documents=iter(
            [
                Document(document_id="a", text="aa", split="train"),
                Document(document_id="b", text="bb", split="train"),
                Document(document_id="c", text="cc", split="train"),
            ],
        ),
        artifact_directory=processed_directory,
        shard_document_limit=1,
    )

    first_selection = bounded_training_document_references(
        artifact_directory=processed_directory,
        split="train",
        max_training_documents=2,
        max_training_bytes=None,
    )
    second_selection = bounded_training_document_references(
        artifact_directory=processed_directory,
        split="train",
        max_training_documents=2,
        max_training_bytes=None,
    )

    first_keys = tuple(
        (reference.shard_reference.path.name, reference.member_name)
        for reference in first_selection.document_references
    )
    second_keys = tuple(
        (reference.shard_reference.path.name, reference.member_name)
        for reference in second_selection.document_references
    )
    assert first_keys == second_keys
    assert len(set(first_keys)) == len(first_keys)


def test_performance_metrics_artifact_is_written(tmp_path: Path) -> None:
    logger = PipelinePerformanceLogger(run_directory=tmp_path)

    with logger.measure_stage(stage_name="tokenizer") as performance_timer:
        pass
    logger.write_stage_timing(
        timing=performance_timer.timing(),
        metrics={
            "training_documents": 2,
            "training_bytes": 10,
            "tokenizer_merges_completed": 4,
            "pair_count_seconds": 0.1,
            "workers": 2,
        },
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "performance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["stage_name"] == "tokenizer"
    assert records[0]["duration_seconds"] >= 0.0
    assert records[0]["worker_count"] == 2
    assert records[0]["metrics"]["pair_count_seconds"] == 0.1


def _documents(artifact_directory: Path) -> list[Document]:
    return list(iter_text_shard_documents(artifact_directory=artifact_directory, split=None))


def _packed_rows(artifact_directory: Path) -> list[list[int]]:
    packed_index = json.loads((artifact_directory / "index.json").read_text(encoding="utf-8"))
    rows: list[list[int]] = []
    for shard in packed_index["shards"]:
        content = (artifact_directory / shard["path"]).read_bytes()
        values = [
            int.from_bytes(content[byte_index : byte_index + 2], byteorder="little")
            for byte_index in range(0, len(content), 2)
        ]
        row_length = packed_index["row_length"]
        rows.extend(
            values[start_index : start_index + row_length]
            for start_index in range(0, len(values), row_length)
        )
    return rows
