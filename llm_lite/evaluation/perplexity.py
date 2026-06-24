import math
from collections.abc import Iterable, Iterator
from datetime import datetime
from time import perf_counter

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    PackingConfiguration,
    PerplexityEvaluationConfiguration,
)
from llm_lite.data.packing import pack_token_sequences
from llm_lite.pipeline.progress import progress_bar
from llm_lite.tokenizer.loading import TextTokenizer
from llm_lite.training.objectives import causal_language_modeling_loss


class PerplexityEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    split: str
    documents: int
    sequences: int
    loss: float
    perplexity: float


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: TextTokenizer,
    texts: Iterable[str],
    evaluation_configuration: PerplexityEvaluationConfiguration,
    packing_configuration: PackingConfiguration,
) -> PerplexityEvaluationResult:
    if tokenizer.pad_token_id is None:
        raise ValueError("Perplexity evaluation requires a configured pad token.")
    started = perf_counter()
    _log(
        "[eval] perplexity_start "
        f"split={evaluation_configuration.split} "
        f"maximum_documents={evaluation_configuration.maximum_documents}"
    )
    document_counter = _DocumentCounter(
        texts=texts,
        maximum_documents=evaluation_configuration.maximum_documents,
    )
    tokenized_document_stream = (
        tokenizer.encode(
            text=text,
            add_bos=packing_configuration.add_bos,
            add_eos=packing_configuration.add_eos,
        )
        for text in document_counter
    )
    sequences = pack_token_sequences(
        tokenized_document_stream=tokenized_document_stream,
        context_length=packing_configuration.context_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    total_loss = 0.0
    sequence_count = 0
    model.eval()
    with torch.no_grad():
        with progress_bar(
            description="eval/perplexity",
            total=evaluation_configuration.maximum_documents,
            unit="seq",
        ) as bar:
            last_document_count = 0
            for sequence in sequences:
                if document_counter.documents != last_document_count:
                    bar.update(document_counter.documents - last_document_count)
                    last_document_count = document_counter.documents
                if sequence_count == 0:
                    _log("[eval] perplexity_first_sequence")
                if sequence_count > 0 and sequence_count % 100 == 0:
                    elapsed = perf_counter() - started
                    _log(
                        "[eval] perplexity_progress "
                        f"documents={document_counter.documents} "
                        f"sequences={sequence_count} "
                        f"seconds={elapsed:.1f}"
                    )
                token_ids = torch.tensor([sequence.token_ids], dtype=torch.long)
                token_ids = token_ids.to(device=next(model.parameters()).device)
                model_output = model(token_ids)
                loss = causal_language_modeling_loss(
                    logits=model_output.logits,
                    token_ids=token_ids,
                )
                total_loss += float(loss.detach().cpu().item())
                sequence_count += 1
            if (
                evaluation_configuration.maximum_documents is not None
                and last_document_count < document_counter.documents
            ):
                bar.update(document_counter.documents - last_document_count)
    if sequence_count == 0:
        raise ValueError(
            "Perplexity evaluation produced no sequences for split "
            f"{evaluation_configuration.split!r} after reading "
            f"{document_counter.documents} documents. Check split names, filtering, "
            "and context/tokenization settings.",
        )
    average_loss = total_loss / sequence_count
    _log(
        "[eval] perplexity_done "
        f"documents={document_counter.documents} "
        f"sequences={sequence_count} "
        f"seconds={perf_counter() - started:.1f}"
    )
    return PerplexityEvaluationResult(
        split=evaluation_configuration.split,
        documents=document_counter.documents,
        sequences=sequence_count,
        loss=average_loss,
        perplexity=math.exp(average_loss),
    )


class _DocumentCounter:
    def __init__(self, texts: Iterable[str], maximum_documents: int | None) -> None:
        self.texts = texts
        self.maximum_documents = maximum_documents
        self.documents = 0

    def __iter__(self) -> Iterator[str]:
        for text in self.texts:
            if self.maximum_documents is not None and self.documents >= self.maximum_documents:
                break
            self.documents += 1
            yield text


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M')}] {message}", flush=True)
