import math
from collections.abc import Iterable, Iterator

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from llm_lite.config.models import (
    PackingConfiguration,
    PerplexityEvaluationConfiguration,
)
from llm_lite.data.packing import pack_token_sequences
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
        for sequence in sequences:
            token_ids = torch.tensor([sequence.token_ids], dtype=torch.long)
            model_output = model(token_ids)
            loss = causal_language_modeling_loss(
                logits=model_output.logits,
                token_ids=token_ids,
            )
            total_loss += float(loss.detach().cpu().item())
            sequence_count += 1
    if sequence_count == 0:
        raise ValueError(
            "Perplexity evaluation produced no sequences for split "
            f"{evaluation_configuration.split!r} after reading "
            f"{document_counter.documents} documents. Check split names, filtering, "
            "and context/tokenization settings.",
        )
    average_loss = total_loss / sequence_count
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
