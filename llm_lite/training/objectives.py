from typing import NamedTuple, Protocol

import torch
from torch import nn


class DpoPreferenceBatch(NamedTuple):
    chosen_token_ids: torch.Tensor
    rejected_token_ids: torch.Tensor
    chosen_completion_mask: torch.Tensor
    rejected_completion_mask: torch.Tensor


TrainingBatch = torch.Tensor | DpoPreferenceBatch


class TrainingObjectiveRunner(Protocol):
    def prepare_batch(self, batch: TrainingBatch, device: torch.device) -> TrainingBatch: ...

    def loss(self, model: nn.Module, batch: TrainingBatch) -> torch.Tensor: ...


class CausalLanguageModelingObjectiveRunner:
    def prepare_batch(self, batch: TrainingBatch, device: torch.device) -> TrainingBatch:
        match batch:
            case torch.Tensor():
                return batch.to(device)
            case DpoPreferenceBatch():
                raise ValueError("Causal language modeling requires token tensor batches.")

    def loss(self, model: nn.Module, batch: TrainingBatch) -> torch.Tensor:
        match batch:
            case torch.Tensor():
                model_output = model(batch)
                return causal_language_modeling_loss(
                    logits=model_output.logits,
                    token_ids=batch,
                )
            case DpoPreferenceBatch():
                raise ValueError("Causal language modeling requires token tensor batches.")


class DirectPreferenceOptimizationObjectiveRunner:
    def __init__(self, reference_model: nn.Module, beta: float) -> None:
        self.reference_model = reference_model
        self.beta = beta
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def prepare_batch(self, batch: TrainingBatch, device: torch.device) -> TrainingBatch:
        match batch:
            case torch.Tensor():
                raise ValueError("DPO requires preference batches.")
            case DpoPreferenceBatch():
                self.reference_model.to(device)
                return DpoPreferenceBatch(
                    chosen_token_ids=batch.chosen_token_ids.to(device),
                    rejected_token_ids=batch.rejected_token_ids.to(device),
                    chosen_completion_mask=batch.chosen_completion_mask.to(device),
                    rejected_completion_mask=batch.rejected_completion_mask.to(device),
                )

    def loss(self, model: nn.Module, batch: TrainingBatch) -> torch.Tensor:
        match batch:
            case torch.Tensor():
                raise ValueError("DPO requires preference batches.")
            case DpoPreferenceBatch():
                policy_chosen_log_prob = sequence_completion_log_probability(
                    model=model,
                    token_ids=batch.chosen_token_ids,
                    completion_mask=batch.chosen_completion_mask,
                )
                policy_rejected_log_prob = sequence_completion_log_probability(
                    model=model,
                    token_ids=batch.rejected_token_ids,
                    completion_mask=batch.rejected_completion_mask,
                )
                with torch.no_grad():
                    reference_chosen_log_prob = sequence_completion_log_probability(
                        model=self.reference_model,
                        token_ids=batch.chosen_token_ids,
                        completion_mask=batch.chosen_completion_mask,
                    )
                    reference_rejected_log_prob = sequence_completion_log_probability(
                        model=self.reference_model,
                        token_ids=batch.rejected_token_ids,
                        completion_mask=batch.rejected_completion_mask,
                    )
                policy_log_ratio = policy_chosen_log_prob - policy_rejected_log_prob
                reference_log_ratio = reference_chosen_log_prob - reference_rejected_log_prob
                return -nn.functional.logsigmoid(
                    self.beta * (policy_log_ratio - reference_log_ratio),
                ).mean()


def causal_language_modeling_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    input_logits = logits[:, :-1, :].contiguous()
    target_token_ids = token_ids[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        input_logits.view(-1, input_logits.shape[-1]),
        target_token_ids.view(-1),
    )


def sequence_completion_log_probability(
    model: nn.Module,
    token_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    model_output = model(token_ids)
    log_probabilities = torch.log_softmax(model_output.logits[:, :-1, :], dim=-1)
    target_token_ids = token_ids[:, 1:]
    target_completion_mask = completion_mask[:, 1:].to(log_probabilities.dtype)
    token_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=target_token_ids.unsqueeze(dim=-1),
    ).squeeze(dim=-1)
    return (token_log_probabilities * target_completion_mask).sum(dim=-1)
