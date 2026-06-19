import math

import torch
from torch import nn

from llm_lite.config.models import DenseGptConfiguration
from llm_lite.model.output import ModelOutput


class DenseGpt(nn.Module):
    def __init__(self, model_configuration: DenseGptConfiguration, vocabulary_size: int) -> None:
        super().__init__()
        if model_configuration.dimension % model_configuration.attention_heads != 0:
            raise ValueError("Model dimension must be divisible by attention heads.")
        self.model_configuration = model_configuration
        self.token_embedding = nn.Embedding(vocabulary_size, model_configuration.dimension)
        self.position_embedding = nn.Embedding(1024, model_configuration.dimension)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(model_configuration=model_configuration)
                for _ in range(model_configuration.layers)
            ]
        )
        self.final_normalization = nn.LayerNorm(model_configuration.dimension)
        self.output_projection = nn.Linear(
            model_configuration.dimension, vocabulary_size, bias=False
        )
        if model_configuration.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> ModelOutput:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = (
            self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        )
        causal_mask = torch.triu(
            torch.full((sequence_length, sequence_length), float("-inf"), device=token_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states=hidden_states, causal_mask=causal_mask)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return ModelOutput(logits=logits)


class TransformerBlock(nn.Module):
    def __init__(self, model_configuration: DenseGptConfiguration) -> None:
        super().__init__()
        self.attention_normalization = nn.LayerNorm(model_configuration.dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_configuration.dimension,
            num_heads=model_configuration.attention_heads,
            dropout=model_configuration.dropout,
            batch_first=True,
        )
        self.feed_forward_normalization = nn.LayerNorm(model_configuration.dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_configuration.dimension, model_configuration.feed_forward_dimension),
            nn.GELU(),
            nn.Linear(model_configuration.feed_forward_dimension, model_configuration.dimension),
            nn.Dropout(model_configuration.dropout),
        )
        self._reset_parameters()

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized_states = self.attention_normalization(hidden_states)
        attention_output, _attention_weights = self.attention(
            normalized_states,
            normalized_states,
            normalized_states,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_normalization(hidden_states),
        )
        return hidden_states

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(parameter.dim()))
