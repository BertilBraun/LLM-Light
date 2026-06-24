from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
from torch import nn

from llm_lite.config.models import DenseGptConfiguration
from llm_lite.model.cache import (
    CachedModelOutput,
    GptInferenceCache,
    TransformerBlockInferenceCache,
)
from llm_lite.model.output import ModelOutput
from llm_lite.model.protocol import CachedAutoregressiveModel


class DenseGpt(CachedAutoregressiveModel):
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
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=token_ids.device,
                dtype=hidden_states.dtype,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states=hidden_states, causal_mask=causal_mask)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return ModelOutput(logits=logits)

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        inference_cache: GptInferenceCache,
    ) -> CachedModelOutput:
        batch_size, sequence_length = token_ids.shape
        cache_sequence_length = _cache_sequence_length(inference_cache=inference_cache)
        positions = torch.arange(
            cache_sequence_length,
            cache_sequence_length + sequence_length,
            device=token_ids.device,
        )
        hidden_states = (
            self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        )
        cache_layers: list[TransformerBlockInferenceCache] = []
        for block, layer_cache in zip(self.blocks, inference_cache.layers, strict=True):
            block_output = block.forward_with_cache(
                hidden_states=hidden_states,
                inference_cache=layer_cache,
            )
            hidden_states = block_output.hidden_states
            cache_layers.append(block_output.inference_cache)
        hidden_states = self.final_normalization(hidden_states)
        logits = self.output_projection(hidden_states)
        assert logits.shape[0] == batch_size
        return CachedModelOutput(
            logits=logits,
            inference_cache=GptInferenceCache(layers=tuple(cache_layers)),
        )

    def empty_inference_cache(self, batch_size: int, device: torch.device) -> GptInferenceCache:
        empty_layers = tuple(
            block.empty_inference_cache(batch_size=batch_size, device=device)
            for block in self.blocks
        )
        return GptInferenceCache(layers=empty_layers)


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

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        inference_cache: TransformerBlockInferenceCache,
    ) -> TransformerBlockOutput:
        normalized_states = self.attention_normalization(hidden_states)
        attention_output, next_cache = self._cached_attention(
            normalized_states=normalized_states,
            inference_cache=inference_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_normalization(hidden_states),
        )
        return TransformerBlockOutput(
            hidden_states=hidden_states,
            inference_cache=next_cache,
        )

    def empty_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransformerBlockInferenceCache:
        head_dimension = self.attention.embed_dim // self.attention.num_heads
        empty_cache_shape = (batch_size, self.attention.num_heads, 0, head_dimension)
        cache_dtype = self.attention.in_proj_weight.dtype
        return TransformerBlockInferenceCache(
            key_states=torch.empty(empty_cache_shape, device=device, dtype=cache_dtype),
            value_states=torch.empty(empty_cache_shape, device=device, dtype=cache_dtype),
        )

    def _cached_attention(
        self,
        normalized_states: torch.Tensor,
        inference_cache: TransformerBlockInferenceCache,
    ) -> tuple[torch.Tensor, TransformerBlockInferenceCache]:
        batch_size, sequence_length, _embedding_dimension = normalized_states.shape
        projected_states = torch_functional.linear(
            normalized_states,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        )
        query_states, key_states, value_states = projected_states.chunk(3, dim=-1)
        query_states = _split_heads(
            states=query_states,
            attention_heads=self.attention.num_heads,
        )
        key_states = _split_heads(
            states=key_states,
            attention_heads=self.attention.num_heads,
        )
        value_states = _split_heads(
            states=value_states,
            attention_heads=self.attention.num_heads,
        )
        cached_key_states = torch.cat(
            (inference_cache.key_states, key_states),
            dim=2,
        )
        cached_value_states = torch.cat(
            (inference_cache.value_states, value_states),
            dim=2,
        )
        attention_scores = (query_states @ cached_key_states.transpose(-2, -1)) / math.sqrt(
            query_states.shape[-1]
        )
        attention_scores = attention_scores.masked_fill(
            _causal_cache_mask(
                past_sequence_length=inference_cache.key_states.shape[2],
                sequence_length=sequence_length,
                total_sequence_length=cached_key_states.shape[2],
                device=normalized_states.device,
            )[None, None, :, :],
            float("-inf"),
        )
        attention_probabilities = torch.softmax(attention_scores, dim=-1)
        attention_probabilities = torch_functional.dropout(
            attention_probabilities,
            p=self.attention.dropout,
            training=self.training,
        )
        attention_output = attention_probabilities @ cached_value_states
        attention_output = _combine_heads(states=attention_output)
        attention_output = self.attention.out_proj(attention_output)
        assert attention_output.shape[0] == batch_size
        return attention_output, TransformerBlockInferenceCache(
            key_states=cached_key_states,
            value_states=cached_value_states,
        )

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(parameter.dim()))


@dataclass(frozen=True)
class TransformerBlockOutput:
    hidden_states: torch.Tensor
    inference_cache: TransformerBlockInferenceCache


def _cache_sequence_length(inference_cache: GptInferenceCache) -> int:
    if not inference_cache.layers:
        return 0
    first_layer_cache = inference_cache.layers[0]
    return int(first_layer_cache.key_states.shape[2])


def _split_heads(states: torch.Tensor, attention_heads: int) -> torch.Tensor:
    batch_size, sequence_length, embedding_dimension = states.shape
    head_dimension = embedding_dimension // attention_heads
    states = states.view(batch_size, sequence_length, attention_heads, head_dimension)
    return states.transpose(1, 2)


def _combine_heads(states: torch.Tensor) -> torch.Tensor:
    batch_size, attention_heads, sequence_length, head_dimension = states.shape
    states = states.transpose(1, 2).contiguous()
    return states.view(batch_size, sequence_length, attention_heads * head_dimension)


def _causal_cache_mask(
    past_sequence_length: int,
    sequence_length: int,
    total_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(
        past_sequence_length,
        past_sequence_length + sequence_length,
        device=device,
    )
    key_positions = torch.arange(total_sequence_length, device=device)
    return key_positions[None, :] > query_positions[:, None]
