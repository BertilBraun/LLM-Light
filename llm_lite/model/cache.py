from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransformerBlockInferenceCache:
    key_states: torch.Tensor
    value_states: torch.Tensor


@dataclass(frozen=True)
class GptInferenceCache:
    layers: tuple[TransformerBlockInferenceCache, ...]


@dataclass(frozen=True)
class CachedModelOutput:
    logits: torch.Tensor
    inference_cache: GptInferenceCache
