from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
