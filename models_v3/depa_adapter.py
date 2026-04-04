"""Adapter utilities for reusing stable DEPA in v3 models."""

from __future__ import annotations

import torch
import torch.nn as nn

from models_v2.depa_stable import StableDEPA


class StableDEPAAdapter(nn.Module):
    """
    Thin wrapper over StableDEPA that exposes stereo residual features.

    The wrapped StableDEPA returns target + delta. This adapter returns delta only,
    making it easier to fuse stereo features with external gates.
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()
        self.depa = StableDEPA(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
        )

    def forward(self, target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        updated = self.depa(target, reference)
        return updated - target
