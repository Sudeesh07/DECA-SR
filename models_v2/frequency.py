"""
Lightweight Frequency-Aware Refinement for Stereo Image Super-Resolution

Two-branch architecture:
- Spatial branch: NAFBlock x 2 for local feature refinement
- HF branch: Fixed Laplacian high-pass filter + learnable 1x1 conv
Merged via concat + conv with residual connection.

No FFT cross-view fusion — keeps this module fast and focused on
enhancing high-frequency details that are critical for SR quality.
"""

import torch
import torch.nn as nn

from models.nafblock import NAFBlock, LayerNorm2d


class LightweightFrequencyRefinement(nn.Module):
    """
    Spatial + high-frequency dual-branch refinement.

    The HF branch uses a fixed (non-learnable) Laplacian kernel to extract
    edges/textures, then a learnable 1x1 conv to weight and transform them.
    This avoids the Laplacian learning away from a good high-pass prior.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Spatial branch
        self.spatial = nn.Sequential(
            NAFBlock(channels),
            NAFBlock(channels),
        )

        # HF branch: fixed Laplacian + learnable 1x1
        self.register_buffer(
            "laplacian",
            self._build_laplacian(channels),
        )
        self.hf_conv = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        # Merge spatial + HF
        self.merge = nn.Sequential(
            LayerNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.beta = nn.Parameter(torch.tensor(0.05))  # Learnable residual scaling

    @staticmethod
    def _build_laplacian(channels: int) -> torch.Tensor:
        """Depthwise Laplacian kernel [channels, 1, 3, 3], fixed."""
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        )
        return (kernel / 4.0).view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_out = self.spatial(x)

        hf = torch.nn.functional.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])
        hf = self.hf_conv(hf)

        merged = self.merge(torch.cat([spatial_out, hf], dim=1))
        return x + merged*self.beta
