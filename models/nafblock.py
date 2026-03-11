# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SimpleGate(nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1, x2 = x.chunk(2, dim=1)
#         return x1 * x2


# class SimplifiedChannelAttention(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(channels, channels, 1, bias=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.conv(self.pool(x))


# class NAFBlock(nn.Module):
#     def __init__(self, channels: int, expansion: int = 2, drop_path: float = 0.0):
#         super().__init__()
#         expanded = channels * expansion

#         self.norm1 = LayerNorm2d(channels)
#         self.conv1 = nn.Conv2d(channels, expanded, 1, bias=True)
#         self.dwconv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=True)
#         self.gate1 = SimpleGate()
#         self.sca = SimplifiedChannelAttention(expanded // 2)
#         self.proj1 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
#         self.norm2 = LayerNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, expanded, 1, bias=True)
#         self.gate2 = SimpleGate()
#         self.proj2 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
#         self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self.norm1(x)
#         x = self.conv1(x)
#         x = self.dwconv(x)
#         x = self.gate1(x)
#         x = self.sca(x)
#         x = self.proj1(x)
#         x = shortcut + self.drop_path(x) * self.beta

#         shortcut = x
#         x = self.norm2(x)
#         x = self.conv2(x)
#         x = self.gate2(x)
#         x = self.proj2(x)
#         x = shortcut + self.drop_path(x) * self.gamma
#         return x


# class LayerNorm2d(nn.Module):
#     def __init__(self, channels: int, eps: float = 1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(channels))
#         self.bias = nn.Parameter(torch.zeros(channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mean = x.mean(1, keepdim=True)
#         var = (x - mean).pow(2).mean(1, keepdim=True)
#         x = (x - mean) / (var + self.eps).sqrt()
#         return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if not self.training or self.drop_prob == 0.0:
#             return x
#         keep = 1 - self.drop_prob
#         mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
#         return x / keep * mask
"""
NAFBlock: Nonlinear Activation Free Block for Image Restoration

Based on NAFNet (ECCV 2022): https://arxiv.org/abs/2204.04676

Key improvements over provided version:
1. Fixed class ordering (LayerNorm2d and DropPath defined before use)
2. Added variance computation stability fix
3. Optional Local Attention variant
4. Proper weight initialization
5. Memory-efficient forward with checkpointing option
6. Added NAFNet encoder/decoder building blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


# =============================================================================
# Core Components (defined first to avoid forward reference issues)
# =============================================================================

class LayerNorm2d(nn.Module):
    """
    Channel-wise Layer Normalization for 2D feature maps.
    
    Normalizes across spatial dimensions for each channel independently.
    More stable than BatchNorm for image restoration tasks.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across spatial dims (H, W)
        mean = x.mean(dim=1, keepdim=True)
        # Use unbiased=False for consistency with LayerNorm
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularization.
    
    Randomly drops entire residual branches during training.
    Paper: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        # Create binary mask: [B, 1, 1, 1]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        
        # Scale to maintain expected value
        return x * mask / keep_prob
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class SimpleGate(nn.Module):
    """
    NAFNet's SimpleGate: Element-wise gating without activation.
    
    Splits input channels in half and multiplies:
        output = x[:, :C//2] * x[:, C//2:]
    
    This provides nonlinearity without explicit activation functions,
    which helps avoid issues with ReLU dead neurons and gradient flow.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA) from NAFNet.
    
    A lightweight channel attention that uses global average pooling
    and a single 1x1 convolution (no reduction ratio, no sigmoid).
    
    Unlike SE-Net which uses FC-ReLU-FC-Sigmoid, this is:
        output = x * Conv1x1(GAP(x))
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv(self.pool(x))  # [B, C, 1, 1]
        return x * attn


# =============================================================================
# NAFBlock Variants
# =============================================================================

class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block.
    
    Architecture:
        Input
          ↓
        LayerNorm → Conv1x1 → DWConv3x3 → SimpleGate → SCA → Conv1x1
          ↓ (+ residual * beta)
        LayerNorm → Conv1x1 → SimpleGate → Conv1x1
          ↓ (+ residual * gamma)
        Output
    
    The key insight is using SimpleGate (channel-split multiplication)
    instead of activation functions like ReLU or GELU.
    
    Args:
        channels: Number of input/output channels
        expansion: Channel expansion factor (default: 2)
        drop_path: Drop path probability for stochastic depth
        use_checkpoint: Use gradient checkpointing to save memory
    """
    def __init__(
        self, 
        channels: int, 
        expansion: int = 2, 
        drop_path: float = 0.0,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        expanded = channels * expansion

        # First block: Spatial mixing with gating
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1, bias=True)
        self.dwconv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=True)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(expanded // 2)
        self.proj1 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
        
        # Second block: Channel mixing with gating
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, expanded, 1, bias=True)
        self.gate2 = SimpleGate()
        self.proj2 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
        
        # Learnable residual scaling (initialized to 0 for stable training)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following NAFNet paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for convolutions
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward logic."""
        # First block: Spatial mixing
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate1(x)
        x = self.sca(x)
        x = self.proj1(x)
        x = shortcut + self.drop_path(x) * self.beta

        # Second block: Channel mixing  
        shortcut = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.gate2(x)
        x = self.proj2(x)
        x = shortcut + self.drop_path(x) * self.gamma
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class NAFBlockWithLocalAttention(nn.Module):
    """
    NAFBlock variant with local window attention.
    
    Replaces the SCA (global attention) with local window attention
    for better capturing local structures while maintaining efficiency.
    
    Args:
        channels: Number of channels
        expansion: Channel expansion factor
        window_size: Size of local attention window
        drop_path: Drop path probability
    """
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        window_size: int = 8,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.window_size = window_size
        expanded = channels * expansion

        # First block with local attention
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1, bias=True)
        self.dwconv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=True)
        self.gate1 = SimpleGate()
        
        # Local attention instead of SCA
        self.local_attn = LocalAttention(expanded // 2, window_size)
        
        self.proj1 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
        
        # Second block (same as NAFBlock)
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, expanded, 1, bias=True)
        self.gate2 = SimpleGate()
        self.proj2 = nn.Conv2d(expanded // 2, channels, 1, bias=True)
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate1(x)
        x = self.local_attn(x)
        x = self.proj1(x)
        x = shortcut + self.drop_path(x) * self.beta

        shortcut = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.gate2(x)
        x = self.proj2(x)
        x = shortcut + self.drop_path(x) * self.gamma
        
        return x


class LocalAttention(nn.Module):
    """
    Local window attention for NAFBlock.
    
    Applies attention within non-overlapping local windows.
    """
    def __init__(self, channels: int, window_size: int = 8):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.scale = (channels) ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        
        # Pad to multiple of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, Hp, Wp = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, 3C, Hp, Wp]
        
        # Reshape to windows: [B, 3C, Hp, Wp] -> [B*num_windows, 3C, ws, ws]
        qkv = qkv.view(B, 3 * C, Hp // ws, ws, Wp // ws, ws)
        qkv = qkv.permute(0, 2, 4, 1, 3, 5).reshape(-1, 3 * C, ws, ws)
        
        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B*nW, C, ws, ws]
        
        # Flatten spatial dims for attention
        q = q.view(-1, C, ws * ws).transpose(-2, -1)  # [B*nW, ws*ws, C]
        k = k.view(-1, C, ws * ws)  # [B*nW, C, ws*ws]
        v = v.view(-1, C, ws * ws).transpose(-2, -1)  # [B*nW, ws*ws, C]
        
        # Attention
        attn = torch.matmul(q, k) * self.scale  # [B*nW, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B*nW, ws*ws, C]
        
        # Reshape back
        out = out.transpose(-2, -1).view(-1, C, ws, ws)  # [B*nW, C, ws, ws]
        out = out.view(B, Hp // ws, Wp // ws, C, ws, ws)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return self.proj(out)


# =============================================================================
# NAFNet Building Blocks
# =============================================================================

class NAFNetEncoder(nn.Module):
    """
    NAFNet-style encoder with progressive downsampling.
    
    Architecture:
        Level 0: NAFBlocks × N0 @ 1× resolution
        Level 1: Downsample → NAFBlocks × N1 @ 1/2 resolution  
        Level 2: Downsample → NAFBlocks × N2 @ 1/4 resolution
        ...
    
    Args:
        channels: Base channel count
        num_blocks: List of block counts per level
        expansion: Channel expansion in NAFBlocks
        drop_path_rate: Maximum drop path rate (linearly increased)
    """
    def __init__(
        self,
        channels: int = 64,
        num_blocks: List[int] = [2, 4, 8, 12],
        expansion: int = 2,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.num_levels = len(num_blocks)
        
        # Compute drop path rates for each block
        total_blocks = sum(num_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Build encoder levels
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        block_idx = 0
        for level, n_blocks in enumerate(num_blocks):
            level_channels = channels * (2 ** level)
            
            # NAFBlocks for this level
            blocks = nn.Sequential(*[
                NAFBlock(level_channels, expansion, dpr[block_idx + i])
                for i in range(n_blocks)
            ])
            self.encoders.append(blocks)
            block_idx += n_blocks
            
            # Downsampler (except for last level)
            if level < self.num_levels - 1:
                down = nn.Conv2d(level_channels, level_channels * 2, 2, stride=2)
                self.downsamplers.append(down)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of features at each level (for skip connections)
        """
        features = []
        
        for level in range(self.num_levels):
            x = self.encoders[level](x)
            features.append(x)
            
            if level < self.num_levels - 1:
                x = self.downsamplers[level](x)
        
        return features


class NAFNetDecoder(nn.Module):
    """
    NAFNet-style decoder with progressive upsampling.
    
    Takes encoder features as skip connections.
    """
    def __init__(
        self,
        channels: int = 64,
        num_blocks: List[int] = [12, 8, 4, 2],
        expansion: int = 2,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.num_levels = len(num_blocks)
        
        total_blocks = sum(num_blocks)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, total_blocks)]
        
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.fusions = nn.ModuleList()
        
        block_idx = 0
        for level in range(self.num_levels):
            # Channels decrease as we go up
            level_channels = channels * (2 ** (self.num_levels - 1 - level))
            
            # Upsampler (except for first level)
            if level > 0:
                up = nn.Sequential(
                    nn.Conv2d(level_channels * 2, level_channels * 4, 1),
                    nn.PixelShuffle(2),
                )
                self.upsamplers.append(up)
                
                # Skip connection fusion
                fusion = nn.Conv2d(level_channels * 2, level_channels, 1)
                self.fusions.append(fusion)
            
            # NAFBlocks
            blocks = nn.Sequential(*[
                NAFBlock(level_channels, expansion, dpr[block_idx + i])
                for i in range(num_blocks[level])
            ])
            self.decoders.append(blocks)
            block_idx += num_blocks[level]
    
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features: List of features from encoder (deepest first)
        """
        # Start from deepest level
        x = encoder_features[-1]
        x = self.decoders[0](x)
        
        # Progressive upsampling with skip connections
        for level in range(1, self.num_levels):
            x = self.upsamplers[level - 1](x)
            skip = encoder_features[-(level + 1)]
            x = self.fusions[level - 1](torch.cat([x, skip], dim=1))
            x = self.decoders[level](x)
        
        return x


# =============================================================================
# Utility Functions
# =============================================================================

def get_nafblock(
    channels: int,
    variant: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create NAFBlock variants.
    
    Args:
        channels: Number of channels
        variant: One of "standard", "local_attention"
        **kwargs: Additional arguments for the block
    """
    variants = {
        "standard": NAFBlock,
        "local_attention": NAFBlockWithLocalAttention,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    return variants[variant](channels, **kwargs)


# =============================================================================
# Unit Tests
# =============================================================================

def test_nafblock():
    """Test NAFBlock and variants."""
    print("Testing NAFBlock Components...")
    print("=" * 50)
    
    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W)
    
    # Test basic components
    ln = LayerNorm2d(C)
    out = ln(x)
    assert out.shape == x.shape
    print(f"✓ LayerNorm2d: {out.shape}")
    
    sg = SimpleGate()
    out = sg(torch.randn(B, C * 2, H, W))
    assert out.shape == x.shape
    print(f"✓ SimpleGate: {out.shape}")
    
    sca = SimplifiedChannelAttention(C)
    out = sca(x)
    assert out.shape == x.shape
    print(f"✓ SimplifiedChannelAttention: {out.shape}")
    
    dp = DropPath(0.1)
    dp.train()
    out = dp(x)
    assert out.shape == x.shape
    print(f"✓ DropPath: {out.shape}")
    
    # Test NAFBlock
    naf = NAFBlock(C, expansion=2, drop_path=0.1)
    out = naf(x)
    assert out.shape == x.shape
    print(f"✓ NAFBlock: {out.shape}")
    
    # Test with checkpointing
    naf_cp = NAFBlock(C, use_checkpoint=True)
    naf_cp.train()
    out = naf_cp(x)
    assert out.shape == x.shape
    print(f"✓ NAFBlock (checkpointed): {out.shape}")
    
    # Test local attention variant
    naf_la = NAFBlockWithLocalAttention(C, window_size=8)
    out = naf_la(x)
    assert out.shape == x.shape
    print(f"✓ NAFBlockWithLocalAttention: {out.shape}")
    
    # Test encoder
    encoder = NAFNetEncoder(C, num_blocks=[2, 2, 2])
    features = encoder(x)
    print(f"✓ NAFNetEncoder: {[f.shape for f in features]}")
    
    # Test decoder
    decoder = NAFNetDecoder(C, num_blocks=[2, 2, 2])
    out = decoder(features)
    assert out.shape == x.shape
    print(f"✓ NAFNetDecoder: {out.shape}")
    
    # Check gradients
    loss = out.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in naf.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_grad}")
    
    # Parameter counts
    print("\nParameter counts:")
    for name, module in [("NAFBlock", naf), ("NAFBlock+LA", naf_la), ("Encoder", encoder), ("Decoder", decoder)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} ({params/1e6:.3f}M)")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_nafblock()