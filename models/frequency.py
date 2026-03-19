"""
Frequency-Aware Refinement Module for Stereo Image Super-Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.weight.view(1, -1, 1, 1) * (x - mean) / (var + self.eps).sqrt() + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet-style SimpleGate: splits channels and multiplies."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Simplified NAFBlock for spatial processing."""
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * expansion, 1)
        self.conv2 = nn.Conv2d(channels * expansion, channels * expansion, 3, 
                               padding=1, groups=channels * expansion)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(channels * expansion // 2, channels, 1)
        
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * expansion, 1)
        self.conv5 = nn.Conv2d(channels * expansion // 2, channels, 1)
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = shortcut + x * self.beta
        
        # Channel mixing
        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = shortcut + x * self.gamma
        
        return x


class FastFourierConvBlock(nn.Module):
    """
    Frequency-domain convolution block.
    
    Processes features in Fourier domain to capture global context
    and enhance high-frequency details.
    
    Args:
        channels: Number of input channels
        use_spectral_attention: Whether to use frequency-band attention
    """
    def __init__(self, channels: int, use_spectral_attention: bool = True):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.use_spectral_attention = use_spectral_attention
        
        # Process real and imaginary parts together
        # Note: rfft2 output has shape [B, C, H, W//2+1] (complex)
        self.conv_freq = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 1),
        )
        
        # Spectral attention: weight different frequency bands
        if use_spectral_attention:
            self.spectral_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels * 2, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, channels * 2),
                nn.Sigmoid(),
            )
        
        # Learnable residual scaling
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        
        B, C, H, W = x.shape
        
        # Transform to frequency domain
        # Output shape: [B, C, H, W//2+1] (complex tensor)
        x_freq = torch.fft.rfft2(x, norm="ortho")
        
        # Split into real and imaginary parts
        # Each has shape [B, C, H, W//2+1]
        x_real = x_freq.real
        x_imag = x_freq.imag
        
        # Concatenate along channel dimension: [B, 2C, H, W//2+1]
        x_cat = torch.cat([x_real, x_imag], dim=1)
        
        # Process in frequency domain
        x_cat = self.conv_freq(x_cat)
        
        # Apply spectral attention if enabled
        if self.use_spectral_attention:
            attn = self.spectral_attn(x_cat)  # [B, 2C]
            attn = attn.view(B, C * 2, 1, 1)
            x_cat = x_cat * attn
        
        # Split back to real and imaginary
        x_real, x_imag = x_cat.chunk(2, dim=1)
        
        # Reconstruct complex tensor
        x_freq = torch.complex(x_real, x_imag)
        
        # Transform back to spatial domain
        # Must specify output size to handle odd dimensions
        x = torch.fft.irfft2(x_freq, s=(H, W), norm="ortho")
        
        return shortcut + x * self.beta


class FrequencyCrossViewFusion(nn.Module):
    """
    Cross-view fusion in frequency domain.
    
    Fuses amplitude information from both views while preserving
    phase from the target view (phase carries structural info).
    
    Key insight: Stereo views share similar low-frequency content
    but differ in high-frequency details due to viewpoint change.
    
    Args:
        channels: Number of feature channels
        use_phase_correlation: Also correlate phase information
    """
    def __init__(self, channels: int, use_phase_correlation: bool = False):
        super().__init__()
        self.use_phase_correlation = use_phase_correlation
        
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        
        # Amplitude correlation network
        # Learns which frequency components to transfer from right view
        self.amp_corr_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        
        # Optional phase correlation
        if use_phase_correlation:
            self.phase_corr_net = nn.Sequential(
                nn.Conv2d(channels * 4, channels, 1),  # real_l, imag_l, real_r, imag_r
                nn.GELU(),
                nn.Conv2d(channels, channels * 2, 1),  # output real and imag offsets
                nn.Tanh(),  # bounded corrections
            )
            self.phase_scale = nn.Parameter(torch.zeros(1))
        
        # High-frequency enhancement
        self.hf_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]
            
        Returns:
            Enhanced left features [B, C, H, W]
        """
        B, C, H, W = f_left.shape
        
        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)
        
        # Transform to frequency domain
        fl_freq = torch.fft.rfft2(fl, norm="ortho")
        fr_freq = torch.fft.rfft2(fr, norm="ortho")
        
        # Extract amplitude and phase
        amp_l = fl_freq.abs()
        amp_r = fr_freq.abs()
        
        # Compute normalized phase (unit complex number)
        # Add small epsilon to avoid division by zero
        phase_l = fl_freq / (amp_l + 1e-8)
        
        # Concatenate amplitudes for correlation
        amp_cat = torch.cat([amp_l, amp_r], dim=1)
        
        # Predict correlation weights (which frequencies to transfer)
        corr_weight = self.amp_corr_net(amp_cat)
        
        # Fuse amplitudes: weighted combination
        # corr_weight=0: keep left, corr_weight=1: use right
        fused_amp = amp_l + corr_weight * (amp_r - amp_l)
        
        # High-frequency enhancement
        # Create high-pass mask (suppress DC and low frequencies)
        freq_h, freq_w = amp_l.shape[2], amp_l.shape[3]
        
        # Distance from DC component (top-left corner for rfft2)
        y_coords = torch.arange(freq_h, device=amp_l.device).float()
        x_coords = torch.arange(freq_w, device=amp_l.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Wrap y coordinates for proper frequency distance
        yy = torch.min(yy, freq_h - yy)
        
        # Normalize distances
        dist = torch.sqrt(yy**2 + xx**2)
        dist = dist / dist.max()
        
        # High-pass weight (higher weight for high frequencies)
        hf_weight = dist.view(1, 1, freq_h, freq_w)
        
        # Enhance high frequencies
        hf_amp = fused_amp * hf_weight
        hf_enhanced = self.hf_enhance(hf_amp)
        fused_amp = fused_amp + hf_enhanced * 0.1  # Small contribution
        
        # Optional phase correlation
        if self.use_phase_correlation:
            phase_cat = torch.cat([
                fl_freq.real, fl_freq.imag,
                fr_freq.real, fr_freq.imag
            ], dim=1)
            phase_offset = self.phase_corr_net(phase_cat)
            phase_real_off, phase_imag_off = phase_offset.chunk(2, dim=1)
            
            # Apply small phase correction
            scale = torch.sigmoid(self.phase_scale) * 0.1
            phase_l = phase_l + torch.complex(
                phase_real_off * scale, 
                phase_imag_off * scale
            )
            # Re-normalize to unit magnitude
            phase_l = phase_l / (phase_l.abs() + 1e-8)
        
        # Reconstruct with fused amplitude and left phase
        fused_freq = fused_amp * phase_l
        
        # Transform back to spatial domain
        fused = torch.fft.irfft2(fused_freq, s=(H, W), norm="ortho")
        
        return f_left + fused * self.beta


class HighFrequencyEnhancementBranch(nn.Module):
    """
    Dedicated branch for high-frequency detail enhancement.
    
    Uses wavelet-like decomposition to explicitly process
    high-frequency components.
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Learnable high-pass filter
        self.hf_filter = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        
        # Initialize as Laplacian-like filter
        with torch.no_grad():
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32)
            kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3) / 4.0
            self.hf_filter.weight.copy_(kernel)
        
        # Process high-frequency features
        self.hf_process = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract high frequencies
        hf = self.hf_filter(x)
        
        # Process
        hf = self.hf_process(hf)
        
        return x + hf * self.beta


class FrequencyAwareRefinement(nn.Module):
    """
    Dual-stream refinement module with spatial and frequency processing.
    
    Architecture:
        Input (L, R)
            |-- Spatial Stream: NAFBlocks x N
            |   |-- Spatial fusion with right view
            |-- Frequency Stream: FFCBlocks x M
            |   |-- Frequency cross-view fusion
            |-- High-Frequency Branch (optional)
                |-- Dedicated HF enhancement
        
        Merge all streams -> Output
    
    Args:
        channels: Number of feature channels
        spatial_blocks: Number of NAFBlocks in spatial stream
        freq_blocks: Number of FFCBlocks in frequency stream
        use_hf_branch: Whether to use dedicated HF enhancement
        use_bidirectional: Process both views
    """
    def __init__(
        self,
        channels: int,
        spatial_blocks: int = 4,
        freq_blocks: int = 2,
        use_hf_branch: bool = True,
        use_bidirectional: bool = False
    ):
        super().__init__()
        self.use_hf_branch = use_hf_branch
        self.use_bidirectional = use_bidirectional
        
        # Spatial stream
        self.spatial_stream = nn.Sequential(
            *[NAFBlock(channels) for _ in range(spatial_blocks)]
        )
        
        # Frequency stream
        self.freq_stream = nn.Sequential(
            *[FastFourierConvBlock(channels) for _ in range(freq_blocks)]
        )
        
        # Cross-view fusion modules
        self.freq_fusion = FrequencyCrossViewFusion(channels)
        
        self.spatial_fusion = nn.Sequential(
            LayerNorm2d(channels * 2),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        # Optional HF branch
        if use_hf_branch:
            self.hf_branch = HighFrequencyEnhancementBranch(channels)
            num_streams = 3
        else:
            num_streams = 2
        
        # Merge all streams
        self.merge = nn.Sequential(
            LayerNorm2d(channels * num_streams),
            nn.Conv2d(channels * num_streams, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor):
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]
            
        Returns:
            Enhanced left features (or tuple of both if bidirectional)
        """
        # Spatial stream: process left, fuse with right
        spatial_l = self.spatial_stream(f_left)
        spatial_fused = self.spatial_fusion(torch.cat([spatial_l, f_right], dim=1))
        
        # Frequency stream: process left, cross-view fusion
        freq_l = self.freq_stream(f_left)
        freq_fused = self.freq_fusion(freq_l, f_right)
        
        # Collect streams to merge
        streams = [spatial_fused, freq_fused]
        
        # Optional HF branch
        if self.use_hf_branch:
            hf_enhanced = self.hf_branch(f_left)
            streams.append(hf_enhanced)
        
        # Merge all streams
        merged = self.merge(torch.cat(streams, dim=1))
        
        if self.use_bidirectional:
            # Process right view symmetrically
            spatial_r = self.spatial_stream(f_right)
            spatial_fused_r = self.spatial_fusion(torch.cat([spatial_r, f_left], dim=1))
            
            freq_r = self.freq_stream(f_right)
            freq_fused_r = self.freq_fusion(freq_r, f_left)
            
            streams_r = [spatial_fused_r, freq_fused_r]
            
            if self.use_hf_branch:
                hf_enhanced_r = self.hf_branch(f_right)
                streams_r.append(hf_enhanced_r)
            
            merged_r = self.merge(torch.cat(streams_r, dim=1))
            
            return merged, merged_r
        
        return merged


# =============================================================================
# Loss Functions for Frequency-Aware Training
# =============================================================================

class WaveletLoss(nn.Module):
    """
    Wavelet subband loss for high-frequency supervision.
    
    Uses Haar wavelet decomposition to compute losses on
    different frequency subbands (LL, LH, HL, HH).
    """
    def __init__(self, weights: Optional[Tuple[float, ...]] = None):
        super().__init__()
        # Default: emphasize high-frequency subbands
        self.weights = weights or (0.1, 0.3, 0.3, 0.3)  # LL, LH, HL, HH
        
        # Haar wavelet filters
        self.register_buffer('ll_filter', torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5]
        ]).view(1, 1, 2, 2))
        
        self.register_buffer('lh_filter', torch.tensor([
            [-0.5, -0.5],
            [0.5, 0.5]
        ]).view(1, 1, 2, 2))
        
        self.register_buffer('hl_filter', torch.tensor([
            [-0.5, 0.5],
            [-0.5, 0.5]
        ]).view(1, 1, 2, 2))
        
        self.register_buffer('hh_filter', torch.tensor([
            [0.5, -0.5],
            [-0.5, 0.5]
        ]).view(1, 1, 2, 2))
    
    def dwt(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply 2D Haar DWT."""
        B, C, H, W = x.shape
        
        # Expand filters for all channels
        ll = self.ll_filter.expand(C, 1, 2, 2)
        lh = self.lh_filter.expand(C, 1, 2, 2)
        hl = self.hl_filter.expand(C, 1, 2, 2)
        hh = self.hh_filter.expand(C, 1, 2, 2)
        
        # Apply filters with stride 2
        ll_out = F.conv2d(x, ll, stride=2, groups=C)
        lh_out = F.conv2d(x, lh, stride=2, groups=C)
        hl_out = F.conv2d(x, hl, stride=2, groups=C)
        hh_out = F.conv2d(x, hh, stride=2, groups=C)
        
        return ll_out, lh_out, hl_out, hh_out
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute wavelet loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
        """
        pred_subbands = self.dwt(pred)
        target_subbands = self.dwt(target)
        
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for w, p_sub, t_sub in zip(self.weights, pred_subbands, target_subbands):
            loss = loss + w * F.l1_loss(p_sub, t_sub)
        
        return loss


class FrequencyL1Loss(nn.Module):
    """L1 loss in frequency domain."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = torch.fft.rfft2(pred, norm="ortho")
        target_freq = torch.fft.rfft2(target, norm="ortho")
        
        # Loss on both amplitude and phase
        amp_loss = F.l1_loss(pred_freq.abs(), target_freq.abs())
        phase_loss = F.l1_loss(pred_freq.angle(), target_freq.angle())
        
        return amp_loss + 0.1 * phase_loss


# =============================================================================
# Unit Tests
# =============================================================================

def test_frequency_modules():
    """Test all frequency-aware modules."""
    print("Testing Frequency-Aware Modules...")
    print("=" * 50)
    
    B, C, H, W = 2, 64, 32, 64
    f_left = torch.randn(B, C, H, W)
    f_right = torch.randn(B, C, H, W)
    
    # Test FastFourierConvBlock
    ffc = FastFourierConvBlock(C)
    out = ffc(f_left)
    assert out.shape == f_left.shape, f"FFC shape mismatch: {out.shape}"
    print(f"✓ FastFourierConvBlock: {out.shape}")
    
    # Test FrequencyCrossViewFusion
    fcvf = FrequencyCrossViewFusion(C)
    out = fcvf(f_left, f_right)
    assert out.shape == f_left.shape, f"FCVF shape mismatch: {out.shape}"
    print(f"✓ FrequencyCrossViewFusion: {out.shape}")
    
    # Test FrequencyAwareRefinement
    far = FrequencyAwareRefinement(C, spatial_blocks=2, freq_blocks=1)
    out = far(f_left, f_right)
    assert out.shape == f_left.shape, f"FAR shape mismatch: {out.shape}"
    print(f"✓ FrequencyAwareRefinement: {out.shape}")
    
    # Test bidirectional
    far_bi = FrequencyAwareRefinement(C, use_bidirectional=True)
    out_l, out_r = far_bi(f_left, f_right)
    assert out_l.shape == f_left.shape and out_r.shape == f_right.shape
    print(f"✓ Bidirectional FAR: L={out_l.shape}, R={out_r.shape}")
    
    # Test losses
    pred = torch.randn(B, 3, H, W)
    target = torch.randn(B, 3, H, W)
    
    wavelet_loss = WaveletLoss()
    wl = wavelet_loss(pred, target)
    print(f"✓ WaveletLoss: {wl.item():.4f}")
    
    freq_loss = FrequencyL1Loss()
    fl = freq_loss(pred, target)
    print(f"✓ FrequencyL1Loss: {fl.item():.4f}")
    
    # Param counts
    print("\nParameter counts:")
    for name, module in [("FFC", ffc), ("FCVF", fcvf), ("FAR", far)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} ({params/1e6:.3f}M)")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_frequency_modules()