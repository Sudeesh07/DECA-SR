"""
HCPF: Hierarchical Cross-View Pyramid Fusion for Stereo Image Super-Resolution

Key improvements over original:
1. Vectorized correlation volume computation (no Python loop)
2. Proper upsampling alignment (learned vs bilinear)
3. Bidirectional support
4. Disparity guidance propagation
5. Memory-efficient correlation
6. Removed einops dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .deca import DECA, LayerNorm2d

class GroupWiseCorrelation(nn.Module):
    """
    Group-wise correlation volume for efficient stereo matching at coarse scale.
    
    Splits features into groups and computes correlation for each disparity level.
    Much more efficient than full attention for large disparity ranges.
    
    Complexity: O(H * W * D * G) where D=max_disp, G=num_groups
    
    Args:
        channels: Number of feature channels
        num_groups: Number of channel groups for correlation
        max_disp: Maximum disparity to search (in pixels at current resolution)
    """
    def __init__(self, channels: int, num_groups: int = 8, max_disp: int = 48):
        super().__init__()
        self.num_groups = num_groups
        self.max_disp = max_disp
        self.group_channels = channels // num_groups
        
        assert channels % num_groups == 0, f"channels {channels} must be divisible by num_groups {num_groups}"

        # Aggregate correlation volume to features
        self.aggregate = nn.Sequential(
            nn.Conv2d(num_groups * max_disp, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Learnable residual scale
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
        G = self.num_groups
        Cg = self.group_channels
        max_d = min(self.max_disp, W)

        # Reshape to groups: [B, G, Cg, H, W]
        fl = f_left.view(B, G, Cg, H, W)
        fr = f_right.view(B, G, Cg, H, W)
        
        # Normalize for stable correlation
        fl = F.normalize(fl, dim=2)
        fr = F.normalize(fr, dim=2)

        # Build correlation volume efficiently
        # Pre-allocate output tensor
        corr_vol = f_left.new_zeros(B, G * self.max_disp, H, W)
        
        for d in range(max_d):
            if d == 0:
                # Zero disparity: direct correlation
                corr = (fl * fr).mean(dim=2)  # [B, G, H, W]
            else:
                # Shift right image by disparity d
                # fl[:,:,:,:,d:] correlates with fr[:,:,:,:,:W-d]
                corr = (fl[:, :, :, :, d:] * fr[:, :, :, :, :W-d]).mean(dim=2)
                # Pad left side (occluded region) with zeros
                corr = F.pad(corr, (d, 0), mode='constant', value=0)
            
            corr_vol[:, d * G:(d + 1) * G, :, :] = corr
        
        # Aggregate to feature space
        out = self.aggregate(corr_vol)
        
        return f_left + out * self.beta


class GroupWiseCorrelationVectorized(nn.Module):
    """
    Fully vectorized group-wise correlation (faster but more memory).
    
    Uses unfold operation to avoid Python loop over disparities.
    """
    def __init__(self, channels: int, num_groups: int = 8, max_disp: int = 48):
        super().__init__()
        self.num_groups = num_groups
        self.max_disp = max_disp
        self.group_channels = channels // num_groups

        self.aggregate = nn.Sequential(
            nn.Conv2d(num_groups * max_disp, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_left.shape
        G = self.num_groups
        Cg = self.group_channels
        D = min(self.max_disp, W)

        # Reshape and normalize
        fl = F.normalize(f_left.view(B, G, Cg, H, W), dim=2)  # [B, G, Cg, H, W]
        fr = F.normalize(f_right.view(B, G, Cg, H, W), dim=2)
        
        # Pad right features on the left for shifting
        fr_padded = F.pad(fr, (D-1, 0), mode='constant', value=0)  # [B, G, Cg, H, W+D-1]
        
        # Unfold to get all disparity shifts: [B, G, Cg, H, D, W]
        fr_unfolded = fr_padded.unfold(dimension=4, size=W, step=1)  # [B, G, Cg, H, D, W]
        fr_unfolded = fr_unfolded.flip(dims=[4])  # Reverse disparity order
        
        # Compute correlation for all disparities at once
        # fl: [B, G, Cg, H, W] -> [B, G, Cg, H, 1, W]
        fl_expanded = fl.unsqueeze(4)
        
        # Correlation: [B, G, Cg, H, D, W] -> mean over Cg -> [B, G, H, D, W]
        corr = (fl_expanded * fr_unfolded).mean(dim=2)
        
        # Rearrange to [B, G*D, H, W]
        corr = corr.permute(0, 1, 3, 2, 4).reshape(B, G * D, H, W)

        # Keep aggregate input channels fixed at num_groups * self.max_disp.
        if D < self.max_disp:
            corr_padded = f_left.new_zeros(B, G * self.max_disp, H, W)
            corr_padded[:, :G * D, :, :] = corr
            corr = corr_padded
        
        # Aggregate
        out = self.aggregate(corr)
        
        return f_left + out * self.beta


class FullEpipolarAttention(nn.Module):
    """
    Full epipolar attention at fine scale.
    
    Computes attention between each position in left view and all positions
    on the same scanline in right view.
    
    Complexity: O(H * W * W) - use only at full resolution or small images
    
    Args:
        channels: Number of feature channels
        num_heads: Number of attention heads
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert channels % num_heads == 0

        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Learnable residual scale
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

        q = self.q_proj(fl)  # [B, C, H, W]
        k = self.k_proj(fr)  # [B, C, H, W]
        v = self.v_proj(fr)  # [B, C, H, W]

        # Reshape for row-wise attention
        # [B, C, H, W] -> [B, num_heads, head_dim, H, W] -> [B*H, num_heads, W, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k = k.view(B, self.num_heads, self.head_dim, H, W)
        v = v.view(B, self.num_heads, self.head_dim, H, W)
        
        q = q.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        k = k.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        v = v.permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        
        # Attention: [B*H, num_heads, W, W]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B*H, num_heads, W, head_dim]
        
        # Reshape back
        out = out.view(B, H, self.num_heads, W, self.head_dim)
        out = out.permute(0, 2, 4, 1, 3).reshape(B, C, H, W)
        
        out = self.out_proj(out)

        return f_left + out * self.beta




class HCPF(nn.Module):
    """
    Hierarchical Cross-View Pyramid Fusion
    
    Multi-scale cross-view attention with coarse-to-fine guidance:
    - 1/4 resolution: Group-wise correlation (efficient, large receptive field)
    - 1/2 resolution: DECA (adaptive sampling)
    - 1/1 resolution: Full epipolar attention (precise)
    
    Architecture:
        Input (L, R)
            ↓ downsample 4×
        Coarse (1/4): GroupWiseCorrelation
            ↓ guide + upsample 2×
        Mid (1/2): DECA
            ↓ guide + upsample 2×
        Fine (1×): FullEpipolarAttention
            ↓
        Merge all scales → Output
    
    Args:
        channels: Number of feature channels
        k: Number of sampling points for DECA
        coarse_max_disp: Maximum disparity at coarse level
        use_vectorized_corr: Use faster vectorized correlation
        use_learned_upsampling: Use learned upsampling instead of bilinear
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        coarse_max_disp: int = 24,
        use_vectorized_corr: bool = False,
        use_learned_upsampling: bool = True
    ):
        super().__init__()
        self.use_learned_upsampling = use_learned_upsampling

        # Downsampling
        self.down_2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            LayerNorm2d(channels),
        )
        self.down_4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            LayerNorm2d(channels),
        )
        
        # Upsampling
        if use_learned_upsampling:
            self.up_2 = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
            self.up_4 = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            self.up_2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            self.up_4 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Multi-scale fusion modules
        CorrelationClass = GroupWiseCorrelationVectorized if use_vectorized_corr else GroupWiseCorrelation
        self.coarse_fusion = CorrelationClass(channels, num_groups=8, max_disp=coarse_max_disp)
        self.mid_fusion = DECA(channels, k=k, max_disparity=coarse_max_disp * 2)  # 2× resolution
        self.fine_fusion = FullEpipolarAttention(channels, num_heads=4)
        
        # Guidance projection (with channel adjustment)
        self.coarse_to_mid = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.mid_to_fine = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Multi-scale merge with attention
        self.scale_attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Softmax(dim=1),
        )
        
        self.merge = nn.Sequential(
            LayerNorm2d(channels * 3),
            nn.Conv2d(channels * 3, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(
        self, 
        f_left: torch.Tensor, 
        f_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = f_left.shape

        # ===== Coarse Level (1/4 resolution) =====
        fl_2 = self.down_2(f_left)
        fr_2 = self.down_2(f_right)
        fl_4 = self.down_4(fl_2)
        fr_4 = self.down_4(fr_2)
        
        coarse_out = self.coarse_fusion(fl_4, fr_4)
        
        # Prepare guidance for mid level
        coarse_guide = self.coarse_to_mid(coarse_out)
        if self.use_learned_upsampling:
            coarse_guide_up = self.up_4(coarse_guide)
            if coarse_guide_up.shape[-2:] != fl_2.shape[-2:]:
                coarse_guide_up = F.interpolate(
                    coarse_guide_up,
                    size=fl_2.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
        else:
            coarse_guide_up = F.interpolate(coarse_guide, size=(H//2, W//2), mode='bilinear', align_corners=False)

        # ===== Mid Level (1/2 resolution) =====
        fl_2_guided = fl_2 + coarse_guide_up
        mid_out = self.mid_fusion(fl_2_guided, fr_2)
        
        # Prepare guidance for fine level
        mid_guide = self.mid_to_fine(mid_out)
        if self.use_learned_upsampling:
            mid_guide_up = self.up_2(mid_guide)
            if mid_guide_up.shape[-2:] != f_left.shape[-2:]:
                mid_guide_up = F.interpolate(
                    mid_guide_up,
                    size=f_left.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
        else:
            mid_guide_up = F.interpolate(mid_guide, size=(H, W), mode='bilinear', align_corners=False)

        # ===== Fine Level (full resolution) =====
        fl_guided = f_left + mid_guide_up
        fine_out = self.fine_fusion(fl_guided, f_right)

        # ===== Multi-scale Merge =====
        # Upsample coarse and mid to full resolution
        coarse_up = F.interpolate(coarse_out, size=(H, W), mode='bilinear', align_corners=False)
        mid_up = F.interpolate(mid_out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate all scales
        multi_scale = torch.cat([coarse_up, mid_up, fine_out], dim=1)
        
        # Learned scale attention
        scale_weights = self.scale_attention(multi_scale)  # [B, 3, H, W]
        
        # Weighted combination + merge
        weighted = (
            scale_weights[:, 0:1] * coarse_up +
            scale_weights[:, 1:2] * mid_up +
            scale_weights[:, 2:3] * fine_out
        )
        
        merged = self.merge(multi_scale) + weighted

        return merged


class BidirectionalHCPF(nn.Module):
    """
    Bidirectional HCPF that processes both views.
    
    Uses weight-shared HCPF modules for efficiency.
    """
    def __init__(self, channels: int, k: int = 8, **kwargs):
        super().__init__()
        self.hcpf = HCPF(channels, k, **kwargs)
        
    def forward(
        self, 
        f_left: torch.Tensor, 
        f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (enhanced_left, enhanced_right)
        """
        # L->R
        enhanced_left = self.hcpf(f_left, f_right)
        
        # R->L (flip, process, flip back)
        f_right_flip = torch.flip(f_right, dims=[-1])
        f_left_flip = torch.flip(f_left, dims=[-1])
        enhanced_right_flip = self.hcpf(f_right_flip, f_left_flip)
        enhanced_right = torch.flip(enhanced_right_flip, dims=[-1])
        
        return enhanced_left, enhanced_right


# =============================================================================
# Unit Tests
# =============================================================================

def test_hcpf():
    """Test all HCPF components."""
    print("Testing HCPF Components...")
    print("=" * 50)
    
    B, C, H, W = 2, 64, 64, 128
    f_left = torch.randn(B, C, H, W)
    f_right = torch.randn(B, C, H, W)
    
    # Test GroupWiseCorrelation
    gwc = GroupWiseCorrelation(C, num_groups=8, max_disp=24)
    out = gwc(f_left, f_right)
    assert out.shape == f_left.shape
    print(f"✓ GroupWiseCorrelation: {out.shape}")
    
    # Test Vectorized version
    gwcv = GroupWiseCorrelationVectorized(C, num_groups=8, max_disp=24)
    out_v = gwcv(f_left, f_right)
    assert out_v.shape == f_left.shape
    print(f"✓ GroupWiseCorrelationVectorized: {out_v.shape}")
    
    # Test FullEpipolarAttention
    fea = FullEpipolarAttention(C, num_heads=4)
    out = fea(f_left, f_right)
    assert out.shape == f_left.shape
    print(f"✓ FullEpipolarAttention: {out.shape}")
    
    # Test DECA
    deca = DECA(C, k=8, num_heads=4)
    out = deca(f_left, f_right)
    assert out.shape == f_left.shape
    print(f"✓ DECA: {out.shape}")
    
    # Test full HCPF
    hcpf = HCPF(C, k=8)
    out = hcpf(f_left, f_right)
    assert out.shape == f_left.shape
    print(f"✓ HCPF: {out.shape}")
    
    # Test bidirectional
    bi_hcpf = BidirectionalHCPF(C, k=8)
    out_l, out_r = bi_hcpf(f_left, f_right)
    assert out_l.shape == f_left.shape and out_r.shape == f_right.shape
    print(f"✓ BidirectionalHCPF: L={out_l.shape}, R={out_r.shape}")
    
    # Check gradients
    loss = out.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in hcpf.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_grad}")
    
    # Parameter counts
    print("\nParameter counts:")
    for name, module in [("GWC", gwc), ("FEA", fea), ("DECA", deca), ("HCPF", hcpf)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} ({params/1e6:.3f}M)")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_hcpf()