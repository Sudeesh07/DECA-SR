"""
DECA: Deformable Epipolar Cross-Attention for Stereo Image Super-Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Einops-free tensor operations
# =============================================================================

def rearrange_to_multihead(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, num_heads, head_dim, H*W]"""
    B, C, H, W = x.shape
    return x.view(B, num_heads, head_dim, H * W)


def rearrange_sampled_to_multihead(x: torch.Tensor, B: int, K: int, num_heads: int, head_dim: int) -> torch.Tensor:
    """[K, B, C, H, W] -> [B, num_heads, head_dim, H*W, K]"""
    # x is a list stacked as [K, B, C, H, W]
    # -> [B, K, C, H, W] -> [B, K, h, d, H, W]
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    x = x.view(B, K, num_heads, head_dim, x.shape[3], x.shape[4])
    # -> [B, h, d, H, W, K] -> [B, h, d, H*W, K]
    x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
    H, W = x.shape[3], x.shape[4]
    return x.reshape(B, num_heads, head_dim, H * W, K)


def rearrange_back_to_spatial(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B, num_heads, head_dim, H*W] -> [B, C, H, W]"""
    B, num_heads, head_dim, N = x.shape
    return x.reshape(B, num_heads * head_dim, H, W)

def groupwise_correlation(q, k, num_groups=8):
    """
    q, k: [B, C, H, W]
    returns: [B, num_groups, H, W]
    """
    B, C, H, W = q.shape
    assert C % num_groups == 0

    q = q.view(B, num_groups, C // num_groups, H, W)
    k = k.view(B, num_groups, C // num_groups, H, W)

    corr = (q * k).mean(dim=2)  # [B, G, H, W]
    return corr


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


class DECAOffsetPredictor(nn.Module):
    """
    Predicts K sampling offsets along the epipolar line (horizontal for rectified stereo).

        channels: Number of input channels
        k: Number of sampling points along epipolar line
        max_disparity: Maximum disparity in pixels (clamps offset range)
        num_corr_shifts: Number of horizontal shifts for correlation computation
    """
    def __init__(self, channels: int, k: int = 8, max_disparity: int = 32, num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.k = k
        self.max_disparity = max_disparity
        self.num_corr_shifts = num_corr_shifts

        if num_corr_shifts < 1:
            raise ValueError(f"num_corr_shifts must be >= 1, got {num_corr_shifts}")

        reduced = max(1, channels // 4)

        self.query_proj = nn.Conv2d(channels, reduced, 1)
        self.key_proj = nn.Conv2d(channels, reduced, 1)

        corr_in_channels = num_corr_shifts + reduced

        self.corr_refine = nn.Sequential(
            nn.Conv2d(corr_in_channels, reduced, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced, reduced, 3, padding=1),
            nn.GELU(),
        )

        self.offset_head = nn.Conv2d(reduced, k, 1)

        nn.init.zeros_(self.offset_head.weight)
        assert self.offset_head.bias is not None
        nn.init.zeros_(self.offset_head.bias)
        with torch.no_grad():
            target_normalized = torch.linspace(-0.8, 0.8, k)
            self.offset_head.bias.copy_(torch.atanh(target_normalized))

        max_shift = max(1, max_disparity // 4)
        self.register_buffer(
            'shifts',
            torch.round(torch.linspace(-max_shift, max_shift, num_corr_shifts)).long()
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]

        Returns:
            offsets: Horizontal offsets in pixels [B, K, H, W]
        """
        B, C, H, W = f_left.shape

        q = self.query_proj(f_left)   # [B, reduced, H, W]
        k = self.key_proj(f_right)    # [B, reduced, H, W]

        corr_planes = []
        shifts = getattr(self, 'shifts')
        for shift in shifts:
            shift_val = int(shift.item())
            if shift_val == 0:
                k_shifted = k
            elif shift_val > 0:
                k_shifted = F.pad(k, (shift_val, 0, 0, 0), mode='replicate')[:, :, :, :W]
            else:
                k_shifted = F.pad(k, (0, -shift_val, 0, 0), mode='replicate')[:, :, :, -shift_val:]
                if k_shifted.shape[-1] < W:
                    k_shifted = F.pad(k_shifted, (0, W - k_shifted.shape[-1]), mode='replicate')
                k_shifted = k_shifted[:, :, :, :W]

            corr_at_shift = (q * k_shifted).sum(dim=1, keepdim=True) * (q.shape[1] ** -0.5)
            corr_planes.append(corr_at_shift)

        corr = torch.cat(corr_planes, dim=1)  # [B, num_corr_shifts, H, W]

        corr = torch.cat([corr, q], dim=1)  # [B, num_corr_shifts + reduced, H, W]

        corr = self.corr_refine(corr)
        offsets = self.offset_head(corr)  # [B, K, H, W]
        # offsets = offsets.clamp(-self.max_disparity, self.max_disparity)
        offsets = torch.tanh(offsets) * self.max_disparity  # Smoothly clamp to [-max_disparity, max_disparity]

        return offsets


class DECA(nn.Module):
    """
    Deformable Epipolar Cross-Attention (DECA)

    Performs cross-view attention along epipolar lines with content-adaptive
    sampling positions. For rectified stereo pairs, epipolar lines are horizontal.

    Complexity: O(H*W*K) vs O(H*W*W) for full epipolar attention
    Args:
        channels: Number of feature channels
        k: Number of sampling points per query (default: 8)
        num_heads: Number of attention heads (default: 4)
        max_disparity: Maximum disparity in pixels (default: 32)
        use_occlusion_gate: Whether to use soft occlusion gating (default: False)
        num_corr_shifts: Number of horizontal shifts for offset predictor correlation
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        use_occlusion_gate: bool = False,
        num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.channels = channels
        self.k = k
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_disparity = max_disparity
        self.use_occlusion_gate = use_occlusion_gate

        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        # Normalization
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        # Q/K/V projections
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # Offset predictor (with fixes #1 and #6)
        self.offset_predictor = DECAOffsetPredictor(
            channels, k, max_disparity, num_corr_shifts
        )

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # Optional occlusion gate
        if use_occlusion_gate:
            self.occlusion_gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels // 4, 1),
                nn.GELU(),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )

        self.log_scale = nn.Parameter(
            torch.tensor(self.head_dim ** -0.5).log()
        )

    def _get_sampling_grid_single(
        self,
        offsets_k: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct sampling grid for a single set of offsets (one of the K points).

        Args:
            offsets_k: [B, H, W] horizontal offsets in pixels for one sampling point
            H, W: Spatial dimensions
            device: Target device

        Returns:
            grid: [B, H, W, 2] sampling grid for grid_sample
        """
        B = offsets_k.shape[0]

        # Create base grid
        grid_y = torch.linspace(-1, 1, H, device=device)
        grid_x = torch.linspace(-1, 1, W, device=device)
        base_y, base_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

        # Normalize offsets to [-1, 1] range (guard W == 1)
        if W > 1:
            offset_norm = offsets_k * (2.0 / (W - 1))  # [B, H, W]
        else:
            offset_norm = torch.zeros_like(offsets_k)

        # Compute sampled x coordinates
        sample_x = base_x.unsqueeze(0) + offset_norm  # [B, H, W]
        sample_x = sample_x.clamp(-1, 1)

        # y stays the same (epipolar constraint)
        sample_y = base_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # Stack: [B, H, W, 2]
        grid = torch.stack([sample_x, sample_y], dim=-1)
        return grid

    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = f_left.shape

        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        offsets = self.offset_predictor(fl, fr)  # [B, K, H, W]

        queries = self.q_proj(fl)
        keys = self.k_proj(fr)
        values = self.v_proj(fr)

        # Create base grid ONCE, reuse for all K sampling points
        grid_y = torch.linspace(-1, 1, H, device=f_left.device)
        grid_x = torch.linspace(-1, 1, W, device=f_left.device)
        base_y, base_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        # Expand to batch: [B, H, W]
        base_x = base_x.unsqueeze(0).expand(B, -1, -1)
        base_y = base_y.unsqueeze(0).expand(B, -1, -1)

        # Precompute the pixel-to-grid conversion factor
        if W > 1:
            px_to_grid = 2.0 / (W - 1)
        else:
            px_to_grid = 0.0

        keys_sampled_list = []
        vals_sampled_list = []
        for i in range(self.k):
            # Only compute the offset-shifted x coordinate per iteration
            sample_x = (base_x + offsets[:, i, :, :] * px_to_grid).clamp(-1, 1)
            grid_i = torch.stack([sample_x, base_y], dim=-1)  # [B, H, W, 2]

            ks = F.grid_sample(
                keys, grid_i,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            vs = F.grid_sample(
                values, grid_i,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            keys_sampled_list.append(ks)
            vals_sampled_list.append(vs)

        # Stack: [K, B, C, H, W]
        keys_stacked = torch.stack(keys_sampled_list, dim=0)
        vals_stacked = torch.stack(vals_sampled_list, dim=0)

        # Reshape for attention: [B, num_heads, head_dim, H*W, K]
        keys_sampled = rearrange_sampled_to_multihead(
            keys_stacked, B, self.k, self.num_heads, self.head_dim
        )
        vals_sampled = rearrange_sampled_to_multihead(
            vals_stacked, B, self.k, self.num_heads, self.head_dim
        )

        # Reshape queries: [B, num_heads, head_dim, H*W]
        q = rearrange_to_multihead(queries, self.num_heads, self.head_dim)

        scale = self.log_scale.exp()

        # Compute attention: Q @ K^T * scale
        # q: [B, h, d, N], k: [B, h, d, N, K] -> attn: [B, h, N, K]
        attn = torch.einsum('bhdn, bhdnk -> bhnk', q, keys_sampled) * scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        # attn: [B, h, N, K], v: [B, h, d, N, K] -> out: [B, h, d, N]
        out = torch.einsum('bhnk, bhdnk -> bhdn', attn, vals_sampled)

        # Reshape back to spatial
        out = rearrange_back_to_spatial(out, H, W)
        out = self.out_proj(out)

        # Optional occlusion gating
        if self.use_occlusion_gate:
            gate = self.occlusion_gate(torch.cat([f_left, out], dim=1))
            out = gate * out

        # Residual connection
        return f_left + out


class BidirectionalDECA(nn.Module):
    """
    Bidirectional DECA that enhances both views simultaneously.
    Performs DECA from left to right and from right to left in parallel, allowing each to learn offsets independently. This can capture asymmetric occlusion patterns and disparities.
     Complexity is roughly 2× a single DECA, but provides richer cross-view interactions and can improve performance in occluded regions by allowing the "weaker" view to also attend to the "stronger" view.
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        use_occlusion_gate: bool = False,
        num_corr_shifts: int = 9,
    ):
        super().__init__()

        # L->R attention: enhance left using right
        self.deca_l2r = DECA(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            use_occlusion_gate=use_occlusion_gate,
            num_corr_shifts=num_corr_shifts,
        )

        # R->L attention: enhance right using left
        # Separate module so it can learn R->L offset patterns independently.
        # Offsets for R->L are typically opposite sign from L->R.
        self.deca_r2l = DECA(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            use_occlusion_gate=use_occlusion_gate,
            num_corr_shifts=num_corr_shifts,
        )

    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]

        Returns:
            Tuple of (enhanced_left, enhanced_right)
        """
        enhanced_left = self.deca_l2r(f_left, f_right)
        enhanced_right = self.deca_r2l(f_right, f_left)

        return enhanced_left, enhanced_right


# =============================================================================
# Unit Tests
# =============================================================================

def test_deca():
    """Basic functionality test."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, C, H, W = 2, 64, 32, 64
    f_left = torch.randn(B, C, H, W, device=device)
    f_right = torch.randn(B, C, H, W, device=device)

    # Test basic DECA
    deca = DECA(channels=C, k=8, num_heads=4).to(device)
    out = deca(f_left, f_right)
    assert out.shape == f_left.shape, f"Shape mismatch: {out.shape} vs {f_left.shape}"
    print(f"✓ DECA forward pass: {out.shape}")

    # Test with occlusion gate
    deca_gate = DECA(channels=C, k=8, num_heads=4, use_occlusion_gate=True).to(device)
    out_gate = deca_gate(f_left, f_right)
    assert out_gate.shape == f_left.shape
    print(f"✓ DECA with occlusion gate: {out_gate.shape}")

    # Test bidirectional
    bideca = BidirectionalDECA(channels=C, k=8, num_heads=4).to(device)
    out_l, out_r = bideca(f_left, f_right)
    assert out_l.shape == f_left.shape and out_r.shape == f_right.shape
    print(f"✓ Bidirectional DECA: L={out_l.shape}, R={out_r.shape}")

    # Check gradients flow
    loss = out.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in deca.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_grad}")

    # Param count
    params_deca = sum(p.numel() for p in deca.parameters())
    params_bideca = sum(p.numel() for p in bideca.parameters())
    print(f"✓ DECA params: {params_deca:,} ({params_deca/1e6:.2f}M)")
    print(f"✓ BidirectionalDECA params: {params_bideca:,} ({params_bideca/1e6:.2f}M)")

    # Test that learnable scale is being used
    assert deca.log_scale.grad is not None, "log_scale should receive gradients"
    print(f"✓ Learnable scale has gradient: {deca.log_scale.grad.item():.6f}")

    # Test offset predictor multi-shift correlation
    predictor = deca.offset_predictor
    offsets_test = predictor(f_left, f_right)
    assert offsets_test.shape == (B, 8, H, W), f"Offset shape: {offsets_test.shape}"
    assert offsets_test.abs().max() <= deca.max_disparity, "Offsets should be clamped"
    print(f"✓ Offset predictor output: {offsets_test.shape}, range [{offsets_test.min():.1f}, {offsets_test.max():.1f}]")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_deca()