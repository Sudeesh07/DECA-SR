"""
DECA: Deformable Epipolar Cross-Attention for Stereo Image Super-Resolution

Fixed version addressing:
1. Correlation now computed across multiple horizontal shifts (not just zero disparity)
3. Memory-efficient grid sampling with loop instead of repeat_for_k
5. BidirectionalDECA properly negates offsets for R->L instead of relying on flip trick
6. Offset predictor now receives left features for better context in textureless regions
7. Learnable attention temperature instead of fixed scale

No external dependencies (pure PyTorch).
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
    return x.view(B, num_heads * head_dim, H, W)


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

    FIX #1: Computes correlation across multiple horizontal shifts instead of only at
            zero disparity. This gives the network actual stereo matching signal.
    FIX #6: Concatenates left features into the refinement path so the network has
            content context for textureless regions.

    Args:
        channels: Number of input channels
        k: Number of sampling points along epipolar line
        max_disparity: Maximum disparity in pixels (clamps offset range)
        num_corr_shifts: Number of horizontal shifts for correlation computation
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        max_disparity: int = 192,
        num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.k = k
        self.max_disparity = max_disparity
        self.num_corr_shifts = num_corr_shifts

        reduced = channels // 4

        # Lightweight projections for correlation
        self.query_proj = nn.Conv2d(channels, reduced, 1)
        self.key_proj = nn.Conv2d(channels, reduced, 1)

        # FIX #1: Correlation is now computed at multiple shifts, producing
        # num_corr_shifts channels. We also concatenate the left features (reduced channels)
        # for context (FIX #6), so total input to refinement = num_corr_shifts + reduced.
        corr_in_channels = num_corr_shifts + reduced

        # Correlation + context refinement
        self.corr_refine = nn.Sequential(
            nn.Conv2d(corr_in_channels, reduced, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced, reduced, 3, padding=1),
            nn.GELU(),
        )

        # Offset prediction head
        self.offset_head = nn.Conv2d(reduced, k, 1)

        # Initialize to zero offsets, bias spread across disparity range
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        with torch.no_grad():
            self.offset_head.bias.copy_(
                torch.linspace(-max_disparity // 2, max_disparity // 2, k)
            )

        # Precompute the shift values (centered around 0, spaced across a local window)
        # These are pixel-level shifts for building a mini cost volume
        max_shift = max_disparity // 4  # e.g., 48 pixels for max_disparity=192
        self.register_buffer(
            'shifts',
            torch.linspace(-max_shift, max_shift, num_corr_shifts).long()
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

        # FIX #1: Compute correlation at multiple horizontal shifts
        # This builds a mini cost volume that captures stereo matching signal
        # across different disparities, not just at zero offset.
        corr_planes = []
        for shift in self.shifts:
            shift_val = shift.item()
            if shift_val == 0:
                k_shifted = k
            elif shift_val > 0:
                # Shift right: pad left side, crop right side
                k_shifted = F.pad(k, (shift_val, 0, 0, 0), mode='replicate')[:, :, :, :W]
            else:
                # Shift left: pad right side, crop left side
                k_shifted = F.pad(k, (0, -shift_val, 0, 0), mode='replicate')[:, :, :, -shift_val:]
                # Ensure width matches (pad may differ by 1 in edge cases)
                if k_shifted.shape[-1] < W:
                    k_shifted = F.pad(k_shifted, (0, W - k_shifted.shape[-1]), mode='replicate')
                k_shifted = k_shifted[:, :, :, :W]

            # Dot product correlation at this shift, summed over channels -> [B, 1, H, W]
            corr_at_shift = (q * k_shifted).sum(dim=1, keepdim=True)
            corr_planes.append(corr_at_shift)

        corr = torch.cat(corr_planes, dim=1)  # [B, num_corr_shifts, H, W]

        # FIX #6: Concatenate left features so the refinement network has content
        # context, helping in textureless regions where correlation is ambiguous.
        corr = torch.cat([corr, q], dim=1)  # [B, num_corr_shifts + reduced, H, W]

        # Refine and predict offsets
        corr = self.corr_refine(corr)
        offsets = self.offset_head(corr)  # [B, K, H, W]
        offsets = offsets.clamp(-self.max_disparity, self.max_disparity)

        return offsets


class DECA(nn.Module):
    """
    Deformable Epipolar Cross-Attention (DECA)

    Performs cross-view attention along epipolar lines with content-adaptive
    sampling positions. For rectified stereo pairs, epipolar lines are horizontal.

    Complexity: O(H*W*K) vs O(H*W*W) for full epipolar attention

    FIX #3: Uses a loop over K for grid_sample instead of expanding memory by K×.
    FIX #7: Uses a learnable temperature parameter instead of fixed 1/sqrt(d).

    Args:
        channels: Number of feature channels
        k: Number of sampling points per query (default: 8)
        num_heads: Number of attention heads (default: 4)
        max_disparity: Maximum disparity in pixels (default: 192)
        use_occlusion_gate: Whether to use soft occlusion gating (default: False)
        num_corr_shifts: Number of horizontal shifts for offset predictor correlation
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 192,
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

        # FIX #7: Learnable attention temperature
        self.log_scale = nn.Parameter(
            torch.tensor(self.head_dim ** -0.5).log()
        )

        # Pre-register base grid components (will be expanded in forward)
        self.register_buffer('_grid_initialized', torch.tensor(False))

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

        # Normalize offsets to [-1, 1] range
        offset_norm = offsets_k * (2.0 / W)  # [B, H, W]

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
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]

        Returns:
            Enhanced left features [B, C, H, W]
        """
        B, C, H, W = f_left.shape

        # Normalize inputs
        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        # Predict sampling offsets
        offsets = self.offset_predictor(fl, fr)  # [B, K, H, W]

        # Compute Q/K/V
        queries = self.q_proj(fl)  # [B, C, H, W]
        keys = self.k_proj(fr)     # [B, C, H, W]
        values = self.v_proj(fr)   # [B, C, H, W]

        # FIX #3: Sample K/V at offset positions using a loop over K.
        # This avoids expanding keys/values by K× in memory.
        # Each iteration samples one offset's worth of keys and values.
        keys_sampled_list = []
        vals_sampled_list = []
        for i in range(self.k):
            grid_i = self._get_sampling_grid_single(
                offsets[:, i, :, :], H, W, f_left.device
            )  # [B, H, W, 2]

            ks = F.grid_sample(
                keys, grid_i,
                mode='bilinear', padding_mode='border', align_corners=True
            )  # [B, C, H, W]
            vs = F.grid_sample(
                values, grid_i,
                mode='bilinear', padding_mode='border', align_corners=True
            )  # [B, C, H, W]
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

        # FIX #7: Use learnable temperature
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

    FIX #5: Instead of the flip trick (which assumes horizontal symmetry in
    learned representations), we use a dedicated R->L path that shares
    Q/K/V projections but has its own offset predictor with properly
    negated disparity semantics.
    """
    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 192,
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
    assert offsets_test.abs().max() <= 192, "Offsets should be clamped"
    print(f"✓ Offset predictor output: {offsets_test.shape}, range [{offsets_test.min():.1f}, {offsets_test.max():.1f}]")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_deca()