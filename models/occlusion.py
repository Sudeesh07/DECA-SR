# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .nafblock import NAFBlock, LayerNorm2d


# class ConfidencePredictor(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.lr_branch = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(channels, channels // 2, 3, padding=1),
#             nn.GELU(),
#         )
#         self.texture_branch = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 2),
#             nn.GELU(),
#         )
#         self.fuse = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, 1),
#             nn.GELU(),
#             nn.Conv2d(channels // 2, 1, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
#         lr_feat = self.lr_branch(torch.cat([f_left, f_right], dim=1))
#         tex_feat = self.texture_branch(f_left)
#         confidence = self.fuse(torch.cat([lr_feat, tex_feat], dim=1))
#         return confidence

# class SelfAttentionBlock(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.block = NAFBlock(channels)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)

# class CrossAttentionBlock(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.norm_l = LayerNorm2d(channels)
#         self.norm_r = LayerNorm2d(channels)
#         self.q = nn.Conv2d(channels, channels, 1)
#         self.k = nn.Conv2d(channels, channels, 1)
#         self.v = nn.Conv2d(channels, channels, 1)
#         self.out = nn.Conv2d(channels, channels, 1)

#         self.local_cross = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 1),
#         )

#     def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
#         fl = self.norm_l(f_left)
#         fr = self.norm_r(f_right)
#         out = self.local_cross(torch.cat([fl, fr], dim=1))
#         return out

# class OcclusionAwareFusion(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.confidence = ConfidencePredictor(channels)
#         self.cross_attn = CrossAttentionBlock(channels)
#         self.self_attn = SelfAttentionBlock(channels)

#     def forward(self, f_left: torch.Tensor, f_right: torch.Tensor):
#         C = self.confidence(f_left, f_right)
#         cross_out = self.cross_attn(f_left, f_right)
#         self_out = self.self_attn(f_left)
#         fused = C * cross_out + (1 - C) * self_out + f_left

#         return fused, C

# class StereoConsistencyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def warp_with_disparity(self, img: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = img.shape
#         grid_y = torch.linspace(-1, 1, H, device=img.device)
#         grid_x = torch.linspace(-1, 1, W, device=img.device)
#         grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
#         disp_norm = disp.squeeze(1) * 2.0 / W
#         grid_x_shifted = grid_x.unsqueeze(0) + disp_norm
#         grid = torch.stack([grid_x_shifted, grid_y.unsqueeze(0).expand(B, -1, -1)], dim=-1)

#         warped = F.grid_sample(img, grid, mode="bilinear",
#                                 padding_mode="border", align_corners=True)
#         return warped

#     def forward(self, sr_left, sr_right, disp_lr, disp_rl, confidence):
#         losses = {}

#         if disp_lr is not None:
#             warped_r = self.warp_with_disparity(sr_right, disp_lr)
#             losses["photometric"] = F.l1_loss(sr_left, warped_r)
#             if disp_rl is not None:
#                 warped_disp = self.warp_with_disparity(disp_rl, disp_lr)
#                 losses["cycle"] = F.l1_loss(disp_lr + warped_disp,
#                                              torch.zeros_like(disp_lr))
#         eps = 1e-7
#         entropy = -(confidence * torch.log(confidence + eps) +
#                     (1 - confidence) * torch.log(1 - confidence + eps))
#         losses["conf_reg"] = entropy.mean()

#         return losses


"""
Occlusion-Aware Fusion Module for Stereo Image Super-Resolution

Handles occluded regions by learning soft confidence maps and using
self-attention for occluded areas, cross-attention for visible areas.

Key improvements over original:
1. Multi-scale confidence prediction
2. Proper cross-attention implementation (not just local convs)
3. Edge-aware confidence refinement
4. Bidirectional occlusion handling
5. Improved stereo consistency loss with occlusion-aware weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# =============================================================================
# Basic Components
# =============================================================================

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Simplified NAFBlock."""
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        expanded = channels * expansion
        
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.dwconv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate1 = SimpleGate()
        self.proj1 = nn.Conv2d(expanded // 2, channels, 1)
        
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.proj2 = nn.Conv2d(expanded // 2, channels, 1)
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate1(x)
        x = self.proj1(x)
        x = shortcut + x * self.beta
        
        shortcut = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.gate2(x)
        x = self.proj2(x)
        x = shortcut + x * self.gamma
        return x


# =============================================================================
# Confidence Prediction
# =============================================================================

class MultiScaleConfidencePredictor(nn.Module):
    """
    Multi-scale confidence predictor for occlusion detection.
    
    Predicts a soft confidence map C ∈ [0, 1] where:
    - C ≈ 1: Pixel is visible in both views (use cross-view attention)
    - C ≈ 0: Pixel is occluded (use self-attention)
    
    Uses multiple cues:
    1. Left-right feature correlation (low = likely occluded)
    2. Texture complexity (high texture = harder to match)
    3. Multi-scale analysis for robust prediction
    
    Args:
        channels: Number of input feature channels
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Left-right correlation branch
        self.lr_branch = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
        )
        
        # Texture/edge complexity branch
        self.texture_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 2),
            nn.GELU(),
        )
        
        # Disparity gradient branch (sharp changes indicate occlusion boundaries)
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.GELU(),
        )
        
        # Multi-scale processing
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.coarse_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
        )
        
        # Final fusion
        total_channels = channels // 2 + channels // 2 + channels // 4 + 1  # lr + tex + grad + coarse
        self.fuse = nn.Sequential(
            nn.Conv2d(total_channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
        
        # Edge-aware refinement
        self.edge_refine = nn.Sequential(
            nn.Conv2d(1 + channels, channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]
            
        Returns:
            confidence: Confidence map [B, 1, H, W]
        """
        B, C, H, W = f_left.shape
        
        # Left-right correlation features
        lr_feat = self.lr_branch(torch.cat([f_left, f_right], dim=1))
        
        # Texture complexity from left view
        tex_feat = self.texture_branch(f_left)
        
        # Gradient features (difference indicates potential occlusion)
        diff = (f_left - f_right).abs()
        grad_feat = self.gradient_branch(diff)
        
        # Coarse prediction at half resolution
        fl_down = self.down(f_left)
        fr_down = self.down(f_right)
        coarse_conf = self.coarse_predictor(torch.cat([fl_down, fr_down], dim=1))
        coarse_conf_up = self.up(coarse_conf)
        
        # Handle size mismatch from downsampling
        if coarse_conf_up.shape[-2:] != (H, W):
            coarse_conf_up = F.interpolate(coarse_conf_up, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse all features
        fused = torch.cat([lr_feat, tex_feat, grad_feat, coarse_conf_up], dim=1)
        confidence = self.fuse(fused)
        
        # Edge-aware refinement (preserve sharp boundaries)
        confidence = self.edge_refine(torch.cat([confidence, f_left], dim=1))
        
        return confidence


class LightweightConfidencePredictor(nn.Module):
    """Lightweight confidence predictor for faster inference."""
    def __init__(self, channels: int):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([f_left, f_right], dim=1))


# =============================================================================
# Attention Blocks
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for processing occluded regions.
    
    Uses local self-attention within the same view to fill in
    information that cannot be obtained from the other view.
    """
    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for non-occluded regions.
    
    Transfers information from the other view using attention
    along epipolar lines.
    
    Args:
        channels: Number of feature channels
        num_heads: Number of attention heads
        use_local: Also use local cross-correlation
    """
    def __init__(self, channels: int, num_heads: int = 4, use_local: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_local = use_local
        
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        
        # Q from left, K/V from right
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Local cross-correlation (for fine details)
        if use_local:
            self.local_cross = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
            )
            self.local_gate = nn.Parameter(torch.zeros(1))
        
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
        
        # Global cross-attention (row-wise for epipolar constraint)
        q = self.q_proj(fl)
        k = self.k_proj(fr)
        v = self.v_proj(fr)
        
        # Reshape for row-wise attention: [B*H, num_heads, W, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H, W).permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim, H, W).permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim, H, W).permute(0, 3, 1, 4, 2).reshape(B * H, self.num_heads, W, self.head_dim)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        global_out = torch.matmul(attn, v)
        
        # Reshape back
        global_out = global_out.view(B, H, self.num_heads, W, self.head_dim)
        global_out = global_out.permute(0, 2, 4, 1, 3).reshape(B, C, H, W)
        global_out = self.out_proj(global_out)
        
        # Local cross-correlation
        if self.use_local:
            local_out = self.local_cross(torch.cat([fl, fr], dim=1))
            gate = torch.sigmoid(self.local_gate)
            out = gate * local_out + (1 - gate) * global_out
        else:
            out = global_out
        
        return f_left + out * self.beta


# =============================================================================
# Occlusion-Aware Fusion
# =============================================================================

class OcclusionAwareFusion(nn.Module):
    """
    Occlusion-Aware Fusion Module.
    
    Predicts soft occlusion confidence and uses it to blend:
    - Cross-view attention (for visible regions)
    - Self-attention (for occluded regions)
    
    Formula:
        output = C * CrossAttn(F_L, F_R) + (1-C) * SelfAttn(F_L) + F_L
    
    where C is the confidence map (1 = visible, 0 = occluded).
    
    Args:
        channels: Number of feature channels
        use_multiscale_conf: Use multi-scale confidence predictor
        num_self_blocks: Number of NAFBlocks for self-attention
    """
    def __init__(
        self,
        channels: int,
        use_multiscale_conf: bool = True,
        num_self_blocks: int = 2
    ):
        super().__init__()
        
        # Confidence prediction
        if use_multiscale_conf:
            self.confidence = MultiScaleConfidencePredictor(channels)
        else:
            self.confidence = LightweightConfidencePredictor(channels)
        
        # Cross-attention for visible regions
        self.cross_attn = CrossAttentionBlock(channels, num_heads=4, use_local=True)
        
        # Self-attention for occluded regions
        self.self_attn = SelfAttentionBlock(channels, num_blocks=num_self_blocks)
        
        # Final refinement
        self.refine = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self, 
        f_left: torch.Tensor, 
        f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_left: Left view features [B, C, H, W]
            f_right: Right view features [B, C, H, W]
            
        Returns:
            fused: Fused features [B, C, H, W]
            confidence: Confidence map [B, 1, H, W]
        """
        # Predict occlusion confidence
        C = self.confidence(f_left, f_right)
        
        # Cross-view attention (for non-occluded regions)
        cross_out = self.cross_attn(f_left, f_right)
        
        # Self-attention (for occluded regions)
        self_out = self.self_attn(f_left)
        
        # Confidence-weighted fusion
        fused = C * cross_out + (1 - C) * self_out
        
        # Add residual and refine
        fused = fused + f_left
        fused = self.refine(fused)
        
        return fused, C


class BidirectionalOcclusionFusion(nn.Module):
    """
    Bidirectional occlusion-aware fusion.
    
    Processes both views with shared weights.
    """
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.fusion = OcclusionAwareFusion(channels, **kwargs)
    
    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (fused_left, fused_right, conf_left, conf_right)
        """
        fused_l, conf_l = self.fusion(f_left, f_right)
        fused_r, conf_r = self.fusion(f_right, f_left)
        return fused_l, fused_r, conf_l, conf_r


# =============================================================================
# Loss Functions
# =============================================================================

class StereoConsistencyLoss(nn.Module):
    """
    Stereo consistency loss for training.
    
    Components:
    1. Photometric loss: L1 between warped views
    2. Cycle consistency: Forward-backward disparity consistency
    3. Confidence regularization: Entropy penalty to avoid trivial solutions
    4. Smoothness loss: Encourage smooth disparities
    
    Args:
        photo_weight: Weight for photometric loss
        cycle_weight: Weight for cycle consistency loss
        conf_weight: Weight for confidence regularization
        smooth_weight: Weight for smoothness loss
    """
    def __init__(
        self,
        photo_weight: float = 1.0,
        cycle_weight: float = 0.5,
        conf_weight: float = 0.1,
        smooth_weight: float = 0.1
    ):
        super().__init__()
        self.photo_weight = photo_weight
        self.cycle_weight = cycle_weight
        self.conf_weight = conf_weight
        self.smooth_weight = smooth_weight
        
        # Sobel filters for edge detection (smoothness loss)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def warp_with_disparity(
        self, 
        img: torch.Tensor, 
        disp: torch.Tensor,
        direction: str = 'left_to_right'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warp image using disparity.
        
        Args:
            img: Image to warp [B, C, H, W]
            disp: Disparity map [B, 1, H, W]
            direction: 'left_to_right' or 'right_to_left'
            
        Returns:
            warped: Warped image
            valid_mask: Mask of valid (non-occluded) pixels
        """
        B, C, H, W = img.shape
        
        # Create base grid
        grid_y = torch.linspace(-1, 1, H, device=img.device)
        grid_x = torch.linspace(-1, 1, W, device=img.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Normalize disparity to [-1, 1] range
        disp_norm = disp.squeeze(1) * 2.0 / W
        
        # Apply disparity offset
        if direction == 'left_to_right':
            grid_x_shifted = grid_x.unsqueeze(0) + disp_norm
        else:
            grid_x_shifted = grid_x.unsqueeze(0) - disp_norm
        
        # Create sampling grid
        grid = torch.stack([
            grid_x_shifted,
            grid_y.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)
        
        # Warp
        warped = F.grid_sample(
            img, grid, 
            mode='bilinear', 
            padding_mode='zeros',  # Use zeros for out-of-bounds
            align_corners=True
        )
        
        # Valid mask (pixels that map to valid locations)
        valid_mask = (grid_x_shifted >= -1) & (grid_x_shifted <= 1)
        valid_mask = valid_mask.unsqueeze(1).float()
        
        return warped, valid_mask

    def compute_smoothness(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Edge-aware smoothness loss.
        
        Encourages smooth disparity except at image edges.
        """
        # Disparity gradients
        disp_grad_x = F.conv2d(disp, self.sobel_x, padding=1)
        disp_grad_y = F.conv2d(disp, self.sobel_y, padding=1)
        
        # Image gradients (for edge-awareness)
        img_gray = img.mean(dim=1, keepdim=True)
        img_grad_x = F.conv2d(img_gray, self.sobel_x, padding=1).abs()
        img_grad_y = F.conv2d(img_gray, self.sobel_y, padding=1).abs()
        
        # Edge-aware weights (low weight at image edges)
        weight_x = torch.exp(-img_grad_x)
        weight_y = torch.exp(-img_grad_y)
        
        # Weighted smoothness
        smooth_x = (disp_grad_x.abs() * weight_x).mean()
        smooth_y = (disp_grad_y.abs() * weight_y).mean()
        
        return smooth_x + smooth_y

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        disp_lr: Optional[torch.Tensor] = None,
        disp_rl: Optional[torch.Tensor] = None,
        confidence_l: Optional[torch.Tensor] = None,
        confidence_r: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stereo consistency losses.
        
        Args:
            sr_left: Super-resolved left image
            sr_right: Super-resolved right image
            disp_lr: Left-to-right disparity (optional)
            disp_rl: Right-to-left disparity (optional)
            confidence_l: Left confidence map (optional)
            confidence_r: Right confidence map (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Photometric loss
        if disp_lr is not None:
            warped_r, valid_mask = self.warp_with_disparity(sr_right, disp_lr, 'left_to_right')
            
            # Occlusion-aware photometric loss
            if confidence_l is not None:
                # Weight by confidence (lower weight for occluded regions)
                weight = confidence_l * valid_mask
                photo_loss = (weight * (sr_left - warped_r).abs()).sum() / (weight.sum() + 1e-8)
            else:
                photo_loss = (valid_mask * (sr_left - warped_r).abs()).sum() / (valid_mask.sum() + 1e-8)
            
            losses['photometric'] = self.photo_weight * photo_loss
            
            # Cycle consistency
            if disp_rl is not None:
                warped_disp, _ = self.warp_with_disparity(disp_rl, disp_lr, 'left_to_right')
                cycle_loss = F.l1_loss(disp_lr + warped_disp, torch.zeros_like(disp_lr))
                losses['cycle'] = self.cycle_weight * cycle_loss
            
            # Smoothness
            smooth_loss = self.compute_smoothness(disp_lr, sr_left)
            losses['smoothness'] = self.smooth_weight * smooth_loss
        
        # Confidence regularization (entropy)
        if confidence_l is not None:
            eps = 1e-7
            entropy_l = -(
                confidence_l * torch.log(confidence_l + eps) +
                (1 - confidence_l) * torch.log(1 - confidence_l + eps)
            )
            losses['conf_entropy_l'] = self.conf_weight * entropy_l.mean()
        
        if confidence_r is not None:
            eps = 1e-7
            entropy_r = -(
                confidence_r * torch.log(confidence_r + eps) +
                (1 - confidence_r) * torch.log(1 - confidence_r + eps)
            )
            losses['conf_entropy_r'] = self.conf_weight * entropy_r.mean()
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1).
    
    L(x) = sqrt(x^2 + eps^2)
    
    More robust to outliers than L2, smoother than L1.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()


# =============================================================================
# Unit Tests
# =============================================================================

def test_occlusion_modules():
    """Test occlusion-aware fusion modules."""
    print("Testing Occlusion-Aware Modules...")
    print("=" * 50)
    
    B, C, H, W = 2, 64, 32, 64
    f_left = torch.randn(B, C, H, W)
    f_right = torch.randn(B, C, H, W)
    
    # Test confidence predictors
    multi_conf = MultiScaleConfidencePredictor(C)
    conf = multi_conf(f_left, f_right)
    assert conf.shape == (B, 1, H, W)
    assert (conf >= 0).all() and (conf <= 1).all()
    print(f"✓ MultiScaleConfidencePredictor: {conf.shape}, range=[{conf.min():.3f}, {conf.max():.3f}]")
    
    light_conf = LightweightConfidencePredictor(C)
    conf = light_conf(f_left, f_right)
    assert conf.shape == (B, 1, H, W)
    print(f"✓ LightweightConfidencePredictor: {conf.shape}")
    
    # Test attention blocks
    self_attn = SelfAttentionBlock(C)
    out = self_attn(f_left)
    assert out.shape == f_left.shape
    print(f"✓ SelfAttentionBlock: {out.shape}")
    
    cross_attn = CrossAttentionBlock(C, num_heads=4)
    out = cross_attn(f_left, f_right)
    assert out.shape == f_left.shape
    print(f"✓ CrossAttentionBlock: {out.shape}")
    
    # Test full fusion
    oaf = OcclusionAwareFusion(C)
    fused, conf = oaf(f_left, f_right)
    assert fused.shape == f_left.shape
    assert conf.shape == (B, 1, H, W)
    print(f"✓ OcclusionAwareFusion: fused={fused.shape}, conf={conf.shape}")
    
    # Test bidirectional
    bi_oaf = BidirectionalOcclusionFusion(C)
    fused_l, fused_r, conf_l, conf_r = bi_oaf(f_left, f_right)
    assert fused_l.shape == f_left.shape and fused_r.shape == f_right.shape
    print(f"✓ BidirectionalOcclusionFusion: L={fused_l.shape}, R={fused_r.shape}")
    
    # Test losses
    sr_left = torch.randn(B, 3, H * 4, W * 4)
    sr_right = torch.randn(B, 3, H * 4, W * 4)
    disp = torch.rand(B, 1, H * 4, W * 4) * 50
    conf = torch.rand(B, 1, H * 4, W * 4)
    
    stereo_loss = StereoConsistencyLoss()
    losses = stereo_loss(sr_left, sr_right, disp, disp, conf, conf)
    print(f"✓ StereoConsistencyLoss: {list(losses.keys())}")
    
    char_loss = CharbonnierLoss()
    loss = char_loss(sr_left, sr_right)
    print(f"✓ CharbonnierLoss: {loss.item():.4f}")
    
    # Parameter counts
    print("\nParameter counts:")
    for name, module in [("MultiScaleConf", multi_conf), ("OAF", oaf)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} ({params/1e6:.3f}M)")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_occlusion_modules()