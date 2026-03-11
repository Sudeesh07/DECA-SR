import torch
import torch.nn as nn
from einops import rearrange

from .nafblock import LayerNorm2d


class TCVA(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * (self.head_dim ** 0.5))

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_left.shape

        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        q = self.q_proj(fl)
        k = self.k_proj(fr)
        v = self.v_proj(fr)

        q = rearrange(q, "b (h d) H W -> b h d (H W)", h=self.num_heads)
        k = rearrange(k, "b (h d) H W -> b h d (H W)", h=self.num_heads)
        v = rearrange(v, "b (h d) H W -> b h d (H W)", h=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h d (H W) -> b (h d) H W", H=H, W=W)
        out = self.out_proj(out)

        return f_left + out
