import torch
import torch.nn.functional as F
from torch import nn

from .grid import scatter
from .serialization import serialize


def sincos(pos, dim, base=100.0):
    freq = base ** (-torch.arange(0, dim // 3, 2, device=pos.device) / (dim // 3))
    angle = pos[..., None] * freq
    return torch.cat([angle.sin(), angle.cos()], dim=-1).flatten(-2)


def serialize_stage(grid, batch, num_objects, shuffle_orders):
    grid = grid - scatter(grid.float(), batch, num_objects, "amin").long()[batch]
    _, order, inverse, _ = serialize(grid, batch, shuffle_orders=shuffle_orders)
    return grid, order, inverse


class PatchAttention(nn.Module):
    def __init__(self, channels, num_heads, patch_size):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, batch, order, inverse):
        n, C = feat.shape
        P, H = self.patch_size, self.num_heads
        pad = -n % P
        feat = F.pad(feat[order], (0, 0, 0, pad))
        batch = F.pad(batch[order], (0, pad), value=-1).reshape(-1, P)

        q, k, v = self.qkv(feat).reshape(-1, P, 3, H, C // H).permute(2, 0, 3, 1, 4)
        mask = (batch[:, :, None] == batch[:, None, :])[:, None]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).reshape(-1, C)[:n]
        return self.proj(x)[inverse]


class Block(nn.Module):
    def __init__(self, channels, num_heads, patch_size, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = PatchAttention(channels, num_heads, patch_size)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * mlp_ratio), nn.GELU(), nn.Linear(channels * mlp_ratio, channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat, batch, order, inverse):
        feat = feat + self.attn(self.norm1(feat), batch, order, inverse)
        return feat + self.mlp(self.norm2(feat))
