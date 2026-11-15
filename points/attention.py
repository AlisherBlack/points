import torch
import torch.nn.functional as F
from torch import nn

from .grid import scatter
from .serialization import serialize


def sincos(pos, dim, base=100.0):
    freq = base ** (-torch.arange(0, dim // 3, 2, device=pos.device) / (dim // 3))
    angle = pos[..., None] * freq
    return torch.cat([angle.sin(), angle.cos()], dim=-1).flatten(-2)


def rope(x, pos, base=100.0):
    freq = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device) / x.shape[-1])
    angle = pos[..., None] * freq
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * angle.cos() - x2 * angle.sin(), x1 * angle.sin() + x2 * angle.cos()], dim=-1)


def point_rope(x, pos, base=100.0):
    return torch.cat([rope(part, pos[..., i], base) for i, part in enumerate(x.chunk(3, dim=-1))], dim=-1)


def serialize_stage(grid, batch, num_objects, shuffle_orders):
    grid = grid - scatter(grid.float(), batch, num_objects, "amin").long()[batch]
    _, order, inverse, _ = serialize(grid, batch, shuffle_orders=shuffle_orders)
    return grid, order, inverse


class PatchAttention(nn.Module):
    def __init__(self, channels, num_heads, patch_size, use_rope):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.use_rope = use_rope
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, grid, batch, order, inverse):
        n, C = feat.shape
        P, H = self.patch_size, self.num_heads
        pad = -n % P
        feat = F.pad(feat[order], (0, 0, 0, pad))
        grid = F.pad(grid[order], (0, 0, 0, pad)).float()
        batch = F.pad(batch[order], (0, pad), value=-1).reshape(-1, P)

        q, k, v = self.qkv(feat).reshape(-1, P, 3, H, C // H).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            pos = grid.reshape(-1, 1, P, 3)
            q, k = point_rope(q, pos), point_rope(k, pos)
        mask = (batch[:, :, None] == batch[:, None, :])[:, None]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).reshape(-1, C)[:n]
        return self.proj(x)[inverse]


class Block(nn.Module):
    def __init__(self, channels, num_heads, patch_size, use_rope, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = PatchAttention(channels, num_heads, patch_size, use_rope)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * mlp_ratio), nn.GELU(), nn.Linear(channels * mlp_ratio, channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat, grid, batch, order, inverse):
        feat = feat + self.attn(self.norm1(feat), grid, batch, order, inverse)
        return feat + self.mlp(self.norm2(feat))
