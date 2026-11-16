import torch
from torch import nn

from .attention import Block, serialize_stage, sincos
from .grid import grid_cluster, grid_sample, scatter


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU())


def linear_bn_relu_dropout(in_features, out_features, dropout):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(dropout))


def shared_mlp(in_channels, widths):
    channels = (in_channels, *widths)
    return nn.Sequential(*[linear_bn_relu(a, b) for a, b in zip(channels[:-1], channels[1:])])


class GridSetAbstraction(nn.Module):
    def __init__(self, in_channels, widths, shift, abs_coords=False):
        super().__init__()
        self.shift = shift
        self.abs_coords = abs_coords
        self.mlp = shared_mlp(in_channels + (6 if abs_coords else 3), widths)

    def forward(self, grid, coord, feat, batch):
        parents, cluster, batch = grid_cluster(grid, batch, self.shift)
        center = scatter(coord, cluster, len(parents), "mean")
        x = [coord - center[cluster]]
        if self.abs_coords:
            x.append(coord)
        if feat is not None:
            x.append(feat)
        x = self.mlp(torch.cat(x, dim=1))
        return parents, center, scatter(x, cluster, len(parents), "amax"), batch


class PointNet2Grid(nn.Module):
    def __init__(
        self,
        num_classes=40,
        grid_size=0.05,
        shifts=(2, 1),
        widths=((64, 64, 128), (128, 128, 256)),
        global_widths=(256, 512, 1024),
        head=(512, 256),
        dropout=0.3,
        attn_depths=(0, 0),
        num_heads=(8, 16),
        patch_size=8,
        pe=None,
        abs_coords=False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.pe = pe
        channels = (0, *[w[-1] for w in widths])
        self.sa = nn.ModuleList([GridSetAbstraction(c, w, s, abs_coords) for c, w, s in zip(channels[:-1], widths, shifts)])
        self.global_mlp = shared_mlp(channels[-1] + 3, global_widths)
        features = (global_widths[-1], *head)
        self.head = nn.Sequential(
            *[linear_bn_relu_dropout(a, b, dropout) for a, b in zip(features[:-1], features[1:])],
            nn.Linear(head[-1], num_classes),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleList([Block(w[-1], h, patch_size) for _ in range(d)])
            for w, h, d in zip(widths, num_heads, attn_depths)
        ])

    def forward(self, points):
        B, n, _ = points.shape
        coord = points.reshape(-1, 3)
        batch = torch.arange(B, device=points.device).repeat_interleave(n)
        grid, pick, batch = grid_sample(coord, self.grid_size, batch, self.training)
        coord = coord[pick]
        feat = None
        for sa, blocks in zip(self.sa, self.blocks):
            grid, coord, feat, batch = sa(grid, coord, feat, batch)
            if len(blocks) == 0:
                continue
            pos, order, inverse = serialize_stage(grid, batch, len(points), self.training)
            if self.pe == "ape":
                feat = feat + sincos(pos.float(), feat.shape[1])
            for i, block in enumerate(blocks):
                feat = block(feat, batch, order[i % len(order)], inverse[i % len(order)])
        x = self.global_mlp(torch.cat([coord, feat], dim=1))
        return self.head(scatter(x, batch, len(points), "amax"))
