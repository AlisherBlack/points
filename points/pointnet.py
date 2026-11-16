import torch
from torch import nn


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels), nn.ReLU())


def linear_bn_relu_dropout(in_features, out_features, dropout):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(), nn.Dropout(dropout))


def edge_conv(x, k, h):                                              # x: (B, n, F)
    idx = torch.cdist(x, x, p=2).topk(k, largest=False).indices      # (B, n, k)
    b = torch.arange(x.shape[0], device=x.device)[:, None, None]     # (B, 1, 1)
    xj = x[b, idx]                                                   # (B, n, k, F)
    xi = x[:, :, None].expand_as(xj)                                 # (B, n, k, F)
    e = h(torch.cat([xi, xj - xi], dim=-1))                          # (B, n, k, F')
    return e.max(dim=2).values                                       # (B, n, F')


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def h(self, edges):
        e = self.linear(edges)
        e = self.bn(e.reshape(-1, e.shape[-1])).reshape(e.shape)
        return torch.relu(e)

    def forward(self, x):
        return edge_conv(x.transpose(1, 2), self.k, self.h).transpose(1, 2)


class PointNet(nn.Module):
    def __init__(self, num_classes=40, in_channels=3, widths=(64, 64, 64, 128, 1024), head=(512, 256), dropout=0.3, edge_layers=0, k=20):
        super().__init__()
        channels = (in_channels, *widths)
        layers = list(zip(channels[:-1], channels[1:]))
        self.point_mlp = nn.Sequential(
            *[EdgeConv(a, b, k) for a, b in layers[:edge_layers]],
            *[conv_bn_relu(a, b) for a, b in layers[edge_layers:]],
        )
        features = (widths[-1], *head)
        self.head = nn.Sequential(
            *[linear_bn_relu_dropout(a, b, dropout) for a, b in zip(features[:-1], features[1:])],
            nn.Linear(head[-1], num_classes),
        )

    def forward(self, points):
        features = self.point_mlp(points.transpose(1, 2))
        pooled = features.max(dim=2).values
        return self.head(pooled)

    def critical_points(self, points):
        return self.point_mlp(points.transpose(1, 2)).argmax(dim=2)
