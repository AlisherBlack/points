# Adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
# MIT License, Copyright (c) Massachusetts Institute of Technology and its affiliates.

import torch
import torch.nn.functional as F
from torch import nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    batch_size, num_dims, num_points = x.shape
    idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    def __init__(self, num_classes=40, in_channels=3, k=20, widths=(64, 64, 128, 256), emb_dims=1024, head=(512, 256), dropout=0.5):
        super().__init__()
        self.k = k
        c1, c2, c3, c4 = widths
        h1, h2 = head

        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c3)
        self.bn4 = nn.BatchNorm2d(c4)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, c1, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(c1 * 2, c2, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2 * 2, c3, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(c3 * 2, c4, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(c1 + c2 + c3 + c4, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(emb_dims * 2, h1, bias=False)
        self.bn6 = nn.BatchNorm1d(h1)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(h1, h2)
        self.bn7 = nn.BatchNorm1d(h2)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(h2, num_classes)

    def forward(self, points):
        x = points.transpose(1, 2)
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)
