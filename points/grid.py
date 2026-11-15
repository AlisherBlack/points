import torch


def scatter(x, index, size, reduce):
    return x.new_zeros(size, x.shape[1]).scatter_reduce_(0, index[:, None].expand_as(x), x, reduce, include_self=False)


def with_batch(grid, batch):
    return grid if batch is None else torch.cat([batch[:, None], grid], dim=1)


def split_batch(key, batch):
    return (key, None) if batch is None else (key[:, 1:], key[:, 0])


def grid_sample(coord, grid_size, batch=None, training=True):
    grid = torch.floor(coord / grid_size).long()
    voxels, inverse = torch.unique(with_batch(grid, batch), dim=0, return_inverse=True)
    score = torch.rand(len(coord), device=coord.device) if training else (coord / grid_size - grid - 0.5).norm(dim=1)
    best = score.new_full((len(voxels),), float("inf")).scatter_reduce_(0, inverse, score, "amin")
    winner = score == best[inverse]
    index = torch.arange(len(coord), device=coord.device)
    pick = index.new_full((len(voxels),), len(coord)).scatter_reduce_(0, inverse[winner], index[winner], "amin")
    grid, batch = split_batch(voxels, batch)
    return grid, pick, batch


def grid_cluster(grid, batch=None, shift=1):
    parents, cluster = torch.unique(with_batch(grid >> shift, batch), dim=0, return_inverse=True)
    parents, batch = split_batch(parents, batch)
    return parents, cluster, batch


def grid_pool(grid, coord, feat, batch=None, shift=1):
    parents, cluster, batch = grid_cluster(grid, batch, shift)
    coord = scatter(coord, cluster, len(parents), "mean")
    feat = scatter(feat, cluster, len(parents), "amax")
    return parents, coord, feat, batch
