import torch

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def interleave(axes, depth):
    code = torch.zeros_like(axes[0])
    for bit in range(depth):
        for i, axis in enumerate(axes):
            code |= ((axis >> bit) & 1) << (len(axes) * bit + len(axes) - 1 - i)
    return code


def z_order_encode(grid_coord, depth):
    return interleave(grid_coord.long().unbind(1), depth)


def hilbert_encode(grid_coord, depth):
    x = list(grid_coord.long().unbind(1))
    q = 1 << (depth - 1)
    while q > 1:
        p = q - 1
        for i in range(len(x)):
            on = (x[i] & q) > 0
            t = (x[0] ^ x[i]) & p
            x[0] = torch.where(on, x[0] ^ p, x[0] ^ t)
            x[i] = torch.where(on, x[i], x[i] ^ t)
        q >>= 1
    for i in range(1, len(x)):
        x[i] = x[i] ^ x[i - 1]
    t = torch.zeros_like(x[0])
    q = 1 << (depth - 1)
    while q > 1:
        t = torch.where((x[-1] & q) > 0, t ^ (q - 1), t)
        q >>= 1
    return interleave([axis ^ t for axis in x], depth)


def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in ORDERS
    if order.endswith("-trans"):
        grid_coord = grid_coord[:, [1, 0, 2]]
    code = z_order_encode(grid_coord, depth) if order.startswith("z") else hilbert_encode(grid_coord, depth)
    if batch is not None:
        code = batch.long() << depth * 3 | code
    return code


def serialize(grid_coord, batch=None, orders=ORDERS, depth=None, shuffle_orders=False):
    if depth is None:
        depth = int(grid_coord.max() + 1).bit_length()
    assert depth <= 16
    code = torch.stack([encode(grid_coord, batch, depth, order) for order in orders])
    order = torch.argsort(code)
    inverse = torch.empty_like(order).scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    if shuffle_orders:
        perm = torch.randperm(len(orders))
        code, order, inverse = code[perm], order[perm], inverse[perm]
    return code, order, inverse, depth
