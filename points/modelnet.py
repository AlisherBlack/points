import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_shape(root, name, num_points=1024):
    cls = name.rsplit("_", 1)[0]
    return np.loadtxt(Path(root) / cls / f"{name}.txt", delimiter=",", max_rows=num_points, dtype=np.float32)


def prepare(root, num_points=1024, cache=None):
    root = Path(root)
    cache = Path(cache) if cache else root.parent / f"modelnet40_{num_points}.npz"
    if cache.exists():
        print(f"Кэш уже есть: {cache.resolve()}")
        return cache

    classes = (root / "modelnet40_shape_names.txt").read_text().split()
    arrays = {"classes": np.array(classes)}
    for split in ["train", "test"]:
        names = (root / f"modelnet40_{split}.txt").read_text().split()
        points = np.empty((len(names), num_points, 6), dtype=np.float32)
        for i, name in enumerate(tqdm(names, desc=split, file=sys.stdout)):
            points[i] = load_shape(root, name, num_points)
        arrays[f"{split}_points"] = points
        arrays[f"{split}_labels"] = np.array([classes.index(name.rsplit("_", 1)[0]) for name in names])

    np.savez(cache, **arrays)
    print(f"Сохранил кэш: {cache.resolve()}")
    return cache


class ModelNet40(Dataset):
    def __init__(self, cache, split="train", normals=False, augment=False):
        data = np.load(cache)
        self.points = torch.from_numpy(data[f"{split}_points"])
        self.labels = torch.from_numpy(data[f"{split}_labels"])
        self.classes = list(data["classes"])
        self.normals = normals
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        points = self.points[i] if self.normals else self.points[i, :, :3]
        if self.augment:
            scale = torch.empty(3).uniform_(0.7, 1.5)
            shift = torch.empty(3).uniform_(-0.2, 0.2)
            points = torch.cat([points[:, :3] * scale + shift, points[:, 3:]], dim=1)
        return points, self.labels[i]


def modelnet40_splits(cache, normals=False):
    return ModelNet40(cache, "train", normals, augment=True), ModelNet40(cache, "test", normals)
