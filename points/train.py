import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def forward_ms(model, points, device, warmup=3, repeats=20):
    model.eval()
    points = points.to(device)
    for _ in range(warmup):
        model(points)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(points)
    synchronize(device)
    return (time.perf_counter() - started) / repeats * 1000


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for points, labels in loader:
        logits = model(points.to(device)).cpu()
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += len(labels)
    return total_loss / total, correct / total


def fit(model, train_set, test_set, device, epochs=200, batch_size=32, lr=1e-3, block_lr=None):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128)
    model.to(device)
    params = model.parameters()
    if block_lr is not None:
        named = list(model.named_parameters())
        params = [
            {"params": [p for n, p in named if "blocks." not in n]},
            {"params": [p for n, p in named if "blocks." in n], "lr": block_lr},
        ]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            loss = F.cross_entropy(model(points.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        train_loss = total_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_acc})
        seconds = time.perf_counter() - started
        print(
            f"эпоха {epoch:3d} | train loss {train_loss:.4f} | test loss {test_loss:.4f} | "
            f"test accuracy {test_acc:.3f} | {seconds:.1f} с"
        )

    return {
        "test_accuracy": test_acc,
        "parameters": count_parameters(model),
        "history": history,
    }


def plot_losses(history):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["test_loss"] for h in history], marker="o", label="test")
    ax.set_xlabel("эпоха")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("cross-entropy")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()
