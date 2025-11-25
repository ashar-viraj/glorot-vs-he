import argparse
import random
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Simple reproducibility helper.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_activation(name: str) -> Tuple[nn.Module, str]:
    if name == "relu":
        return nn.ReLU(), "relu"
    if name == "tanh":
        return nn.Tanh(), "tanh"
    raise ValueError(f"Unsupported activation {name}")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str) -> None:
        super().__init__()
        act_layer, _ = get_activation(activation)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(module: nn.Module, scheme: str, activation_name: str) -> None:
    if not isinstance(module, nn.Linear):
        return
    gain = nn.init.calculate_gain(activation_name)
    if scheme == "xavier":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif scheme == "he":
        nn.init.kaiming_uniform_(module.weight, nonlinearity=activation_name)
    elif scheme == "uniform":
        fan_in = module.weight.size(1)
        bound = 1.0 / fan_in ** 0.5
        nn.init.uniform_(module.weight, -bound, bound)
    else:
        raise ValueError(f"Unknown init scheme {scheme}")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


@dataclass
class RunConfig:
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    hidden_dim: int = 256
    init: str = "xavier"  # xavier | he | uniform
    activation: str = "relu"  # relu | tanh
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def get_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: Callable, optimizer: torch.optim.Optimizer, device: str
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: Callable, device: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss / len(loader.dataset), correct / total


def run(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    train_loader, test_loader = get_data(cfg.batch_size)
    _, act_name = get_activation(cfg.activation)
    model = MLP(input_dim=28 * 28, hidden_dim=cfg.hidden_dim, num_classes=10, activation=cfg.activation)
    model.apply(lambda m: init_weights(m, cfg.init, act_name))
    model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, cfg.device)
        print(
            f"epoch={epoch+1:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc*100:.2f}% "
            f"init={cfg.init}"
        )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="MLP init comparison on MNIST")
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--lr", type=float, default=RunConfig.lr)
    parser.add_argument("--hidden-dim", type=int, default=RunConfig.hidden_dim)
    parser.add_argument("--init", choices=["xavier", "he", "uniform"], default=RunConfig.init)
    parser.add_argument("--activation", choices=["relu", "tanh"], default=RunConfig.activation)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    args = parser.parse_args()
    return RunConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        init=args.init,
        activation=args.activation,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
