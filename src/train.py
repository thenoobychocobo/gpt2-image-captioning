import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CocoDataset


def train(
    dataset: CocoDataset,
    model: nn.Module,
    batch_size: int,
    num_epochs: int,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # TODO: Add learning rate scheduler

    # Training loop
    for epoch in range(num_epochs):
        pass
        # TODO: WIP
