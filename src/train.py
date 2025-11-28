import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.utils import save_loss_curves


def train(
    dataset: CocoDataset,
    model: nn.Module,
    batch_size: int,
    num_epochs: int,
    device: torch.device | None = None,
    outputs_dir: str = "graphs",
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    os.makedirs(outputs_dir, exist_ok=True)

    # Data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Add learning rate scheduler
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):

            # Log batch contents for debugging
            print(f"Batch {batch_idx} contents:"
                  f"\n token_ids shape: {batch['token_ids'].shape}"
                  f"\n clip_embedding shape: {batch['clip_embedding'].shape}"
                  f"\n attention_mask shape: {batch['attention_mask'].shape}"
            )

            # Unpack the batch dictionary
            token_ids = batch["token_ids"].to(device)
            clip_embeddings = batch["clip_embedding"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            
            # Forward pass, model tries to predict the next token given previous tokens and image embeddings
            outputs = model(
                caption_token_ids=token_ids[:, :-1],
                clip_image_embeddings=clip_embeddings,
                attention_mask=attention_mask[:, :-1],
                labels=token_ids[:, 1:]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss_values.append(loss.item())
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )


        scheduler1.step()
        scheduler2.step()

    # Save loss curves
    loss_curve_path = os.path.join("graphs", "loss_curve.png")
    save_loss_curves(loss_values, loss_curve_path)

    print("Training complete.")
    
    return model
