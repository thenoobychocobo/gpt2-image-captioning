import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.dataset import CocoDataset
from src.models import ImageCaptioningModel
from src.utils import save_loss_curves


def train(
    train_dataset: CocoDataset,
    model: ImageCaptioningModel,
    batch_size: int,
    num_epochs: int,
    num_workers: int = 4,
    learning_rate: float = 2e-5,
    num_warmup_steps: int = 0,
    save_every_epoch: int = 5,
    device: torch.device | None = None,
    outputs_dir: str = "checkpoints",
) -> None:
    """
    Trains the `ImageCaptioningModel` on the provided `CocoDataset`.

    Args:
        train_dataset (CocoDataset): The training dataset.
        model (ImageCaptioningModel): The model to be trained.
        batch_size (int): The batch size for training.
        num_epochs (int): The number of training epochs.
        num_workers (int, optional): The number of CPU threads for data loading. Should be 0 on Windows. Defaults to 4.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 2e-5.
        num_warmup_steps (int, optional): The number of warmup steps for the learning rate scheduler.
            Learning rate increases linearly from 0 to specified value during warmup. Defaults to 0.
        save_every_epoch (int, optional): The frequency (in epochs) to save the model checkpoint. Defaults to 5.
        device (torch.device | None, optional): The device to run the training on. Defaults to None, which selects CUDA if available.
        outputs_dir (str, optional): The directory to save model checkpoints. Defaults to "checkpoints".
    """
    # Create output directory if it doesn't exist
    os.makedirs(outputs_dir, exist_ok=True)

    # Device and model setup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()  # Set model to training mode

    # Data loader
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Add learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * num_epochs,
    )

    # Track the average loss per epoch
    epoch_loss_values: list[float] = []

    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        current_epoch_loss = 0.0

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch and move to device
            token_ids = batch["token_ids"].to(device)
            image_embeddings = batch["image_embedding"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            # Model predicts the next token given previous tokens and image embeddings
            # We do not have to shift the token_ids here because the model's forward method handles that internally
            outputs = model.forward(
                caption_token_ids=token_ids,
                image_embeddings=image_embeddings,
                attention_mask=attention_mask,
                labels=token_ids,
            )

            # Compute loss and gradients
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Scheduler step (update learning rate)
            scheduler.step()

            # Logging
            batch_loss = loss.item()
            current_epoch_loss += batch_loss
            progress_bar.set_postfix(
                {f"Batch {batch_idx + 1} Loss": f"{batch_loss:.4f}"}
            )

        # End of Epoch Stats
        avg_loss = current_epoch_loss / len(dataloader)
        epoch_loss_values.append(avg_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % save_every_epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(outputs_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    # Save loss curves
    loss_curve_path = os.path.join(outputs_dir, "loss_curve.png")
    save_loss_curves(epoch_loss_values, loss_curve_path)

    print("Training complete.")
