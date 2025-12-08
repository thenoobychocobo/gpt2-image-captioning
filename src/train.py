import os
from typing import Any

import torch
from objectbox import Store
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.dataset import CocoDataset
from src.eval import evaluate_epoch, evaluate_rat_epoch, save_eval_summary
from src.models import ImageCaptioningModel, RetrievalAugmentedTransformer
from src.utils import save_eval_metric_curves, save_loss_curves

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

def train(
    train_dataset: CocoDataset,
    model: ImageCaptioningModel,
    batch_size: int,
    num_epochs: int,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    num_warmup_steps: int = 0,
    save_every_epoch: int = 5,
    device: torch.device | None = None,
    outputs_dir: str = "checkpoints",
    grad_accum_steps: int = 1,
    # Evaluation parameters
    val_dataset: CocoDataset | None = None,
    val_annotations_path: str | None = None,
    eval_every_epoch: int = 1,
    eval_batch_size: int | None = None,
    eval_max_length: int = 50,
    eval_temperature: float = 0.0,
    eval_top_p: float = 0.9,
) -> dict[str, Any]:
    """
    Trains the `ImageCaptioningModel` on the provided `CocoDataset`.
    Optionally runs evaluation on the validation set after each epoch.

    Args:
        train_dataset (CocoDataset): The training dataset.
        model (ImageCaptioningModel): The model to be trained.
        batch_size (int): The batch size for training.
        num_epochs (int): The number of training epochs.
        num_workers (int, optional): The number of CPU threads for data loading. Should be 0 on Windows. Defaults to 4.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        num_warmup_steps (int, optional): The number of warmup steps for the learning rate scheduler.
            Learning rate increases linearly from 0 to specified value during warmup. Defaults to 0.
        save_every_epoch (int, optional): The frequency (in epochs) to save the model checkpoint. Defaults to 5.
        device (torch.device | None, optional): The device to run the training on. Defaults to None, which selects CUDA if available.
        outputs_dir (str, optional): The directory to save model checkpoints. Defaults to "checkpoints".
        grad_accum_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
        val_dataset (CocoDataset | None, optional): Validation dataset for evaluation. Defaults to None.
        val_annotations_path (str | None, optional): Path to validation annotations JSON. Required if val_dataset is provided.
        eval_every_epoch (int, optional): Frequency of evaluation in epochs. Defaults to 1.
        eval_batch_size (int | None, optional): Batch size for evaluation. Defaults to training batch_size.
        eval_max_length (int, optional): Maximum caption length during generation. Defaults to 50.
        eval_temperature (float, optional): Sampling temperature for generation. Defaults to 0.0.
        eval_top_p (float, optional): Nucleus sampling probability. Defaults to 0.9.

    Returns:
        dict: Training history containing losses and evaluation metrics.
    """
    # Create output directories
    os.makedirs(outputs_dir, exist_ok=True)
    eval_dir = os.path.join(outputs_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Validate evaluation parameters
    if val_dataset is not None and val_annotations_path is None:
        raise ValueError(
            "val_annotations_path is required when val_dataset is provided"
        )

    # Set defaults
    eval_batch_size = eval_batch_size or batch_size

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

    # Track training history
    epoch_loss_values: list[float] = []
    val_metrics_history: list[dict[str, Any]] = []
    best_val_cider: float = -1.0
    best_epoch: int = 0

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        current_epoch_loss = 0.0

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch and move to device
            token_ids = batch["token_ids"].to(device)
            labels = batch["labels"].to(
                device
            )  # padding tokens set to -100 to ignore loss computation
            image_embeddings = batch["image_embedding"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Clear gradients ONLY when starting a new accumulation cycle
            # We don't need to call optimizer.zero_grad() here since it's called below

            # Forward pass
            # Model predicts the next token given previous tokens and image embeddings
            # We do not have to shift the token_ids here because the model's forward method handles that internally
            outputs = model.forward(
                caption_token_ids=token_ids,
                image_embeddings=image_embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )

            # SCALE LOSS: Divide the loss by the accumulation steps
            loss = outputs.loss / grad_accum_steps

            # Compute loss and accumulate gradients
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (
                batch_idx + 1 == len(dataloader)
            ):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update parameters (optimizer step)
                optimizer.step()

                # Scheduler step (update learning rate)
                scheduler.step()

                # Clear accumulated gradients
                optimizer.zero_grad()

            # Logging (Use the unscaled loss for logging simplicity)
            batch_loss = outputs.loss.item()
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
            checkpoint_path = os.path.join(outputs_dir, f"model_epoch_{epoch + 1}.pt")
            model.save_parameters(checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # Run Evaluation
        if (epoch + 1) % eval_every_epoch == 0:
            # Evaluate on validation set
            if val_dataset is not None:
                val_metrics = evaluate_epoch(
                    model=model,
                    dataset=val_dataset,
                    annotations_path=val_annotations_path,
                    epoch=epoch + 1,
                    split_name="val",
                    batch_size=eval_batch_size,
                    num_workers=num_workers,
                    max_length=eval_max_length,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    device=device,
                    output_dir=eval_dir,
                )
                val_metrics_dict = {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    **val_metrics.to_dict(),
                }
                val_metrics_history.append(val_metrics_dict)

                writer.add_scalar("CIDEr/val", val_metrics.cider, epoch + 1)
                writer.add_scalar("BLEU-1/val", val_metrics.bleu_1, epoch + 1)
                writer.add_scalar("BLEU-4/val", val_metrics.bleu_4, epoch + 1)
                writer.add_scalar("ROUGE-L/val", val_metrics.rouge_l, epoch + 1)

                # Track best model by CIDEr score
                if val_metrics.cider > best_val_cider:
                    best_val_cider = val_metrics.cider
                    best_epoch = epoch + 1
                    best_model_path = os.path.join(
                        outputs_dir, f"best_model_epoch_{best_epoch}.pt"
                    )
                    model.save_parameters(best_model_path)
                    print(
                        f"New best model! CIDEr: {best_val_cider:.4f} (saved to {best_model_path})"
                    )

            # Set model back to training mode
            model.train()

    # Save loss curves
    loss_curve_path = os.path.join(outputs_dir, "loss_curve.png")
    save_loss_curves(epoch_loss_values, loss_curve_path)

    # Save evaluation summaries and metric curves
    if val_metrics_history:
        save_eval_summary(
            val_metrics_history,
            os.path.join(eval_dir, "val_metrics_summary.json"),
        )
        save_eval_metric_curves(
            val_metrics_history,
            os.path.join(eval_dir, "val_metrics_curve.png"),
            title="Validation Metrics Over Epochs",
        )

    writer.close()

    # Print final summary
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best validation CIDEr: {best_val_cider:.4f} at epoch {best_epoch}")
    print("=" * 60)

    return {
        "epoch_losses": epoch_loss_values,
        "val_metrics": val_metrics_history,
        "best_val_cider": best_val_cider,
        "best_epoch": best_epoch,
    }


def train_rat(
    train_dataset: CocoDataset,
    model: RetrievalAugmentedTransformer,
    db_store: Store,
    top_k: int,
    top_i: int,
    batch_size: int,
    num_epochs: int,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    num_warmup_steps: int = 0,
    save_every_epoch: int = 5,
    device: torch.device | None = None,
    outputs_dir: str = "checkpoints",
    grad_accum_steps: int = 1,
    # Evaluation parameters
    val_dataset: CocoDataset | None = None,
    val_annotations_path: str | None = None,
    eval_every_epoch: int = 1,
    eval_batch_size: int | None = None,
    eval_max_length: int = 50,
    eval_temperature: float = 0.0,
    eval_top_p: float = 0.9,
) -> None:
    """
    Trains the `RetrievalAugmentedTransformer` on the provided `CocoDataset`.

    Args:
        train_dataset (CocoDataset): The training dataset.
        model (RetrievalAugmentedTransformer): The model to be trained.
        db_store (Store): The ObjectBox store for retrieval.
        top_k (int): The number of top similar images to retrieve from the database.
        top_i (int): The number of top captions to retrieve for each image.
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
    # Create output directories
    os.makedirs(outputs_dir, exist_ok=True)
    eval_dir = os.path.join(outputs_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Validate evaluation parameters
    if val_dataset is not None and val_annotations_path is None:
        raise ValueError(
            "val_annotations_path is required when val_dataset is provided"
        )

    # Set defaults
    eval_batch_size = eval_batch_size or batch_size

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

    # Track training history
    epoch_loss_values: list[float] = []
    val_metrics_history: list[dict[str, Any]] = []
    best_val_cider: float = -1.0
    best_epoch: int = 0

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        current_epoch_loss = 0.0

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch and move to device
            token_ids = batch["token_ids"].to(device)
            labels = batch["labels"].to(
                device
            )  # padding tokens set to -100 to ignore loss computation
            image_embeddings = batch["image_embedding"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Clear gradients ONLY when starting a new accumulation cycle
            # We don't need to call optimizer.zero_grad() here since it's called below

            # Forward pass
            # Model predicts the next token given previous tokens and image embeddings
            # We do not have to shift the token_ids here because the model's forward method handles that internally
            outputs = model.forward(
                db_store=db_store,
                top_i=top_i,
                top_k=top_k,
                caption_token_ids=token_ids,
                image_embeddings=image_embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )

            # SCALE LOSS: Divide the loss by the accumulation steps
            loss = outputs.loss / grad_accum_steps

            # Compute loss and accumulate gradients
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (
                batch_idx + 1 == len(dataloader)
            ):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update parameters (optimizer step)
                optimizer.step()

                # Scheduler step (update learning rate)
                scheduler.step()

                # Clear accumulated gradients
                optimizer.zero_grad()

            # Logging (Use the unscaled loss for logging simplicity)
            batch_loss = outputs.loss.item()
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
            checkpoint_path = os.path.join(outputs_dir, f"model_epoch_{epoch + 1}.pt")
            model.save_parameters(checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # Run Evaluation
        if (epoch + 1) % eval_every_epoch == 0:
            # Evaluate on validation set
            if val_dataset is not None:
                val_metrics = evaluate_rat_epoch(
                    model=model,
                    db_store=db_store,
                    top_k=top_k,
                    top_i=top_i,
                    dataset=val_dataset,
                    annotations_path=val_annotations_path,
                    epoch=epoch + 1,
                    split_name="val",
                    batch_size=eval_batch_size,
                    num_workers=num_workers,
                    max_length=eval_max_length,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    device=device,
                    output_dir=eval_dir,
                )
                val_metrics_dict = {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    **val_metrics.to_dict(),
                }
                val_metrics_history.append(val_metrics_dict)

                # Track best model by CIDEr score
                if val_metrics.cider > best_val_cider:
                    best_val_cider = val_metrics.cider
                    best_epoch = epoch + 1
                    best_model_path = os.path.join(
                        outputs_dir, f"best_rat_model_epoch_{best_epoch}.pt"
                    )
                    model.save_parameters(best_model_path)
                    print(
                        f"New best model! CIDEr: {best_val_cider:.4f} (saved to {best_model_path})"
                    )

            # Set model back to training mode
            model.train()

    # Save loss curves
    loss_curve_path = os.path.join(outputs_dir, "loss_curve.png")
    save_loss_curves(epoch_loss_values, loss_curve_path)

    # Save evaluation summaries and metric curves
    if val_metrics_history:
        save_eval_summary(
            val_metrics_history,
            os.path.join(eval_dir, "val_metrics_summary.json"),
        )
        save_eval_metric_curves(
            val_metrics_history,
            os.path.join(eval_dir, "val_metrics_curve.png"),
            title="Validation Metrics Over Epochs",
        )

    # Print final summary
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best validation CIDEr: {best_val_cider:.4f} at epoch {best_epoch}")
    print("=" * 60)

    return {
        "epoch_losses": epoch_loss_values,
        "val_metrics": val_metrics_history,
        "best_val_cider": best_val_cider,
        "best_epoch": best_epoch,
    }

