import os

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def save_loss_curves(loss_values: list[float], filepath: str) -> None:
    """Saves a graph of loss values over training iterations.

    Args:
        loss_values (list[float]): List of loss values recorded during training.
        filepath (str): The file path to save the graph image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()


def save_eval_metric_curves(
    metrics_history: list[dict],
    filepath: str,
    title: str = "Evaluation Metrics Over Epochs",
) -> None:
    """Saves a graph of evaluation metrics over epochs.

    Args:
        metrics_history (list[dict]): List of metric dicts from each epoch.
            Each dict should have 'epoch' key and metric keys like 'BLEU-4', 'CIDEr', etc.
        filepath (str): The file path to save the graph image.
        title (str): Title for the plot.
    """
    if not metrics_history:
        return

    epochs = [m["epoch"] for m in metrics_history]

    # Define metrics to plot and their colors
    metrics_config = [
        ("BLEU-1", "tab:blue", "--"),
        ("BLEU-2", "tab:blue", "-."),
        ("BLEU-3", "tab:blue", ":"),
        ("BLEU-4", "tab:blue", "-"),
        ("METEOR", "tab:orange", "-"),
        ("ROUGE-L", "tab:green", "-"),
        ("CIDEr", "tab:red", "-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot BLEU scores on left subplot
    for metric_name, color, linestyle in metrics_config[:4]:
        if metric_name in metrics_history[0]:
            values = [m[metric_name] for m in metrics_history]
            ax1.plot(
                epochs, values, label=metric_name, color=color, linestyle=linestyle
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_title("BLEU Scores")
    ax1.legend()
    ax1.grid(True)

    # Plot other metrics on right subplot
    for metric_name, color, linestyle in metrics_config[4:]:
        if metric_name in metrics_history[0]:
            values = [m[metric_name] for m in metrics_history]
            ax2.plot(
                epochs, values, label=metric_name, color=color, linestyle=linestyle
            )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("METEOR, ROUGE-L, CIDEr")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def load_gpt2_tokenizer() -> GPT2Tokenizer:
    """Loads the GPT-2 tokenizer from Hugging Face transformers.

    Returns:
        GPT2Tokenizer: The GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # Set padding token to EOS token (GPT-2 has no dedicated pad token)
    return tokenizer


class ImageDirectoryDataset(Dataset):
    """
    Simple Dataset to load images from a flat directory.
    This is a helper dataset for efficient batch processing.
    """

    def __init__(self, directory: str) -> None:
        """
        Args:
            directory (str): Path to the folder containing images.
        """
        self.directory = directory
        # Filter for valid image extensions
        self.valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.filenames = [
            f
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in self.valid_exts
        ]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[str, Image.Image]: Filename and corresponding PIL Image object.
        """
        filename = self.filenames[idx]
        path = os.path.join(self.directory, filename)

        # Convert to RGB to ensure 3 channels (handles Greyscale/RGBA)
        image = Image.open(path).convert("RGB")
        return filename, image

    @staticmethod
    def collate_fn(
        batch: list[tuple[str, Image.Image]],
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Custom collate function for DataLoader to handle batches of (filename, image) tuples.
        This lets us keep image loading in `__getitem__`, allowing CPU workers to handle image loading in parallel.

        Args:
            batch (list[tuple[str, Image.Image]]): List of (filename, image) tuples from `__getitem__`.

        Returns:
            tuple[list[str], list[Image.Image]]: Tuple containing a list of filenames and a list of corresponding PIL Image objects.
        """
        # zip(*batch) unzips [(f1, img1), (f2, img2)] into [(f1, f2), (img1, img2)]
        batch_filenames, batch_images = zip(*batch)
        return list(batch_filenames), list(batch_images)
