import matplotlib.pyplot as plt
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
