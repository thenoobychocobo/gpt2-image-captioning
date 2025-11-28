def save_loss_curves(loss_values: list[float], filename: str) -> None:
    """Saves a graph of loss values over training iterations.

    Args:
        loss_values (list[float]): List of loss values recorded during training.
        filename (str): The filename to save the graph image.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()