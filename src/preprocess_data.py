import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


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


def extract_clip_embeddings(
    image_dir: str,
    output_path: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """
    Extracts CLIP embeddings for all images in a directory and saves them to a `.pt` file.

    Args:
        image_dir (str): Path to the directory containing images.
        output_path (str): Path to save the resulting `.pt` file.
        clip_model_name (str, optional): HuggingFace model ID for CLIP. Defaults to "openai/clip-vit-base-patch32".
        batch_size (int, optional): Number of images to process at once. Higher is faster but uses more VRAM. Defaults to 32.
        num_workers (int, optional): Number of CPU threads for image loading (0 on Windows). Defaults to 0.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (auto-detect).
    """

    # Setup Device, Model, and Processor
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()  # Inference mode
    processor = AutoProcessor.from_pretrained(clip_model_name)

    # Setup Data Loading
    dataset = ImageDirectoryDataset(image_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_embeddings: list[torch.Tensor] = []
    all_filenames: list[str] = []

    print(f"Starting CLIP embedding extraction for {len(dataset)} images...")

    # Extraction Loop
    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(
            dataloader, desc="CLIP Embedding Extraction"
        ):
            # Preprocess images (Resize, Normalize)
            # processor() handles the list of PIL images automatically
            inputs = processor(images=batch_images, return_tensors="pt").to(device)

            # Forward pass through CLIP Vision Model
            image_features = model.get_image_features(**inputs)

            # Normalize embeddings (L2 Norm)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

            # Move to CPU to save RAM and store
            all_embeddings.append(image_features.cpu())
            all_filenames.extend(batch_filenames)

    # Concatenate Embeddings and Save
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Saving {final_embeddings.shape[0]} embeddings to {output_path}...")
    torch.save(
        {"filenames": all_filenames, "embeddings": final_embeddings}, output_path
    )


if __name__ == "__main__":
    # Example usage for Train Set
    extract_clip_embeddings(
        image_dir="./data/coco/train2014",  # <--- Update this
        output_path="./clip_train2014_embeddings.pt",
        batch_size=32,  # Higher is faster but uses more VRAM
        num_workers=4,  # Set to 0 if you are on Windows!
    )

    # Example usage for Validation Set (You need this for inference!)
    # extract_embeddings(
    #     image_dir="./data/coco/val2014",
    #     output_path="./clip_val2014_embeddings.pt",
    #     batch_size=32,
    #     num_workers=4
    # )
