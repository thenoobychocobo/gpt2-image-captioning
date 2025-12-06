import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, CLIPProcessor

from src.utils import ImageDirectoryDataset


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: torch.device | None = None,
) -> tuple[CLIPModel, CLIPProcessor]:
    """
    Loads the CLIP model and processor from Hugging Face transformers.
    Moves the model to the specified device and sets it to evaluation mode.

    Args:
        model_name (str, optional): HuggingFace model ID for CLIP. Defaults to "openai/clip-vit-base-patch32".
        device (torch.device | None, optional): Device to run the model on. Defaults to None (auto-detect).

    Returns:
        tuple[CLIPModel, CLIPProcessor]: Loaded CLIP model and processor.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP model '{model_name}' on device: {device}...")

    # CLIP Model
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()  # Set to inference mode

    # CLIP Processor
    processor: CLIPProcessor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def extract_clip_embedding_from_image(
    image: str | Image.Image,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extracts the CLIP embedding for a single image.

    Args:
        image (str | Image.Image): Path to an image file or a PIL Image object.
        clip_model (CLIPModel): Pre-loaded CLIP model.
        clip_processor (CLIPProcessor): Pre-loaded CLIP processor.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).

    Returns:
        torch.Tensor: The normalized CLIP image embedding with shape (embedding_dim,).
    """
    device = device or clip_model.device

    # Load image if a path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    with torch.no_grad():
        # Preprocess image
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        # Forward pass through CLIP Vision Model
        image_features = clip_model.get_image_features(**inputs)

        # Normalize embedding (L2 Norm)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.squeeze(
            0
        )  # Remove batch dimension; shape (embedding_dim,)


def extract_clip_embeddings(
    image_dir: str,
    output_path: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> None:
    """
    Extracts CLIP embeddings for all images in a directory and saves them to a `.pt` file.
    Use this over `extract_clip_embedding_from_image` for more efficient batch processing.

    Args:
        image_dir (str): Path to the directory containing images.
        output_path (str): Path to save the resulting `.pt` file.
        clip_model (CLIPModel): Pre-loaded CLIP model.
        clip_processor (CLIPProcessor): Pre-loaded CLIP processor.
        batch_size (int, optional): Number of images to process at once. Higher is faster but uses more VRAM. Defaults to 32.
        num_workers (int, optional): Number of CPU threads for image loading. Should be 0 on Windows. Defaults to 4.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).
    """

    # Setup Device
    device = device or clip_model.device

    # Precaution: Ensure model is in eval mode
    clip_model.eval()

    # Setup Data Loading
    dataset = ImageDirectoryDataset(image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ImageDirectoryDataset.collate_fn,
    )

    all_embeddings: list[torch.Tensor] = []
    all_filenames: list[str] = []

    # Extraction Loop
    print(f"Starting CLIP embedding extraction for {len(dataset)} images...")
    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(
            dataloader, desc="CLIP Embedding Extraction"
        ):
            # Preprocess images (Resize, Normalize)
            # clip_processor() handles the list of PIL images automatically
            inputs = clip_processor(images=batch_images, return_tensors="pt").to(device)

            # Forward pass through CLIP Vision Model
            image_features = clip_model.get_image_features(**inputs)

            # Normalize embeddings (L2 Norm)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

            # Move to CPU to save VRAM and store
            all_embeddings.append(image_features.cpu())
            all_filenames.extend(batch_filenames)

    # Concatenate Embeddings and Save
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Saving {final_embeddings.shape[0]} embeddings to {output_path}...")
    torch.save(
        {"filenames": all_filenames, "embeddings": final_embeddings}, output_path
    )


# === Example Usage ===
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP Model and Processor
    clip_model, clip_processor = load_clip_model(device=device)

    # Extract and Save Embeddings (batch processing)
    extract_clip_embeddings(
        image_dir="path/to/image/directory",
        output_path="clip_image_embeddings.pt",
        clip_model=clip_model,
        clip_processor=clip_processor,
        batch_size=64,
        num_workers=4,  # Set to 0 on Windows
    )
