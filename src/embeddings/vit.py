import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from src.utils import ImageDirectoryDataset


def load_vit_model(
    model_name: str = "google/vit-base-patch16-224",
    device: torch.device | None = None,
) -> tuple[ViTModel, ViTImageProcessor]:
    """
    Loads the ViT model and processor from Hugging Face transformers.
    Moves the model to the specified device and sets it to evaluation mode.

    Args:
        model_name (str, optional): Name of the ViT model to load. Defaults to "google/vit-base-patch16-224".
        device (torch.device | None, optional): Device to load the model onto. Defaults to None.

    Returns:
        tuple[ViTModel, ViTImageProcessor]: Loaded ViT model and image processor.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ViT model '{model_name}' on device: {device}...")

    # Load Model (ViTModel outputs raw hidden states, not classification logits)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()

    # Load Processor (Handles resizing to 224x224 and Normalization)
    processor = ViTImageProcessor.from_pretrained(model_name)

    return model, processor


def extract_vit_embedding_from_image(
    image: str | Image.Image,
    vit_model: ViTModel,
    vit_processor: ViTImageProcessor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extracts the ViT embedding (pooler_output/[CLS]) for a single image.

    Args:
        image (str | Image.Image): Path to an image file or a PIL Image object.
        vit_model (ViTModel): Pre-loaded ViT model.
        vit_processor (ViTImageProcessor): Pre-loaded ViT processor.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).

    Returns:
        torch.Tensor: The normalized ViT image embedding with shape (embedding_dim,).
    """
    device = device or vit_model.device

    # Load image if a path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    with torch.no_grad():
        # Preprocess image
        inputs = vit_processor(images=image, return_tensors="pt").to(device)

        # Forward pass
        outputs = vit_model(**inputs)

        # Extract Pooler Output (The [CLS] token representation)
        embeddings = outputs.pooler_output

        # Normalize embedding (L2 Norm) to match CLIP behavior
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings.squeeze(0)  # Remove batch dimension; shape (embedding_dim,)


def extract_vit_embeddings(
    image_dir: str,
    output_path: str,
    vit_model: ViTModel,
    vit_processor: ViTImageProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> None:
    """
    Extracts ViT [CLS] token embeddings for all images in a directory using batch processing.
    """
    device = device or vit_model.device

    # Precaution: Ensure model is in eval mode
    vit_model.eval()

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

    print(f"Starting ViT embedding extraction for {len(dataset)} images...")

    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(
            dataloader, desc="ViT Embedding Extraction"
        ):
            # ViT Processor expects a list of images
            inputs = vit_processor(images=batch_images, return_tensors="pt").to(device)

            # Forward pass
            outputs = vit_model(**inputs)

            # Extract Pooler Output (The [CLS] token representation)
            embeddings = outputs.pooler_output

            # Normalize embedding (L2 Norm)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
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

    # Load ViT Model and Processor
    vit_model, vit_processor = load_vit_model(device=device)

    # Extract and Save Embeddings (batch processing)
    extract_vit_embeddings(
        image_dir="path/to/image/directory",
        output_path="vit_image_embeddings.pt",
        vit_model=vit_model,
        vit_processor=vit_processor,
        batch_size=64,
        num_workers=4,  # Set to 0 on Windows
    )
