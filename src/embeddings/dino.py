import math 
from typing import Union, Tuple, Any
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the DINOv3 ViT-L/16 vision backbone architecture and its pre-trained weight options
from dinov3.hub.backbones import dinov3_vitl16, Weights as BackboneWeights
# Import the main DINO-TXT model class and its configuration settings
from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
# Import the text encoder component required for DINO-TXT model initialization
from dinov3.eval.text.text_transformer import TextTransformer
# Import the pre-trained weight options for the complete DINO-TXT model
from dinov3.eval.text.dinotxt_model import DINOTxtWeights

# TODO: move this base URL to a proper location
DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"

from src.utils import ImageDirectoryDataset


def load_dinotxt_image_encoder(
    device: torch.device | None = None,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    backbone_weights: Union[BackboneWeights, str] = BackboneWeights.LVD1689M,
) -> Tuple[torch.nn.Module, Any]:
    """
    Loads the DINOv3-TXT image encoder (ViT-L/16 backbone) with official pre-trained weights.

    Initializes the full DINOv3-TXT cross-modal model (image + text), loads pre-trained weights,
    then extracts and returns only the vision encoder for image embedding extraction. The text encoder
    is initialized as a placeholder to maintain model architecture integrity (not used for inference).

    Args:
        device (torch.device | None, optional): Device to run the model on (GPU/CPU).
            Defaults to None (auto-detect: CUDA if available, otherwise CPU).
        weights (Union[DINOTxtWeights, str], optional): Pre-trained weights version for DINOv3-TXT.
            Defaults to DINOTxtWeights.LVTD2300M (the only official pre-trained version).
        backbone_weights (Union[BackboneWeights, str], optional): Pre-trained weights for the ViT-L/16 vision backbone.
            Defaults to BackboneWeights.LVD1689M.

    Returns:
        tuple[torch.nn.Module, Any]: 
            - torch.nn.Module: Isolated DINOv3-TXT vision encoder (outputs 2048-dimensional image embeddings)
            - Any: Dummy processor (None) for interface compatibility with original CLIP-based code
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating dinotxt model on device: {device}")

    # config of dino-txt
    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )

    # Load backbones
    vision_backbone = dinov3_vitl16(pretrained=True, weights=backbone_weights).to(device)
    # text encoder Not used, but load to keep model architecture complete
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        dropout_prob=0.0,
    ).to(device)

    model = DINOTxt(
        model_config=dinotxt_config,
        vision_backbone=vision_backbone,
        text_backbone=text_backbone
    ).to(device)

    url = f"{DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

    state_dict = torch.hub.load_state_dict_from_url(url, check_hash=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    image_encoder = model.visual_model
    dummy_processor = None

    return image_encoder, dummy_processor


def extract_dinotxt_embedding_from_image(
    image: str | Image.Image,
    dinotxt_encoder: torch.nn.Module,
    dummy_processor: Any,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extracts the normalized DINOv3-TXT image embedding for a single image.

    Preprocesses the input image (resize to 224×224, normalize to [-1, 1]) and runs it through
    the pre-loaded DINOv3-TXT vision encoder to generate a 2048-dimensional embedding, then
    applies L2 normalization to the result.

    Args:
        image (str | Image.Image): Path to an image file (str) or a PIL Image object (Image.Image).
            The image will be automatically converted to RGB format.
        dinotxt_encoder (torch.nn.Module): Pre-loaded DINOv3-TXT vision encoder (from load_dinotxt_image_encoder).
        dummy_processor (Any): Dummy processor (None) for interface compatibility with original CLIP-based code.
            This parameter is unused in the function logic.
        device (torch.device | None, optional): Device to run the model on (GPU/CPU).
            Defaults to None (uses the device of the dinotxt_encoder).

    Returns:
        torch.Tensor: The L2-normalized DINOv3-TXT image embedding with shape (2048,).
            Batch dimension is removed (squeezed) for direct use.
    """
    device = device or next(dinotxt_encoder.parameters()).device

    # Load image if a path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    
    with torch.no_grad():
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        inputs = preprocess(image).unsqueeze(0).to(device)

        image_features = dinotxt_encoder(inputs)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.squeeze(0) 


def extract_dinotxt_embeddings(
    image_dir: str,
    output_path: str,
    dinotxt_encoder: torch.nn.Module,
    dummy_processor: Any,
    batch_size: int = 16,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> None:
    """
    Extracts normalized DINOv3-TXT image embeddings for all images in a directory and saves results to a `.pt` file.

    Processes images in batches for efficiency (far faster than single-image extraction). 
    Each image is preprocessed (resize to 224×224, normalize to [-1, 1]), run through the DINOv3-TXT vision encoder
    to generate 2048-dimensional embeddings (L2-normalized), then all embeddings + corresponding filenames are saved
    as a PyTorch `.pt` file for later use.

    Args:
        image_dir (str): Path to the directory containing images (supports common formats like JPG/PNG).
        output_path (str): Path to save the output `.pt` file (e.g., "dinotxt_embeddings.pt").
            The saved file contains a dictionary with keys: "filenames" (list of image paths) and "embeddings" (tensor of shape [N, 2048]).
        dinotxt_encoder (torch.nn.Module): Pre-loaded DINOv3-TXT vision encoder (from load_dinotxt_image_encoder).
        dummy_processor (Any): Dummy processor (None) for interface compatibility with original CLIP-based code.
            This parameter is unused in the function logic.
        batch_size (int, optional): Number of images to process per batch. Higher values are faster but consume more VRAM.
            Defaults to 16 (optimized for 8GB GPU; adjust to 32 for 12GB+ GPUs, 8 for 4GB GPUs).
        num_workers (int, optional): Number of CPU worker threads for parallel image loading.
            Set to 0 on Windows (avoids multiprocessing errors); use 4-8 on Linux/macOS. Defaults to 4.
        device (torch.device | None, optional): Device to run the model on (GPU/CPU).
            Defaults to None (uses the device of the dinotxt_encoder).

    Note:
        Ensure all images in `image_dir` are readable (no corrupted files) to avoid DataLoader errors.
        Embeddings are moved to CPU before saving to reduce VRAM usage during processing.
    """

    # Setup Device
    device = device or next(dinotxt_encoder.parameters()).device
    dinotxt_encoder.eval()

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

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Extraction Loop
    print(f"Starting dinotxt embedding extraction for {len(dataset)} images...")
    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(
            dataloader, desc="DINO-TXT Embedding Extraction"
        ):
            batch_tensors = torch.stack([preprocess(img.convert("RGB")) for img in batch_images]).to(device)

            image_features = dinotxt_encoder(batch_tensors) 

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

    dinotxt_encoder, dummy_processor = load_dinotxt_image_encoder(device=device)

    # Extract and Save Embeddings (batch processing)
    extract_dinotxt_embeddings(
        image_dir="path/to/image/directory",
        output_path="dinotxt_image_embeddings.pt",
        dinotxt_encoder=dinotxt_encoder,
        dummy_processor=dummy_processor,
        batch_size=16,
        num_workers=4,
    )
