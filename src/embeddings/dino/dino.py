import math 
from typing import Union, Tuple, Any, List
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# Import the DINOv3 ViT-L/16 vision backbone architecture and its pre-trained weight options
from src.embeddings.dino.dinov3.dinov3.hub.backbones import dinov3_vitl16, Weights as BackboneWeights
# Import the main DINO-TXT model class and its configuration settings
from src.embeddings.dino.dinov3.dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
# Import the text encoder component required for DINO-TXT model initialization
from src.embeddings.dino.dinov3.dinov3.eval.text.text_transformer import TextTransformer
# Import the pre-trained weight options for the complete DINO-TXT model
from src.embeddings.dino.dinov3.dinov3.eval.text.dinotxt_model import DINOTxtWeights

# TODO: move this base URL to a proper location
DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"

from src.utils import ImageDirectoryDataset

def load_dinotxt_image_encoder(
    device: torch.device | None = None,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    backbone_weights: Union[BackboneWeights, str] = BackboneWeights.LVD1689M,
) -> Tuple[torch.nn.Module, DINOTxtImageProcessor]:
    """
    Loads the DINOv3-TXT image encoder + reusable image processor.

    Initializes the full DINOv3-TXT cross-modal model (image + text), loads pre-trained weights,
    then extracts and returns only the vision encoder + a dedicated preprocessing processor.
    The text encoder is initialized as a placeholder to maintain model architecture integrity.

    Args:
        device (torch.device | None, optional): Device to run the model on (GPU/CPU).
            Defaults to None (auto-detect: CUDA if available, otherwise CPU).
        weights (Union[DINOTxtWeights, str], optional): Pre-trained weights version for DINOv3-TXT.
            Defaults to DINOTxtWeights.LVTD2300M (the only official pre-trained version).
        backbone_weights (Union[BackboneWeights, str], optional): Pre-trained weights for the ViT-L/16 vision backbone.
            Defaults to BackboneWeights.LVD1689M.

    Returns:
        tuple[torch.nn.Module, DINOTxtImageProcessor]: 
            - torch.nn.Module: Isolated DINOv3-TXT vision encoder (outputs 2048-dimensional image embeddings)
            - DINOTxtImageProcessor: Reusable processor for DINO-TXT image preprocessing (consistent with CLIP API)
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
    # Create and return the reusable DINO-TXT processor (instead of dummy None)
    dinotxt_processor = DINOTxtImageProcessor(image_size=dinotxt_config.vision_model_train_img_size)

    return image_encoder, dinotxt_processor


def extract_dinotxt_embedding_from_image(
    image: str | Image.Image,
    dinotxt_encoder: torch.nn.Module,
    dinotxt_processor: DINOTxtImageProcessor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extracts the normalized DINOv3-TXT image embedding for a single image.

    Uses the reusable DINO-TXT processor for preprocessing (consistent with CLIP workflow).

    Args:
        image (str | Image.Image): Path to an image file or a PIL Image object.
        dinotxt_encoder (torch.nn.Module): Pre-loaded DINOv3-TXT vision encoder.
        dinotxt_processor (DINOTxtImageProcessor): Reusable DINO-TXT image processor (from load_dinotxt_image_encoder).
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).

    Returns:
        torch.Tensor: The L2-normalized DINOv3-TXT image embedding with shape (2048,).
    """
    device = device or next(dinotxt_encoder.parameters()).device

    with torch.no_grad():
        # Use the reusable processor (consistent with CLIP's processor API)
        inputs = dinotxt_processor(images=image, return_tensors="pt").to(device)
        image_features = dinotxt_encoder(inputs["pixel_values"])

        # Normalize embedding (L2 Norm)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.squeeze(0)  # Remove batch dimension


def extract_dinotxt_embeddings(
    image_dir: str,
    output_path: str,
    dinotxt_encoder: torch.nn.Module,
    dinotxt_processor: DINOTxtImageProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> None:
    """
    Extracts normalized DINOv3-TXT image embeddings for all images in a directory and saves results to a `.pt` file.
    Uses the reusable DINO-TXT processor (eliminates code duplication).

    Args:
        image_dir (str): Path to the directory containing images.
        output_path (str): Path to save the resulting `.pt` file.
        dinotxt_encoder (torch.nn.Module): Pre-loaded DINOv3-TXT vision encoder.
        dinotxt_processor (DINOTxtImageProcessor): Reusable DINO-TXT image processor (from load_dinotxt_image_encoder).
        batch_size (int, optional): Number of images to process at once. Defaults to 32.
        num_workers (int, optional): Number of CPU threads for image loading. Defaults to 4.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).
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

    # Extraction Loop
    print(f"Starting dinotxt embedding extraction for {len(dataset)} images...")
    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(
            dataloader, desc="DINO-TXT Embedding Extraction"
        ):
            # Use the reusable processor for batch processing (no duplicated preprocess code)
            inputs = dinotxt_processor(images=batch_images, return_tensors="pt").to(device)
            image_features = dinotxt_encoder(inputs["pixel_values"])

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

class DINOTxtImageProcessor:
    """
    Reusable image processor for DINO-TXT (mimics Hugging Face CLIPProcessor API).
    Encapsulates all official DINO-TXT preprocessing steps for consistency.
    """
    def __init__(self, image_size: int = 224):
        """
        Initialize the DINO-TXT image processor.
        
        Args:
            image_size (int): Target size for image resizing (matches DINO-TXT training config).
                              Defaults to 224 (official DINO-TXT setting).
        """
        self.image_size = image_size
        # Official DINO-TXT preprocessing pipeline (normalization to [-1, 1])
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, images: Union[str, Image.Image, List[Image.Image]], return_tensors: str = "pt") -> dict:
        """
        Process single/multiple images (mimics CLIPProcessor's API for consistency).
        
        Args:
            images: Input to process - can be:
                    - Path to single image (str)
                    - Single PIL Image object
                    - List of PIL Image objects (for batch processing)
            return_tensors (str): Type of tensors to return (only "pt" (PyTorch) is supported).
        
        Returns:
            dict: Dictionary with "pixel_values" key containing processed tensor(s).
        """
        if return_tensors != "pt":
            raise ValueError(f"Only 'pt' (PyTorch) tensors are supported, got {return_tensors}")
        
        # Handle single image path
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")
        
        # Process single image
        if isinstance(images, Image.Image):
            images = images.convert("RGB")
            pixel_values = self.preprocess(images).unsqueeze(0)  # Add batch dim
        
        # Process batch of images
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            images = [img.convert("RGB") for img in images]
            pixel_values = torch.stack([self.preprocess(img) for img in images])
        
        else:
            raise TypeError(
                f"Unsupported image type(s): {type(images)}. "
                "Supported types: str (image path), PIL.Image.Image, list of PIL.Image.Image"
            )
        
        return {"pixel_values": pixel_values}

# === Example Usage ===
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder + reusable processor (no more dummy None!)
    dinotxt_encoder, dinotxt_processor = load_dinotxt_image_encoder(device=device)

    # Extract single image embedding (using the processor)

    # single_embedding = extract_dinotxt_embedding_from_image(
    #     image="path/to/single/image.jpg",
    #     dinotxt_encoder=dinotxt_encoder,
    #     dinotxt_processor=dinotxt_processor,
    #     device=device
    # )
    # print(f"Single image embedding shape: {single_embedding.shape}")  # Should be (2048,)

    # Extract and Save Batch Embeddings (using the same processor)

    extract_dinotxt_embeddings(
        image_dir="test_image_dir",
        output_path="dinotxt_image_embeddings.pt",
        dinotxt_encoder=dinotxt_encoder,
        dinotxt_processor=dinotxt_processor,
        batch_size=16,
        num_workers=4,
    )
