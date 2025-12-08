import os
import torch
import torchvision.transforms.v2 as v2
from typing import Literal, Sequence
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import ImageDirectoryDataset

WEIGHTS_FILE = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BACKBONE_WEIGHTS_FILE = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
RESIZE_DEFAULT_SIZE = 256
CROP_DEFAULT_SIZE = 224
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225) 


def load_dinov3_models(
    model_weights_dir: str,
    repo_or_dir: str = "facebookresearch/dinov3",
    source: Literal["local", "github"] = "github",
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, object]:
    """
    Loads the DINOv3.txt model and tokenizer.
    Moves the model to the specified device and sets it to evaluation mode.

    Args:
        model_weights_dir (str): The directory containing the DINOv3 model weight (`.pth`) files.
            Should contain both `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`
            and `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
        repo_or_dir (str, optional): If `source` is 'github', this should correspond to a github repo with format
            `repo_owner/repo_name[:ref]` with an optional ref (tag or branch), for example 'pytorch/vision:0.10'.
            If `ref` is not specified, the default branch is assumed to be `main` if it exists, and otherwise `master`.
            If `source` is 'local'  then it should be a path to a local directory.
            Defaults to "facebookresearch/dinov3" (the official DINOv3 github repository).
        source (Literal["local", "github"], optional): Specifies whether `repo_or_dir` is interpreted as a local directory
            path or a github repository. Defaults to "github".
        device (torch.device | None, optional): Device to move the model to. Defaults to None (auto-detect).

    Raises:
        FileNotFoundError: If model weight files are not found in the specified directory.

    Returns:
        tuple[torch.nn.Module, object]: The loaded DINOv3 model and tokenizer.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINOv3 model from '{model_weights_dir}' on device: {device}...")

    # Look into specified directory for both model parameter `.pth` files
    vision_head_and_text_encoder_path = None
    pretrain_lvd1689m_path = None
    for file_name in os.listdir(model_weights_dir):
        if file_name == WEIGHTS_FILE:
            vision_head_and_text_encoder_path = os.path.join(
                model_weights_dir, file_name
            )
        elif file_name == BACKBONE_WEIGHTS_FILE:
            pretrain_lvd1689m_path = os.path.join(model_weights_dir, file_name)

    if vision_head_and_text_encoder_path is None:
        raise FileNotFoundError(
            f"Could not find '{WEIGHTS_FILE}' in directory '{model_weights_dir}'"
        )
    if pretrain_lvd1689m_path is None:
        raise FileNotFoundError(
            f"Could not find '{BACKBONE_WEIGHTS_FILE}' in directory '{model_weights_dir}'"
        )

    # Load DINOv3 txt model from repository
    model, tokenizer = torch.hub.load(
        repo_or_dir,
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source=source,
        weights=vision_head_and_text_encoder_path,
        backbone_weights=pretrain_lvd1689m_path,
        trust_repo=True,
    )
    model.eval()

    return model.to(device), tokenizer

# why do i directly copy the code for preprocessor from the repo? because the torch.hub.load does not help on getting it, it only helps on getting the model and tokenizer.
# if can find another way to get the preprocessor from the repo, then can replace all these code
def make_eval_transform(
    *,
    resize_size: int,
    crop_size: int,
    interpolation,
    mean: Sequence[float],
    std: Sequence[float],
    resize_square: bool = False,
    resize_large_side: bool = False,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    
    if resize_square:
        transforms_list.append(v2.Resize((resize_size, resize_size), interpolation=interpolation))
    elif resize_large_side:
        transforms_list.append(v2.Resize(resize_size, interpolation=interpolation, antialias=True))
    else:
        transforms_list.append(v2.Resize(resize_size, interpolation=interpolation))

    transforms_list.append(v2.CenterCrop(crop_size))
    
    transforms_list.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    
    return v2.Compose(transforms_list)

def get_dinov3_preprocessor(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )

def extract_dino_embeddings(
    image_dir: str,
    output_path: str,
    dino_model: torch.nn.Module,
    dino_processor: v2.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> None:
    # load Image
    device = device or dino_model.device

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
    print(f"Starting DINO embedding extraction for {len(dataset)} images...")
    with torch.no_grad():
        for batch_filenames, batch_images in tqdm(dataloader, desc="DINO Embedding Extraction"):
            # batch_images is a list of PIL images (or whatever ImageDirectoryDataset yields)
            # apply the single-image processor to each image, then stack to get (B, C, H, W)
            processed = [dino_processor(img) for img in batch_images]
            # ensure each processed item is a tensor CxHxW
            image_tensor = torch.stack(processed, dim=0).to(device)

            image_features = dino_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu())
            all_filenames.extend(batch_filenames)

    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Saving {final_embeddings.shape[0]} embeddings to {output_path}...")
    torch.save(
        {"filenames": all_filenames, "embeddings": final_embeddings}, output_path
    )

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model, _ = load_dinov3_models(model_weights_dir="dino_weights", device=device)
    dino_processor = get_dinov3_preprocessor()

    # Extract and Save Embeddings (batch processing)
    extract_dino_embeddings(
        image_dir="test_image",
        output_path="dino_image_embeddings.pt",
        dino_model=dino_model,
        dino_processor=dino_processor,
        batch_size=64,
        num_workers=4,  # Set to 0 on Windows
        device=device,
    )