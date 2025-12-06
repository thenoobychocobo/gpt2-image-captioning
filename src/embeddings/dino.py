import os
from typing import Literal

import torch

WEIGHTS_FILE = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BACKBONE_WEIGHTS_FILE = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


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

    # Load DINOv3 txt model from repositroy
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

    # tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
    # with torch.autocast('cuda', dtype=torch.float):
    #     with torch.no_grad():
    #         image_features = model.encode_image(image_tensor)
    #         text_features = model.encode_text(tokenized_texts_tensor)
    #     # image_features /= image_features.norm(dim=-1, keepdim=True)
    #     # text_features /= text_features.norm(dim=-1, keepdim=True)
