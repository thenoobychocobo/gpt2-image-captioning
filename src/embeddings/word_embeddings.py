import json
from collections import defaultdict

import torch
import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.embeddings.clip import load_clip_model


def load_image_embeddings_file_names(
    image_dir: str,
) -> list[str]:
    """
    Loads image embedding filenames from a .pt file.
    """
    # load the pt file
    data_pt = torch.load(image_dir)
    return data_pt["filenames"]


def load_captions_annotations(
    file_name: str,
) -> list[str]:
    with open(file_name, "r") as f:
        data = json.load(f)
    return data["annotations"]


def get_image_id_from_filename(filename: str) -> int:
    """Extracts the image ID from a COCO filename.
    E.g., 'COCO_train2014_000000123456.jpg' -> 123456
    """
    base_name = filename.split("_")[-1]  # '000000123456.jpg'
    image_id_str = base_name.split(".")[0]  # '000000123456'
    return int(image_id_str)


def map_caption_id_to_caption(annotations_list):
    image_to_captions = defaultdict(list)
    for ann in annotations_list:
        image_to_captions[ann["image_id"]].append(
            {"caption_id": ann["id"], "caption": ann["caption"]}
        )
    return image_to_captions


def extract_clip_embedding_from_caption(
    caption: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extracts the CLIP embedding for a single caption.

    Args:
        caption (str): The caption text.
        clip_model (CLIPModel): Pre-loaded CLIP model.
        clip_processor (CLIPProcessor): Pre-loaded CLIP processor.
        device (torch.device | None, optional): Device to run the model on. Defaults to None (uses model's device).

    Returns:
        torch.Tensor: The normalized CLIP text embedding with shape (embedding_dim,).
    """
    device = device or clip_model.device

    with torch.no_grad():
        # Preprocess caption
        inputs = clip_processor(text=[caption], return_tensors="pt", padding=True).to(
            device
        )

        # Forward pass through CLIP Text Model
        text_features = clip_model.get_text_features(**inputs)

        # Normalize embedding (L2 Norm)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.squeeze(
            0
        )  # Remove batch dimension; shape (embedding_dim,)


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get file names from embeddings file
    embedding_file = "file_path_to_train_val_clip_embeddings.pt"
    file_names = load_image_embeddings_file_names(embedding_file)

    # Load CLIP Model and Processor
    clip_model, clip_processor = load_clip_model(device=device)

    # Load captions annotations
    annotations_file = "file_path_to_captions_train2014.json"
    annotations_list = load_captions_annotations(annotations_file)

    # Map image IDs to captions
    image_to_captions = map_caption_id_to_caption(annotations_list)

    # Collect all captions first for batch processing
    all_captions_data = []
    for filename in file_names:
        image_id = get_image_id_from_filename(filename)
        captions_info = image_to_captions.get(image_id, [])

        if not captions_info:
            continue

        for cap_info in captions_info:
            all_captions_data.append(
                {
                    "filename": filename,
                    "caption_id": cap_info["caption_id"],
                    "caption": cap_info["caption"],
                }
            )

    print(f"Total captions to process: {len(all_captions_data)}")

    # Process captions in batches
    batch_size = 32  # ADDED: Batch size for efficiency
    processed_embeddings = {}

    # Batch processing loop with progress bar
    for i in tqdm(
        range(0, len(all_captions_data), batch_size), desc="Processing batches"
    ):
        batch = all_captions_data[i : i + batch_size]
        captions = [item["caption"] for item in batch]

        # Process entire batch at once
        with torch.no_grad():
            inputs = clip_processor(
                text=captions, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            embeddings = clip_model.get_text_features(**inputs)
            embeddings = embeddings.cpu().numpy()

        # Organize embeddings by filename
        for j, item in enumerate(batch):
            filename = item["filename"]
            if filename not in processed_embeddings:
                processed_embeddings[filename] = []

            processed_embeddings[filename].append(
                {"caption_id": item["caption_id"], "embedding": embeddings[j]}
            )

    # Convert to final structure
    processed_data = [
        {"filenames": filename, "embeddings": embeddings}
        for filename, embeddings in processed_embeddings.items()
    ]

    print(f"Processed {len(processed_data)} images")

    # Save the processed data
    output_path = "file_path_to_caption_embeddings_structured.pt"
    torch.save(processed_data, output_path)
    print(f"Saved to {output_path}")

    # Verify the structure
    sample = processed_data[0]
    print("\nSample structure:")
    print(f"Filename (image_id): {sample['filenames']}")
    print(f"Number of captions: {len(sample['embeddings'])}")
    print(f"First caption_id: {sample['embeddings'][0]['caption_id']}")
    print(f"Embedding shape: {sample['embeddings'][0]['embedding'].shape}")
