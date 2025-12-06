import json
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils import load_gpt2_tokenizer


def split_coco_annotations(
    annotations_path: str, output_dir: str, split_ratio: float = 0.8, seed: int = 42
) -> None:
    """
    Split the COCO annotations JSON file into training and validation sets based on image IDs (not split by captions).
    When creating a `CocoDataset` instance, use the resulting JSON files (but the same `.pt` embeddings file).

    Args:
        annotations_path (str): Path to original COCO annotations JSON file.
        output_dir (str): Directory to save the split JSON files.
        split_ratio (float, optional): Ratio of training data (e.g., a ratio of 0.8 means 80% training data and 20% validation data).
            Defaults to 0.8.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # 1. Extract unique Image IDs
    # We split by Image ID (not caption) to ensure no data leakage.
    # An image and ALL its captions must go together into either Train or Val.
    images = coco_data["images"]  # Image metadata list
    annotations = coco_data["annotations"]  # Caption list

    unique_img_ids: list[int] = [img["id"] for img in images]

    # 2. Shuffle and Split IDs
    random.seed(seed)
    random.shuffle(unique_img_ids)

    cutoff = int(len(unique_img_ids) * split_ratio)
    train_ids = set(unique_img_ids[:cutoff])
    val_ids = set(unique_img_ids[cutoff:])

    print(f"Splitting: {len(train_ids)} Train images, {len(val_ids)} Val images.")

    # 3. Filter the Content
    # Create subset lists for Train
    train_images = [img for img in images if img["id"] in train_ids]
    train_anns = [ann for ann in annotations if ann["image_id"] in train_ids]

    # Create subset lists for Val
    val_images = [img for img in images if img["id"] in val_ids]
    val_anns = [ann for ann in annotations if ann["image_id"] in val_ids]

    # 4. Construct and Save new JSONs
    # We keep the original 'info' and 'licenses' to maintain COCO format
    common_info = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
    }

    train_data = {**common_info, "images": train_images, "annotations": train_anns}
    val_data = {**common_info, "images": val_images, "annotations": val_anns}

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_split.json")
    val_path = os.path.join(output_dir, "val_split.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f)

    with open(val_path, "w") as f:
        json.dump(val_data, f)

    print(f"Created:\n- {train_path}\n- {val_path}")


@dataclass
class CaptionData:
    """
    Data class to hold caption information.
    Each caption is associated with an image. Do note that multiple captions can correspond to the same image.

    Attributes:
        image_id (int): The id of the caption's associated image.
        embedding_index (int): Index to retrieve the image embedding of the caption's associated image.
        caption_text (str): The caption text.
    """

    image_id: int
    embedding_index: int
    caption_text: str


class CocoDataset(Dataset):
    """
    COCO Dataset for image captioning.
    Each item in the dataset corresponds to a caption, in the form of a `CaptionData` instance.
    Each caption is associated with an image, and multiple captions can correspond to the same image.
    """

    def __init__(
        self,
        embeddings_path: str,
        annotations_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        max_length: int = 50,
        normalize_embeddings: bool = False,
    ):
        """
        Args:
            embeddings_path (str): Path to the .pt file containing image filenames and pre-computed image embeddings.
            annotations_path (str): Path to captions_train2014.json.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for processing captions. Defaults to None, in which case the GPT-2 tokenizer is loaded.
            max_length (int, optional): Maximum length for tokenized captions. Defaults to 50.
            normalize_embeddings (bool, optional): If True, re-normalizes embeddings (usually not needed if already done).
                Defaults to False.
        """
        self.tokenizer = tokenizer or load_gpt2_tokenizer()
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Load the Pre-Computed Embeddings (.pt file)
        data: dict = torch.load(embeddings_path)
        self.image_embeddings: torch.Tensor = data["embeddings"]
        self.image_filenames: list[str] = data["filenames"]
        # an image with filename at index i in image_filenames
        # has its corresponding embedding at index i in image_embeddings

        # Map each image ID to its corresponding index in the image_embeddings tensor
        self.image_id_to_index: dict[int, int] = {
            self.get_image_id_from_filename(fname): idx
            for idx, fname in enumerate(self.image_filenames)
        }

        # Load COCO Annotations
        with open(annotations_path, "r") as f:
            coco_data: dict = json.load(f)

        # Build the list of CaptionData entries
        self.captions: list[CaptionData] = [
            CaptionData(
                image_id=ann["image_id"],
                embedding_index=self.image_id_to_index[ann["image_id"]],
                caption_text=ann["caption"],
            )
            for ann in coco_data["annotations"]
        ]

        print(f"Dataset ready: {len(self.captions)} captions.")

    @staticmethod
    def get_image_id_from_filename(filename: str) -> int:
        """
        Extracts the image ID from a COCO filename.
        E.g., 'COCO_train2014_000000123456.jpg' -> 123456

        Args:
            filename (str): The COCO filename.

        Returns:
            int: The extracted image ID.
        """
        return int(filename.split("_")[-1].split(".")[0])

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        caption_data = self.captions[idx]

        # Get image embedding
        image_embedding = self.image_embeddings[caption_data.embedding_index]
        if self.normalize_embeddings:
            image_embedding = image_embedding / image_embedding.norm(2, -1)

        # Tokenize caption text
        encoding = self.tokenizer(
            caption_data.caption_text
            + self.tokenizer.eos_token,  # Explicitly add EOS token
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer: (1, max_length) to (max_length,)
        token_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # Create labels as a copy of token ids
        labels = token_ids.clone()

        # Replace padding token ids in labels with -100 to ignore them in loss computation
        labels[attention_mask == 0] = -100

        # NOTE: The EOS token that was explicitly added will have an attention mask value of 1
        # This is the case even if the GPT-2's tokenizer's pad_token is set to be its eos_token (by standard convention).
        # The tokenizer will still treat the explicitly added EOS token as a valid token (attention mask = 1),
        # while any padding tokens added for max_length will have attention mask = 0, even though they share the same token ID.
        # In this way, the model learns to predict the EOS token, but not the padding tokens.
        # Similarly, with our code above, the EOS token will have its label preserved (and will contribute to loss computation),
        # while padding tokens will have label -100 (ignored by loss computation).

        return {
            "token_ids": token_ids,
            "labels": labels,
            "image_embedding": image_embedding,
            "attention_mask": attention_mask,
            "caption_text": caption_data.caption_text,
            "image_id": caption_data.image_id,
        }
