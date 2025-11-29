import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils import load_gpt2_tokenizer


@dataclass
class CaptionData:
    """
    Data class to hold caption information.
    Each caption is associated with an image. Do note that multiple captions can correspond to the same image.

    Attributes:
        image_id (int): The id of the caption's associated image.
        embedding_index (int): Index to retrieve the CLIP embedding of the caption's associated image.
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
        tokenizer: PreTrainedTokenizer = load_gpt2_tokenizer(),
        max_length: int = 50,
        normalize_embeddings: bool = False,
    ):
        """
        Args:
            embeddings_path (str): Path to the .pt file containing image filenames and pre-computed CLIP embeddings.
            annotations_path (str): Path to captions_train2014.json.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing captions. Defaults to GPT-2 tokenizer.
            max_length (int, optional): Maximum length for tokenized captions. Defaults to 50.
            normalize_embeddings (bool, optional): If True, re-normalizes embeddings (usually not needed if already done).
                Defaults to False.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Load the Pre-Computed Embeddings (.pt file)
        data: dict = torch.load(embeddings_path)
        self.clip_embeddings: torch.Tensor = data["embeddings"]
        self.image_filenames: list[str] = data["filenames"]
        # an image with filename at index i in image_filenames
        # has its corresponding embedding at index i in clip_embeddings

        # Map each image ID to its corresponding index in the clip_embeddings tensor
        self.image_id_to_index: dict[int, int] = {
            self.get_image_id_from_filename(fname): idx
            for idx, fname in enumerate(self.image_filenames)
        }

        # Load COCO Annotations
        with open(annotations_path, "r") as f:
            coco_json: dict = json.load(f)

        # Build the list of CaptionData entries
        self.captions: list[CaptionData] = [
            CaptionData(
                image_id=ann["image_id"],
                embedding_index=self.image_id_to_index[ann["image_id"]],
                caption_text=ann["caption"],
            )
            for ann in coco_json["annotations"]
        ]

        print(
            f"Dataset ready: {len(self.captions)} captions for {len(self.image_id_to_index)} images."
        )

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

    @staticmethod
    def get_filename_from_image_id(image_id: int) -> str:
        """
        Constructs the COCO filename from an image ID.
        E.g., 123456 -> 'COCO_train2014_000000123456.jpg'

        Args:
            image_id (int): The COCO image ID.

        Returns:
            str: The constructed filename.
        """
        return f"COCO_train2014_{image_id:012d}.jpg"

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        caption_data = self.captions[idx]

        # Get CLIP image embedding
        clip_embedding = self.clip_embeddings[caption_data.embedding_index]
        if self.normalize_embeddings:
            clip_embedding = clip_embedding / clip_embedding.norm(2, -1)

        # Tokenize caption text
        encoding = self.tokenizer(
            caption_data.caption_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer: (1, max_length) to (max_length,)
        token_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        return {
            "token_ids": token_ids,
            "clip_embedding": clip_embedding,
            "attention_mask": attention_mask,
            "caption_text": caption_data.caption_text,
            "image_id": caption_data.image_id,
        }
