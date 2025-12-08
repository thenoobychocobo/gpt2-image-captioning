import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CocoDataset
from src.models import ImageCaptioningModel, RetrievalAugmentedTransformer


def generate_test_caption_predictions(
    model: ImageCaptioningModel,
    test_dataset: CocoDataset,
    batch_size: int,
    num_workers: int = 4,
    max_length: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    output_json_path: str = "results.json",
    device: torch.device | None = None,
) -> list[dict]:
    """
    Uses the provided `ImageCaptioningModel` to generate captions for all images in the test dataset.

    Args:
        model (ImageCaptioningModel): The trained captioning model.
        test_dataset (CocoDataset): The test dataset containing images and metadata.
        batch_size (int): The batch size for generating captions.
        num_workers (int, optional): The number of CPU threads for data loading. Should be 0 on Windows. Defaults to 4.
        max_length (int, optional): The maximum length of generated captions. Defaults to 50.
        top_p (float, optional): The nucleus sampling probability for generation. Defaults to 0.9.
        temperature (float, optional): The temperature for sampling during generation. Defaults to 1.0.
        output_json_path (str, optional): The file path to save the generated captions in JSON format. Defaults to "results.json".
        device (torch.device | None, optional): The device to run the model on. Defaults to None (auto-detect).

    Returns:
        list[dict]: A list of dictionaries containing image IDs and their corresponding generated captions.
    """
    # Device and model setup
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    model.eval()  # This is already done when model.generate() is called, but just to be explicit

    # Data Loader
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Generate captions for each image in the test dataset
    results: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Captions"):
            # Unpack batch and move to device
            # We only need image_ids and image_embeddings for generation
            image_ids: torch.Tensor = batch["image_id"]  # Can keep on CPU
            image_embeddings: torch.Tensor = batch["image_embedding"].to(device)

            # Generate captions
            generated_ids = model.generate(
                image_embeddings=image_embeddings,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
            )

            # Decode generated captions
            tokenizer = test_dataset.tokenizer
            generated_captions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Store results in COCO format
            for image_id, caption in zip(image_ids, generated_captions):
                results.append({"image_id": image_id.item(), "caption": caption})

    # Save results to JSON file
    with open(output_json_path, "w") as f:
        json.dump(results, f)

    return results


def generate_test_caption_predictions_rat(
    model: RetrievalAugmentedTransformer,
    db_store,
    top_k: int,
    top_i: int,
    test_dataset: CocoDataset,
    batch_size: int,
    num_workers: int = 4,
    max_length: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    output_json_path: str = "results.json",
    device: torch.device | None = None,
) -> list[dict]:
    """
    Uses the provided `RetrievalAugmentedTransformer` to generate captions for all images in the test dataset.

    Args:
        model (RetrievalAugmentedTransformer): The trained captioning model.
        db_store: The vector db store containing the image and caption embeddings.
        top_k (int): The number of top similar images to retrieve from the database.
        top_i (int): The number of top captions to retrieve from the database.
        test_dataset (CocoDataset): The test dataset containing images and metadata.
        batch_size (int): The batch size for generating captions.
        num_workers (int, optional): The number of CPU threads for data loading. Should be 0 on Windows. Defaults to 4.
        max_length (int, optional): The maximum length of generated captions. Defaults to 50.
        top_p (float, optional): The nucleus sampling probability for generation. Defaults to 0.9.
        temperature (float, optional): The temperature for sampling during generation. Defaults to 1.0.
        output_json_path (str, optional): The file path to save the generated captions in JSON format. Defaults to "results.json".
        device (torch.device | None, optional): The device to run the model on. Defaults to None (auto-detect).

    Returns:
        list[dict]: A list of dictionaries containing image IDs and their corresponding generated captions.
    """
    # Device and model setup
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    model.eval()  # This is already done when model.generate() is called, but just to be explicit

    # Data Loader
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Generate captions for each image in the test dataset
    results: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Captions"):
            # Unpack batch and move to device
            # We only need image_ids and image_embeddings for generation
            image_ids: torch.Tensor = batch["image_id"]  # Can keep on CPU
            image_embeddings: torch.Tensor = batch["image_embedding"].to(device)

            # Generate captions
            generated_ids = model.generate(
                db_store=db_store,
                top_k=top_k,
                top_i=top_i,
                image_embeddings=image_embeddings,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
            )

            # Decode generated captions
            tokenizer = test_dataset.tokenizer
            generated_captions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Store results in COCO format
            for image_id, caption in zip(image_ids, generated_captions):
                results.append({"image_id": image_id.item(), "caption": caption})

    # Save results to JSON file
    with open(output_json_path, "w") as f:
        json.dump(results, f)

    return results
