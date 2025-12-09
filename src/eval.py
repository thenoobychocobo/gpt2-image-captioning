"""
Evaluation module for image captioning using COCO metrics.
Based on: https://cocodataset.org/#captions-eval

Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (n-gram precision)
- METEOR (semantic matching with synonyms/stemming)
- CIDEr (consensus-based TF-IDF weighted)
- ROUGE-L (longest common subsequence)
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvalMetrics:
    """Container for caption evaluation metrics."""

    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    # meteor: float
    rouge_l: float
    cider: float

    def to_dict(self) -> dict[str, float]:
        return {
            "BLEU-1": self.bleu_1,
            "BLEU-2": self.bleu_2,
            "BLEU-3": self.bleu_3,
            "BLEU-4": self.bleu_4,
            # "METEOR": self.meteor,
            "ROUGE-L": self.rouge_l,
            "CIDEr": self.cider,
        }

    def __str__(self) -> str:
        return (
            f"BLEU-1: {self.bleu_1:.4f} | BLEU-2: {self.bleu_2:.4f} | "
            f"BLEU-3: {self.bleu_3:.4f} | BLEU-4: {self.bleu_4:.4f} | "
            # f"METEOR: {self.meteor:.4f} |"
            f"ROUGE-L: {self.rouge_l:.4f} | CIDEr: {self.cider:.4f}"
        )


def compute_caption_metrics(
    predictions: dict[int, list[str]],
    references: dict[int, list[str]],
) -> EvalMetrics:
    """
    Compute COCO caption evaluation metrics.

    Args:
        predictions: Dict mapping image_id to list of generated captions (usually 1 caption per image).
        references: Dict mapping image_id to list of reference/ground-truth captions.

    Returns:
        EvalMetrics dataclass containing all metric scores.
    """
    # Filter to only images present in both predictions and references
    common_ids = set(predictions.keys()) & set(references.keys())
    if len(common_ids) == 0:
        raise ValueError("No common image IDs found between predictions and references")

    preds = {k: predictions[k] for k in common_ids}
    refs = {k: references[k] for k in common_ids}

    # Initialize scorers
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    results: dict[str, float] = {}

    for scorer, method in scorers:
        score, _ = scorer.compute_score(refs, preds)
        if isinstance(method, list):
            # BLEU returns a list of scores for BLEU-1 through BLEU-4
            for metric_name, metric_score in zip(method, score):
                results[metric_name] = metric_score
        else:
            results[method] = score

    return EvalMetrics(
        bleu_1=results["BLEU-1"],
        bleu_2=results["BLEU-2"],
        bleu_3=results["BLEU-3"],
        bleu_4=results["BLEU-4"],
        # meteor=results["METEOR"],
        rouge_l=results["ROUGE-L"],
        cider=results["CIDEr"],
    )


def load_coco_references(annotations_path: str) -> dict[int, list[str]]:
    """
    Load reference captions from a COCO annotations JSON file.

    Args:
        annotations_path: Path to COCO annotations JSON file.

    Returns:
        Dict mapping image_id to list of reference captions.
    """
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    references: dict[int, list[str]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in references:
            references[img_id] = []
        references[img_id].append(ann["caption"])

    return references


def evaluate_captions(
    predictions: list[dict[str, Any]],
    annotations_path: str,
) -> EvalMetrics:
    """
    Evaluate generated captions against ground truth using COCO metrics.

    Args:
        predictions: List of dicts with 'image_id' and 'caption' keys.
        annotations_path: Path to ground truth COCO annotations JSON.

    Returns:
        EvalMetrics dataclass containing all metric scores.
    """
    # Convert predictions list to dict format
    preds_dict: dict[int, list[str]] = {}
    for pred in predictions:
        img_id = pred["image_id"]
        preds_dict[img_id] = [pred["caption"]]

    # Load references
    refs_dict = load_coco_references(annotations_path)

    return compute_caption_metrics(preds_dict, refs_dict)


def generate_and_evaluate(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    annotations_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], EvalMetrics]:
    """
    Generate captions for a dataset and compute evaluation metrics.

    Args:
        model: The trained ImageCaptioningModel.
        dataset: CocoDataset instance to evaluate on.
        annotations_path: Path to ground truth annotations JSON.
        batch_size: Batch size for generation.
        num_workers: Number of CPU workers for data loading.
        max_length: Maximum length of generated captions.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions list, EvalMetrics).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    predictions: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            image_ids: torch.Tensor = batch["image_id"]
            image_embeddings: torch.Tensor = batch["image_embedding"].to(device)

            # Generate captions
            generated_ids = model.generate(
                image_embeddings=image_embeddings,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )

            # Decode generated captions
            tokenizer = dataset.tokenizer
            generated_captions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,  # strips away special tokens like <eos>
            )

            # Store results (avoid duplicates - one caption per image)
            for img_id, caption in zip(image_ids, generated_captions):
                img_id_int = img_id.item()
                if img_id_int not in seen_image_ids:
                    predictions.append({"image_id": img_id_int, "caption": caption})
                    seen_image_ids.add(img_id_int)

    # Compute metrics
    metrics = evaluate_captions(predictions, annotations_path)

    return predictions, metrics


def generate_and_evaluate_rat(
    model: torch.nn.Module,
    db_store,
    top_k: int,
    top_i: int,
    dataset: torch.utils.data.Dataset,
    annotations_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], EvalMetrics]:
    """
    Generate captions for a dataset and compute evaluation metrics.
    This is for the Retrieval-Augmented Transformer (RAT) model.

    Args:
        model: The trained Retrieval-Augmented Transformer .
        dataset: CocoDataset instance to evaluate on.
        annotations_path: Path to ground truth annotations JSON.
        batch_size: Batch size for generation.
        num_workers: Number of CPU workers for data loading.
        max_length: Maximum length of generated captions.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions list, EvalMetrics).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    predictions: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            image_ids: torch.Tensor = batch["image_id"]
            image_embeddings: torch.Tensor = batch["image_embedding"].to(device)

            # Generate captions
            generated_ids = model.generate(
                db_store=db_store,
                top_k=top_k,
                top_i=top_i,
                image_embeddings=image_embeddings,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )

            # Decode generated captions
            tokenizer = dataset.tokenizer
            generated_captions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,  # strips away special tokens like <eos>
            )

            # Store results (avoid duplicates - one caption per image)
            for img_id, caption in zip(image_ids, generated_captions):
                img_id_int = img_id.item()
                if img_id_int not in seen_image_ids:
                    predictions.append({"image_id": img_id_int, "caption": caption})
                    seen_image_ids.add(img_id_int)

    # Compute metrics
    metrics = evaluate_captions(predictions, annotations_path)

    return predictions, metrics


def evaluate_epoch(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    annotations_path: str,
    epoch: int,
    split_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
    output_dir: str = "eval_results",
) -> EvalMetrics:
    """
    Run evaluation for a single epoch on a given dataset split.
    Saves predictions and metrics to JSON files.

    Args:
        model: The trained ImageCaptioningModel.
        dataset: CocoDataset instance to evaluate on.
        annotations_path: Path to ground truth annotations JSON.
        epoch: Current epoch number (for logging/saving).
        split_name: Name of the split (e.g., "val", "test").
        batch_size: Batch size for generation.
        num_workers: Number of CPU workers for data loading.
        max_length: Maximum length of generated captions.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        device: Device to run inference on.
        output_dir: Directory to save predictions and metrics.

    Returns:
        EvalMetrics dataclass containing all metric scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch} | {split_name.upper()} Evaluation")
    print(f"{'=' * 60}")

    # Generate and evaluate
    predictions, metrics = generate_and_evaluate(
        model=model,
        dataset=dataset,
        annotations_path=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    # Log results
    print(f"\nResults: {metrics}")

    # Save predictions to JSON
    predictions_path = os.path.join(
        output_dir, f"epoch_{epoch}_{split_name}_predictions.json"
    )
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {predictions_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f"epoch_{epoch}_{split_name}_metrics.json")
    metrics_data = {
        "epoch": epoch,
        "split": split_name,
        "num_images": len(predictions),
        **metrics.to_dict(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def evaluate_rat_epoch(
    model: torch.nn.Module,
    db_store,
    top_k: int,
    top_i: int,
    dataset: torch.utils.data.Dataset,
    annotations_path: str,
    epoch: int,
    split_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device | None = None,
    output_dir: str = "eval_results",
) -> EvalMetrics:
    """
    Run evaluation for a single epoch on a given dataset split.
    Saves predictions and metrics to JSON files.

    Args:
        model: The trained RetrievalAugmentedTransformer.
        dataset: CocoDataset instance to evaluate on.
        annotations_path: Path to ground truth annotations JSON.
        epoch: Current epoch number (for logging/saving).
        split_name: Name of the split (e.g., "val", "test").
        batch_size: Batch size for generation.
        num_workers: Number of CPU workers for data loading.
        max_length: Maximum length of generated captions.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        device: Device to run inference on.
        output_dir: Directory to save predictions and metrics.

    Returns:
        EvalMetrics dataclass containing all metric scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch} | {split_name.upper()} Evaluation")
    print(f"{'=' * 60}")

    # Generate and evaluate
    predictions, metrics = generate_and_evaluate_rat(
        model=model,
        db_store=db_store,
        top_k=top_k,
        top_i=top_i,
        dataset=dataset,
        annotations_path=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    # Log results
    print(f"\nResults: {metrics}")

    # Save predictions to JSON
    predictions_path = os.path.join(
        output_dir, f"epoch_{epoch}_{split_name}_predictions_rat.json"
    )
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {predictions_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(
        output_dir, f"epoch_{epoch}_{split_name}_metrics_rat.json"
    )
    metrics_data = {
        "epoch": epoch,
        "split": split_name,
        "num_images": len(predictions),
        **metrics.to_dict(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def save_eval_summary(
    all_metrics: list[dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save a summary of all evaluation metrics across epochs.

    Args:
        all_metrics: List of metrics dicts from each epoch.
        output_path: Path to save the summary JSON.
    """
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Evaluation summary saved to: {output_path}")
