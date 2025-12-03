"""
FiftyOne visualization module for image captioning evaluation.

This module provides tools to:
- Create FiftyOne datasets from caption predictions and references
- Visualize generated vs. reference captions side-by-side
- Filter and analyze samples by evaluation metrics
- Interactively explore model performance
"""

import json
import os
from typing import Any

import fiftyone as fo
from fiftyone import ViewField as F


def create_captioning_dataset(
    images_dir: str,
    predictions_path: str,
    annotations_path: str,
    metrics_per_image: dict[int, dict[str, float]] | None = None,
    dataset_name: str = "caption_eval",
    overwrite: bool = True,
) -> fo.Dataset:
    """
    Create a FiftyOne dataset for caption visualization and analysis.

    Args:
        images_dir: Directory containing the images.
        predictions_path: Path to JSON file with predictions [{"image_id": int, "caption": str}, ...].
        annotations_path: Path to COCO annotations JSON with ground truth captions.
        metrics_per_image: Optional dict mapping image_id to per-image metrics.
        dataset_name: Name for the FiftyOne dataset.
        overwrite: Whether to overwrite existing dataset with same name.

    Returns:
        FiftyOne Dataset object ready for visualization.
    """
    # Load predictions
    with open(predictions_path, "r") as f:
        predictions_list = json.load(f)
    predictions = {p["image_id"]: p["caption"] for p in predictions_list}

    # Load COCO annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Build mappings from annotations
    image_files: dict[int, str] = {}
    for img in coco_data["images"]:
        image_files[img["id"]] = img["file_name"]

    references: dict[int, list[str]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in references:
            references[img_id] = []
        references[img_id].append(ann["caption"])

    # Create or get dataset
    if overwrite and fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    # Build samples
    samples = []
    for img_id, generated_caption in predictions.items():
        if img_id not in image_files:
            continue

        filepath = os.path.join(images_dir, image_files[img_id])
        if not os.path.exists(filepath):
            continue

        sample = fo.Sample(filepath=filepath)

        # Add captions
        sample["generated_caption"] = generated_caption
        sample["reference_captions"] = references.get(img_id, [])
        sample["num_references"] = len(references.get(img_id, []))
        sample["image_id"] = img_id

        # Add per-image metrics if available
        if metrics_per_image and img_id in metrics_per_image:
            for metric_name, score in metrics_per_image[img_id].items():
                # Normalize field name (lowercase, replace - with _)
                field_name = metric_name.lower().replace("-", "_")
                sample[field_name] = score

        samples.append(sample)

    dataset.add_samples(samples)

    print(f"Created FiftyOne dataset '{dataset_name}' with {len(samples)} samples")
    return dataset


def create_comparison_dataset(
    images_dir: str,
    predictions_paths: dict[str, str],
    annotations_path: str,
    dataset_name: str = "caption_comparison",
    overwrite: bool = True,
) -> fo.Dataset:
    """
    Create a FiftyOne dataset comparing captions from multiple models/epochs.

    Args:
        images_dir: Directory containing the images.
        predictions_paths: Dict mapping model/epoch name to predictions JSON path.
            E.g., {"epoch_5": "path/to/epoch_5.json", "epoch_10": "path/to/epoch_10.json"}
        annotations_path: Path to COCO annotations JSON.
        dataset_name: Name for the FiftyOne dataset.
        overwrite: Whether to overwrite existing dataset.

    Returns:
        FiftyOne Dataset with captions from all specified sources.
    """
    # Load all predictions
    all_predictions: dict[str, dict[int, str]] = {}
    for name, path in predictions_paths.items():
        with open(path, "r") as f:
            preds_list = json.load(f)
        all_predictions[name] = {p["image_id"]: p["caption"] for p in preds_list}

    # Load COCO annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    image_files = {img["id"]: img["file_name"] for img in coco_data["images"]}
    references: dict[int, list[str]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in references:
            references[img_id] = []
        references[img_id].append(ann["caption"])

    # Get common image IDs across all prediction sets
    common_ids = set(image_files.keys())
    for preds in all_predictions.values():
        common_ids &= set(preds.keys())

    # Create dataset
    if overwrite and fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    samples = []
    for img_id in common_ids:
        filepath = os.path.join(images_dir, image_files[img_id])
        if not os.path.exists(filepath):
            continue

        sample = fo.Sample(filepath=filepath)
        sample["image_id"] = img_id
        sample["reference_captions"] = references.get(img_id, [])

        # Add caption from each source
        for name, preds in all_predictions.items():
            field_name = f"caption_{name}".replace("-", "_").replace(" ", "_")
            sample[field_name] = preds[img_id]

        samples.append(sample)

    dataset.add_samples(samples)

    print(f"Created comparison dataset '{dataset_name}' with {len(samples)} samples")
    return dataset


def add_metrics_to_dataset(
    dataset: fo.Dataset,
    metrics_path: str,
    prefix: str = "",
) -> fo.Dataset:
    """
    Add per-image metrics to an existing FiftyOne dataset.

    Args:
        dataset: Existing FiftyOne dataset.
        metrics_path: Path to JSON with per-image metrics.
        prefix: Optional prefix for metric field names.

    Returns:
        Updated dataset.
    """
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    # Assuming metrics_data is {image_id: {metric_name: score}}
    if isinstance(metrics_data, list):
        # Convert list format to dict
        metrics_dict = {m["image_id"]: m for m in metrics_data}
    else:
        metrics_dict = metrics_data

    for sample in dataset:
        img_id = sample["image_id"]
        if img_id in metrics_dict:
            for metric_name, score in metrics_dict[img_id].items():
                if metric_name == "image_id":
                    continue
                field_name = f"{prefix}{metric_name}".lower().replace("-", "_")
                sample[field_name] = score
            sample.save()

    return dataset


def get_low_score_view(
    dataset: fo.Dataset,
    metric_field: str = "cider",
    threshold: float = 0.5,
) -> fo.DatasetView:
    """
    Get a view of samples with low scores for a given metric.

    Args:
        dataset: FiftyOne dataset.
        metric_field: Name of the metric field to filter on.
        threshold: Samples with scores below this will be included.

    Returns:
        Filtered DatasetView.
    """
    return dataset.match(F(metric_field) < threshold)


def get_high_score_view(
    dataset: fo.Dataset,
    metric_field: str = "cider",
    threshold: float = 1.0,
) -> fo.DatasetView:
    """
    Get a view of samples with high scores for a given metric.

    Args:
        dataset: FiftyOne dataset.
        metric_field: Name of the metric field to filter on.
        threshold: Samples with scores above this will be included.

    Returns:
        Filtered DatasetView.
    """
    return dataset.match(F(metric_field) > threshold)


def launch_app(
    dataset: fo.Dataset | fo.DatasetView,
    port: int = 5151,
    address: str = "0.0.0.0",
    wait: bool = True,
) -> fo.Session:
    """
    Launch the FiftyOne web app for interactive visualization.

    Args:
        dataset: FiftyOne dataset or view to visualize.
        port: Port to run the app on.
        address: Address to bind to.
        wait: Whether to block until the app is closed.

    Returns:
        FiftyOne Session object.
    """
    session = fo.launch_app(dataset, port=port, address=address)
    print(f"\nFiftyOne app launched at http://localhost:{port}")
    print("Press Ctrl+C to stop the app\n")

    if wait:
        session.wait()

    return session


def export_samples(
    view: fo.DatasetView,
    export_dir: str,
    include_images: bool = True,
) -> None:
    """
    Export a subset of samples to a directory.

    Args:
        view: FiftyOne DatasetView to export.
        export_dir: Directory to export to.
        include_images: Whether to copy images to export directory.
    """
    os.makedirs(export_dir, exist_ok=True)

    # Export metadata as JSON
    samples_data = []
    for sample in view:
        sample_data = {
            "filepath": sample.filepath,
            "image_id": sample.get("image_id"),
            "generated_caption": sample.get("generated_caption"),
            "reference_captions": sample.get("reference_captions"),
        }

        # Add any metric fields
        for field in sample.field_names:
            if field not in sample_data and not field.startswith("_"):
                sample_data[field] = sample.get(field)

        samples_data.append(sample_data)

    metadata_path = os.path.join(export_dir, "samples.json")
    with open(metadata_path, "w") as f:
        json.dump(samples_data, f, indent=2)

    print(f"Exported {len(samples_data)} samples to {export_dir}")

    if include_images:
        import shutil

        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for sample in view:
            if os.path.exists(sample.filepath):
                dst = os.path.join(images_dir, os.path.basename(sample.filepath))
                shutil.copy2(sample.filepath, dst)
        print(f"Copied images to {images_dir}")


# === Example Usage ===
if __name__ == "__main__":
    # Example: Create and visualize a captioning dataset

    # Paths (adjust these to your setup)
    IMAGES_DIR = "data/coco/val2017"
    PREDICTIONS_PATH = "eval_results/epoch_10_test_predictions.json"
    ANNOTATIONS_PATH = "data/coco/annotations/captions_val2017.json"

    # Create dataset
    dataset = create_captioning_dataset(
        images_dir=IMAGES_DIR,
        predictions_path=PREDICTIONS_PATH,
        annotations_path=ANNOTATIONS_PATH,
        dataset_name="my_caption_eval",
    )

    # Print dataset info
    print(dataset)

    # Get samples sorted by a metric (if available)
    # sorted_view = dataset.sort_by("cider", reverse=True)

    # Get low-scoring samples for analysis
    # low_score_view = get_low_score_view(dataset, "bleu_4", threshold=0.3)

    # Launch interactive app
    launch_app(dataset)

