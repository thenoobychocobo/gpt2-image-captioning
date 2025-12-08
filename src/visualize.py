"""
FiftyOne visualization module for image captioning evaluation.

This module provides tools to:
- Create FiftyOne datasets from caption predictions and references
- Launch the FiftyOne web app for interactive visualization
"""

import json
import os

import fiftyone as fo


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
