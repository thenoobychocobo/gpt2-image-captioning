#!/usr/bin/env python3
"""Quick visualization of caption results using FiftyOne."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualize import create_captioning_dataset, launch_app

print("🎨 Creating FiftyOne dataset for visualization...")
print()

# Create dataset
dataset = create_captioning_dataset(
    images_dir="data/coco/val2017",
    predictions_path="sample_output/results.json",
    annotations_path="data/coco/annotations/captions_val2017.json",
    dataset_name="my_captions",$
    overwrite=True,
)

print(f"\n✅ Created dataset with {len(dataset)} samples")
print(f"\nDataset contains:")
print(f"  • Generated captions (your model)")
print(f"  • Reference captions (ground truth)")
print(f"  • Image IDs")
print()

# Launch FiftyOne app
print("🚀 Launching FiftyOne interactive viewer...")
print(f"   → Open your browser to: http://localhost:5151")
print()
print("💡 Tips:")
print("  • Click on any image to see full captions side-by-side")
print("  • Use the search box to find specific images")
print("  • Add filters/stages to analyze subsets")
print("  • Press Ctrl+C here to stop the server")
print()

launch_app(dataset, port=5151, wait=True)

