"""Quick visualization of caption results using FiftyOne."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualize import create_captioning_dataset, launch_app

def main():
    parser = argparse.ArgumentParser(
        description="Visualize image captions with FiftyOne",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments with defaults
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/coco/val2014",
        help="Path to images directory"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/coco/annotations/captions_val2014.json",
        help="Path to COCO annotations JSON"
    )
    
    # Optional arguments (do not to changr mostly)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="my_captions",
        help="Name for FiftyOne dataset"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for FiftyOne app"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing dataset"
    )
    
    args = parser.parse_args()
    
    print("Creating FiftyOne dataset for visualization...")
    print(f"  Images: {args.images_dir}")
    print(f"  Predictions: {args.predictions}")
    print(f"  Annotations: {args.annotations}")
    print()
    
    # Create dataset
    dataset = create_captioning_dataset(
        images_dir=args.images_dir,
        predictions_path=args.predictions,
        annotations_path=args.annotations,
        dataset_name=args.dataset_name,
        overwrite=not args.no_overwrite
    )
    
    print(f"\n✅ Created dataset with {len(dataset)} samples")
    print(f"\nDataset contains:")
    print(f"  • Generated captions (your model)")
    print(f"  • Reference captions (ground truth)")
    print(f"  • Image IDs")
    print()
    
    # Launch FiftyOne app
    print(" Launching FiftyOne interactive viewer...")
    print(f"   → Open your browser to: http://localhost:{args.port}")
    print()
    print("💡 Tips:")
    print("  • Click on any image to see full captions side-by-side")
    print("  • Use the search box to find specific images")
    print("  • Add filters/stages to analyze subsets")
    print("  • Press Ctrl+C here to stop the server")
    print()
    
    launch_app(dataset, port=args.port, wait=True)


if __name__ == "__main__":
    main()