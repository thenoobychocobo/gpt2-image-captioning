"""
src/database/faiss_indexing.py

Indexing pipeline for building FAISS indices from embedding files.
Uses approximate nearest neighbor search (HNSW) for faster retrieval.
"""

import gc
import os

import faiss
import numpy as np
import torch
from tqdm import tqdm

from src.database.faiss_store import FAISSStore, save_faiss_store


def run_faiss_indexing_pipeline(
    db_directory: str,
    image_embedding_file_path: str,
    caption_embedding_file_path: str,
    use_approximate: bool = True,
    hnsw_m: int = 32,  # Number of connections per layer (higher = more accurate but slower)
    hnsw_efConstruction: int = 200,  # Size of dynamic candidate list (higher = better quality index)
    hnsw_efSearch: int = 64,  # Size of search candidates (higher = more accurate but slower search)
) -> None:
    """
    Build FAISS indices from embedding files.

    Args:
        db_directory (str): Directory to save FAISS indices and metadata.
        image_embedding_file_path (str): Path to image embeddings .pt file.
        caption_embedding_file_path (str): Path to caption embeddings .pt file.
        use_approximate (bool): If True, use HNSW approximate search. If False, use exact search.
        hnsw_m (int): HNSW parameter - number of bidirectional links per node.
            Typical: 16-64. Higher = more accurate but slower build & more memory.
        hnsw_efConstruction (int): HNSW parameter - size of dynamic candidate list during construction.
            Typical: 40-500. Higher = better index quality but slower build.
        hnsw_efSearch (int): HNSW parameter - size of dynamic candidate list during search.
            Typical: 16-512. Higher = more accurate search but slower.
            Can be changed after index is built.
    """
    print("Starting FAISS indexing pipeline...")
    print(f"Mode: {'APPROXIMATE (HNSW)' if use_approximate else 'EXACT (Flat)'}")

    print("Loading data...")
    # Load .pt files
    image_data = torch.load(image_embedding_file_path, weights_only=True)
    caption_data = torch.load(caption_embedding_file_path, weights_only=False)

    print("Converting Image tensors to NumPy...")
    # Convert all images to Float32 NumPy array (embeddings are already normalized)
    all_image_embeddings = image_data["embeddings"].numpy().astype(np.float32)
    image_filenames = image_data["filenames"]

    # Get dimensions
    num_images, embed_dim = all_image_embeddings.shape
    print(f"Image embeddings: {num_images} vectors of dimension {embed_dim}")

    # Create image FAISS index
    print("Building image FAISS index...")
    if use_approximate:
        # HNSW (Hierarchical Navigable Small World) - Fast approximate search
        # Inner product version for normalized vectors
        image_index = faiss.IndexHNSWFlat(embed_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        image_index.hnsw.efConstruction = hnsw_efConstruction
        image_index.hnsw.efSearch = hnsw_efSearch
        print(
            f"  Using HNSW with M={hnsw_m}, efConstruction={hnsw_efConstruction}, efSearch={hnsw_efSearch}"
        )
    else:
        # Flat index - Exact but slower search
        image_index = faiss.IndexFlatIP(embed_dim)
        print("  Using Flat (exact search)")

    image_index.add(all_image_embeddings)
    print(f"  Indexed {image_index.ntotal} image vectors")

    # Free up memory
    del image_data
    gc.collect()

    print("Processing caption embeddings...")
    caption_embeddings_list = []
    caption_metadata = []

    # Create filename map for faster lookup
    filename_to_index = {filename: idx for idx, filename in enumerate(image_filenames)}

    for caption_entry in tqdm(caption_data, desc="Processing captions"):
        filename = caption_entry["filenames"]

        # Skip if image not in dataset
        if filename not in filename_to_index:
            continue

        current_caps = caption_entry["embeddings"]

        for cap in current_caps:
            # Convert tensor to numpy if needed
            if torch.is_tensor(cap["embedding"]):
                embedding = cap["embedding"].numpy().astype(np.float32)
            else:
                embedding = np.array(cap["embedding"], dtype=np.float32)

            caption_embeddings_list.append(embedding)
            caption_metadata.append(
                {
                    "filename": filename,
                    "caption_id": cap["caption_id"],
                }
            )

    # Convert to numpy array
    caption_embeddings = np.array(caption_embeddings_list, dtype=np.float32)
    num_captions = len(caption_embeddings)
    print(f"Caption embeddings: {num_captions} vectors of dimension {embed_dim}")

    # Create caption FAISS index
    print("Building caption FAISS index...")
    if use_approximate:
        caption_index = faiss.IndexHNSWFlat(
            embed_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT
        )
        caption_index.hnsw.efConstruction = hnsw_efConstruction
        caption_index.hnsw.efSearch = hnsw_efSearch
        print(
            f"  Using HNSW with M={hnsw_m}, efConstruction={hnsw_efConstruction}, efSearch={hnsw_efSearch}"
        )
    else:
        caption_index = faiss.IndexFlatIP(embed_dim)
        print("  Using Flat (exact search)")

    caption_index.add(caption_embeddings)
    print(f"  Indexed {caption_index.ntotal} caption vectors")

    # Free up memory
    del caption_data
    del caption_embeddings_list
    gc.collect()

    # Create FAISSStore object
    print("Creating FAISS store...")
    faiss_store = FAISSStore(
        image_index=image_index,
        caption_index=caption_index,
        image_metadata=image_filenames,
        caption_metadata=caption_metadata,
    )

    # Save to disk
    print(f"Saving FAISS store to {db_directory}...")
    save_faiss_store(faiss_store, db_directory)

    print("\nFAISS indexing pipeline complete!")
    print(f"  - Mode: {'APPROXIMATE (HNSW)' if use_approximate else 'EXACT (Flat)'}")
    print(f"  - Images indexed: {num_images}")
    print(f"  - Captions indexed: {num_captions}")
    print(f"  - Index files saved to: {db_directory}")

    if use_approximate:
        print("\nHNSW Parameters:")
        print(f"  - M (connections): {hnsw_m}")
        print(f"  - efConstruction: {hnsw_efConstruction}")
        print(f"  - efSearch: {hnsw_efSearch}")
        print("\nNote: You can adjust efSearch later for speed/accuracy trade-off:")
        print("  Lower efSearch → Faster but less accurate")
        print("  Higher efSearch → Slower but more accurate")


if __name__ == "__main__":
    db_path = "faiss_db"

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    run_faiss_indexing_pipeline(
        db_directory=db_path,
        image_embedding_file_path="path_to_train_clip_embeddings.pt",
        caption_embedding_file_path="path_to_gpt2-image-captioning/data/data/coco/embeddings/train_caption_embeddings.pt",
        use_approximate=True,  # Enable HNSW
        hnsw_m=32,  # 32 links like in the paper
        hnsw_efConstruction=200,  # High quality index
        hnsw_efSearch=64,  # Good search accuracy
    )
