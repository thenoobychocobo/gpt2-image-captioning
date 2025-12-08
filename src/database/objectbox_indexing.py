import gc
import os

import numpy as np
import torch
from objectbox import Box
from tqdm import tqdm

from src.database.entities import Caption, Image
from database.objectbox_store import create_objectbox_store


def safe_put(box: Box, entities: list, chunk_size: int = 9000):
    """
    Bypasses ObjectBox's 10,000 item limit per put() call.
    Splits large lists into chunks automatically.
    Running this inside a transaction keeps it extremely fast.
    """
    total = len(entities)
    if total <= chunk_size:
        box.put(entities)
    else:
        # Slice list into safe chunks (e.g., 0-9000, 9000-18000...).
        # This is because ObjectBox put() crashes if >10,000 items are given at once.
        for i in range(0, total, chunk_size):
            box.put(entities[i : i + chunk_size])


def run_objectbox_indexing_pipeline(
    db_directory: str,
    image_embedding_file_path: str,
    caption_embedding_file_path: str,
    batch_size: int = 5000,  # Increased to 5000 because safe_put handles the volume
) -> None:
    """Run the indexing pipeline"""

    print("Starting indexing pipeline...")
    # Initialize ObjectBox store
    db_store = create_objectbox_store(db_directory=db_directory)
    image_box: Box = db_store.box(Image)
    caption_box: Box = db_store.box(Caption)

    print("Loading data...")
    # Load .pt files
    image_data = torch.load(image_embedding_file_path, weights_only=True)
    caption_data = torch.load(caption_embedding_file_path, weights_only=False)

    print("Converting Image tensors to NumPy...")
    # Convert all images to Float32 NumPy array once. Embeddings are already normalized.
    all_image_embeddings = image_data["embeddings"].numpy().astype(np.float32)

    # Create filename map
    filename_to_index = {
        filename: idx for idx, filename in enumerate(image_data["filenames"])
    }

    # DELETE the original PyTorch tensor to free up space
    del image_data["embeddings"]
    del image_data["filenames"]  # Filenames were copied to filename_to_index
    del image_data

    # Pre-clean the captions for faster iteration and reduced memory
    print("Pre-converting caption tensors...")
    for item in tqdm(caption_data, desc="Pre-converting"):
        for cap in item["embeddings"]:
            # Convert any remaining torch Tensors to NumPy upfront
            if torch.is_tensor(cap["embedding"]):
                cap["embedding"] = cap["embedding"].numpy().astype(np.float32)

    # Force garbage collection to free up memory
    gc.collect()

    images_to_add = []
    captions_to_add = []

    print(f"Processing entities (Batch Size: {batch_size})...")

    for caption_entry in tqdm(caption_data, desc="Indexing"):
        filename = caption_entry["filenames"]

        image_idx = filename_to_index.get(filename)
        if image_idx is None:
            continue

        # Fetch pre-normalized embeddings from the .pt files
        current_image_embedding = all_image_embeddings[image_idx]
        current_caps = caption_entry["embeddings"]

        # Extract embeddings into a 2D NumPy Matrix (already normalized)
        cap_vecs = np.array([x["embedding"] for x in current_caps], dtype=np.float32)
        cap_ids = [x["caption_id"] for x in current_caps]

        # Dot Product (Matrix x Vector) for Similarity - no normalization needed
        sim_scores = np.dot(cap_vecs, current_image_embedding)

        # Bulk convert matrix to Python list (faster than row-by-row .tolist())
        cap_vecs_list = cap_vecs.tolist()

        # Create Image Object
        image_entity = Image(
            file_name=filename,
            image_embedding_vector=current_image_embedding.tolist(),
        )
        images_to_add.append(image_entity)

        # Create Caption Objects
        for i, cap_id in enumerate(cap_ids):
            captions_to_add.append(
                Caption(
                    file_name=filename,
                    caption_id=cap_id,
                    caption_embedding_vector=cap_vecs_list[i],
                    similarity_scores=float(sim_scores[i]),
                )
            )

        # Transactional Bulk Insert
        if len(images_to_add) >= batch_size:
            # Open ONE transaction for the whole batch
            with db_store.write_tx():
                # Use safe_put to avoid the 10k limit crash
                safe_put(image_box, images_to_add)
                safe_put(caption_box, captions_to_add)

            # Clear memory
            images_to_add = []
            captions_to_add = []

    # Final flush for remaining items
    if images_to_add or captions_to_add:
        with db_store.write_tx():
            if images_to_add:
                safe_put(image_box, images_to_add)
            if captions_to_add:
                safe_put(caption_box, captions_to_add)

    db_store.close()
    print("Indexing pipeline complete!")

# EXAMPLE USAGE
if __name__ == "__main__":
    db_path = "vector_db"
    
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    run_objectbox_indexing_pipeline(
        db_directory=db_path,
        image_embedding_file_path="path_to_train_clip_embeddings.pt",
        caption_embedding_file_path="path_to_train_caption_embeddings.pts",
    )
