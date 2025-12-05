import logging
import numpy as np
import os
import sys
from objectbox import Box, Store
from tqdm import tqdm
import torch
from src.database.entities import Image, Caption
from src.database.image_store import create_objectbox_store
import gc

# --- OPTIMIZATION 1: The "Safe Put" Helper ---
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
        # Slice list into safe chunks (e.g., 0-9000, 9000-18000...)
        for i in range(0, total, chunk_size):
            box.put(entities[i : i + chunk_size])

def run_indexing_pipeline(
    db_directory: str,
    image_embedding_file_path: str,
    caption_embedding_file_path: str,
    batch_size: int = 5000, # Increased to 5000 because safe_put handles the volume
) -> None:
    """Run the indexing pipeline"""

    # --- OPTIMIZATION 2: WSL Filesystem Check ---
    # Writing database files to /mnt/c/ from WSL is 50x-100x slower than native Linux.
    if "/mnt/c/" in db_directory or "/mnt/d/" in db_directory:
        print("\n" + "!"*60)
        print("CRITICAL PERFORMANCE WARNING:")
        print(f"You are saving the DB to a Windows mount: {db_directory}")
        print("This causes extreme slowness (IO Bottleneck).")
        print("SOLUTION: Change db_directory to a Linux path like './objectbox_db'")
        print("!"*60 + "\n")
        # I am not stopping the script, but I strongly advise you to kill it and change the path.

    # Initialize ObjectBox store
    db_store = create_objectbox_store(db_directory=db_directory)
    image_box: Box = db_store.box(Image)
    caption_box: Box = db_store.box(Caption)

    print("Loading data...")
    # Load .pt files
    image_data = torch.load(image_embedding_file_path, weights_only=True) 
    caption_data = torch.load(caption_embedding_file_path, weights_only=False)

    # --- OPTIMIZATION 3: Global Numpy Conversion ---
    print("Converting Image tensors to NumPy...")
    # Convert all images to Float32 NumPy array once
    all_image_embeddings = image_data['embeddings'].numpy().astype(np.float32)
        
    # Create filename map
    filename_to_index = {filename: idx for idx, filename in enumerate(image_data['filenames'])}

    # ⭐ ACTION: DELETE the original PyTorch tensor to free up space
    del image_data['embeddings']
    del image_data['filenames'] # Filenames were copied to filename_to_index
    del image_data

    # ⭐ ACTION: Pre-clean the captions for faster iteration and reduced memory
    print("Pre-converting caption tensors...")
    for item in tqdm(caption_data, desc="Pre-converting"):
        for cap in item['embeddings']:
            # Convert any remaining torch Tensors to NumPy upfront
            if torch.is_tensor(cap['embedding']):
                cap['embedding'] = cap['embedding'].numpy().astype(np.float32)

    gc.collect() # Force Python to clean up

    # --- OPTIMIZATION 4: Global Normalization ---
    print("Normalizing image embeddings...")
    # Vectorized normalization of 100k+ images in one go (uses C-level BLAS)
    norms = np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10 
    image_embeddings_normalized = all_image_embeddings / norms

    images_to_add = []
    captions_to_add = []

    print(f"Processing entities (Batch Size: {batch_size})...")
    
    # TODO: for now, we will only index around 10000 training images. There is a bottleneck that needs to be fixed
    i = 0
    # Iterate directly over the list (fastest iteration method in Python)
    for caption_entry in tqdm(caption_data, desc="Indexing"):
        i += 1
        if i % 10000 == 0:
            print(f"Processed {i} caption entries...exiting for test")
            break  # TEMPORARY LIMIT for testing purposes
        
        filename = caption_entry['filenames']
        
        # O(1) Dictionary Lookup
        image_idx = filename_to_index.get(filename)
        if image_idx is None:
            continue
            
        # Fetch pre-calculated arrays
        current_image_norm = image_embeddings_normalized[image_idx]
        current_image_raw = all_image_embeddings[image_idx]
        current_caps = caption_entry['embeddings']
        
        # --- OPTIMIZATION 5: Vectorized Caption Processing ---
        # 1. Extract raw embeddings into a 2D NumPy Matrix
        cap_vecs_raw = np.array([x['embedding'] for x in current_caps], dtype=np.float32)
        cap_ids = [x['caption_id'] for x in current_caps]

        # 2. Normalize Matrix (Single CPU Operation)
        cap_norms = np.linalg.norm(cap_vecs_raw, axis=1, keepdims=True)
        cap_norms[cap_norms == 0] = 1e-10
        cap_vecs_norm = cap_vecs_raw / cap_norms

        # 3. Dot Product (Matrix x Vector) for Similarity
        sim_scores = np.dot(cap_vecs_norm, current_image_norm)

        # 4. Bulk convert matrix to Python list (faster than row-by-row .tolist())
        cap_vecs_list = cap_vecs_raw.tolist()

        # Create Image Object
        image_entity = Image(
            file_name=filename,
            image_embedding_vector=current_image_raw.tolist(), 
        )
        images_to_add.append(image_entity)

        # Create Caption Objects
        for i, cap_id in enumerate(cap_ids):
            captions_to_add.append(Caption(
                file_name=filename,
                caption_id=cap_id,
                caption_embedding_vector=cap_vecs_list[i], 
                similarity_scores=float(sim_scores[i]),
            ))

        # --- OPTIMIZATION 6: Transactional Bulk Insert ---
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

if __name__ == "__main__":
    # 1. Get the path to your high-speed Linux Home Directory
    # This resolves to something like "/home/hoxia/objectbox_db"
    fast_db_path = os.path.join(os.path.expanduser("~"), "vector_db")
    
    print(f"SAVING DATABASE TO: {fast_db_path}")

    run_indexing_pipeline(
        db_directory=fast_db_path,  # <--- FORCE LINUX PATH
        image_embedding_file_path="/mnt/c/Users/hoxia/Documents/NLDeeznuts/gpt2-image-captioning/data/data/coco/embeddings/train_clip_embeddings.pt",
        caption_embedding_file_path="/mnt/c/Users/hoxia/Documents/NLDeeznuts/gpt2-image-captioning/data/data/coco/embeddings/train_caption_embeddings.pt"
    )