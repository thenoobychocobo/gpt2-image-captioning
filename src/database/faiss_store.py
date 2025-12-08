"""
src/database/faiss_store.py

Contains functions for creating and managing FAISS indices,
and performing similarity searches as an alternative to ObjectBox.
"""

import os
import pickle
from typing import Optional

import faiss
import numpy as np

class FAISSStore:
    """
    Wrapper class for FAISS indices and metadata.
    Provides similar interface to ObjectBox Store for easy replacement.

    Attributes:
        image_index (faiss.Index): FAISS index for image embeddings.
        caption_index (faiss.Index): FAISS index for caption embeddings.
        image_metadata (list[str]): List of image filenames corresponding to image_index.
        caption_metadata (list[dict]): List of dicts with caption metadata corresponding to caption_index.
    """

    def __init__(
        self,
        image_index: faiss.Index,
        caption_index: faiss.Index,
        image_metadata: list[str],
        caption_metadata: list[dict],
    ):
        self.image_index = image_index
        self.caption_index = caption_index
        self.image_metadata = image_metadata  # List of filenames
        self.caption_metadata = caption_metadata  # List of dicts with filename, caption_id

        # Create reverse lookup: filename -> list of caption indices
        self.filename_to_caption_indices = {}
        for idx, meta in enumerate(caption_metadata):
            filename = meta["filename"]
            if filename not in self.filename_to_caption_indices:
                self.filename_to_caption_indices[filename] = []
            self.filename_to_caption_indices[filename].append(idx)

    def close(self):
        """Placeholder for API compatibility with ObjectBox Store"""
        pass


def create_faiss_store(
    db_directory: str,
    clear_db: bool = False,
) -> Optional[FAISSStore]:
    """
    Load an existing FAISS store from disk.

    Args:
        db_directory (str): Directory path for the FAISS indices and metadata.
        clear_db (bool, optional): Whether to clear the existing database.
            If True, returns None (no store to load).

    Returns:
        FAISSStore or None: The loaded FAISS store, or None if clear_db=True or not found.
    """
    if clear_db and os.path.exists(db_directory):
        import shutil
        shutil.rmtree(db_directory)
        os.makedirs(db_directory)
        return None

    # Check if indices exist
    image_index_path = os.path.join(db_directory, "image_index.faiss")
    caption_index_path = os.path.join(db_directory, "caption_index.faiss")
    image_meta_path = os.path.join(db_directory, "image_metadata.pkl")
    caption_meta_path = os.path.join(db_directory, "caption_metadata.pkl")

    if not all(
        os.path.exists(p)
        for p in [image_index_path, caption_index_path, image_meta_path, caption_meta_path]
    ):
        return None

    # Load indices
    image_index = faiss.read_index(image_index_path)
    caption_index = faiss.read_index(caption_index_path)

    # Load metadata
    with open(image_meta_path, "rb") as f:
        image_metadata = pickle.load(f)
    with open(caption_meta_path, "rb") as f:
        caption_metadata = pickle.load(f)

    return FAISSStore(image_index, caption_index, image_metadata, caption_metadata)


def save_faiss_store(store: FAISSStore, db_directory: str) -> None:
    """
    Save FAISS store to disk.

    Args:
        store (FAISSStore): The FAISS store to save.
        db_directory (str): Directory path to save the indices and metadata.
    """
    os.makedirs(db_directory, exist_ok=True)

    # Save indices
    faiss.write_index(store.image_index, os.path.join(db_directory, "image_index.faiss"))
    faiss.write_index(store.caption_index, os.path.join(db_directory, "caption_index.faiss"))

    # Save metadata
    with open(os.path.join(db_directory, "image_metadata.pkl"), "wb") as f:
        pickle.dump(store.image_metadata, f)
    with open(os.path.join(db_directory, "caption_metadata.pkl"), "wb") as f:
        pickle.dump(store.caption_metadata, f)


def retrieve_images_by_vector_similarity(
    faiss_store: FAISSStore, 
    query_embedding_vectors: np.ndarray, 
    top_i: int
) -> list[list[tuple[str, float]]]:
    """
    Retrieve images via batch similarity search using query image embedding vectors.
    Filters out near-perfect matches (same image).

    Args:
        faiss_store (FAISSStore): The FAISS store containing the images.
        query_embedding_vectors (np.ndarray): Query image embedding vectors of shape (batch_size, embed_dim).
        top_i (int): Number of top similar images to retrieve per query.

    Returns:
        List of lists, where each inner list contains (filename, score) tuples for one query.
        Outer list length = batch_size, inner list length = top_i.
    """
    # Ensure query is 2D array (FAISS requires batch dimension)
    if query_embedding_vectors.ndim == 1:
        query_embedding_vectors = query_embedding_vectors.reshape(1, -1)

    # Batch search - search for top_i + 10 to account for filtering
    distances, indices = faiss_store.image_index.search(
        query_embedding_vectors.astype(np.float32), top_i + 10
    )

    batch_results = []
    
    # Process each query's results
    for batch_idx in range(len(query_embedding_vectors)):
        query_results = []
        
        for dist, idx in zip(distances[batch_idx], indices[batch_idx]):
            # Skip invalid indices
            if idx == -1:
                continue

            # Convert distance to similarity score
            # For IndexFlatIP (inner product), distance is already the dot product (similarity)
            similarity = float(dist)

            # Skip near-perfect matches (same image)
            # Since embeddings are normalized, dot product of 1.0 means identical
            if similarity > 0.9999:
                continue

            filename = faiss_store.image_metadata[idx]
            query_results.append((filename, similarity))

            if len(query_results) >= top_i:
                break
        
        batch_results.append(query_results)

    return batch_results


def get_caption_embeddings(
    faiss_store: FAISSStore, 
    top_k: int, 
    batch_filenames: list[list[str]], 
    embed_dim: int = 512
) -> np.ndarray:
    """
    Given batch of filename lists, retrieve caption embeddings and return exactly top_k captions per query.

    Args:
        faiss_store (FAISSStore): The FAISS store.
        top_k (int): Number of captions to retrieve per query.
        batch_filenames (list[list[str]]): Batch of filename lists, shape (batch_size, variable).
        embed_dim (int): Embedding dimension for padding if needed.

    Returns:
        np.ndarray: Array of shape (batch_size, top_k, embed_dim) containing caption embeddings.
    """
    batch_caption_embeddings = []

    for filenames in batch_filenames:
        all_caption_indices = []

        # Collect caption indices for each filename
        for filename in filenames:
            if filename in faiss_store.filename_to_caption_indices:
                caption_indices = faiss_store.filename_to_caption_indices[filename]
                all_caption_indices.extend(caption_indices)

                # Break early if we have enough
                if len(all_caption_indices) >= top_k:
                    break

        # Handle empty case
        if not all_caption_indices:
            batch_caption_embeddings.append(np.zeros((top_k, embed_dim), dtype=np.float32))
            continue

        # Take first top_k caption indices
        selected_indices = all_caption_indices[:top_k]

        # Reconstruct embeddings from FAISS index
        caption_embeddings = []
        for idx in selected_indices:
            # FAISS reconstruct() returns the embedding at given index
            embedding = faiss_store.caption_index.reconstruct(int(idx))
            caption_embeddings.append(embedding)

        caption_embeddings = np.array(caption_embeddings, dtype=np.float32)

        # Pad if needed
        num_retrieved = len(caption_embeddings)
        if num_retrieved < top_k:
            padding = np.zeros((top_k - num_retrieved, embed_dim), dtype=np.float32)
            caption_embeddings = np.vstack([caption_embeddings, padding])

        batch_caption_embeddings.append(caption_embeddings)

    # Stack into batch
    return np.array(batch_caption_embeddings, dtype=np.float32)  # (batch_size, top_k, embed_dim)