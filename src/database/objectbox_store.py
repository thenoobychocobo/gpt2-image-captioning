"""
src/document_store.py

Contains functions for creating and managing the ObjectBox document store,
adding documents and images, and performing similarity searches.
"""

import os

import numpy as np
import torch
from objectbox import Box, Model, Store

from src.database.entities import Caption, Image

# Shared ObjectBox model for all created stores (to maintain the same objectbox-model.json)
OBJECTBOX_MODEL = Model()
OBJECTBOX_MODEL.entity(Image)
OBJECTBOX_MODEL.entity(Caption)

# 5 GB size limit for the database
DATABASE_SIZE_LIMIT_KB = 1024 * 1024 * 5


def create_objectbox_store(
    db_directory: str,
    clear_db: bool = False,
) -> Store:
    """
    Create and return an ObjectBox store.

    Args:
        db_directory (str): Directory path for the ObjectBox database.
        clear_db (bool, optional): Whether to clear the existing database before creating a new one.
            Defaults to False.

    Returns:
        Store: The prepared ObjectBox store.
    """

    if clear_db and os.path.exists(db_directory):
        import shutil

        shutil.rmtree(db_directory)

    db_store = Store(
        model=OBJECTBOX_MODEL,
        directory=db_directory,
        max_db_size_in_kb=DATABASE_SIZE_LIMIT_KB,
    )
    return db_store


def retrieve_images_by_vector_similarity(
    db_store: Store, query_embedding_vector: np.ndarray, top_i: int
) -> list[tuple[str, float]]:
    """
    Retrieve images via similarity search with its associated embedded caption using the query image embedding vector from the given ObjectBox store.
    We do not consider similarity score that is perfectly equal to 1.0 (it means that it is the same image).

    Args:
        db_store (Store): The ObjectBox store containing the images.
        query_embedding_vector (np.ndarray): Query image embedding vector to search for similar images.
        top_i (int): Number of top similar images to retrieve.

    Returns:
        List of filenames obtained from the search
    """
    query = (
        db_store.box(Image)
        .query(
            Image.image_embedding_vector.nearest_neighbor(query_embedding_vector, top_i)
        )
        .build()
    )
    results = query.find_with_scores()

    # If score is exactly 0, it means it's the same image, we skip it
    # Since the embeddings are normalised dot product, ObjectBox returns 1.0 for the same image (which is 1 - 1.0 = 0 distance)
    results = [result for result in results if result[1] > 0.0001]

    return [(result[0].file_name, result[1]) for result in results]


def get_caption_embeddings(
    db_store: Store, top_k: int, filenames: list[str], embed_dim: int = 512
) -> np.ndarray:
    """
    Given list of filenames, retrieve caption embeddings and return exactly top_k captions.
    """
    caption_box: Box = db_store.box(Caption)
    all_captions = []

    # Query ALL filenames (don't break early)
    for filename in filenames:
        query = caption_box.query(Caption.file_name.equals(filename)).build()
        results = query.find()
        all_captions.extend(results)

        # Only break if we have ENOUGH captions already
        if len(all_captions) >= top_k:
            break

    # Handle empty case
    if not all_captions:
        return np.zeros((top_k, embed_dim), dtype=np.float32)

    # Take first top_k captions
    selected_captions = all_captions[:top_k]

    # Extract embeddings
    caption_embeddings = np.array(
        [np.array(caption.caption_embedding_vector) for caption in selected_captions]
    )

    # Pad if needed
    num_retrieved = len(caption_embeddings)
    if num_retrieved < top_k:
        embed_dim = caption_embeddings.shape[1]
        padding = np.zeros((top_k - num_retrieved, embed_dim), dtype=np.float32)
        caption_embeddings = np.vstack([caption_embeddings, padding])

    return caption_embeddings


def retrieve_for_single_embedding(
    single_embedding: torch.Tensor,
    db_store: Store,
    top_i: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    query_vector_for_db = single_embedding.squeeze(0).cpu().numpy().tolist()

    filenames_with_scores = retrieve_images_by_vector_similarity(
        db_store=db_store,
        query_embedding_vector=query_vector_for_db,
        top_i=top_i,
    )

    caption_embeds = get_caption_embeddings(
        db_store=db_store,
        top_k=top_k,
        filenames=[filename for filename, _ in filenames_with_scores],
    )

    return torch.from_numpy(caption_embeds).to(device)
