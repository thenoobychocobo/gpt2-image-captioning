"""
src/document_store.py

Contains functions for creating and managing the ObjectBox document store,
adding documents and images, and performing similarity searches.
"""

import os

import numpy as np
from objectbox import Box, Model, Store

from src.database.entities import Image, Caption


# Shared ObjectBox model for all created stores (to maintain the same objectbox-model.json)
OBJECTBOX_MODEL = Model()
OBJECTBOX_MODEL.entity(Image)
OBJECTBOX_MODEL.entity(Caption)

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

    db_store = Store(model=OBJECTBOX_MODEL, directory=db_directory)
    return db_store


def add_images_to_store(db_store: Store, images: list[Image]) -> list[int]:
    """
    Add images to the given ObjectBox store.

    Args:
        db_store (Store): The ObjectBox store to add images to.
        images (list[Image]): List of images to be added.

    Returns:
        list[int]: List of IDs of the added images.
    """
    image_box: Box = db_store.box(Image)
    ids: list[int] = add_entities_to_store(db_store, image_box, images)
    return ids

def add_captions_to_store(db_store: Store, captions: list[Caption]) -> list[int]:
    """
    Add captions to the given ObjectBox store.

    Args:
        db_store (Store): The ObjectBox store to add captions to.
        captions (list[Caption]): List of captions to be added.

    Returns:
        list[int]: List of IDs of the added captions.
    """
    caption_box: Box = db_store.box(Caption)
    ids: list[int] = add_entities_to_store(db_store, caption_box, captions)
    return ids

def add_entities_to_store(db_store: Store, box: Box, entities: list) -> list[int]:
    """
    Add items to the specified box.

    Args:
        db_store (Store): The ObjectBox store.
        box (Box): The ObjectBox box to add items to (from the same store).
        entities (list): List of entities to be added to the box. Assumes all items
            are of the same type (Entity class). Infers type from the first item.
    Returns:
        list[int]: List of IDs of the added items.
    """
    # Passing a list to box.put() triggers an internal C++ bulk insert.
    # It automatically handles the transaction and returns the list of IDs.
    return box.put(entities)

# TODO: top-k images is not equivalent to top k images, might need different handling
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
        .query(Image.image_embedding_vector.nearest_neighbor(query_embedding_vector, top_i))
        .build()
    )
    results = query.find_with_scores()
    # if score is exactly 1.0, it means it's the same image, we skip it
    results = [result for result in results if result[1] < 0.9999]
    # Print score along with filename for debugging
    print("Retrieved Images and Similarity Scores:")
    for result in results:
        print(f"Filename: {result[0].file_name}, Similarity Score: {result[1]}")
    return [(result[0].file_name, result[1]) for result in results]

def get_caption_embeddings(
    db_store: Store,
    top_k: int,
    filenames: list[str]
) -> np.ndarray | None:
    """
    Given list of filenames, retrieve the associated caption embedding vectors from the ObjectBox store.
    Args:
        db_store (Store): The ObjectBox store containing the images.
        top_k (int): Number of top captions to retrieve in total. Returned captions may be lesser than k if not enough captions are found.
        filenames (list[str]): List of filenames of images to retrieve caption embeddings for.

    Returns:
        np.ndarray | None: The associated caption embedding vector if found, otherwise None.
    """
    caption_box: Box = db_store.box(Caption)
    all_captions = []
    
    # Query each filename until we have top_k captions
    for filename in filenames:
        if len(all_captions) >= top_k:
            break
            
        query = caption_box.query(Caption.file_name.equals(filename)).build()
        results = query.find()
        
        # Add captions from this image
        all_captions.extend(results)
    
    if not all_captions:
        return None
    
    # Get only first k items from list if there are more than k, safely return fewer if not enough captions
    selected_captions = all_captions[:top_k]

    # Extract embeddings
    caption_embeddings = np.array([np.array(caption.caption_embedding_vector) for caption in selected_captions])
    
    return caption_embeddings