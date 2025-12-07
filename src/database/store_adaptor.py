# src/database/store_adapter.py
"""
Adapters to make ObjectBox and FAISS stores conform to the VectorStore protocol.
"""

import numpy as np
import torch
from objectbox import Store

from src.database.faiss_store import FAISSStore
from src.database import image_store, faiss_store


class ObjectBoxStoreAdapter:
    """Adapter for ObjectBox Store to conform to VectorStore protocol."""
    
    def __init__(self, store: Store):
        self.store = store
    
    def close(self) -> None:
        self.store.close()
    
    def retrieve_images_by_vector_similarity(
        self, query_embedding_vector: np.ndarray, top_i: int
    ) -> list[tuple[str, float]]:
        return image_store.retrieve_images_by_vector_similarity(
            self.store, query_embedding_vector, top_i
        )
    
    def get_caption_embeddings(
        self, top_k: int, filenames: list[str], embed_dim: int = 512
    ) -> np.ndarray:
        return image_store.get_caption_embeddings(
            self.store, top_k, filenames, embed_dim
        )
    
    def retrieve_for_single_embedding(
        self,
        single_embedding: torch.Tensor,
        top_i: int,
        top_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        return image_store.retrieve_for_single_embedding(
            single_embedding, self.store, top_i, top_k, device
        )


class FAISSStoreAdapter:
    """Adapter for FAISS Store to conform to VectorStore protocol."""
    
    def __init__(self, store: FAISSStore):
        self.store = store
    
    def close(self) -> None:
        self.store.close()
    
    def retrieve_images_by_vector_similarity(
        self, query_embedding_vector: np.ndarray, top_i: int
    ) -> list[tuple[str, float]]:
        return faiss_store.retrieve_images_by_vector_similarity(
            self.store, query_embedding_vector, top_i
        )
    
    def get_caption_embeddings(
        self, top_k: int, filenames: list[str], embed_dim: int = 512
    ) -> np.ndarray:
        return faiss_store.get_caption_embeddings(
            self.store, top_k, filenames, embed_dim
        )
    
    def retrieve_for_single_embedding(
        self,
        single_embedding: torch.Tensor,
        top_i: int,
        top_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        return faiss_store.retrieve_for_single_embedding(
            single_embedding, self.store, top_i, top_k, device
        )


def create_vector_store(
    backend: str = "faiss",
    db_directory: str = None,
    clear_db: bool = False,
):
    """
    Factory function to create a vector store with the specified backend.
    
    Args:
        backend: Either "faiss" or "objectbox"
        db_directory: Path to the database directory
        clear_db: Whether to clear existing database
    
    Returns:
        Adapter conforming to VectorStore protocol
    """
    if backend.lower() == "faiss":
        from src.database.faiss_store import create_faiss_store
        store = create_faiss_store(db_directory, clear_db)
        return FAISSStoreAdapter(store)
    
    elif backend.lower() == "objectbox":
        from src.database.image_store import create_objectbox_store
        store = create_objectbox_store(db_directory, clear_db)
        return ObjectBoxStoreAdapter(store)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'faiss' or 'objectbox'")