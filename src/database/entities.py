"""
src/entities.py

Defines ObjectBox entities for storing documents and chunks with vector embeddings.
"""

from objectbox import (
    Entity,
    Float32Vector,
    HnswIndex,
    Id,
    Float64,
    VectorDistanceType,
)

from database.config import (
    IMAGE_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_DIMENSIONS,
    INDEXING_SEARCH_COUNT,
    NEIGHBORS_PER_NODE,
    NORMALIZE_EMBEDDINGS,
)

@Entity()
class Image:
    """
    Represents an image in the dataset with its text and image embedding vector.
    """

    id = Id()
    image_embedding_vector = Float32Vector(
        index=HnswIndex(
            dimensions=IMAGE_EMBEDDING_DIMENSIONS,
            distance_type=VectorDistanceType.DOT_PRODUCT
            if NORMALIZE_EMBEDDINGS
            else VectorDistanceType.EUCLIDEAN,
            neighbors_per_node=NEIGHBORS_PER_NODE,
            indexing_search_count=INDEXING_SEARCH_COUNT,
        )
    )
    text_embedding_vector = Float32Vector(
        index=HnswIndex(
            dimensions=TEXT_EMBEDDING_DIMENSIONS,
            distance_type=VectorDistanceType.DOT_PRODUCT
            if NORMALIZE_EMBEDDINGS
            else VectorDistanceType.EUCLIDEAN,
            neighbors_per_node=NEIGHBORS_PER_NODE,
            indexing_search_count=INDEXING_SEARCH_COUNT,
        )
    )
    similarity_score = Float64()  # Store similarity scores between text and image embeddings