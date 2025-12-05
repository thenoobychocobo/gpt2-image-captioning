"""
src/entities.py

Defines ObjectBox entities for storing documents and chunks with vector embeddings.
"""

from objectbox import (
    Entity,
    Int64,
    String,
    Float32Vector,
    HnswIndex,
    Id,
    Float64,
    VectorDistanceType,
)

from src.database.config import (
    IMAGE_EMBEDDING_DIMENSIONS,
    CAPTION_EMBEDDING_DIMENSIONS,
    INDEXING_SEARCH_COUNT,
    NEIGHBORS_PER_NODE,
    NORMALIZE_EMBEDDINGS,
)

@Entity()
class Image:
    """
    Represents an image in the dataset with its image embedding vector.
    """

    id = Id()
    file_name = String()
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

@Entity()
class Caption:
    """
    Represents a caption in the dataset with reference to image embedding vector.
    """

    id = Id()
    file_name = String()
    caption_id = Int64()
    caption_embedding_vector = Float32Vector(
        index=HnswIndex(
            dimensions=CAPTION_EMBEDDING_DIMENSIONS,
            distance_type=VectorDistanceType.DOT_PRODUCT
            if NORMALIZE_EMBEDDINGS
            else VectorDistanceType.EUCLIDEAN,
            neighbors_per_node=NEIGHBORS_PER_NODE,
            indexing_search_count=INDEXING_SEARCH_COUNT,
        )
    )
    similarity_scores = Float64()  # Store similarity score of the captions to the source image