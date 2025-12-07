"""
src/config.py

Holds configuration constants for the indexing pipeline.
"""

#: Whether to normalize embeddings to unit length.
#: If normalized, dot-product similarity is used for nearest neighbor search
#: as it will be equivalent to cosine similarity while being faster to compute.
NORMALIZE_EMBEDDINGS: bool = True

#: Dimensionality of the CLIP image embedding vectors
IMAGE_EMBEDDING_DIMENSIONS: int = 512

#: Dimensionality of the CLIP caption embedding vectors
CAPTION_EMBEDDING_DIMENSIONS: int = 512

#: Maximum number of connections per node in the HNSW graph.
#: Higher number increases the graph connectivity which can lead to better results, but higher resource usage.
#: Try 16 for faster but less accurate results, or 64 for more accurate results.
NEIGHBORS_PER_NODE: int | None = None

#: Number of neighbors to search for during querying (efConstruction in HSNW terms).
#: Higher value lead to more accurate search, but slower indexing.
#: If indexing time is not a major concern, a value of at least 200 is recommended to improve search quality
INDEXING_SEARCH_COUNT: int | None = None
