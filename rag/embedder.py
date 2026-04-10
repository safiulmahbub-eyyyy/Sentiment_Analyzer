"""
Query Embedding Module
Converts user questions into vector embeddings for semantic search
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

from rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    MAX_TEXT_LENGTH,
    CACHE_EMBEDDING_MODEL,
    VERBOSE
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model

    Uses lru_cache to load model once and reuse it
    (This is the "heat the oven once" optimization!)

    Returns:
        Loaded SentenceTransformer model
    """
    if VERBOSE:
        print(f"[EMBEDDER] Loading model: {EMBEDDING_MODEL}...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    if VERBOSE:
        print(f"[EMBEDDER] Model loaded! Dimension: {EMBEDDING_DIMENSION}")

    return model


def prepare_query_text(query: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Prepare query text for embedding

    Args:
        query: User's question/query
        max_length: Maximum character length

    Returns:
        Cleaned and truncated query text
    """
    # Remove extra whitespace
    text = " ".join(query.strip().split())

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    return text


def embed_query(query: str) -> List[float]:
    """
    Convert a single query into an embedding vector

    This is the main function you'll use in the RAG pipeline.
    It uses the cached model (fast after first call!)

    Args:
        query: User's question as text

    Returns:
        Embedding vector as a list of floats (384 dimensions)

    Example:
        >>> embedding = embed_query("What do people think about iPhone 15?")
        >>> len(embedding)
        384
    """
    # Get cached model (fast!)
    model = get_embedding_model()

    # Prepare text
    text = prepare_query_text(query)

    if VERBOSE:
        print(f"[EMBEDDER] Embedding query: '{text[:100]}...'")

    # Generate embedding
    embedding = model.encode([text], convert_to_numpy=True)

    # Return as list (not numpy array)
    return embedding[0].tolist()


def embed_queries_batch(queries: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Convert multiple queries into embeddings efficiently

    Useful for batch processing or testing multiple queries at once.

    Args:
        queries: List of question strings
        batch_size: Number of queries to process at once

    Returns:
        List of embedding vectors

    Example:
        >>> queries = [
        ...     "Are gaming laptops worth it?",
        ...     "What's the best smartphone?",
        ...     "Is iPhone better than Android?"
        ... ]
        >>> embeddings = embed_queries_batch(queries)
        >>> len(embeddings)
        3
    """
    # Get cached model
    model = get_embedding_model()

    # Prepare all texts
    texts = [prepare_query_text(q) for q in queries]

    if VERBOSE:
        print(f"[EMBEDDER] Embedding {len(texts)} queries in batch...")

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=VERBOSE
    )

    # Convert to list of lists
    return embeddings.tolist()


def validate_embedding(embedding: List[float]) -> bool:
    """
    Validate that an embedding has the correct format

    Args:
        embedding: Vector to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(embedding, (list, np.ndarray)):
        raise ValueError(f"Embedding must be a list or numpy array, got {type(embedding)}")

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embedding)}")

    # Check if all values are numbers
    try:
        float_embedding = [float(x) for x in embedding]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Embedding contains non-numeric values: {e}")

    return True


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings

    Useful for debugging or comparing query similarity

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score between 0 and 1 (higher = more similar)
    """
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm1 * norm2)

    return float(similarity)


def clear_model_cache():
    """
    Clear the cached embedding model from memory

    Useful if you need to free up memory or reload the model
    """
    get_embedding_model.cache_clear()
    if VERBOSE:
        print("[EMBEDDER] Model cache cleared")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the embedder with example queries"""
    print("="*60)
    print("QUERY EMBEDDER TEST")
    print("="*60)

    # Test single query
    print("\n[TEST 1] Single query embedding:")
    query = "What do people think about iPhone 15?"
    print(f"Query: '{query}'")

    embedding = embed_query(query)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Validate
    try:
        validate_embedding(embedding)
        print("Validation: PASSED")
    except ValueError as e:
        print(f"Validation: FAILED - {e}")

    # Test batch queries
    print("\n[TEST 2] Batch query embedding:")
    queries = [
        "Are gaming laptops worth it?",
        "What's the best smartphone?",
        "Is iPhone better than Android?"
    ]

    embeddings = embed_queries_batch(queries)
    print(f"Embedded {len(embeddings)} queries")

    # Test similarity
    print("\n[TEST 3] Query similarity:")
    sim = compute_similarity(embeddings[1], embeddings[2])
    print(f"Similarity between '{queries[1]}' and '{queries[2]}': {sim:.3f}")

    print("\n[TEST 4] Model caching:")
    print("Calling embed_query again (should be instant)...")
    import time
    start = time.time()
    _ = embed_query("Test query")
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f}s (model was cached!)")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
