"""
Embedding Generation Utilities
Shared functions for vector embeddings across the project
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


def prepare_text_for_embedding(title: str, body: str = None, max_length: int = 512) -> str:
    """
    Prepare post text for embedding generation

    Combines title and body text, truncates if too long.
    Shared utility used by:
    - embeddings/generate_embeddings.py (batch processing)
    - collector/supabase_pipeline.py (inline embedding generation)

    Args:
        title: Post title (required)
        body: Post body/selftext (optional)
        max_length: Maximum character length

    Returns:
        Combined and truncated text ready for embedding
    """
    # Combine title and body
    text = title or ""

    if body and body.strip():
        text += "\n" + body

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def generate_embeddings_batch(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts

    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model instance
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar

    Returns:
        List of embedding vectors (as lists of floats)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )

    # Convert numpy arrays to Python lists for JSON serialization
    return embeddings.tolist()


def enrich_post_with_embedding(
    post: Dict[str, Any],
    model: SentenceTransformer,
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Add embedding vector to a single post

    Args:
        post: Post dictionary
        model: SentenceTransformer model instance
        max_length: Maximum text length for embedding

    Returns:
        Post dictionary with 'embedding' field added
    """
    # Prepare text
    text = prepare_text_for_embedding(
        title=post.get('title', ''),
        body=post.get('selftext', ''),
        max_length=max_length
    )

    # Generate embedding
    embedding = model.encode([text], convert_to_numpy=True)

    # Add to post (convert to list for JSON)
    post['embedding'] = embedding[0].tolist()

    return post


def enrich_posts_with_embeddings(
    posts: List[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 32,
    max_length: int = 512,
    show_progress: bool = False
) -> List[Dict[str, Any]]:
    """
    Add embeddings to a list of posts efficiently (batched)

    Args:
        posts: List of post dictionaries
        model: SentenceTransformer model instance
        batch_size: Number of texts to embed at once
        max_length: Maximum text length
        show_progress: Whether to show progress bar

    Returns:
        Posts with 'embedding' field added to each
    """
    if not posts:
        return posts

    # Prepare all texts
    texts = [
        prepare_text_for_embedding(
            title=post.get('title', ''),
            body=post.get('selftext', ''),
            max_length=max_length
        )
        for post in posts
    ]

    # Generate embeddings in batch (more efficient)
    embeddings = generate_embeddings_batch(
        texts=texts,
        model=model,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Add embeddings to posts
    enriched_posts = []
    for i, post in enumerate(posts):
        post['embedding'] = embeddings[i]
        enriched_posts.append(post)

    return enriched_posts
