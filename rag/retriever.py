"""
Vector Retrieval Module
Searches Supabase for semantically similar posts using vector embeddings
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from supabase_db.db_client import get_client

from rag.config import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_DATE_RANGE_DAYS,
    VERBOSE
)


def retrieve_similar_posts(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    subreddit_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    days_ago: int = DEFAULT_DATE_RANGE_DAYS
) -> List[Dict[str, Any]]:
    """
    Retrieve posts similar to the query embedding

    This is the main retrieval function for the RAG pipeline.
    It searches Supabase using vector similarity (pgvector + cosine distance)

    Args:
        query_embedding: Query vector (384 dimensions)
        top_k: Number of similar posts to retrieve
        similarity_threshold: Minimum similarity score (0-1, higher = more similar)
        subreddit_filter: Optional subreddit to filter by (e.g., "iphone")
        sentiment_filter: Optional sentiment to filter by ("positive", "negative", "neutral")
        days_ago: Only search posts from last N days (default: 365 = 1 year)

    Returns:
        List of post dictionaries with similarity scores, sorted by relevance

    Example:
        >>> from rag.embedder import embed_query
        >>> query_emb = embed_query("What do people think about iPhone 15?")
        >>> posts = retrieve_similar_posts(query_emb, top_k=10)
        >>> print(f"Found {len(posts)} relevant posts")
        Found 10 relevant posts
    """
    if VERBOSE:
        print(f"[RETRIEVER] Searching for top {top_k} posts...")
        print(f"[RETRIEVER] Similarity threshold: {similarity_threshold}")
        if subreddit_filter:
            print(f"[RETRIEVER] Subreddit filter: r/{subreddit_filter}")
        if sentiment_filter:
            print(f"[RETRIEVER] Sentiment filter: {sentiment_filter}")

    # Get Supabase client
    supabase = get_client()

    # Search using vector similarity
    posts = supabase.search_similar_posts(
        query_embedding=query_embedding,
        match_threshold=similarity_threshold,
        match_count=top_k,
        filter_subreddit=subreddit_filter,
        filter_sentiment=sentiment_filter,
        days_ago=days_ago
    )

    if VERBOSE:
        print(f"[RETRIEVER] Retrieved {len(posts)} posts")
        if posts:
            print(f"[RETRIEVER] Top similarity score: {posts[0].get('similarity', 0):.3f}")

    return posts


def filter_posts_by_date(
    posts: List[Dict[str, Any]],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Filter posts by date range (client-side filtering)

    Args:
        posts: List of posts to filter
        start_date: Include posts after this date
        end_date: Include posts before this date

    Returns:
        Filtered list of posts
    """
    filtered = posts

    if start_date:
        filtered = [
            p for p in filtered
            if datetime.fromisoformat(p['created_utc'].replace('Z', '+00:00')) >= start_date
        ]

    if end_date:
        filtered = [
            p for p in filtered
            if datetime.fromisoformat(p['created_utc'].replace('Z', '+00:00')) <= end_date
        ]

    return filtered


def filter_posts_by_score(
    posts: List[Dict[str, Any]],
    min_score: Optional[int] = None,
    max_score: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter posts by Reddit score (upvotes)

    Args:
        posts: List of posts to filter
        min_score: Minimum Reddit score
        max_score: Maximum Reddit score

    Returns:
        Filtered list of posts
    """
    filtered = posts

    if min_score is not None:
        filtered = [p for p in filtered if p.get('score', 0) >= min_score]

    if max_score is not None:
        filtered = [p for p in filtered if p.get('score', 0) <= max_score]

    return filtered


def rerank_by_relevance(
    posts: List[Dict[str, Any]],
    boost_score_weight: float = 0.1,
    boost_comments_weight: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Re-rank posts by combining similarity with Reddit engagement metrics

    Combines:
    - Vector similarity (main factor)
    - Reddit score (upvotes - minor boost)
    - Number of comments (discussion activity - minor boost)

    Args:
        posts: List of posts with similarity scores
        boost_score_weight: Weight for Reddit score (0-1)
        boost_comments_weight: Weight for comment count (0-1)

    Returns:
        Re-ranked list of posts
    """
    for post in posts:
        similarity = post.get('similarity', 0)
        reddit_score = max(0, post.get('score', 0))
        num_comments = post.get('num_comments', 0)

        # Normalize Reddit score (logarithmic scaling to prevent outliers)
        import math
        score_factor = math.log1p(reddit_score) / 10  # Log scale, cap at ~1
        comments_factor = math.log1p(num_comments) / 10

        # Combined relevance score
        post['relevance_score'] = (
            similarity +
            (score_factor * boost_score_weight) +
            (comments_factor * boost_comments_weight)
        )

    # Sort by relevance
    ranked = sorted(posts, key=lambda p: p.get('relevance_score', 0), reverse=True)

    return ranked


def get_diverse_posts(
    posts: List[Dict[str, Any]],
    max_per_subreddit: int = 3
) -> List[Dict[str, Any]]:
    """
    Get diverse posts (limit posts from same subreddit)

    Useful to avoid all results coming from one subreddit

    Args:
        posts: List of posts
        max_per_subreddit: Maximum posts per subreddit

    Returns:
        Diversified list of posts
    """
    subreddit_counts = {}
    diverse_posts = []

    for post in posts:
        subreddit = post.get('subreddit', 'unknown')
        count = subreddit_counts.get(subreddit, 0)

        if count < max_per_subreddit:
            diverse_posts.append(post)
            subreddit_counts[subreddit] = count + 1

    return diverse_posts


def format_post_preview(post: Dict[str, Any], max_length: int = 200) -> str:
    """
    Format a post as a preview string

    Useful for debugging and logging

    Args:
        post: Post dictionary
        max_length: Maximum character length

    Returns:
        Formatted preview string
    """
    title = post.get('title', 'No title')
    subreddit = post.get('subreddit', 'unknown')
    similarity = post.get('similarity', 0)
    sentiment = post.get('sentiment_label', 'unknown')

    preview = f"[r/{subreddit}] [{sentiment.upper()}] (sim: {similarity:.2f}) {title}"

    if len(preview) > max_length:
        preview = preview[:max_length-3] + "..."

    return preview


# ============================================================
# ADVANCED RETRIEVAL STRATEGIES
# ============================================================

def hybrid_retrieval(
    query_embedding: List[float],
    keywords: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: Vector similarity + keyword matching

    Combines semantic search with keyword filtering for better precision

    Args:
        query_embedding: Query vector
        keywords: Optional keywords to boost (e.g., ["iphone", "15"])
        top_k: Number of results
        **kwargs: Additional arguments for retrieve_similar_posts()

    Returns:
        Retrieved posts
    """
    # Get semantic results
    posts = retrieve_similar_posts(query_embedding, top_k=top_k * 2, **kwargs)

    # Boost posts containing keywords (if provided)
    if keywords:
        keywords_lower = [k.lower() for k in keywords]

        for post in posts:
            title_lower = post.get('title', '').lower()
            body_lower = post.get('selftext', '').lower()

            # Count keyword matches
            matches = sum(
                1 for kw in keywords_lower
                if kw in title_lower or kw in body_lower
            )

            # Boost similarity by keyword matches
            if matches > 0:
                boost = min(0.1 * matches, 0.3)  # Max boost of 0.3
                post['similarity'] = min(1.0, post.get('similarity', 0) + boost)

        # Re-sort by boosted similarity
        posts = sorted(posts, key=lambda p: p.get('similarity', 0), reverse=True)

    # Return top-k after boosting
    return posts[:top_k]


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the retriever with a sample query"""
    print("="*60)
    print("RETRIEVER TEST")
    print("="*60)

    # Import embedder for testing
    from rag.embedder import embed_query

    # Test query
    query = "What do people think about iPhone 15 camera quality?"
    print(f"\nQuery: '{query}'")

    # Embed query
    print("\n[1/3] Embedding query...")
    query_embedding = embed_query(query)
    print(f"Embedding dimension: {len(query_embedding)}")

    # Retrieve posts
    print("\n[2/3] Retrieving similar posts...")
    posts = retrieve_similar_posts(
        query_embedding=query_embedding,
        top_k=5,
        similarity_threshold=0.3,
        days_ago=365
    )

    print(f"\nRetrieved {len(posts)} posts")

    # Display results
    print("\n[3/3] Top results:")
    for i, post in enumerate(posts, 1):
        preview = format_post_preview(post)
        print(f"{i}. {preview}")

    # Test filtering
    if posts:
        print("\n[TEST] Filtering by score (min_score=10):")
        filtered = filter_posts_by_score(posts, min_score=10)
        print(f"Filtered to {len(filtered)} posts")

    # Test diversity
    if posts:
        print("\n[TEST] Getting diverse posts (max 2 per subreddit):")
        diverse = get_diverse_posts(posts, max_per_subreddit=2)
        print(f"Diversified to {len(diverse)} posts")

    print("\n" + "="*60)
    print("RETRIEVER TEST COMPLETE")
    print("="*60)
