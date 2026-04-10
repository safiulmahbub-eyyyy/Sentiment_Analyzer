"""
Unified Supabase Pipeline for GitHub Actions
Orchestrates: Collection → Sentiment Analysis → Embedding Generation → Insertion

This file is a pure orchestrator - all logic is imported from existing modules.
Zero duplication. Maximum maintainability.

Pipeline Steps:
1. Collect posts from Reddit (delegates to collector.github_collector)
2. Add sentiment analysis (delegates to analyzer.sentiment_utils)
3. Generate embeddings (delegates to embeddings.embedding_utils)
4. Insert to Supabase (delegates to supabase_db.db_client)
5. Report statistics
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from supabase_db.db_client import get_client
from reddit_config import get_reddit_client

# Import collector functions
from collector.github_collector import (
    collect_from_subreddit,
    SUBREDDITS,
    FEED_LIMITS
)

# Import sentiment utilities
from analyzer.sentiment_utils import (
    calculate_sentiment,
    prepare_text_for_sentiment
)

# Import embedding utilities
from embeddings.embedding_utils import enrich_posts_with_embeddings
from embeddings.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    DEVICE
)

# Configuration
BATCH_SIZE = 100
ENABLE_EMBEDDINGS = True  # Toggle automated embedding generation


def enrich_posts_with_sentiment(
    posts: List[Dict[str, Any]],
    analyzer: SentimentIntensityAnalyzer
) -> List[Dict[str, Any]]:
    """
    Add sentiment analysis to collected posts

    This is the ONLY custom logic in this file - it bridges collection and sentiment.

    Args:
        posts: List of post dictionaries from collector
        analyzer: VADER analyzer instance

    Returns:
        Posts with sentiment fields added
    """
    enriched_posts = []

    for post in posts:
        # Use shared utility to prepare text
        text = prepare_text_for_sentiment(
            title=post.get('title', ''),
            body=post.get('selftext', '')
        )

        # Use shared utility to calculate sentiment
        sentiment = calculate_sentiment(text, analyzer)

        # Convert Unix timestamps to ISO 8601 format for PostgreSQL
        if 'created_utc' in post:
            post['created_utc'] = datetime.fromtimestamp(post['created_utc']).isoformat()
        if 'collected_at' in post:
            post['collected_at'] = datetime.fromtimestamp(post['collected_at']).isoformat()

        # Merge sentiment into post
        enriched_posts.append({**post, **sentiment})

    return enriched_posts


def collect_all_posts(reddit) -> List[Dict[str, Any]]:
    """
    Collect posts from all configured subreddits
    
    Args:
        reddit: Reddit client instance

    Returns:
        List of collected posts (raw, no sentiment), deduplicated by post_id
    """
    print("\n[COLLECTION] Gathering posts from subreddits...")
    all_posts = []

    for subreddit_name in SUBREDDITS:
        posts = collect_from_subreddit(reddit, subreddit_name)
        all_posts.extend(posts)
        time.sleep(2)  # Rate limiting

    # Deduplicate by post_id (same post can appear in new/hot/rising feeds)
    seen_ids = set()
    unique_posts = []
    duplicates = 0

    for post in all_posts:
        post_id = post.get('post_id')
        if post_id not in seen_ids:
            seen_ids.add(post_id)
            unique_posts.append(post)
        else:
            duplicates += 1

    print(f"[OK] Collected {len(all_posts):,} posts total")
    if duplicates > 0:
        print(f"[INFO] Removed {duplicates:,} duplicate posts from same collection")
    print(f"[OK] {len(unique_posts):,} unique posts ready for insertion")

    return unique_posts


def insert_posts_to_supabase(supabase, posts: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert posts to Supabase in batches
    
    Args:
        supabase: Supabase client
        posts: List of posts to insert
    
    Returns:
        Dictionary with success/error counts
    """
    print(f"\n[INSERTION] Uploading {len(posts):,} posts to Supabase...")
    result = supabase.insert_posts(posts, batch_size=BATCH_SIZE)
    
    print(f"[OK] Inserted {result['success']:,} posts")
    if result['errors'] > 0:
        print(f"[INFO] {result['errors']:,} posts skipped (likely duplicates)")
    
    return result




def print_statistics(supabase, collected_posts: List[Dict[str, Any]]):
    """
    Print pipeline statistics and breakdowns
    
    Args:
        supabase: Supabase client
        collected_posts: Posts collected in this run
    """
    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    
    # Database statistics
    stats = supabase.get_stats()
    if stats:
        print(f"\n[DATABASE] Overall stats:")
        print(f"  Total posts:          {stats.get('total_posts', 0):,}")
        print(f"  Posts with sentiment: {stats.get('posts_with_sentiment', 0):,}")
        print(f"  Posts with embeddings:{stats.get('posts_with_embeddings', 0):,}")
        print(f"  Avg sentiment:        {stats.get('avg_sentiment_compound', 0):.3f}")
    
    # Breakdown by subreddit (this batch)
    subreddit_counts = Counter(post['subreddit'] for post in collected_posts)
    print(f"\n[THIS BATCH] Posts by subreddit:")
    for sub, count in subreddit_counts.most_common():
        print(f"  r/{sub:20s} {count:4d} posts")
    
    # Sentiment distribution (this batch)
    if collected_posts and 'sentiment_label' in collected_posts[0]:
        sentiment_counts = Counter(post['sentiment_label'] for post in collected_posts)
        print(f"\n[THIS BATCH] Sentiment distribution:")
        total = len(collected_posts)
        for label, count in sentiment_counts.items():
            percent = (count / total) * 100
            print(f"  {label.upper():10s}: {count:4d} posts ({percent:5.1f}%)")


def main():
    """
    Main pipeline orchestrator

    This is a pure orchestration function - it has NO business logic.
    All work is delegated to imported modules.

    Steps:
    1. Initialize clients (Reddit, Supabase, VADER, Embedding Model)
    2. Collect posts (delegates to github_collector)
    3. Analyze sentiment (delegates to sentiment_utils)
    4. Generate embeddings (delegates to embedding_utils) - OPTIONAL
    5. Insert to database (delegates to db_client)
    6. Report statistics

    Note:
    - ENABLE_EMBEDDINGS flag controls whether embeddings are generated inline
    - Embeddings can also be generated in batch via embeddings/generate_embeddings.py
    """
    print("="*60)
    print("SUPABASE PIPELINE - AUTOMATED COLLECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Embedding Generation: {'ENABLED' if ENABLE_EMBEDDINGS else 'DISABLED'}")
    print("="*60)

    embedding_model = None

    try:
        # [1] Initialize clients
        step_count = 5 if ENABLE_EMBEDDINGS else 4
        print(f"\n[1/{step_count}] Initializing clients...")
        reddit = get_reddit_client()
        supabase = get_client()
        analyzer = SentimentIntensityAnalyzer()
        print("[OK] Reddit, Supabase, and VADER ready")

        # Initialize embedding model if enabled
        if ENABLE_EMBEDDINGS:
            print(f"[INIT] Loading embedding model: {EMBEDDING_MODEL}")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
            print(f"[OK] Embedding model loaded (dimension: 384)")

        # [2] Collect posts
        print(f"\n[2/{step_count}] Collecting posts...")
        raw_posts = collect_all_posts(reddit)

        if not raw_posts:
            print("\n[WARNING] No posts collected! Exiting.")
            return

        # [3] Analyze sentiment
        print(f"\n[3/{step_count}] Analyzing sentiment...")
        posts_with_sentiment = enrich_posts_with_sentiment(raw_posts, analyzer)
        print(f"[OK] Analyzed sentiment for {len(posts_with_sentiment):,} posts")

        # [4] Generate embeddings (optional)
        if ENABLE_EMBEDDINGS and embedding_model:
            print(f"\n[4/{step_count}] Generating embeddings...")
            start_time = time.time()

            posts_with_embeddings = enrich_posts_with_embeddings(
                posts=posts_with_sentiment,
                model=embedding_model,
                batch_size=EMBEDDING_BATCH_SIZE,
                max_length=MAX_TEXT_LENGTH,
                show_progress=False  # Disable progress bar in GitHub Actions
            )

            embedding_time = time.time() - start_time
            print(f"[OK] Generated {len(posts_with_embeddings):,} embeddings in {embedding_time:.1f}s")
            print(f"[SPEED] ~{len(posts_with_embeddings)/embedding_time:.0f} posts/second")

            final_posts = posts_with_embeddings
        else:
            final_posts = posts_with_sentiment

        # [5] Insert to database
        print(f"\n[{step_count}/{step_count}] Inserting to database...")
        insert_posts_to_supabase(supabase, final_posts)

        # [6] Report statistics
        print_statistics(supabase, final_posts)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()