"""
Generate vector embeddings for all posts in Supabase

This script processes existing posts in batches to generate embeddings.
For inline embedding generation during collection, see collector/supabase_pipeline.py
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import torch
from supabase_db.db_client import get_client
from embeddings.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    UPDATE_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    DEVICE,
    SHOW_PROGRESS
)
from embeddings.embedding_utils import (
    prepare_text_for_embedding,
    generate_embeddings_batch
)


class EmbeddingGenerator:
    """
    Generate embeddings for Reddit posts in Supabase

    This class handles batch processing of existing posts.
    All embedding logic is delegated to embedding_utils for reusability.
    """

    def __init__(self):
        """Initialize embedding generator with model and database client"""
        print(f"[INIT] Loading model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print(f"[OK] Model loaded! Embedding dimension: {EMBEDDING_DIMENSION}")

        self.supabase = get_client()
        print(f"[OK] Supabase client ready")

    def process_all_posts(self):
        """
        Generate embeddings for all posts without embeddings

        This method:
        1. Fetches posts missing embeddings from Supabase
        2. Prepares texts using shared utility
        3. Generates embeddings in batches using shared utility
        4. Uploads embeddings back to Supabase
        """
        print("\n[1/4] Fetching posts without embeddings...")
        posts = self.supabase.get_posts_without_embeddings()
        total_posts = len(posts)

        if total_posts == 0:
            print("[OK] All posts already have embeddings!")
            return

        print(f"[OK] Found {total_posts:,} posts to process\n")

        # Prepare texts using shared utility
        print("[2/4] Preparing texts...")
        texts = [
            prepare_text_for_embedding(
                title=post.get('title', ''),
                body=post.get('selftext', ''),
                max_length=MAX_TEXT_LENGTH
            )
            for post in posts
        ]
        print(f"[OK] Prepared {len(texts):,} texts")

        # Generate embeddings using shared utility
        print(f"\n[3/4] Generating embeddings (batch size: {EMBEDDING_BATCH_SIZE})...")
        start_time = time.time()

        embeddings = generate_embeddings_batch(
            texts=texts,
            model=self.model,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress=SHOW_PROGRESS
        )

        embedding_time = time.time() - start_time
        posts_per_second = total_posts / embedding_time

        print(f"[OK] Generated {total_posts:,} embeddings in {embedding_time:.1f}s")
        print(f"[SPEED] ~{posts_per_second:.0f} posts/second")

        # Prepare updates
        print("\n[4/4] Uploading embeddings to Supabase...")
        updates = [
            {
                'post_id': posts[i]['post_id'],
                'embedding': embeddings[i]
            }
            for i in range(total_posts)
        ]
        print(f"[OK] Prepared {len(updates):,} updates")

        # Upload to Supabase in batches
        upload_start = time.time()
        total_batches = (total_posts + UPDATE_BATCH_SIZE - 1) // UPDATE_BATCH_SIZE
        success_count = 0

        print(f"[UPLOAD] Processing in batches of {UPDATE_BATCH_SIZE}...")

        for i in range(0, total_posts, UPDATE_BATCH_SIZE):
            batch = updates[i:i + UPDATE_BATCH_SIZE]
            batch_num = i // UPDATE_BATCH_SIZE + 1

            # Show progress bar
            progress = (i + len(batch)) / total_posts * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '=' * filled + '-' * (bar_length - filled)

            print(f"\rBatch {batch_num}/{total_batches} [{bar}] {progress:.1f}%", end='', flush=True)

            try:
                result = self.supabase.update_embeddings(batch, batch_size=len(batch))
                success_count += result['success']
            except Exception as e:
                print(f"\n[ERROR] Batch {batch_num} failed: {e}")

        print()  # New line after progress bar

        upload_time = time.time() - upload_start
        total_time = time.time() - start_time

        # Print final summary
        print("\n" + "="*60)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"[OK] Successfully processed: {success_count:,} posts")
        print(f"[TIME] Embedding generation: {int(embedding_time//60)}m {int(embedding_time%60)}s")
        print(f"[TIME] Database upload: {int(upload_time//60)}m {int(upload_time%60)}s")
        print(f"[TIME] Total time: {int(total_time//60)}m {int(total_time%60)}s")
        print(f"[SPEED] Overall: ~{success_count/total_time:.0f} posts/second")
        print("="*60)

    def generate_for_new_posts(self, post_ids: List[str] = None):
        """
        Generate embeddings for specific posts or recent posts

        Args:
            post_ids: Optional list of post IDs to process

        Note:
            If no post_ids provided, processes all posts without embeddings.
            For inline embedding during collection, use collector/supabase_pipeline.py
        """
        if post_ids:
            print(f"[INFO] Generating embeddings for {len(post_ids)} specific posts...")
            # TODO: Implement fetching and processing specific posts by ID
            print("[WARNING] Specific post ID processing not yet implemented")
            print("[INFO] Falling back to processing all posts without embeddings")

        # Process all posts without embeddings
        self.process_all_posts()


def main():
    """
    Main entry point for batch embedding generation

    This script is used for:
    1. Initial embedding generation for all existing posts
    2. Backfilling embeddings for posts that don't have them
    3. Manual re-generation of embeddings if needed

    For automated embedding generation during collection:
    See collector/supabase_pipeline.py
    """
    print("="*60)
    print("BATCH EMBEDDING GENERATION")
    print("="*60)

    try:
        # Initialize and run generator
        generator = EmbeddingGenerator()
        generator.process_all_posts()

        # Display final database statistics
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        stats = generator.supabase.get_stats()
        if stats:
            total = stats.get('total_posts', 0)
            with_embeddings = stats.get('posts_with_embeddings', 0)
            coverage = (with_embeddings / total * 100) if total > 0 else 0

            print(f"Total posts:          {total:,}")
            print(f"Posts with embeddings: {with_embeddings:,}")
            print(f"Coverage:             {coverage:.1f}%")

        print("\n[OK] Embedding generation completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
