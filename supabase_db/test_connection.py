"""
Test Supabase connection and verify setup
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from supabase_db.db_client import get_client


def test_connection():
    """Test Supabase connection"""
    print("="*60)
    print("Testing Supabase Connection")
    print("="*60)

    try:
        # Create client
        print("\n1. Creating Supabase client...")
        client = get_client()
        print("   [OK] Client created successfully")

        # Test database stats
        print("\n2. Testing database statistics...")
        stats = client.get_stats()

        if stats:
            print("   [OK] Database connection successful!")
            print("\n   Database Statistics:")
            print(f"   - Total posts: {stats.get('total_posts', 0):,}")
            print(f"   - Posts with embeddings: {stats.get('posts_with_embeddings', 0):,}")
            print(f"   - Posts with sentiment: {stats.get('posts_with_sentiment', 0):,}")
            print(f"   - Avg sentiment: {stats.get('avg_sentiment_compound', 0):.3f}")

            if stats.get('earliest_post'):
                print(f"   - Earliest post: {stats['earliest_post']}")
            if stats.get('latest_post'):
                print(f"   - Latest post: {stats['latest_post']}")
        else:
            print("   [WARN] Database is empty or stats function not available")

        # Test post count
        print("\n3. Testing post count query...")
        count = client.get_post_count()
        print(f"   [OK] Total posts: {count:,}")

        # Summary
        print("\n" + "="*60)
        print("Connection Test Results")
        print("="*60)

        if stats:
            if stats.get('total_posts', 0) > 0:
                print("[OK] PASSED: Database is populated and accessible")
            else:
                print("[WARN] WARNING: Database is empty (run migration first)")
        else:
            print("[WARN] WARNING: Could not retrieve statistics")

        print("\n[NOTE] Next steps:")
        if not stats or stats.get('total_posts', 0) == 0:
            print("   1. Run migration: python supabase_db/migrate.py")
            print("   2. Generate embeddings: python embeddings/generate_embeddings.py")
        elif stats.get('posts_with_embeddings', 0) == 0:
            print("   1. Generate embeddings: python embeddings/generate_embeddings.py")
        else:
            print("   1. Test search: python supabase_db/test_search.py")
            print("   2. Build RAG pipeline: rag/retriever.py")

    except Exception as e:
        print(f"\n[ERROR] Connection test failed: {e}")
        print("\n[FIX] Troubleshooting:")
        print("   1. Check .env file has correct Supabase credentials")
        print("   2. Verify SUPABASE_URL and SUPABASE_SERVICE_KEY are set")
        print("   3. Ensure schema.sql was run in Supabase SQL Editor")
        print("   4. Check Supabase project is not paused (free tier)")
        raise


if __name__ == "__main__":
    test_connection()
