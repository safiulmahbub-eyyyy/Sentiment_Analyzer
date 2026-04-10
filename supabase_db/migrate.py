"""
Migrate data from SQLite to Supabase (PostgreSQL)
"""

import sys
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from supabase_db.db_client import get_client


class Migration:
    """Handle SQLite to Supabase migration"""

    def __init__(self, sqlite_path: str = "database/tech_sentiment.db"):
        """Initialize migration

        Args:
            sqlite_path: Path to SQLite database
        """
        self.sqlite_path = Path(sqlite_path)
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

        self.supabase = get_client()
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries

    def fetch_all_posts(self) -> List[Dict[str, Any]]:
        """Fetch all posts from SQLite"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM raw_posts")

        posts = []
        for row in cursor.fetchall():
            post = dict(row)

            # Convert Unix timestamp to ISO format for PostgreSQL
            post['created_utc'] = datetime.fromtimestamp(post['created_utc']).isoformat()
            post['collected_at'] = datetime.fromtimestamp(post['collected_at']).isoformat()

            posts.append(post)

        return posts

    def migrate(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Perform migration from SQLite to Supabase

        Args:
            batch_size: Number of posts per batch

        Returns:
            Migration statistics
        """
        print("="*60)
        print("Starting SQLite to Supabase migration...")
        print("="*60)
        print(f"Reading from SQLite: {self.sqlite_path}")

        # Fetch all posts
        start_time = time.time()
        posts = self.fetch_all_posts()
        total_posts = len(posts)

        print(f"Total posts to migrate: {total_posts:,}")
        print(f"\nMigrating in batches of {batch_size}...")

        # Migrate in batches
        success_count = 0
        error_count = 0

        for i in range(0, total_posts, batch_size):
            batch = posts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_posts + batch_size - 1) // batch_size

            # Show progress
            progress = (i + len(batch)) / total_posts * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '=' * filled + '-' * (bar_length - filled)

            print(f"\rBatch {batch_num}/{total_batches} [{bar}] {progress:.1f}%", end='', flush=True)

            # Insert batch
            try:
                result = self.supabase.insert_posts(batch, batch_size=len(batch))
                success_count += result['success']
                error_count += result['errors']
            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")
                error_count += len(batch)

        print()  # New line after progress bar

        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        # Print summary
        print("\n" + "="*60)
        print("Migration complete!")
        print("="*60)
        print(f"[OK] Successfully migrated: {success_count:,} posts")
        print(f"[FAIL] Failed: {error_count:,} posts")
        print(f"[TIME] Time taken: {minutes}m {seconds}s")

        if error_count > 0:
            print(f"\n[WARN] Warning: {error_count} posts failed to migrate")
            print("   Check Supabase logs for details")

        return {
            'total': total_posts,
            'success': success_count,
            'errors': error_count,
            'duration_seconds': duration
        }

    def verify_migration(self) -> bool:
        """Verify migration completed successfully"""
        print("\n" + "="*60)
        print("Verifying migration...")
        print("="*60)

        # Count posts in SQLite
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_posts")
        sqlite_count = cursor.fetchone()[0]

        # Count posts in Supabase
        supabase_count = self.supabase.get_post_count()

        print(f"SQLite posts: {sqlite_count:,}")
        print(f"Supabase posts: {supabase_count:,}")

        if sqlite_count == supabase_count:
            print("[OK] Verification passed! All posts migrated successfully.")
            return True
        else:
            print(f"[ERROR] Verification failed! Missing {sqlite_count - supabase_count:,} posts.")
            return False

    def close(self):
        """Close SQLite connection"""
        self.conn.close()


def main():
    """Main migration function"""
    try:
        # Create migration instance
        migration = Migration()

        # Perform migration
        stats = migration.migrate(batch_size=100)

        # Verify migration
        migration.verify_migration()

        # Show final stats
        print("\n" + "="*60)
        print("Migration Statistics:")
        print("="*60)
        stats_data = migration.supabase.get_stats()
        if stats_data:
            print(f"Total posts: {stats_data.get('total_posts', 0):,}")
            print(f"Posts with sentiment: {stats_data.get('posts_with_sentiment', 0):,}")
            print(f"Posts with embeddings: {stats_data.get('posts_with_embeddings', 0):,}")
            print(f"Avg sentiment: {stats_data.get('avg_sentiment_compound', 0):.3f}")

        # Cleanup
        migration.close()

        print("\n[OK] Migration completed successfully!")
        print("\n[NOTE] Next steps:")
        print("   1. Run: python embeddings/generate_embeddings.py")
        print("   2. Test search: python supabase_db/test_search.py")
        print("   3. Update GitHub Actions: .github/workflows/sync_to_supabase.yml")

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
