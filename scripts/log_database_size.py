"""
Database Size Logger
Logs database statistics to track growth over time
Used in GitHub Actions to monitor automation success
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from supabase_db.db_client import get_client


def main():
    """Log current database size and statistics"""
    try:
        client = get_client()

        # Get counts
        total_posts = client.get_post_count()
        stats = client.get_stats()

        # Create log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_posts': total_posts,
            'posts_with_sentiment': stats.get('posts_with_sentiment', 0) if stats else 0,
            'posts_with_embeddings': stats.get('posts_with_embeddings', 0) if stats else 0,
            'avg_sentiment': stats.get('avg_sentiment_compound', 0) if stats else 0,
        }

        # Print as JSON for GitHub Actions to capture
        print(json.dumps(log_entry))

        # Also print human-readable summary
        print(f"\n📊 Database Size: {total_posts:,} posts", file=sys.stderr)
        print(f"✅ Logged at: {log_entry['timestamp']}", file=sys.stderr)

    except Exception as e:
        print(f"❌ Error logging database size: {e}", file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    main()
