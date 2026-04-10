"""
Database Status Checker
Quick script to monitor database growth and verify automation success
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from supabase_db.db_client import get_client


def format_number(num):
    """Format number with commas"""
    return f"{num:,}"


def print_separator(char='=', length=60):
    """Print a separator line"""
    print(char * length)


def get_recent_posts_count(client, hours=3):
    """Get count of posts collected in last N hours"""
    try:
        # Calculate timestamp for N hours ago
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.isoformat()

        response = client.client.table('reddit_posts').select(
            'post_id',
            count='exact'
        ).gte('collected_at', cutoff_timestamp).execute()

        return response.count
    except Exception as e:
        print(f"Warning: Could not get recent posts count: {e}")
        return None


def get_subreddit_breakdown(client, limit=10):
    """Get post count by subreddit"""
    try:
        response = client.client.table('reddit_posts').select('subreddit').execute()

        # Count posts per subreddit
        from collections import Counter
        subreddit_counts = Counter(post['subreddit'] for post in response.data)

        return subreddit_counts.most_common(limit)
    except Exception as e:
        print(f"Warning: Could not get subreddit breakdown: {e}")
        return []


def get_latest_post_time(client):
    """Get timestamp of most recent post"""
    try:
        response = client.client.table('reddit_posts').select(
            'collected_at'
        ).order('collected_at', desc=True).limit(1).execute()

        if response.data:
            return response.data[0]['collected_at']
        return None
    except Exception as e:
        print(f"Warning: Could not get latest post time: {e}")
        return None


def main():
    """Check database status"""
    print_separator()
    print("DATABASE STATUS CHECK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()

    try:
        # Initialize client
        print("\n[1/5] Connecting to Supabase...")
        client = get_client()
        print("✓ Connected")

        # Get total count
        print("\n[2/5] Getting total post count...")
        total_posts = client.get_post_count()
        print(f"✓ Total posts in database: {format_number(total_posts)}")

        # Get detailed stats
        print("\n[3/5] Getting database statistics...")
        stats = client.get_stats()

        if stats:
            print(f"✓ Posts with sentiment: {format_number(stats.get('posts_with_sentiment', 0))}")
            print(f"✓ Posts with embeddings: {format_number(stats.get('posts_with_embeddings', 0))}")
            print(f"✓ Average sentiment: {stats.get('avg_sentiment_compound', 0):.3f}")
        else:
            print("⚠ Could not retrieve detailed stats")

        # Get recent posts (last 3 hours - matches automation interval)
        print("\n[4/5] Checking recent activity...")
        recent_count = get_recent_posts_count(client, hours=3)

        if recent_count is not None:
            print(f"✓ Posts collected in last 3 hours: {format_number(recent_count)}")

            if recent_count > 0:
                print("  ✅ Automation is working!")
            else:
                print("  ⚠ No new posts in last 3 hours - check GitHub Actions")

        # Get latest post time
        latest_time = get_latest_post_time(client)
        if latest_time:
            # Parse ISO timestamp
            try:
                latest_dt = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))
                time_ago = datetime.now(latest_dt.tzinfo) - latest_dt
                hours_ago = time_ago.total_seconds() / 3600

                print(f"✓ Most recent post: {hours_ago:.1f} hours ago")

                if hours_ago > 4:
                    print("  ⚠ Last collection was over 4 hours ago - automation may be failing")
            except:
                print(f"✓ Most recent post: {latest_time}")

        # Get subreddit breakdown
        print("\n[5/5] Getting subreddit breakdown...")
        breakdown = get_subreddit_breakdown(client, limit=10)

        if breakdown:
            print("✓ Top 10 subreddits by post count:")
            for subreddit, count in breakdown:
                print(f"  r/{subreddit:20s} {format_number(count):>8s} posts")

        # Summary
        print_separator()
        print("SUMMARY")
        print_separator()
        print(f"Total Posts:     {format_number(total_posts)}")

        if recent_count is not None:
            print(f"Recent (3h):     {format_number(recent_count)}")

            if recent_count > 0:
                print("Status:          ✅ HEALTHY - Automation working")
            else:
                print("Status:          ⚠ WARNING - No recent posts")

        print_separator()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
