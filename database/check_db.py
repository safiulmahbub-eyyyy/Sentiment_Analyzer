"""
Database Inspection Tool
Shows what's in the database
"""
import sqlite3
from datetime import datetime, timedelta

DATABASE_PATH = 'database/tech_sentiment.db'

def check_database():
    """Display database statistics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        print("="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        
        # Total posts
        cursor.execute("SELECT COUNT(*) FROM raw_posts")
        total = cursor.fetchone()[0]
        print(f"\nTotal posts: {total:,}")
        
        if total == 0:
            print("\nDatabase is empty. Run the collector first!")
            return
        
        # Posts by subreddit
        print("\nPosts by subreddit:")
        cursor.execute("""
            SELECT subreddit, COUNT(*) as count 
            FROM raw_posts 
            GROUP BY subreddit 
            ORDER BY count DESC
        """)
        for sub, count in cursor.fetchall():
            print(f"  r/{sub:20s} {count:5,} posts")
        
        # Recent posts
        print("\nRecent posts (last 5):")
        cursor.execute("""
            SELECT title, subreddit, created_utc, score
            FROM raw_posts
            ORDER BY created_utc DESC
            LIMIT 5
        """)
        for i, (title, sub, created, score) in enumerate(cursor.fetchall(), 1):
            date = datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M')
            print(f"  {i}. [{date}] r/{sub}")
            print(f"     {title[:60]}...")
            print(f"     Score: {score}")
        
        # Posts from last 7 days
        cutoff = (datetime.now() - timedelta(days=7)).timestamp()
        cursor.execute("SELECT COUNT(*) FROM raw_posts WHERE created_utc >= ?", (cutoff,))
        last_7d = cursor.fetchone()[0]
        print(f"\nPosts from last 7 days: {last_7d:,}")
        
        conn.close()
        print("\n" + "="*60)
        
    except sqlite3.Error as e:
        print(f"Error: {e}")
        print("Make sure the database exists. Run the collector first!")

if __name__ == "__main__":
    check_database()