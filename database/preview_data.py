"""
Quick Data Preview
Shows sample posts so you can verify quality
"""
import sqlite3
from datetime import datetime

def preview_data():
    conn = sqlite3.connect('database/tech_sentiment.db')
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("DATA QUALITY PREVIEW - Random Sample of 10 Posts")
    print("="*80)
    
    # Get 10 random posts
    cursor.execute("""
        SELECT subreddit, title, selftext, score, num_comments, created_utc
        FROM raw_posts
        ORDER BY RANDOM()
        LIMIT 10
    """)
    
    posts = cursor.fetchall()
    
    for i, (sub, title, text, score, comments, created) in enumerate(posts, 1):
        date = datetime.fromtimestamp(created).strftime('%Y-%m-%d')
        
        print(f"\n[{i}] r/{sub} | {date} | ↑{score} | 💬{comments}")
        print(f"    Title: {title[:70]}...")
        
        if text and len(text) > 0:
            print(f"    Text:  {text[:100]}...")
        else:
            print(f"    Text:  [No text content]")
    
    conn.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    preview_data()