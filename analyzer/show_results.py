import sqlite3
import os

DB_PATH = os.path.join('database', 'tech_sentiment.db')

def show_sentiment_results():
    """Display sentiment analysis results"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 70)
    
    # Overall statistics
    cursor.execute("""
        SELECT 
            sentiment_label,
            COUNT(*) as count
        FROM raw_posts
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
        ORDER BY count DESC
    """)
    
    results = cursor.fetchall()
    total = sum(count for _, count in results)
    
    print(f"\nTotal Posts Analyzed: {total:,}")
    print("\nSENTIMENT BREAKDOWN:")
    print("-" * 70)
    
    for label, count in results:
        percent = (count / total) * 100
        bar_length = int(percent / 2)
        bar = "█" * bar_length
        print(f"{label.upper():10s} | {bar:50s} | {count:6,d} ({percent:5.1f}%)")
    
    # Top positive posts
    print("\n" + "=" * 70)
    print("TOP 5 MOST POSITIVE POSTS:")
    print("=" * 70)
    
    cursor.execute("""
        SELECT title, sentiment_compound, subreddit
        FROM raw_posts
        WHERE sentiment_compound IS NOT NULL
        AND title IS NOT NULL
        ORDER BY sentiment_compound DESC
        LIMIT 5
    """)
    
    for i, (title, compound, subreddit) in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. [{subreddit}] (Score: {compound:.3f})")
        print(f"   {title[:80]}...")
    
    # Top negative posts
    print("\n" + "=" * 70)
    print("TOP 5 MOST NEGATIVE POSTS:")
    print("=" * 70)
    
    cursor.execute("""
        SELECT title, sentiment_compound, subreddit
        FROM raw_posts
        WHERE sentiment_compound IS NOT NULL
        AND title IS NOT NULL
        ORDER BY sentiment_compound ASC
        LIMIT 5
    """)
    
    for i, (title, compound, subreddit) in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. [{subreddit}] (Score: {compound:.3f})")
        print(f"   {title[:80]}...")
    
    # Sentiment by subreddit (top 10)
    print("\n" + "=" * 70)
    print("SENTIMENT BY SUBREDDIT (Top 10 by post count):")
    print("=" * 70)
    
    cursor.execute("""
        SELECT 
            subreddit,
            COUNT(*) as total,
            AVG(sentiment_compound) as avg_sentiment,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral
        FROM raw_posts
        WHERE sentiment_label IS NOT NULL
        GROUP BY subreddit
        ORDER BY total DESC
        LIMIT 10
    """)
    
    print(f"\n{'Subreddit':<25} {'Posts':>7} {'Avg Score':>10} {'Pos':>6} {'Neg':>6} {'Neu':>6}")
    print("-" * 70)
    
    for subreddit, total, avg_sent, pos, neg, neu in cursor.fetchall():
        print(f"{subreddit:<25} {total:>7,d} {avg_sent:>10.3f} {pos:>6,d} {neg:>6,d} {neu:>6,d}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    show_sentiment_results()