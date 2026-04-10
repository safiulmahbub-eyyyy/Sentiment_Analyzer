"""
Process posts with VADER sentiment analysis
Refactored to use shared sentiment utilities - zero duplication
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analyzer.sentiment_utils import calculate_sentiment, prepare_text_for_sentiment

# Database path
DB_PATH = os.path.join('database', 'tech_sentiment.db')


def process_posts_with_vader(limit=None):
    """
    Process posts with VADER sentiment analysis
    Now uses shared sentiment utilities for consistency across the project
    
    Args:
        limit: Number of posts to process (None = all posts without sentiment)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print("=" * 70)
    print("VADER SENTIMENT ANALYSIS - PROCESSING POSTS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get posts that haven't been processed yet
        query = """
            SELECT post_id, title, selftext 
            FROM raw_posts 
            WHERE sentiment_compound IS NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        posts = cursor.fetchall()
        
        total_posts = len(posts)
        
        if total_posts == 0:
            print("\nNo posts to process (all already have sentiment scores)")
            conn.close()
            return True
        
        print(f"\nFound {total_posts:,} posts to process")
        print("=" * 70)
        
        processed = 0
        skipped = 0
        
        # Process each post
        for post_id, title, selftext in posts:
            processed += 1
            
            # Use shared utility to prepare text
            text_to_analyze = prepare_text_for_sentiment(
                title=title or '',
                body=selftext
            )
            
            # Skip if no text at all
            if not text_to_analyze:
                skipped += 1
                continue
            
            # Use shared utility to calculate sentiment
            sentiment = calculate_sentiment(text_to_analyze, analyzer)
            
            # Update database
            cursor.execute("""
                UPDATE raw_posts 
                SET sentiment_pos = ?,
                    sentiment_neg = ?,
                    sentiment_neu = ?,
                    sentiment_compound = ?,
                    sentiment_label = ?
                WHERE post_id = ?
            """, (
                sentiment['sentiment_pos'],
                sentiment['sentiment_neg'],
                sentiment['sentiment_neu'],
                sentiment['sentiment_compound'],
                sentiment['sentiment_label'],
                post_id
            ))
            
            # Progress indicator every 100 posts
            if processed % 100 == 0:
                conn.commit()  # Save progress
                percent = (processed / total_posts) * 100
                label = sentiment['sentiment_label']
                compound = sentiment['sentiment_compound']
                print(f"Progress: {processed}/{total_posts} ({percent:.1f}%) - "
                      f"Last: {label.upper()} ({compound:.3f})")
        
        # Final commit
        conn.commit()
        
        # Get summary statistics
        cursor.execute("""
            SELECT 
                sentiment_label,
                COUNT(*) as count
            FROM raw_posts
            WHERE sentiment_label IS NOT NULL
            GROUP BY sentiment_label
        """)
        
        results = cursor.fetchall()
        
        # Print results
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nProcessed: {processed - skipped:,} posts")
        if skipped > 0:
            print(f"Skipped (no text): {skipped:,} posts")
        
        print(f"\nSENTIMENT BREAKDOWN:")
        print("-" * 40)
        
        total_analyzed = sum(count for _, count in results)
        
        for label, count in results:
            print(f"  {label.upper():10s}: {count:6,d} posts")
        
        print("-" * 40)
        print(f"  {'TOTAL':10s}: {total_analyzed:6,d} posts")
        
        # Calculate percentages
        if total_analyzed > 0:
            print(f"\nPERCENTAGES:")
            print("-" * 40)
            for label, count in results:
                percent = (count / total_analyzed) * 100
                print(f"  {label.upper():10s}: {percent:5.1f}%")
        
        print("\n" + "=" * 70)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        if 'conn' in locals():
            conn.close()
        
        return False


def get_sentiment_statistics(detailed=False):
    """
    Get sentiment statistics from the database
    
    Args:
        detailed: If True, show detailed breakdown by subreddit
    
    Returns:
        dict: Statistics dictionary
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_posts,
                COUNT(sentiment_compound) as posts_with_sentiment,
                AVG(sentiment_compound) as avg_compound,
                AVG(sentiment_pos) as avg_positive,
                AVG(sentiment_neg) as avg_negative,
                AVG(sentiment_neu) as avg_neutral
            FROM raw_posts
        """)
        
        row = cursor.fetchone()
        stats = {
            'total_posts': row[0],
            'posts_with_sentiment': row[1],
            'avg_compound': row[2],
            'avg_positive': row[3],
            'avg_negative': row[4],
            'avg_neutral': row[5],
            'coverage_percent': (row[1] / row[0] * 100) if row[0] > 0 else 0
        }
        
        # Sentiment label distribution
        cursor.execute("""
            SELECT sentiment_label, COUNT(*) as count
            FROM raw_posts
            WHERE sentiment_label IS NOT NULL
            GROUP BY sentiment_label
        """)
        
        stats['label_distribution'] = {label: count for label, count in cursor.fetchall()}
        
        # Detailed breakdown by subreddit
        if detailed:
            cursor.execute("""
                SELECT 
                    subreddit,
                    COUNT(*) as total,
                    AVG(sentiment_compound) as avg_compound,
                    SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral
                FROM raw_posts
                WHERE sentiment_label IS NOT NULL
                GROUP BY subreddit
                ORDER BY total DESC
            """)
            
            stats['by_subreddit'] = [
                {
                    'subreddit': row[0],
                    'total': row[1],
                    'avg_compound': row[2],
                    'positive': row[3],
                    'negative': row[4],
                    'neutral': row[5]
                }
                for row in cursor.fetchall()
            ]
        
        conn.close()
        return stats
        
    except Exception as e:
        print(f"[ERROR] Failed to get statistics: {e}")
        return None


def print_sentiment_report():
    """Print a comprehensive sentiment analysis report"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS REPORT")
    print("=" * 70)
    
    stats = get_sentiment_statistics(detailed=True)
    
    if not stats:
        print("[ERROR] Could not retrieve statistics")
        return
    
    # Overall statistics
    print(f"\n[OVERALL STATISTICS]")
    print(f"  Total posts:          {stats['total_posts']:,}")
    print(f"  Posts with sentiment: {stats['posts_with_sentiment']:,}")
    print(f"  Coverage:             {stats['coverage_percent']:.1f}%")
    print(f"  Average compound:     {stats['avg_compound']:.3f}")
    
    # Distribution
    print(f"\n[SENTIMENT DISTRIBUTION]")
    total_with_sentiment = stats['posts_with_sentiment']
    for label, count in stats['label_distribution'].items():
        percent = (count / total_with_sentiment * 100) if total_with_sentiment > 0 else 0
        print(f"  {label.upper():10s}: {count:6,d} posts ({percent:5.1f}%)")
    
    # By subreddit
    if 'by_subreddit' in stats:
        print(f"\n[TOP SUBREDDITS BY SENTIMENT]")
        print(f"{'Subreddit':<20} {'Posts':>8} {'Avg':>6} {'Pos%':>6} {'Neg%':>6} {'Neu%':>6}")
        print("-" * 70)
        
        for sub in stats['by_subreddit'][:15]:  # Top 15
            pos_pct = (sub['positive'] / sub['total'] * 100) if sub['total'] > 0 else 0
            neg_pct = (sub['negative'] / sub['total'] * 100) if sub['total'] > 0 else 0
            neu_pct = (sub['neutral'] / sub['total'] * 100) if sub['total'] > 0 else 0
            
            print(f"r/{sub['subreddit']:<18} {sub['total']:>8,d} "
                  f"{sub['avg_compound']:>6.3f} "
                  f"{pos_pct:>5.1f}% "
                  f"{neg_pct:>5.1f}% "
                  f"{neu_pct:>5.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n[SENTIMENT ANALYSIS] Starting...")
    print("This may take a few minutes for large datasets...\n")
    
    # Process all posts without sentiment
    success = process_posts_with_vader()
    
    if success:
        print("\n[SUCCESS] All posts have been analyzed!")
        
        # Show comprehensive report
        print_sentiment_report()
        
    else:
        print("\n[FAILED] Processing encountered errors. Check logs above.")