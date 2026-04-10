"""
Continuous Reddit Data Collector - Simplified Version
Collects posts from tech subreddits and stores in SQLite
"""
import sqlite3
import time
import logging
import os
from datetime import datetime, timedelta
from reddit_config import get_reddit_client

# Configuration
SUBREDDITS = [
    # Mobile & Wearables
    'apple', 'iphone', 'android', 'GooglePixel', 'samsung', 'GalaxyWatch',
    
    # Computers & Laptops  
    'laptops', 'buildapc', 'pcgaming', 'pcmasterrace', 'battlestations',
    
    # Peripherals
    'mechanicalkeyboards', 'Monitors', 'headphones',
    
    # Gaming Handhelds
    'SteamDeck',
    
    # Smart Home
    'HomeAutomation', 'smarthome',
    
    # General & Support
    'technology', 'gadgets', 'TechSupport'
]

FEED_LIMITS = {
    'new': 100,
    'hot': 50,
    'rising': 25
}

DATABASE_PATH = 'database/tech_sentiment.db'

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_database():
    """Create database and tables"""
    os.makedirs('database', exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_posts (
            post_id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            title TEXT NOT NULL,
            selftext TEXT,
            author TEXT,
            created_utc REAL NOT NULL,
            score INTEGER,
            num_comments INTEGER,
            url TEXT,
            permalink TEXT,
            collected_at REAL NOT NULL
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON raw_posts(title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON raw_posts(created_utc DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subreddit ON raw_posts(subreddit)')
    
    conn.commit()
    conn.close()
    logger.info(f"[OK] Database ready: {DATABASE_PATH}")


def is_valid_post(post):
    """Check if post should be collected"""
    try:
        if not post.title or len(post.title) < 10:
            return False
        if post.selftext in ['[removed]', '[deleted]']:
            return False
        if post.author is None:
            return False
        if post.score < -5:
            return False
        return True
    except:
        return False


def collect_from_subreddit(reddit, subreddit_name):
    """Collect posts from one subreddit"""
    logger.info(f"[>>] Collecting from r/{subreddit_name}")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    total_new = 0
    total_filtered = 0
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Collect from each feed type
        for feed_type, limit in FEED_LIMITS.items():
            if feed_type == 'new':
                posts = subreddit.new(limit=limit)
            elif feed_type == 'hot':
                posts = subreddit.hot(limit=limit)
            elif feed_type == 'rising':
                posts = subreddit.rising(limit=limit)
            
            for post in posts:
                if not is_valid_post(post):
                    total_filtered += 1
                    continue
                
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO raw_posts 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        post.id,
                        subreddit_name,
                        post.title,
                        post.selftext,
                        str(post.author),
                        post.created_utc,
                        post.score,
                        post.num_comments,
                        post.url,
                        post.permalink,
                        time.time()
                    ))
                    
                    if cursor.rowcount > 0:
                        total_new += 1
                except:
                    continue
            
            time.sleep(1)  # Small delay between feeds
        
        conn.commit()
        logger.info(f"    r/{subreddit_name}: [OK] {total_new} new, [X] {total_filtered} filtered")
        
    except Exception as e:
        logger.error(f"[ERROR] r/{subreddit_name}: {e}")
    finally:
        conn.close()
    
    return total_new, total_filtered


def run_collection():
    """Run one complete collection cycle"""
    logger.info("="*60)
    logger.info("[START] Starting collection batch")
    logger.info("="*60)
    
    reddit = get_reddit_client()
    start_time = time.time()
    
    total_collected = 0
    total_filtered = 0
    
    # Collect from each subreddit
    for subreddit_name in SUBREDDITS:
        new, filtered = collect_from_subreddit(reddit, subreddit_name)
        total_collected += new
        total_filtered += filtered
        time.sleep(2)  # Delay between subreddits
    
    # Get database stats
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_posts")
    total_posts = cursor.fetchone()[0]
    conn.close()
    
    duration = time.time() - start_time
    
    # Summary
    logger.info("="*60)
    logger.info("[DONE] Collection complete")
    logger.info(f"[TIME] Duration: {duration:.1f} seconds")
    logger.info(f"[NEW] New posts: {total_collected}")
    logger.info(f"[FILTERED] Filtered: {total_filtered}")
    logger.info(f"[DATABASE] Total in DB: {total_posts:,}")
    logger.info("="*60)


def main():
    """Main entry point"""
    logger.info("\n[WELCOME] Reddit Continuous Collector")
    logger.info("[VERSION] 1.0 - Simplified")
    
    # Setup database
    setup_database()
    
    # Run collection
    run_collection()
    
    logger.info("\n[COMPLETE] Collection finished!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n[STOPPED] Collector stopped by user")
    except Exception as e:
        logger.error(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()