"""
GitHub Actions Collector
Collects Reddit posts and saves as JSON for later import to SQLite
"""
import os
import json
import time
from datetime import datetime
from reddit_config import get_reddit_client

# Configuration
SUBREDDITS = [
    # Mobile & Wearables
    'apple', 'iphone', 'android', 'GooglePixel', 'samsung', 'GalaxyWatch',
    
    # Computers & Gaming  
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

OUTPUT_DIR = 'data/collected'


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
    print(f"[>>] Collecting from r/{subreddit_name}")
    
    collected_posts = []
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
                
                # Convert post to dictionary
                post_data = {
                    'post_id': post.id,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'author': str(post.author),
                    'created_utc': post.created_utc,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'permalink': post.permalink,
                    'collected_at': time.time()
                }
                
                collected_posts.append(post_data)
            
            time.sleep(1)  # Small delay between feeds
        
        print(f"    r/{subreddit_name}: [OK] {len(collected_posts)} collected, [X] {total_filtered} filtered")
        
    except Exception as e:
        print(f"[ERROR] r/{subreddit_name}: {e}")
    
    return collected_posts


def save_to_json(all_posts):
    """Save collected posts to JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{OUTPUT_DIR}/reddit_posts_{timestamp}.json"
    
    # Save to JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVED] Data saved to: {filename}")
    print(f"[TOTAL] {len(all_posts)} posts saved")
    
    return filename


def main():
    """Main collection function"""
    print("="*60)
    print("[START] GitHub Actions Collection")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Get Reddit client
    reddit = get_reddit_client()
    
    # Collect from all subreddits
    all_posts = []
    
    for subreddit_name in SUBREDDITS:
        posts = collect_from_subreddit(reddit, subreddit_name)
        all_posts.extend(posts)
        time.sleep(2)  # Rate limiting
    
    # Save to JSON
    if all_posts:
        filename = save_to_json(all_posts)
        
        # Summary
        print("\n" + "="*60)
        print("[DONE] Collection complete")
        print(f"[FILE] {filename}")
        print(f"[POSTS] {len(all_posts)} posts collected")
        
        # Count by subreddit
        from collections import Counter
        subreddit_counts = Counter(post['subreddit'] for post in all_posts)
        print("\n[BREAKDOWN] Posts by subreddit:")
        for sub, count in subreddit_counts.most_common():
            print(f"  r/{sub:20s} {count:4d} posts")
        
        print("="*60)
    else:
        print("\n[WARNING] No posts collected!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)  # Exit with error code so GitHub knows it failed