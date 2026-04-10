"""
Import JSON Data to Local SQLite Database
Pulls data from GitHub and imports to local database
"""
import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_PATH = 'database/tech_sentiment.db'
JSON_DIR = 'data/collected'


def setup_database():
    """Ensure database and tables exist"""
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
    print("[OK] Database ready")


def pull_from_github():
    """Pull latest data from GitHub"""
    print("\n[PULL] Pulling latest data from GitHub...")
    try:
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to pull from GitHub: {e}")
        print(f"[INFO] Continuing with local files...")
        return False


def import_json_files():
    """Import all JSON files to database"""
    
    # Get all JSON files
    json_files = list(Path(JSON_DIR).glob('*.json'))
    
    if not json_files:
        print(f"[WARNING] No JSON files found in {JSON_DIR}")
        return
    
    print(f"\n[FOUND] {len(json_files)} JSON files to process")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    total_imported = 0
    total_skipped = 0
    
    for json_file in sorted(json_files):
        print(f"\n[PROCESSING] {json_file.name}")
        
        try:
            # Read JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                posts = json.load(f)
            
            imported = 0
            skipped = 0
            
            # Import each post
            for post in posts:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO raw_posts 
                        (post_id, subreddit, title, selftext, author,
                         created_utc, score, num_comments, url, permalink, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        post['post_id'],
                        post['subreddit'],
                        post['title'],
                        post['selftext'],
                        post['author'],
                        post['created_utc'],
                        post['score'],
                        post['num_comments'],
                        post['url'],
                        post['permalink'],
                        post['collected_at']
                    ))
                    
                    if cursor.rowcount > 0:
                        imported += 1
                    else:
                        skipped += 1
                        
                except sqlite3.Error as e:
                    print(f"  [ERROR] Failed to import post {post.get('post_id', 'unknown')}: {e}")
            
            conn.commit()
            
            print(f"  [IMPORTED] {imported} new posts")
            print(f"  [SKIPPED] {skipped} duplicates")
            
            total_imported += imported
            total_skipped += skipped
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {json_file.name}: {e}")
    
    conn.close()
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] Import Complete")
    print(f"[NEW] {total_imported} posts imported")
    print(f"[DUPLICATES] {total_skipped} posts skipped")
    
    # Get total count
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_posts")
    total = cursor.fetchone()[0]
    conn.close()
    
    print(f"[TOTAL] {total:,} posts in database")
    print("="*60)


def main():
    """Main import process"""
    print("="*60)
    print("[START] GitHub to SQLite Import")
    print("="*60)
    
    # Pull latest from GitHub
    pull_from_github()
    
    # Setup database
    setup_database()
    
    # Import JSON files
    import_json_files()
    
    print("\n[COMPLETE] Import finished!")
    print("[TIP] Run 'python database/check_db.py' to view your data")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Import cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()