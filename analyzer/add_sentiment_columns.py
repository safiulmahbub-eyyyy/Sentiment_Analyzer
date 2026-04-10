# analyzer/add_sentiment_columns.py

import sqlite3
import os

# Database path
DB_PATH = os.path.join('database', 'tech_sentiment.db')

def add_sentiment_columns():
    """Add sentiment analysis columns to the raw_posts table"""
    
    print("=" * 60)
    print("ADDING SENTIMENT COLUMNS TO DATABASE")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check current columns
        cursor.execute("PRAGMA table_info(raw_posts)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        print(f"\nExisting columns: {', '.join(existing_columns)}")
        
        # Add sentiment columns if they don't exist
        new_columns = {
            'sentiment_pos': 'REAL',
            'sentiment_neg': 'REAL',
            'sentiment_neu': 'REAL',
            'sentiment_compound': 'REAL',
            'sentiment_label': 'TEXT'
        }
        
        columns_added = 0
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                print(f"Adding column: {col_name} ({col_type})")
                cursor.execute(f"ALTER TABLE raw_posts ADD COLUMN {col_name} {col_type}")
                columns_added += 1
            else:
                print(f"Column already exists: {col_name}")
        
        conn.commit()
        
        # Verify new structure
        cursor.execute("PRAGMA table_info(raw_posts)")
        all_columns = [column[1] for column in cursor.fetchall()]
        
        print(f"\nUpdated columns: {', '.join(all_columns)}")
        print("\n" + "=" * 60)
        print(f"SUCCESS! Added {columns_added} new columns")
        print("=" * 60)
        
        conn.close()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = add_sentiment_columns()
    if success:
        print("\nReady to process posts with VADER!")
    else:
        print("\nFailed to update schema. Check error above.")