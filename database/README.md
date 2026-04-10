# Database Module (Legacy SQLite)

**Historical File-Based Storage System - Archived in Week 4**

This module contains the original SQLite database implementation used during Weeks 1-3 of the project (October 19 - November 7, 2025). The database was migrated to Supabase (PostgreSQL + pgvector) in Week 4 due to scalability limitations, but the module is preserved for local development, testing, and historical reference.

---

## Overview

The legacy database module implemented a file-based SQLite database (`tech_sentiment.db`) for storing Reddit posts with metadata and sentiment scores. While functional for initial development, the approach encountered scalability constraints that necessitated migration to a cloud-based solution.

**Module Status:** 🗄️ **ARCHIVED** (replaced by `supabase_db/` in Week 4)

**Preservation Rationale:**
- Historical record of project evolution
- Useful for local development without cloud dependencies
- Testing environment for database-agnostic code
- Educational reference for understanding migration decisions

**Module Components:**
1. `tech_sentiment.db` - SQLite database file (gitignored, ~67MB)
2. `check_db.py` - Database statistics and inspection tool
3. `preview_data.py` - Data quality verification utility

---

## Introduction

### Original Design (Weeks 1-3)

During the initial development phase, SQLite was selected as the database solution based on the following criteria:

**Selection Rationale:**
- **Zero Configuration:** Single-file database, no server setup required
- **Simplicity:** Built into Python standard library (`sqlite3` module)
- **Portability:** Database file can be versioned with code (initially)
- **Development Speed:** Immediate prototyping without cloud account setup
- **Cost:** Completely free, no external dependencies

**Architecture:**

```
┌─────────────────────────────────────────┐
│     Reddit API Collection               │
│     (collector/continuous_collector.py) │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│   SQLite Database (tech_sentiment.db)   │
│                                          │
│   Table: raw_posts                      │
│   • 31,097 posts (at migration)         │
│   • No vector embeddings (Week 1-3)     │
│   • VADER sentiment scores (Week 3)     │
│   • File size: ~67 MB                   │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│     Analysis Scripts                     │
│     (analyzer/, scripts/)                │
└──────────────────────────────────────────┘
```

### Limitations Encountered (Week 4)

**Critical Issue:** Unsustainable growth rate approaching GitHub's 100MB repository limit

**Growth Metrics:**
```
Dataset size:     31,097 posts
File size:        67 MB
Growth rate:      ~2,000 posts/day
Storage per post: ~2.2 KB
Daily growth:     ~4.4 MB/day (~31 MB/week)

Projected timeline to 100MB limit:
(100 MB - 67 MB) / 31 MB/week ≈ 1 week remaining capacity
```

**Additional Limitations:**

1. **No Native Vector Search:** SQLite lacks native vector similarity search
   - Would require separate extension (sqlite-vss)
   - Performance not optimized for high-dimensional vectors
   - Complicates deployment and dependencies

2. **Single-Writer Concurrency:**
   - Only one process can write at a time
   - GitHub Actions automation would block local development
   - No connection pooling support

3. **Scalability Ceiling:**
   - Performance degrades with large file sizes (>1 GB)
   - No built-in sharding or partitioning
   - Entire database loaded into memory for some operations

4. **Cloud Deployment Complexity:**
   - File-based database difficult to share across services
   - No built-in backup/replication
   - Manual file management required

**Decision:** Migrate to Supabase (PostgreSQL + pgvector) for production scalability

---

## Database Schema

### Table: `raw_posts`

**Schema Definition:**

```sql
CREATE TABLE raw_posts (
    -- Primary key
    post_id TEXT PRIMARY KEY,

    -- Reddit metadata
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    created_utc REAL NOT NULL,           -- Unix timestamp (differs from Supabase TIMESTAMPTZ)
    score INTEGER,
    num_comments INTEGER,
    url TEXT,
    permalink TEXT,
    collected_at REAL NOT NULL,          -- Unix timestamp

    -- Sentiment analysis (added Week 3)
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT
);
```

**Schema Differences from Supabase:**

| Column | SQLite Type | Supabase Type | Notes |
|--------|-------------|---------------|-------|
| `created_utc` | REAL (Unix timestamp) | TIMESTAMPTZ (ISO 8601) | Requires conversion for migration |
| `collected_at` | REAL (Unix timestamp) | TIMESTAMPTZ (ISO 8601) | Requires conversion for migration |
| `sentiment_label` | TEXT | TEXT CHECK (...) | Supabase adds constraint validation |
| `embedding` | *(Not present)* | vector(384) | Added in Week 4 for Supabase only |

**Indexes:**

```sql
-- Performance indexes (created in Week 2)
CREATE INDEX idx_title ON raw_posts(title);
CREATE INDEX idx_created ON raw_posts(created_utc DESC);
CREATE INDEX idx_subreddit ON raw_posts(subreddit);
CREATE INDEX idx_sentiment_label ON raw_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON raw_posts(sentiment_compound DESC);
```

### Data Evolution Timeline

**Week 1-2 (October 19-26):**
- Initial schema created
- Only metadata columns populated
- No sentiment analysis
- Database size: ~15 MB (7,000 posts)

**Week 3 (October 27 - November 1):**
- Added sentiment columns
- Backfilled VADER scores for all existing posts
- Database size: ~45 MB (21,000 posts)

**Week 4 (November 2-7):**
- Final snapshot: 31,097 posts, ~67 MB
- Migration to Supabase completed
- SQLite database archived

---

## Utility Scripts

### Module: `check_db.py`

**Purpose:** Display database statistics and health metrics

**Usage:**

```bash
python database/check_db.py
```

**Output:**

```
============================================================
DATABASE STATISTICS
============================================================

Total posts: 31,097

Posts by subreddit:
  r/pcmasterrace            5,930 posts
  r/buildapc                5,572 posts
  r/TechSupport             4,757 posts
  r/iphone                  3,007 posts
  r/laptops                 2,407 posts
  r/android                 1,845 posts
  r/apple                   1,567 posts
  r/samsung                 1,234 posts
  r/mechanicalkeyboards       987 posts
  r/headphones                876 posts
  [...]

Recent posts (last 5):
  1. [2025-11-07 14:32] r/iphone
     "iPhone 15 Pro battery life is impressive"
     Score: 245 | Comments: 89

  2. [2025-11-07 14:28] r/buildapc
     "First PC build - need advice on GPU"
     Score: 12 | Comments: 34

  [...]

Sentiment Distribution:
  Positive: 14,896 posts (47.9%)
  Neutral:  10,012 posts (32.2%)
  Negative:  6,189 posts (19.9%)

============================================================
```

**Implementation Details:**

```python
def check_database():
    """Display database statistics"""
    conn = sqlite3.connect('database/tech_sentiment.db')
    cursor = conn.cursor()

    # Total posts
    cursor.execute("SELECT COUNT(*) FROM raw_posts")
    total = cursor.fetchone()[0]

    # Posts by subreddit
    cursor.execute("""
        SELECT subreddit, COUNT(*) as count
        FROM raw_posts
        GROUP BY subreddit
        ORDER BY count DESC
    """)

    # Sentiment distribution
    cursor.execute("""
        SELECT sentiment_label, COUNT(*) as count
        FROM raw_posts
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
    """)

    # Recent posts
    cursor.execute("""
        SELECT title, subreddit, created_utc, score
        FROM raw_posts
        ORDER BY created_utc DESC
        LIMIT 5
    """)
```

**Code Location:** `database/check_db.py`

---

### Module: `preview_data.py`

**Purpose:** Data quality verification and integrity checks

**Usage:**

```bash
python database/preview_data.py
```

**Output:**

```
============================================================
DATA QUALITY REPORT
============================================================

Total Posts: 31,097

Data Completeness:
  ✓ All posts have post_id (primary key)
  ✓ All posts have title
  ✓ All posts have subreddit
  ✓ 98.5% posts have selftext (30,629 / 31,097)
  ✓ 99.1% posts have author (30,817 / 31,097)
  ✓ All posts have created_utc
  ✓ All posts have collected_at

Sentiment Analysis Coverage:
  ✓ 31,097 posts with sentiment scores (100.0%)
  ✓ Average compound score: +0.23 (slightly positive)

Data Quality Issues:
  ⚠ 468 posts missing selftext (link posts, expected)
  ⚠ 280 posts with deleted/removed author (expected)
  ✓ No duplicate post_ids found
  ✓ No posts with invalid timestamps

Validation:
  ✓ All post_ids are alphanumeric (Reddit format)
  ✓ All created_utc timestamps in valid range (2025-10-19 to 2025-11-07)
  ✓ All sentiment_compound values in [-1, 1]
  ✓ All sentiment_label values in {positive, negative, neutral}

============================================================
DATA QUALITY: EXCELLENT ✓
============================================================
```

**Implementation:**

```python
def preview_data():
    """Preview and validate database contents"""
    conn = sqlite3.connect('database/tech_sentiment.db')
    cursor = conn.cursor()

    # Check data completeness
    cursor.execute("SELECT COUNT(*) FROM raw_posts WHERE title IS NULL")
    missing_titles = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM raw_posts WHERE selftext IS NULL OR selftext = ''")
    missing_selftext = cursor.fetchone()[0]

    # Validate sentiment scores
    cursor.execute("""
        SELECT COUNT(*) FROM raw_posts
        WHERE sentiment_compound < -1 OR sentiment_compound > 1
    """)
    invalid_sentiment = cursor.fetchone()[0]

    # Check for duplicates
    cursor.execute("""
        SELECT post_id, COUNT(*) as count
        FROM raw_posts
        GROUP BY post_id
        HAVING count > 1
    """)
    duplicates = cursor.fetchall()

    # Display sample posts
    cursor.execute("SELECT * FROM raw_posts LIMIT 5")
    for post in cursor.fetchall():
        print(f"Post ID: {post[0]}")
        print(f"Title: {post[2]}")
        print(f"Sentiment: {post[14]} ({post[17]})")
        print("-" * 60)
```

**Code Location:** `database/preview_data.py`

---

## Migration to Supabase

### Migration Process (Week 4)

The SQLite → Supabase migration was executed using `supabase_db/migrate.py`, which performed the following transformations:

#### Data Format Conversions

**1. Timestamp Format:**

```python
# SQLite format (Unix timestamp)
created_utc = 1762049662.8408403  # Float

# Supabase format (ISO 8601 with timezone)
created_utc = "2025-11-02T08:34:22.840840+00:00"  # String

# Conversion logic
from datetime import datetime
supabase_timestamp = datetime.fromtimestamp(sqlite_timestamp).isoformat()
```

**2. Sentiment Label Validation:**

```python
# SQLite: No constraint (any string allowed)
sentiment_label = "very positive"  # Would be accepted (wrong!)

# Supabase: CHECK constraint
sentiment_label TEXT CHECK (sentiment_label IN ('positive', 'negative', 'neutral'))

# Migration validation
valid_labels = {'positive', 'negative', 'neutral'}
assert post['sentiment_label'] in valid_labels, f"Invalid label: {post['sentiment_label']}"
```

**3. Embedding Addition:**

```sql
-- SQLite: No embedding column

-- Supabase: Added 384-dimensional vector column
ALTER TABLE reddit_posts ADD COLUMN embedding vector(384);

-- Migration: Embeddings generated post-migration in batches
```

#### Migration Statistics

**Source Database (SQLite):**
- File: `database/tech_sentiment.db`
- Size: 67 MB
- Posts: 31,097
- Date range: October 19 - November 7, 2025

**Destination Database (Supabase):**
- Platform: Supabase (PostgreSQL 15.x)
- Initial size: ~68 MB (before embeddings)
- Posts: 31,097 (100% migrated)
- Validation: All row counts matched, sample data verified

**Migration Duration:** ~4 minutes 32 seconds

**Data Integrity:**
- ✅ All 31,097 posts successfully migrated
- ✅ All sentiment scores preserved
- ✅ All metadata intact (title, author, timestamps, etc.)
- ✅ Zero data loss

### Post-Migration Status

**SQLite Database:**
- **Status:** Archived (read-only)
- **Location:** `database/tech_sentiment.db` (gitignored)
- **Size:** Frozen at ~67 MB
- **Use Case:** Local testing, historical reference

**Supabase Database:**
- **Status:** Active production database
- **Growth:** Continuing at ~7,200 posts/day
- **Size:** 152 MB (as of November 15, 2025)
- **Features:** Vector embeddings, optimized indexes, cloud hosting

---

## Local Development Usage

### When to Use SQLite Database

The legacy SQLite database remains useful for:

1. **Offline Development:**
   ```bash
   # No internet connection? Use SQLite for testing
   python analyzer/process_posts.py  # Works with SQLite
   ```

2. **Local Testing:**
   ```python
   # Test database queries without cloud API calls
   import sqlite3
   conn = sqlite3.connect('database/tech_sentiment.db')
   cursor = conn.execute("SELECT * FROM raw_posts LIMIT 10")
   ```

3. **Historical Analysis:**
   ```bash
   # Analyze Week 1-3 data snapshot
   python database/check_db.py
   python database/preview_data.py
   ```

4. **Schema Experimentation:**
   ```sql
   -- Test schema changes without affecting production
   ALTER TABLE raw_posts ADD COLUMN test_field TEXT;
   ```

### Recreating SQLite Database

If you need to recreate the SQLite database from Supabase:

```python
# supabase_db/export_to_sqlite.py (hypothetical script)

from supabase_db.db_client import get_client
import sqlite3

def export_supabase_to_sqlite():
    """Export Supabase data to SQLite"""
    # Fetch all posts from Supabase
    supabase = get_client()
    posts = supabase.table('reddit_posts').select('*').execute()

    # Create SQLite database
    conn = sqlite3.connect('database/tech_sentiment_export.db')
    cursor = conn.cursor()

    # Create schema (without embedding column)
    cursor.execute("""
        CREATE TABLE raw_posts (
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
            collected_at REAL NOT NULL,
            sentiment_pos REAL,
            sentiment_neg REAL,
            sentiment_neu REAL,
            sentiment_compound REAL,
            sentiment_label TEXT
        )
    """)

    # Convert and insert data
    for post in posts.data:
        # Convert ISO 8601 back to Unix timestamp
        from datetime import datetime
        created_utc = datetime.fromisoformat(post['created_utc']).timestamp()
        collected_at = datetime.fromisoformat(post['collected_at']).timestamp()

        cursor.execute("""
            INSERT INTO raw_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post['post_id'],
            post['subreddit'],
            post['title'],
            post['selftext'],
            post['author'],
            created_utc,  # Converted to Unix timestamp
            post['score'],
            post['num_comments'],
            post['url'],
            post['permalink'],
            collected_at,  # Converted to Unix timestamp
            post['sentiment_pos'],
            post['sentiment_neg'],
            post['sentiment_neu'],
            post['sentiment_compound'],
            post['sentiment_label']
        ))

    conn.commit()
    conn.close()
```

---

## Performance Characteristics

### SQLite Performance Benchmarks

**Environment:** Windows 11, i5 CPU, 8GB RAM, SSD

| Operation | SQLite Time | Supabase Time | Comparison |
|-----------|-------------|---------------|------------|
| **Insert 100 posts** | 45ms | 487ms | SQLite 10× faster (local) |
| **Insert 1,000 posts** | 320ms | 4.2s | SQLite 13× faster (local) |
| **Select by post_id** | 1ms | 15ms | SQLite faster (index lookup) |
| **Select by subreddit (100 posts)** | 12ms | 28ms | SQLite faster (local) |
| **Aggregate query (count by subreddit)** | 34ms | 89ms | SQLite 2.6× faster |
| **Full table scan (31K posts)** | 145ms | 312ms | SQLite 2× faster |
| **Vector search (not applicable)** | N/A | 200ms | Supabase only feature |

**Analysis:**

**SQLite Advantages:**
- No network latency (local file access)
- Faster for small datasets (<100K rows)
- Lower overhead for simple queries
- Instant writes (no API calls)

**SQLite Disadvantages:**
- No vector search capability
- Poor scalability beyond 1 million rows
- No concurrent writes
- Manual backup/replication required

**Conclusion:** SQLite optimal for development/testing, Supabase optimal for production with vector search.

---

## Troubleshooting

### Common Issues

#### Issue 1: "Database file not found"

**Symptom:**
```python
sqlite3.OperationalError: unable to open database file
```

**Cause:** Database file (`tech_sentiment.db`) not present (gitignored)

**Solution:**

```bash
# Option 1: Download from migration snapshot
# (if available in project documentation)

# Option 2: Export from Supabase (see "Recreating SQLite Database" section)

# Option 3: Start fresh collection (not recommended - historical data lost)
python collector/continuous_collector.py
```

#### Issue 2: "Database is locked"

**Symptom:**
```python
sqlite3.OperationalError: database is locked
```

**Cause:** Another process has the database open with an exclusive lock

**Solution:**

```bash
# 1. Close all programs accessing the database
# 2. If stuck, identify and kill process:
lsof database/tech_sentiment.db  # macOS/Linux
# Or manually check task manager (Windows)

# 3. If corruption suspected:
sqlite3 database/tech_sentiment.db "PRAGMA integrity_check;"
```

#### Issue 3: "Sentiment columns missing"

**Symptom:**
```sql
SELECT sentiment_compound FROM raw_posts;
-- Error: no such column: sentiment_compound
```

**Cause:** Database from Week 1-2 (before sentiment columns added)

**Solution:**

```sql
-- Add sentiment columns manually
ALTER TABLE raw_posts ADD COLUMN sentiment_pos REAL;
ALTER TABLE raw_posts ADD COLUMN sentiment_neg REAL;
ALTER TABLE raw_posts ADD COLUMN sentiment_neu REAL;
ALTER TABLE raw_posts ADD COLUMN sentiment_compound REAL;
ALTER TABLE raw_posts ADD COLUMN sentiment_label TEXT;

-- Backfill sentiment scores
-- (requires running analyzer/process_posts.py)
```

---

## Historical Context

### Project Evolution Timeline

**Week 1 (October 19-26, 2025):**
- SQLite database created
- Initial schema designed
- 7,000 posts collected
- Database size: ~15 MB

**Week 2 (October 27 - November 1):**
- Indexes added for performance
- 21,000 posts collected (cumulative)
- Database size: ~45 MB

**Week 3 (November 2-7):**
- Sentiment analysis implemented
- VADER scores backfilled
- 31,097 posts collected (cumulative)
- Database size: ~67 MB

**Week 4 (November 8-14):**
- Scalability concerns identified
- Migration to Supabase planned and executed
- SQLite database archived
- Vector embeddings added (Supabase only)

### Lessons Learned

**1. File Size Growth Underestimated:**

Initial projection (Week 1):
```
~1,000 posts/day × 2 KB/post = 2 MB/day
2 MB/day × 365 days = 730 MB/year
Conclusion: "SQLite will be fine for the semester"
```

Actual growth (Week 3):
```
~7,200 posts/day × 2.2 KB/post = 15.8 MB/day
15.8 MB/day × 7 days/week = 111 MB/week
Conclusion: "Would exceed 100MB GitHub limit in 1 week!"
```

**Lesson:** Always plan for 10× actual data growth, not projected growth.

**2. Vector Search Requirement Not Anticipated:**

Week 1-3 plan:
- Simple keyword search would be sufficient
- No embedding models planned

Week 4 reality:
- RAG system requires semantic similarity search
- Embeddings are essential for quality results
- SQLite has no native vector search support

**Lesson:** Research full project scope before selecting database.

**3. Migration Earlier Is Better:**

Actual migration: Week 4 (31,097 posts)
- Migration time: ~5 minutes
- Validation time: ~10 minutes
- Risk: Low

Hypothetical late migration: Week 10 (150,000 posts)
- Estimated migration time: ~30 minutes
- Estimated validation time: ~1 hour
- Risk: High (more failure points)

**Lesson:** Migrate before database becomes mission-critical.

---

## Comparison: SQLite vs. Supabase

### Feature Matrix

| Feature | SQLite | Supabase (PostgreSQL) |
|---------|--------|----------------------|
| **Storage Type** | File-based | Cloud-hosted |
| **Setup Complexity** | Zero config | Cloud account required |
| **Cost** | Free | Free tier (500MB) |
| **Vector Search** | ❌ No (extension needed) | ✅ Yes (pgvector native) |
| **Concurrent Writes** | ❌ No | ✅ Yes |
| **Connection Pooling** | ❌ No | ✅ Yes |
| **Max Database Size** | ~1 TB (practical: <10 GB) | 500 MB (free), 8 GB (pro) |
| **Backup/Replication** | Manual file copy | Automated |
| **Query Performance (small dataset)** | ⚡ Faster (local) | Slower (network latency) |
| **Query Performance (large dataset)** | Slow (>1M rows) | Fast (optimized indexes) |
| **Scalability** | Limited | High |
| **Deployment** | File management | API-based |

### Use Case Recommendations

**Use SQLite When:**
- ✅ Prototyping and local development
- ✅ Offline capability required
- ✅ Dataset <100K rows
- ✅ Single-user application
- ✅ No vector search needed

**Use Supabase When:**
- ✅ Production deployment
- ✅ Vector similarity search required
- ✅ Dataset >100K rows or growing rapidly
- ✅ Multi-user concurrent access
- ✅ Cloud-native architecture

**Hybrid Approach (Current Implementation):**
- SQLite for local development/testing
- Supabase for production and vector search
- Migration scripts to sync between environments

---

## Maintenance

### Database File Management

**Current State:**
- File location: `database/tech_sentiment.db`
- Git status: Ignored (`.gitignore` entry)
- Size: ~67 MB (frozen at Week 4 snapshot)

**Backup Strategy:**

```bash
# Create compressed backup
tar -czf tech_sentiment_backup_$(date +%Y%m%d).tar.gz database/tech_sentiment.db

# Restore from backup
tar -xzf tech_sentiment_backup_20251107.tar.gz
```

**Cleanup:**

```bash
# Vacuum database to reclaim space
sqlite3 database/tech_sentiment.db "VACUUM;"

# Analyze query performance
sqlite3 database/tech_sentiment.db "ANALYZE;"
```

---

## Future Considerations

### Potential Reactivation Scenarios

The SQLite database could be reactivated if:

1. **Offline-First Architecture:** Building a desktop application requiring local storage
2. **Edge Deployment:** Running on devices without reliable internet
3. **Cost Constraints:** Supabase free tier exhausted, need zero-cost fallback
4. **Data Sovereignty:** Regulations require local data storage

### Modernization Path

If SQLite were to be modernized for vector search:

**Option 1: sqlite-vss Extension**
```sql
-- Install sqlite-vss (vector similarity search extension)
-- https://github.com/asg017/sqlite-vss

.load vss0

-- Create virtual table for embeddings
CREATE VIRTUAL TABLE vss_posts USING vss0(
    embedding(384)
);

-- Insert embeddings
INSERT INTO vss_posts(rowid, embedding)
SELECT rowid, embedding FROM raw_posts;

-- Search similar vectors
SELECT * FROM vss_posts
WHERE vss_search(embedding, query_vector)
LIMIT 20;
```

**Option 2: DuckDB Migration**
- Modern SQLite alternative
- Better analytical query performance
- Native Parquet support
- Still file-based, no server required

---

## References

### SQLite Documentation
- **Official Docs:** https://www.sqlite.org/docs.html
- **Python sqlite3:** https://docs.python.org/3/library/sqlite3.html

### Related Modules
- `supabase_db/` - Current production database (replacement for this module)
- `collector/` - Originally populated this database (Weeks 1-3)
- `analyzer/` - Sentiment analysis scripts compatible with SQLite

### Migration Resources
- Migration script: `supabase_db/migrate.py`
- Supabase documentation: `supabase_db/README.md`

---

**Last Updated:** November 15, 2025
**Module Status:** 🗄️ Archived (Week 4+)
**Original Use:** Weeks 1-3 (October 19 - November 7, 2025)
**Replacement:** `supabase_db/` (PostgreSQL + pgvector)
**Database File:** `tech_sentiment.db` (67 MB, 31,097 posts)
**Maintainer:** Sumayer Khan Sajid (ID: 2221818642)
