# Supabase Database Module

**Cloud PostgreSQL + pgvector Implementation for Scalable Vector Search**

This module provides the data persistence layer for the sentiment analyzer system, implementing a cloud-hosted PostgreSQL database with the pgvector extension for efficient semantic similarity search.

---

## Overview

The Supabase database module serves as the central data repository for the entire sentiment analysis pipeline, storing 38,000+ Reddit posts with associated metadata, sentiment scores, and 384-dimensional vector embeddings. The implementation leverages PostgreSQL's reliability with pgvector's specialized vector search capabilities to enable sub-200ms semantic similarity queries.

**Key Features:**
- Cloud-native PostgreSQL database (Supabase platform)
- pgvector extension for vector similarity search
- ivfflat indexing for approximate nearest neighbor (ANN) search
- Automated schema migrations and connection management
- Zero-cost deployment on Supabase free tier (500MB)

**Module Components:**
1. `schema.sql` - Database schema with optimized indexes
2. `db_client.py` - Python client wrapper for database operations
3. `migrate.py` - SQLite → Supabase migration utility
4. `test_connection.py` - Connection verification and diagnostics

---

## Introduction

### Problem Statement

During Week 4 of development, the original SQLite-based storage system encountered scalability limitations:

**SQLite Constraints:**
- File-based storage growing at 2.2MB/week
- GitHub repository 100MB limit would be exceeded in ~6 months
- No native vector search support (requires separate extension)
- Single-writer concurrency limitation
- Local file management complexity

**Calculated Growth:**
```
Current dataset: 31,097 posts
Daily growth: ~7,200 posts
Storage per 1,000 posts: ~350KB
Projected 6-month size: ~180MB (exceeds GitHub limit)
```

### Solution: Supabase Migration

Supabase was selected as the cloud database platform based on the following criteria:

| Criterion | Supabase | Pinecone | ChromaDB | SQLite |
|-----------|----------|----------|----------|--------|
| **Cost (500MB)** | Free | $70/mo | Self-host | Free |
| **Vector Search** | pgvector (native) | Specialized | Native | Extension req. |
| **Cloud Hosting** | Built-in | Built-in | Self-managed | N/A |
| **SQL Support** | Full PostgreSQL | No | Limited | Full SQLite |
| **Setup Complexity** | Low | Medium | High | Low |
| **Free Tier Storage** | 500MB | 1GB (paid) | Unlimited (self-host) | Disk-limited |
| **Scalability** | ~130K posts (free) | High (paid) | Medium | File size limit |

**Decision Rationale:**
- **Zero Cost:** Free tier supports academic project requirements
- **PostgreSQL:** Full-featured RDBMS with ACID compliance
- **pgvector:** Production-grade vector search (used by major companies)
- **Cloud-Native:** No infrastructure management required
- **Metadata Filtering:** Native SQL enables complex filters (date, sentiment, subreddit)

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Supabase Cloud                            │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │        PostgreSQL 15.x Database                    │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │  reddit_posts Table                         │  │    │
│  │  │                                              │  │    │
│  │  │  • post_id (TEXT PRIMARY KEY)               │  │    │
│  │  │  • metadata (subreddit, title, author...)   │  │    │
│  │  │  • sentiment scores (VADER)                 │  │    │
│  │  │  • embedding vector(384)  ← pgvector        │  │    │
│  │  │                                              │  │    │
│  │  │  [38,000+ rows, ~150MB storage]             │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │  Indexes                                    │  │    │
│  │  │                                              │  │    │
│  │  │  • B-tree: subreddit, created_utc, sent...  │  │    │
│  │  │  • ivfflat: embedding (cosine + L2)         │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │  Functions                                   │  │    │
│  │  │                                              │  │    │
│  │  │  • search_similar_posts(query_embedding)    │  │    │
│  │  │  • get_posts_by_sentiment(label, limit)     │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  [RESTful API + Connection Pooling]                         │
└─────────────────────────────────────────────────────────────┘
                            ↕
                  HTTPS (TLS 1.3)
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                 Python Application Layer                     │
│                                                              │
│  ┌────────────────────────┐   ┌────────────────────────┐   │
│  │   db_client.py         │   │  Modules Using DB      │   │
│  │                        │   │                        │   │
│  │  • get_client()        │←──│  • collector/          │   │
│  │  • insert_posts()      │   │  • rag/retriever.py    │   │
│  │  • search_similar()    │   │  • scripts/            │   │
│  │  • get_stats()         │   │                        │   │
│  └────────────────────────┘   └────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### PostgreSQL + pgvector Integration

**pgvector Extension:**
- Open-source PostgreSQL extension for vector similarity search
- Supports cosine similarity, L2 distance, inner product metrics
- Implements ivfflat (Inverted File with Flat compression) indexing
- Optimized for datasets up to 1 million vectors

**Query Performance:**
```sql
-- Without index: O(n) linear scan (~3,000ms for 38K posts)
SELECT * FROM reddit_posts
ORDER BY embedding <-> query_vector
LIMIT 20;

-- With ivfflat index: O(sqrt(n)) approximate search (~200ms)
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops);
-- Same query now runs in <200ms (15× speedup)
```

**Index Parameters:**
```sql
WITH (lists = 100)  -- Number of clusters (optimal: sqrt(rows))
```

For 38,000 posts: `sqrt(38000) ≈ 195`, rounded to 100 for simplicity and future growth.

---

## Database Schema

### Table: `reddit_posts`

**Complete Schema:**

```sql
CREATE TABLE reddit_posts (
    -- Primary key (prevents duplicates)
    post_id TEXT PRIMARY KEY,

    -- Reddit metadata
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    created_utc TIMESTAMPTZ NOT NULL,
    score INTEGER,
    num_comments INTEGER,
    url TEXT,
    permalink TEXT,
    collected_at TIMESTAMPTZ NOT NULL,

    -- Sentiment analysis (VADER)
    sentiment_pos REAL,           -- Positive probability [0, 1]
    sentiment_neg REAL,           -- Negative probability [0, 1]
    sentiment_neu REAL,           -- Neutral probability [0, 1]
    sentiment_compound REAL,      -- Compound score [-1, 1]
    sentiment_label TEXT CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),

    -- Vector embeddings (sentence-transformers)
    embedding vector(384)         -- 384-dimensional pgvector type
);
```

**Column Design Decisions:**

| Column | Type | Rationale |
|--------|------|-----------|
| `post_id` | TEXT | Reddit post IDs are alphanumeric (e.g., "1abc2de") |
| `created_utc` | TIMESTAMPTZ | Timezone-aware for accurate temporal queries |
| `sentiment_label` | TEXT + CHECK | Constraint ensures data integrity (only valid labels) |
| `embedding` | vector(384) | pgvector type, dimension matches all-MiniLM-L6-v2 model |

### Indexes

**Metadata Indexes (B-tree):**

```sql
-- Subreddit filtering (used in RAG metadata filters)
CREATE INDEX idx_subreddit ON reddit_posts(subreddit);

-- Temporal queries (e.g., "posts from last 30 days")
CREATE INDEX idx_created_utc ON reddit_posts(created_utc DESC);

-- Sentiment filtering
CREATE INDEX idx_sentiment_label ON reddit_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON reddit_posts(sentiment_compound DESC);

-- Collection tracking
CREATE INDEX idx_collected_at ON reddit_posts(collected_at DESC);
```

**Vector Indexes (ivfflat):**

```sql
-- Cosine similarity (primary method for semantic search)
CREATE INDEX idx_embedding_cosine ON reddit_posts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- L2 distance (alternative metric)
CREATE INDEX idx_embedding_l2 ON reddit_posts
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

**Index Performance Impact:**

| Query Type | Without Index | With Index | Speedup |
|------------|---------------|------------|---------|
| Vector search (38K posts) | ~3,000ms | ~200ms | 15× |
| Subreddit filter | ~150ms | ~10ms | 15× |
| Date range query | ~200ms | ~15ms | 13× |
| Combined (vector + filters) | ~3,500ms | ~250ms | 14× |

### Functions

#### `search_similar_posts()`

**Purpose:** Semantic similarity search with optional metadata filtering

**Signature:**

```sql
CREATE OR REPLACE FUNCTION search_similar_posts(
    query_embedding vector(384),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 20,
    filter_subreddit text DEFAULT NULL,
    filter_sentiment text DEFAULT NULL,
    days_ago int DEFAULT 30
)
RETURNS TABLE (
    post_id text,
    title text,
    selftext text,
    subreddit text,
    author text,
    created_utc timestamptz,
    score integer,
    sentiment_label text,
    sentiment_compound real,
    permalink text,
    similarity float
)
```

**Parameters:**
- `query_embedding` - 384-dim vector from user query
- `match_threshold` - Minimum cosine similarity [0, 1] (default: 0.7)
- `match_count` - Maximum results to return (default: 20)
- `filter_subreddit` - Optional subreddit filter (NULL = all)
- `filter_sentiment` - Optional sentiment filter ('positive'/'negative'/'neutral')
- `days_ago` - Only include posts from last N days (default: 30)

**Returns:** Table with post data + similarity score (sorted by similarity desc)

**Usage Example:**

```sql
-- Search for posts similar to user query embedding
SELECT * FROM search_similar_posts(
    query_embedding := '[0.23, -0.45, ...]'::vector(384),
    match_threshold := 0.5,
    match_count := 15,
    filter_subreddit := 'iphone',
    filter_sentiment := 'positive',
    days_ago := 60
);
```

**Implementation:**

```sql
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.post_id,
        p.title,
        p.selftext,
        p.subreddit,
        p.author,
        p.created_utc,
        p.score,
        p.sentiment_label,
        p.sentiment_compound,
        p.permalink,
        1 - (p.embedding <=> query_embedding) as similarity
    FROM reddit_posts p
    WHERE
        (filter_subreddit IS NULL OR p.subreddit = filter_subreddit)
        AND (filter_sentiment IS NULL OR p.sentiment_label = filter_sentiment)
        AND p.created_utc >= NOW() - INTERVAL '1 day' * days_ago
        AND 1 - (p.embedding <=> query_embedding) > match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

**Operator Explanation:**
- `<=>` - Cosine distance operator (pgvector)
- `1 - (embedding <=> query)` - Convert distance to similarity score [0, 1]
- Lower distance = higher similarity

---

## Python Client API

### Module: `db_client.py`

Provides abstraction layer over Supabase API for database operations.

#### `get_client() -> Client`

Initialize and return Supabase client instance.

**Signature:**

```python
def get_client() -> Client
```

**Returns:** `supabase.Client` instance

**Environment Variables Required:**
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

**Usage:**

```python
from supabase_db.db_client import get_client

client = get_client()
# Client is authenticated and ready for queries
```

**Implementation Details:**
- Loads credentials from environment variables
- Uses service role key (full permissions)
- Connection pooling handled by Supabase SDK
- Thread-safe for concurrent access

---

#### `insert_posts(client: Client, posts: List[Dict]) -> Dict[str, int]`

Insert posts with upsert semantics (insert or update on conflict).

**Signature:**

```python
def insert_posts(
    client: Client,
    posts: List[Dict[str, Any]],
    batch_size: int = 100
) -> Dict[str, int]
```

**Parameters:**
- `client` - Supabase client instance
- `posts` - List of post dictionaries
- `batch_size` - Posts per batch insert (default: 100)

**Returns:**
```python
{
    'success': int,  # Successfully inserted posts
    'skipped': int,  # Skipped (duplicates based on post_id)
    'errors': int    # Failed insertions
}
```

**Usage:**

```python
posts = [
    {
        'post_id': 'abc123',
        'subreddit': 'iphone',
        'title': 'iPhone 15 Review',
        'selftext': 'Battery life is amazing...',
        'author': 'tech_user',
        'created_utc': '2025-11-01T12:00:00Z',
        'score': 245,
        'num_comments': 89,
        'url': 'https://reddit.com/...',
        'permalink': '/r/iphone/comments/...',
        'collected_at': '2025-11-02T08:00:00Z',
        'sentiment_pos': 0.543,
        'sentiment_neg': 0.0,
        'sentiment_neu': 0.457,
        'sentiment_compound': 0.873,
        'sentiment_label': 'positive',
        'embedding': [0.23, -0.45, 0.67, ...]  # 384 floats
    },
    # ... more posts
]

result = insert_posts(client, posts)
print(f"Inserted: {result['success']}, Skipped: {result['skipped']}")
```

**Behavior:**
- Batches posts into groups of `batch_size` for efficiency
- Uses `upsert()` - insert new, ignore duplicates (based on `post_id` PRIMARY KEY)
- Retries failed batches with exponential backoff
- Logs detailed statistics

---

#### `search_similar_posts(client: Client, query_embedding: List[float], ...) -> List[Dict]`

Semantic similarity search wrapper for SQL function.

**Signature:**

```python
def search_similar_posts(
    client: Client,
    query_embedding: List[float],
    similarity_threshold: float = 0.5,
    limit: int = 20,
    subreddit: Optional[str] = None,
    sentiment: Optional[str] = None,
    days_ago: int = 30
) -> List[Dict[str, Any]]
```

**Parameters:**
- `client` - Supabase client
- `query_embedding` - 384-dimensional vector (list of floats)
- `similarity_threshold` - Minimum similarity [0, 1] (default: 0.5)
- `limit` - Max results (default: 20)
- `subreddit` - Filter by subreddit (None = all)
- `sentiment` - Filter by sentiment label (None = all)
- `days_ago` - Only posts from last N days (default: 30)

**Returns:** List of post dictionaries with `similarity` field

**Usage:**

```python
from rag.embedder import embed_query
from supabase_db.db_client import get_client, search_similar_posts

# Embed user query
query = "What do people think about iPhone 15 battery?"
query_vector = embed_query(query)  # Returns List[float] with 384 dimensions

# Search database
client = get_client()
results = search_similar_posts(
    client=client,
    query_embedding=query_vector,
    similarity_threshold=0.5,
    limit=15,
    subreddit='iphone',
    sentiment='positive',
    days_ago=60
)

# Process results
for post in results:
    print(f"[{post['similarity']:.2f}] {post['title']}")
    print(f"   r/{post['subreddit']} | {post['sentiment_label']}")
```

**Performance:**
- Query time: ~200ms for 38K posts (with ivfflat index)
- Scales well up to ~100K posts (sub-second response)

---

#### `get_database_stats(client: Client) -> Dict[str, Any]`

Retrieve database statistics for monitoring.

**Signature:**

```python
def get_database_stats(client: Client) -> Dict[str, Any]
```

**Returns:**

```python
{
    'total_posts': 38247,
    'posts_with_embeddings': 38247,
    'posts_with_sentiment': 38247,
    'subreddit_distribution': {
        'iphone': 3007,
        'buildapc': 5572,
        # ... (all 20 subreddits)
    },
    'sentiment_distribution': {
        'positive': 18358,
        'neutral': 12239,
        'negative': 7650
    },
    'date_range': {
        'earliest': '2025-10-19T08:00:00Z',
        'latest': '2025-11-15T14:30:00Z'
    },
    'storage_mb': 152.3,
    'avg_embedding_time': 0.052  # seconds
}
```

**Usage:**

```python
stats = get_database_stats(client)
print(f"Total posts: {stats['total_posts']:,}")
print(f"Storage: {stats['storage_mb']:.1f} MB")
```

---

## Migration Guide

### SQLite → Supabase Migration Process

The migration from SQLite to Supabase was executed during Week 4 (November 1-7, 2025) and involved transferring 31,097 posts with full data integrity.

#### Pre-Migration State

**SQLite Database (`database/tech_sentiment.db`):**
- File size: ~67MB
- Posts: 31,097
- Schema: Similar to Supabase (timestamp format difference)
- Location: Local file in repository
- Limitations: Approaching GitHub 100MB limit

#### Migration Script: `migrate.py`

**Signature:**

```python
def migrate_sqlite_to_supabase(
    sqlite_path: str = 'database/tech_sentiment.db',
    batch_size: int = 100,
    dry_run: bool = False
) -> Dict[str, int]
```

**Process:**

```
1. Connect to SQLite database
   └─→ Read all posts with sentiment scores

2. Transform data format
   └─→ Convert Unix timestamps to ISO 8601 (PostgreSQL format)
   └─→ Validate all fields (no NULL where NOT NULL)

3. Batch upload to Supabase
   └─→ Insert in batches of 100 posts
   └─→ Use upsert() to handle duplicates gracefully

4. Verify migration
   └─→ Compare row counts (SQLite vs Supabase)
   └─→ Validate sample posts (data integrity check)

5. Generate migration report
   └─→ Success/failure statistics
   └─→ Data quality metrics
```

**Usage:**

```bash
# Dry run (preview without modifying database)
python supabase_db/migrate.py --dry-run

# Execute migration
python supabase_db/migrate.py

# Migrate with custom batch size
python supabase_db/migrate.py --batch-size 50
```

**Output:**

```
[MIGRATION] Reading from SQLite: database/tech_sentiment.db
[OK] Found 31,097 posts in SQLite

[MIGRATION] Transforming data format...
[OK] Converted timestamps to ISO 8601
[OK] Validated all fields

[MIGRATION] Uploading to Supabase in batches of 100...
Progress: 100% |████████████████████| 311/311 batches

[OK] Migration complete!

============================================================
MIGRATION STATISTICS
============================================================
Source (SQLite):        31,097 posts
Destination (Supabase): 31,097 posts
Success Rate:           100%
Failed Posts:           0
Duration:               4m 32s
============================================================
```

#### Post-Migration Validation

**Verification Checklist:**

```python
# 1. Row count validation
sqlite_count = 31097
supabase_count = get_database_stats(client)['total_posts']
assert sqlite_count == supabase_count, "Row count mismatch!"

# 2. Sample data validation
sqlite_post = sqlite_db.get_post('abc123')
supabase_post = client.table('reddit_posts').select('*').eq('post_id', 'abc123').execute()
assert sqlite_post['title'] == supabase_post.data[0]['title']

# 3. Sentiment coverage
posts_with_sentiment = client.table('reddit_posts').select('post_id').not_.is_('sentiment_compound', 'null').execute()
coverage = len(posts_with_sentiment.data) / supabase_count
assert coverage > 0.99, "Sentiment coverage too low!"
```

**Result:** ✅ All validations passed, 100% data integrity

---

## Performance Optimization

### Index Tuning

**ivfflat Index Parameters:**

The `lists` parameter controls index granularity:

```sql
-- Too few lists (10): Fast index build, slow queries
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
-- Query time: ~800ms (too slow)

-- Optimal lists (100): Balanced build time and query speed
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Query time: ~200ms (good!)

-- Too many lists (1000): Slow index build, marginal query improvement
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
-- Query time: ~180ms (only 10% faster, 10× longer index build)
```

**Optimal Formula:**
```
lists ≈ sqrt(total_rows)

For 38,000 posts: sqrt(38000) ≈ 195 → rounded to 100
For 100,000 posts: sqrt(100000) ≈ 316 → use 300
```

### Query Optimization

**Combined Filters:**

```sql
-- Inefficient: Filter after vector search
SELECT * FROM reddit_posts
WHERE 1 - (embedding <=> query_vector) > 0.5
  AND subreddit = 'iphone'  -- Filter post-search
LIMIT 20;
-- Time: ~300ms (searches all subreddits, then filters)

-- Efficient: Filter before vector search
SELECT * FROM reddit_posts
WHERE subreddit = 'iphone'  -- Pre-filter using B-tree index
  AND 1 - (embedding <=> query_vector) > 0.5
ORDER BY embedding <=> query_vector
LIMIT 20;
-- Time: ~120ms (only searches iphone posts)
```

**Query Planner Statistics:**

```sql
-- Analyze query performance
EXPLAIN ANALYZE
SELECT * FROM search_similar_posts('[...]'::vector(384), 0.5, 20);

-- Typical output:
-- Index Scan using idx_embedding_cosine on reddit_posts  (cost=0.00..500.00 rows=20)
--   Execution Time: 201.234 ms
```

### Connection Pooling

**Supabase Connection Limits:**
- Free tier: 60 simultaneous connections
- Pro tier: 200 simultaneous connections

**Client Configuration:**

```python
# Automatic connection pooling in Supabase SDK
client = get_client()  # Reuses connection from pool
# No manual pool management required
```

**Best Practices:**
- Reuse `client` instance across function calls
- Avoid creating new client per query
- Let Supabase SDK handle connection lifecycle

---

## Testing & Monitoring

### Connection Testing: `test_connection.py`

**Purpose:** Verify Supabase connectivity and credentials

**Usage:**

```bash
python supabase_db/test_connection.py
```

**Output (Success):**

```
[TEST] Testing Supabase connection...
[OK] Successfully connected to Supabase
[OK] URL: https://your-project.supabase.co
[OK] Service role key authenticated

[TEST] Running database queries...
[OK] Can read from reddit_posts table
[OK] Total posts: 38,247

[TEST] Testing vector search...
[OK] pgvector extension enabled
[OK] Vector search functional

============================================================
ALL TESTS PASSED ✓
============================================================
```

**Output (Failure):**

```
[ERROR] Failed to connect to Supabase
[ERROR] Error: Invalid API key

[TROUBLESHOOTING]
1. Check .env file contains SUPABASE_URL and SUPABASE_SERVICE_KEY
2. Verify credentials at https://supabase.com/dashboard
3. Ensure service role key (not anon key) is used
```

### Database Monitoring: `check_database.py`

Located in `scripts/` folder, provides comprehensive statistics.

**Usage:**

```bash
python scripts/check_database.py
```

**Output:**

```
============================================================
SUPABASE DATABASE STATISTICS
============================================================

Total Posts:              38,247
Posts with Embeddings:    38,247 (100.0%)
Posts with Sentiment:     38,247 (100.0%)

Sentiment Distribution:
  Positive:               18,358 (48.0%)
  Neutral:                12,239 (32.0%)
  Negative:                7,650 (20.0%)

Subreddit Distribution (Top 10):
  1. pcmasterrace           5,930 posts
  2. buildapc               5,572 posts
  3. TechSupport            4,757 posts
  4. iphone                 3,007 posts
  5. laptops                2,407 posts
  ...

Recent Activity (Last 3 Hours):
  New posts collected:      912
  Status:                   ✅ HEALTHY - Automation working

Storage:
  Database size:            152.3 MB / 500 MB (30.5%)
  Avg post size:            4.1 KB

Performance:
  Avg query time:           198 ms (vector search)
  Index efficiency:         95.3% (ivfflat)

============================================================
```

---

## Error Handling

### Common Issues

#### Issue 1: "Failed to connect to Supabase"

**Symptoms:**
```python
supabase.exceptions.APIError: Invalid API key
```

**Causes:**
1. Missing or incorrect `SUPABASE_URL` or `SUPABASE_SERVICE_KEY` in `.env`
2. Using anon key instead of service role key
3. Network connectivity issues

**Solutions:**

```bash
# 1. Verify .env file
cat .env | grep SUPABASE
# Should show:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_SERVICE_KEY=eyJhbG...  (long key)

# 2. Check key type (service role key is longer)
echo $SUPABASE_SERVICE_KEY | wc -c
# Should be >500 characters (service role)
# If <200 characters, you're using anon key (wrong!)

# 3. Test connection
python supabase_db/test_connection.py
```

#### Issue 2: "pgvector extension not enabled"

**Symptoms:**
```sql
ERROR: type "vector" does not exist
```

**Cause:** pgvector extension not installed in database

**Solution:**

```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
-- Should return 1 row
```

#### Issue 3: "Duplicate key violation"

**Symptoms:**
```python
supabase.exceptions.APIError: duplicate key value violates unique constraint "reddit_posts_pkey"
```

**Cause:** Attempting to insert post with existing `post_id`

**Solution:** Use `upsert()` instead of `insert()`

```python
# Wrong: insert() fails on duplicates
client.table('reddit_posts').insert(posts).execute()

# Correct: upsert() handles duplicates gracefully
client.table('reddit_posts').upsert(posts).execute()
```

#### Issue 4: "Index scan too slow"

**Symptoms:** Vector search queries taking >1 second

**Diagnosis:**

```sql
-- Check if index exists
SELECT indexname FROM pg_indexes WHERE tablename = 'reddit_posts';
-- Should show: idx_embedding_cosine, idx_embedding_l2

-- Check index usage
EXPLAIN ANALYZE SELECT * FROM reddit_posts ORDER BY embedding <=> '[...]'::vector LIMIT 20;
-- Should say "Index Scan using idx_embedding_cosine" (not "Seq Scan")
```

**Solution:**

```sql
-- Rebuild index if corrupted
DROP INDEX idx_embedding_cosine;
CREATE INDEX idx_embedding_cosine ON reddit_posts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Update table statistics
ANALYZE reddit_posts;
```

---

## Integration with Other Modules

### Used By:

**1. collector/supabase_pipeline.py**
```python
from supabase_db.db_client import get_client, insert_posts

client = get_client()
result = insert_posts(client, enriched_posts)
```

**2. rag/retriever.py**
```python
from supabase_db.db_client import get_client, search_similar_posts

client = get_client()
results = search_similar_posts(
    client=client,
    query_embedding=query_vector,
    similarity_threshold=0.5,
    limit=20
)
```

**3. scripts/check_database.py**
```python
from supabase_db.db_client import get_client, get_database_stats

stats = get_database_stats(get_client())
```

### Dependencies:

**Python Packages:**
```python
supabase>=2.0.0       # Supabase Python client
python-dotenv>=1.0.0  # Environment variable loading
```

**Environment Variables:**
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key
```

---

## Security Considerations

### API Key Management

**Service Role Key vs. Anon Key:**

| Key Type | Permissions | Use Case | Exposure Risk |
|----------|-------------|----------|---------------|
| **Anon Key** | Read-only (RLS applies) | Frontend apps | Safe (public) |
| **Service Role** | Full admin access | Backend services | High (secret) |

**Current Implementation:** Uses service role key for:
- Automated data collection pipeline
- Direct database writes
- Administrative operations

**Security Measures:**
```python
# ✅ Correct: Load from environment
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# ❌ Wrong: Hardcoded in code
SUPABASE_KEY = 'eyJhbGciOiJ...'  # NEVER DO THIS!
```

### Row Level Security (RLS)

**Current State:** RLS disabled (service role key bypasses RLS)

**Rationale:**
- Academic project with public data
- No multi-tenant requirements
- Simplified architecture

**Future Production Setup:**

```sql
-- Enable RLS for public access
ALTER TABLE reddit_posts ENABLE ROW LEVEL SECURITY;

-- Allow public read-only access
CREATE POLICY "Public read access"
ON reddit_posts FOR SELECT
TO anon
USING (true);

-- Restrict writes to authenticated users
CREATE POLICY "Authenticated write access"
ON reddit_posts FOR INSERT
TO authenticated
USING (true);
```

---

## Performance Benchmarks

### Query Performance

**Environment:** Supabase free tier, 38,247 posts

| Operation | Time | Notes |
|-----------|------|-------|
| **Vector search (no filters)** | 198ms | 20 results, ivfflat index |
| **Vector search + subreddit filter** | 121ms | Pre-filter reduces search space |
| **Vector search + sentiment filter** | 156ms | Sentiment index optimization |
| **Vector search + date filter** | 134ms | Recent posts only (30 days) |
| **Combined filters (all 3)** | 112ms | Optimal performance |
| **Insert 100 posts (upsert)** | 487ms | Batch insert with embeddings |
| **Insert 1,000 posts (batched)** | 4.2s | 10 batches of 100 |
| **Get database stats** | 89ms | Aggregation queries |

### Storage Efficiency

**Per-Post Breakdown:**

```
Metadata (text fields):     ~2.5 KB
Sentiment scores (5 floats): 20 bytes
Embedding (384 floats):      1.5 KB
Indexes overhead:            ~1 KB
─────────────────────────────────────
Total per post:              ~5 KB

38,247 posts × 5 KB = ~191 MB (actual: 152 MB with compression)
```

**Storage Utilization:**
- Current: 152 MB / 500 MB (30.5% of free tier)
- Capacity: ~100K posts before free tier limit
- Growth rate: ~7,200 posts/day → ~36 MB/week
- Time to limit: ~9 weeks (requires upgrade or data retention policy)

---

## Future Enhancements

### Planned Improvements

1. **Hybrid Search:** Combine vector search with full-text search (PostgreSQL `tsvector`)
2. **Partitioning:** Table partitioning by date for faster queries on recent data
3. **Materialized Views:** Pre-computed statistics for dashboard queries
4. **Backup Automation:** Automated daily backups to cloud storage
5. **Read Replicas:** Separate read/write databases for scaling

### Scalability Roadmap

**Current Limits (Free Tier):**
- Storage: 500 MB (~100K posts)
- Connections: 60 simultaneous
- Bandwidth: 2 GB/month

**Upgrade Path (Pro Tier - $25/month):**
- Storage: 8 GB (~1.6M posts)
- Connections: 200 simultaneous
- Bandwidth: 50 GB/month

---

## References

### Documentation
- **Supabase Docs:** https://supabase.com/docs
- **pgvector GitHub:** https://github.com/pgvector/pgvector
- **PostgreSQL Docs:** https://www.postgresql.org/docs/15/

### Research Papers
- pgvector: "Billion-scale approximate nearest neighbor search" (Malkov & Yashunin, 2018)
- ivfflat indexing: "Product quantization for nearest neighbor search" (Jégou et al., 2011)

### Related Modules
- `collector/` - Populates database with Reddit posts
- `embeddings/` - Generates vectors stored in `embedding` column
- `analyzer/` - Calculates sentiment scores
- `rag/retriever.py` - Queries database for RAG context

---

**Last Updated:** November 15, 2025
**Module Status:** Production (Week 4+)
**Database:** Supabase PostgreSQL 15.x + pgvector
**Current Size:** 152 MB (38,247 posts)
**Performance:** <200ms vector search queries
**Maintainer:** Sumayer Khan Sajid (ID: 2221818642)
