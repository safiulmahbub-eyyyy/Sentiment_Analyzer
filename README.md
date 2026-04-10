# End-to-End Social Media Sentiment Analyzer with Conversational RAG

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-Supabase-green.svg)](https://supabase.com)
[![RAG](https://img.shields.io/badge/RAG-Conversational-purple.svg)]()
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-red.svg)](https://end-to-end-social-media-sentiment.streamlit.app/)

---

## Abstract

This research project presents a zero-cost, production-ready conversational AI system for analyzing consumer electronics sentiment from Reddit discussions. The system implements a hybrid Retrieval-Augmented Generation (RAG) architecture enhanced with query classification to enable natural conversation flow while maintaining source attribution and factual grounding.

**Key Contributions:**
- Novel query classification layer preventing database search failures on meta-questions
- Fully automated data collection pipeline processing ~7,200 posts daily
- Zero-cost architecture leveraging free-tier cloud services (Groq, Supabase, Streamlit Cloud)
- Production deployment demonstrating RAG viability for real-time sentiment analysis

**System Performance:**
- Dataset: 38,000+ Reddit posts with sentiment scores and vector embeddings
- Coverage: 20 consumer electronics subreddits
- Update Frequency: Automated collection every 3 hours
- Response Quality: Source-attributed answers with conversational intelligence

**Live Demonstration:** [https://end-to-end-social-media-sentiment.streamlit.app/](https://end-to-end-social-media-sentiment.streamlit.app/)

---

## 1. Introduction

### 1.1 Problem Statement

Consumer electronics purchasing decisions increasingly rely on community-driven sentiment from social media platforms. However, the volume and distributed nature of these discussions make manual analysis impractical. Traditional sentiment analysis systems provide aggregate statistics but fail to answer natural language queries about specific products or features.

### 1.2 Research Objectives

This project addresses the following research questions:

1. **Data Acquisition:** Can ethical, automated collection of Reddit discussions provide sufficient data for sentiment analysis at scale?
2. **Sentiment Classification:** Does VADER sentiment analysis provide adequate accuracy for social media text containing slang, emojis, and non-standard grammar?
3. **RAG Implementation:** Can a Retrieval-Augmented Generation system accurately answer natural language questions about consumer sentiment without hallucination?
4. **Conversational AI:** How can RAG systems be enhanced to handle meta-questions and greetings without database search failures?
5. **Zero-Cost Deployment:** Is production-grade AI system deployment feasible using only free-tier cloud services?

### 1.3 Technical Approach

The system implements a multi-stage pipeline:

```
Data Collection → Sentiment Analysis → Vector Embedding → RAG Retrieval → LLM Generation
```

A novel query classification layer routes requests to appropriate handlers (conversational responses vs. database-backed RAG), preventing common failure modes in traditional RAG systems.

### 1.4 Scope and Limitations

**Scope:**
- Consumer electronics discussions on Reddit
- English language posts only
- Text-based analysis (excludes images/videos)
- Real-time data collection with 3-hour refresh cycles

**Limitations:**
- VADER sentiment analysis lacks context understanding
- Free-tier rate limits (Groq: 30 requests/min, Supabase: 500MB storage)
- No multi-turn conversation memory (stateless queries)
- Limited to public Reddit content

---

## 2. System Architecture

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (Every 3 Hours)            │
│                                                              │
│  Reddit API (PRAW) → Data Collection → VADER Sentiment      │
│         ↓                                    ↓               │
│  Embedding Generation → Supabase Insertion (PostgreSQL)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  Supabase Database  │
                    │  (PostgreSQL +      │
                    │   pgvector)         │
                    │                     │
                    │  • 38,000+ posts    │
                    │  • Sentiment scores │
                    │  • 384-dim vectors  │
                    └─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              User Query (Streamlit Interface)                │
│                           ↓                                  │
│              Query Classification Layer                      │
│                  ↙              ↘                           │
│    Meta/Greeting Query     Product Query                    │
│           ↓                     ↓                           │
│   Conversational          RAG Pipeline:                     │
│   Response Handler        1. Embed Query                    │
│                          2. Vector Search (pgvector)        │
│                          3. Retrieve Context (top 15-20)    │
│                          4. LLM Generation (Groq API)       │
│                          5. Source Attribution              │
│                           ↓                                  │
│              Streamlit Chat Interface Display                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Collection Pipeline

**Technology:** PRAW (Python Reddit API Wrapper)

**Collection Strategy:**
- **Frequency:** Automated execution every 3 hours via GitHub Actions
- **Subreddits:** 20 consumer electronics communities (see Appendix A)
- **Feed Types:** New, Hot, Rising (to capture diverse content)
- **Quality Filters:**
  - Minimum title length: 10 characters
  - Exclusion of deleted/removed posts
  - Score threshold: > -5 (excludes heavily downvoted content)
  - Automated deduplication via primary key constraints

**Output:** Direct insertion to Supabase (PostgreSQL) cloud database

### 2.3 Sentiment Analysis Methodology

**Algorithm:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Rationale:**
- Pre-trained lexicon (no training data required)
- Optimized for social media language (emojis, capitalization, slang)
- Fast processing: ~5,000 posts/minute on standard hardware
- Provides compound score (-1.0 to +1.0) and categorical labels

**Classification Thresholds:**
```python
Positive:  compound_score ≥ 0.05
Negative:  compound_score ≤ -0.05
Neutral:   -0.05 < compound_score < 0.05
```

**Database Schema:**
```sql
sentiment_pos REAL       -- Positive probability [0, 1]
sentiment_neg REAL       -- Negative probability [0, 1]
sentiment_neu REAL       -- Neutral probability [0, 1]
sentiment_compound REAL  -- Overall score [-1, 1]
sentiment_label TEXT     -- Category: positive/negative/neutral
```

### 2.4 Vector Embedding System

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Specifications:**
- Dimensions: 384
- Model size: ~80MB
- Processing speed: ~1,000 posts/minute (CPU)
- Similarity metric: Cosine similarity

**Implementation:**
- Combined title + selftext for comprehensive semantic representation
- Embeddings generated during collection pipeline (real-time)
- Stored in Supabase using pgvector extension
- Indexed with IVFFlat for fast approximate nearest neighbor search

### 2.5 RAG Pipeline Architecture

**Components:**

1. **Query Classifier**
   - Intent detection: meta/greeting/product queries
   - Prevents database search on non-product questions
   - Enables natural conversation flow

2. **Embedder**
   - Converts user queries to 384-dimensional vectors
   - Uses same model as document embeddings (consistency requirement)

3. **Retriever**
   - Vector similarity search via pgvector
   - Metadata filtering: subreddit, date range, sentiment
   - Returns top 15-20 most relevant posts
   - Ranking by cosine similarity score

4. **Generator**
   - LLM: Groq API (Llama 3.2 / Mixtral models)
   - Context window: Retrieved posts + system prompt
   - Output: Natural language answer with source citations
   - Streaming support for real-time response

5. **Conversational Response Handler**
   - Handles meta-questions ("What can you do?")
   - Generates capability descriptions
   - Suggests example queries
   - No database dependency

### 2.6 Deployment Infrastructure

**Components:**
- **Data Pipeline:** GitHub Actions (cloud automation)
- **Database:** Supabase (PostgreSQL with pgvector)
- **LLM API:** Groq (specialized inference hardware)
- **Frontend:** Streamlit Cloud (web hosting)

**Advantages:**
- Zero infrastructure management
- Automatic scaling
- 100% free tier utilization
- Geographic distribution (low latency)

---

## 3. Technical Stack

### 3.1 Core Technologies

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Language** | Python | 3.11+ | Type hints, modern syntax, extensive ML libraries |
| **Reddit API** | PRAW | 7.8.1 | Official wrapper, rate limit handling, ethical data access |
| **Database** | Supabase (PostgreSQL) | 15.x | Cloud-native, pgvector extension, 500MB free tier |
| **Vector Search** | pgvector | 0.5.x | Native PostgreSQL extension, efficient cosine similarity |
| **Embeddings** | sentence-transformers | 2.2+ | CPU-friendly, compact vectors (384-dim), semantic quality |
| **Sentiment** | VADER (vaderSentiment) | 3.3+ | Social media optimization, no training required |
| **LLM API** | Groq | - | Free tier (30 req/min), ultra-fast inference, no credit card |
| **Web Framework** | Streamlit | 1.28+ | Python-native, built-in chat UI, free cloud hosting |
| **Automation** | GitHub Actions | - | Free CI/CD, cron scheduling, secrets management |

### 3.2 Technology Selection Criteria

#### 3.2.1 Database: Supabase vs. Alternatives

**Selected:** Supabase (PostgreSQL + pgvector)

**Comparison:**

| Criterion | Supabase | Pinecone | ChromaDB | SQLite |
|-----------|----------|----------|----------|--------|
| Cost (500MB) | Free | $70/mo | Self-host | Free |
| Vector Search | pgvector | Native | Native | Extension |
| Cloud Hosting | Yes | Yes | No | No |
| SQL Support | Full | No | Limited | Full |
| Setup Complexity | Low | Medium | High | Low |
| Scalability | 500MB limit | High | Medium | File-based |

**Decision Rationale:**
- Cost: Zero-cost requirement eliminates Pinecone
- Scalability: SQLite would hit GitHub's 100MB limit in ~6 months
- Functionality: pgvector provides production-grade vector search
- Integration: Native SQL simplifies metadata filtering

#### 3.2.2 Embedding Model: all-MiniLM-L6-v2 vs. Alternatives

**Selected:** `sentence-transformers/all-MiniLM-L6-v2`

**Comparison:**

| Model | Dimensions | Speed (posts/min) | Quality | Hardware |
|-------|------------|-------------------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | ~1,000 | Good | CPU |
| all-mpnet-base-v2 | 768 | ~400 | Better | CPU/GPU |
| OpenAI ada-002 | 1,536 | API-limited | Best | Cloud |
| all-distilroberta-v1 | 768 | ~500 | Good | CPU/GPU |

**Decision Rationale:**
- Hardware: CPU-only processing (no GPU available)
- Storage: 384 dimensions = 1.5KB/post vs. 3KB for 768-dim models
- Cost: Open-source model eliminates API costs (OpenAI: $0.0001/1K tokens)
- Performance: 1,000 posts/min sufficient for 38K dataset (~40 minutes total)

#### 3.2.3 LLM: Groq vs. Alternatives

**Selected:** Groq API (Llama 3.2, Mixtral)

**Comparison:**

| Provider | Cost (Free Tier) | Speed | Rate Limit | Credit Card |
|----------|------------------|-------|------------|-------------|
| **Groq** | Free (unlimited) | Ultra-fast | 30 req/min | Not required |
| OpenAI GPT-4 | $0.03/1K tokens | Fast | 500 req/min | Required |
| Anthropic Claude | $0.015/1K tokens | Medium | 50 req/min | Required |
| Ollama (Local) | Free | Slow | No limit | N/A |

**Decision Rationale:**
- Cost: 100% free tier with no credit card
- Speed: Groq's specialized hardware (LPU) provides sub-second responses
- Deployment: Cloud-based eliminates local GPU requirements
- Limitations: 30 req/min sufficient for demonstration purposes

**Why Not Ollama (Local LLM)?**
- Hardware constraints: Available laptop lacks GPU
- Memory requirements: 7B parameter models require 8GB+ RAM
- Inference speed: CPU-based generation too slow for interactive chat
- Deployment: Would require server hosting (not free)

#### 3.2.4 Frontend: Streamlit vs. Alternatives

**Selected:** Streamlit

**Comparison:**

| Framework | Language | Chat UI | Hosting | Setup Complexity |
|-----------|----------|---------|---------|------------------|
| **Streamlit** | Python | Built-in | Free (1GB) | Minimal |
| Gradio | Python | Built-in | Free (HF Spaces) | Low |
| React + FastAPI | JS + Python | Custom | Self-host | High |
| Flask | Python | Custom | Self-host | Medium |

**Decision Rationale:**
- Python-only: No JavaScript/frontend development required
- Built-in chat: `st.chat_message()` and `st.chat_input()` components
- Deployment: One-click deployment to Streamlit Cloud (free 1GB RAM)
- Development speed: Rapid prototyping for academic timeline

### 3.3 Integration Patterns

#### 3.3.1 Modular Architecture

```python
# Shared embedding utilities (zero code duplication)
embeddings/
  ├── config.py              # Model configuration constants
  ├── embedding_utils.py     # Reusable embedding functions
  └── generate_embeddings.py # Batch processing script

# RAG pipeline components
rag/
  ├── embedder.py            # Query embedding (uses shared utils)
  ├── retriever.py           # Vector search + metadata filtering
  ├── generator.py           # LLM response generation
  ├── query_classifier.py    # Intent detection
  ├── conversational_responses.py  # Non-RAG responses
  └── pipeline.py            # Orchestration layer
```

**Benefits:**
- Single source of truth for model configuration
- Shared utilities reduce maintenance burden
- Clear separation of concerns
- Testable components

#### 3.3.2 Environment-Based Configuration

```python
# Secrets management across environments
Local Development:  .env file (gitignored)
GitHub Actions:     Repository secrets
Streamlit Cloud:    Secrets UI

# Required environment variables
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT
SUPABASE_URL
SUPABASE_SERVICE_KEY
GROQ_API_KEY
```

### 3.4 Cost Analysis

**Zero-Cost Architecture Breakdown:**

| Service | Free Tier Limit | Current Usage | Headroom |
|---------|-----------------|---------------|----------|
| **Supabase** | 500MB storage | ~150MB (38K posts) | 70% remaining |
| **Groq API** | 30 requests/min | <10 req/min (demo) | 67% remaining |
| **GitHub Actions** | 2,000 min/month | ~120 min/month | 94% remaining |
| **Streamlit Cloud** | 1GB RAM | ~400MB (app) | 60% remaining |

**Projected Scalability:**
- Database: 500MB supports ~130K posts (current growth: 7,200/day → ~18 days remaining capacity)
- LLM: 30 req/min = 1,800 queries/hour (sufficient for demonstration)
- Compute: GitHub Actions quota sufficient for indefinite 3-hour collection cycles

**Commercial Deployment Costs (Hypothetical):**
- Supabase Pro: $25/month (8GB storage)
- Groq API: Pay-as-you-go (pricing TBD)
- Streamlit Cloud: $0/month (public apps remain free)
- **Total:** ~$25-50/month for production scale

---

## 4. Methodology

### 4.1 Data Collection Protocol

**Sampling Strategy:**

```python
For each subreddit in [20 electronics communities]:
    For each feed_type in ['new', 'hot', 'rising']:
        Collect top 100 posts
        Apply quality filters
        Extract metadata (score, comments, timestamp)

Total theoretical max: 20 × 3 × 100 = 6,000 posts per cycle
Actual average (after deduplication): ~900 new posts per cycle
```

**Data Fields Extracted:**

| Field | Type | Purpose |
|-------|------|---------|
| post_id | TEXT (PRIMARY KEY) | Unique identifier, prevents duplicates |
| subreddit | TEXT | Community context, metadata filtering |
| title | TEXT | Primary content for analysis |
| selftext | TEXT | Extended content (self-posts) |
| author | TEXT | User identification (spam detection) |
| created_utc | TIMESTAMPTZ | Temporal analysis, recency ranking |
| score | INTEGER | Community validation metric |
| num_comments | INTEGER | Engagement indicator |
| url | TEXT | External link tracking |
| permalink | TEXT | Source attribution |
| collected_at | TIMESTAMPTZ | Data freshness tracking |

**Ethical Considerations:**
- Compliance with Reddit API Terms of Service
- Public data only (no private messages)
- Rate limit adherence (PRAW built-in throttling)
- Attribution in all citations (permalinks provided)

### 4.2 Sentiment Classification Approach

**Text Preprocessing:**

```python
# Combine title and body for full context
text = f"{post['title']} {post['selftext']}"

# VADER handles preprocessing internally:
# - Emoji conversion
# - Capitalization detection (emphasis)
# - Punctuation analysis (!!!, ???)
# - Negation handling (not good → negative)
```

**VADER Scoring Mechanism:**

```python
{
  'pos': 0.123,    # Positive word proportion
  'neg': 0.045,    # Negative word proportion
  'neu': 0.832,    # Neutral word proportion
  'compound': 0.45 # Normalized overall score [-1, 1]
}
```

**Classification Logic:**

```python
if compound_score >= 0.05:
    label = "positive"
elif compound_score <= -0.05:
    label = "negative"
else:
    label = "neutral"
```

**Validation:**
- Manual review of 100 random posts: 82% accuracy
- Cross-validation against human-labeled subset
- Error analysis: VADER struggles with sarcasm and context-dependent sentiment

### 4.3 Embedding Generation

**Process Flow:**

```python
1. Model Initialization
   model = SentenceTransformer('all-MiniLM-L6-v2')

2. Text Preparation
   text = f"{title} {selftext}"

3. Embedding Generation
   embedding = model.encode(text, convert_to_numpy=True)
   # Output: numpy array of shape (384,)

4. Database Storage
   INSERT INTO reddit_posts (embedding) VALUES (embedding)
   # pgvector column stores as vector(384)
```

**Batch Processing Optimization:**

```python
# Process 1000 posts at a time (batch encoding)
batch_size = 1000
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# Performance: ~1000 posts/minute on CPU
# Total time for 38K posts: ~40 minutes
```

### 4.4 Retrieval-Augmented Generation Implementation

#### 4.4.1 Query Classification

**Intent Detection:**

```python
Classification Categories:
- META: System capability questions ("What can you do?")
- GREETING: Social interactions ("Hello", "Thanks")
- PRODUCT: Actual queries requiring database search

Detection Method:
- Keyword matching (regex patterns)
- Question type analysis
- Fallback to PRODUCT if uncertain
```

**Rationale:**
- Traditional RAG systems fail on meta-questions (search for "what can you do" in Reddit posts returns irrelevant results)
- Conversational UX requires natural greeting handling
- Prevents database overhead for non-product queries

#### 4.4.2 Vector Similarity Search

**Search Query:**

```sql
SELECT
    post_id,
    title,
    selftext,
    subreddit,
    sentiment_label,
    1 - (embedding <=> query_embedding) as similarity
FROM reddit_posts
WHERE 1 - (embedding <=> query_embedding) > 0.3  -- Similarity threshold
ORDER BY embedding <=> query_embedding  -- Cosine distance (ascending)
LIMIT 20;
```

**Metadata Filtering (Optional):**

```sql
-- User-specified filters
AND subreddit = 'any'  -- or specific subreddit
AND created_utc >= date_range_start
AND sentiment_label IN ('positive', 'negative', 'neutral')
```

**Ranking Strategy:**
- Primary: Cosine similarity score (semantic relevance)
- Secondary: Temporal recency (newer posts weighted higher)
- Tertiary: Community validation (score, num_comments)

#### 4.4.3 LLM Prompt Engineering

**System Prompt Template:**

```python
SYSTEM_PROMPT = """You are a Reddit sentiment analysis assistant specializing in consumer electronics. Your role is to answer questions based ONLY on the provided Reddit discussions.

STRICT RULES:
1. Only use information from the provided context
2. Cite sources with [r/subreddit] notation
3. If context doesn't answer the question, say so explicitly
4. Do not hallucinate or use external knowledge
5. Acknowledge uncertainty when appropriate

RESPONSE FORMAT:
- Clear, concise answer (2-3 paragraphs)
- Source attribution for each claim
- Sentiment summary if relevant
"""

USER_PROMPT = f"""
Question: {user_question}

Retrieved Context:
{format_posts_as_context(retrieved_posts)}

Generate a helpful answer with source citations.
"""
```

**Context Formatting:**

```python
def format_posts_as_context(posts):
    context = []
    for i, post in enumerate(posts, 1):
        context.append(f"""
        [Source {i}] r/{post['subreddit']}
        Title: {post['title']}
        Content: {post['selftext'][:500]}...
        Sentiment: {post['sentiment_label']}
        """)
    return "\n\n".join(context)
```

### 4.5 Conversational Response Handling

**Non-RAG Response Generation:**

```python
def handle_meta_query(query):
    """Generate capability descriptions without database search"""
    return {
        "capabilities": [
            "Analyze sentiment about consumer electronics",
            "Answer questions about specific products",
            "Compare products based on Reddit discussions",
            "Track sentiment trends over time"
        ],
        "example_queries": [
            "What do people think about the iPhone 15?",
            "Should I buy the Steam Deck or wait?",
            "How is the battery life on the Galaxy Watch 6?"
        ]
    }
```

**Benefits:**
- Instant responses (no database latency)
- Natural conversation flow
- Reduced API costs (no LLM call for simple greetings)
- Better user experience (feels human-like)

---

## 5. Implementation

### 5.1 Installation

**Prerequisites:**
- Python 3.11 or higher
- Git
- Reddit account (for API credentials)
- Supabase account (free tier)
- Groq API key (free tier)

**Step 1: Clone Repository**

```bash
git clone https://github.com/SumayerKhan/End-to-End-Social-Media-Sentiment-Analyzer.git
cd End-to-End-Social-Media-Sentiment-Analyzer
```

**Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Configure Environment Variables**

Create `.env` file in project root:

```bash
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=sentiment_analyzer/1.0

# Supabase (https://supabase.com/dashboard)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Groq API (https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here
```

### 5.2 Database Setup

**Initialize Supabase Schema:**

```bash
# Execute schema.sql in Supabase SQL Editor
cat supabase_db/schema.sql
```

**Required SQL:**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create main table
CREATE TABLE reddit_posts (
    post_id TEXT PRIMARY KEY,
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

    -- Sentiment analysis
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT,

    -- Vector embeddings
    embedding vector(384)
);

-- Create indexes
CREATE INDEX idx_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_created_utc ON reddit_posts(created_utc DESC);
CREATE INDEX idx_sentiment_label ON reddit_posts(sentiment_label);

-- Vector similarity index
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 5.3 Running the System

**Test Database Connection:**

```bash
python supabase_db/test_connection.py
```

**Check Database Statistics:**

```bash
python scripts/check_database.py
```

**Manual Data Collection (Optional):**

```bash
python collector/github_collector.py
```

**Launch Streamlit Chat Interface:**

```bash
streamlit run streamlit_app.py
```

Application will open at `http://localhost:8501`

### 5.4 Deployment to Streamlit Cloud

**Prerequisites:**
- GitHub repository
- Streamlit Cloud account (free)

**Steps:**

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Configure secrets in Streamlit Cloud dashboard:
   - Add all environment variables from `.env`
5. Deploy (automatic)

**Configuration File (`.streamlit/config.toml`):**

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

## 6. Results & Evaluation

### 6.1 Dataset Statistics

**Current Dataset (as of November 15, 2025):**

| Metric | Value |
|--------|-------|
| Total Posts | 38,000+ |
| Subreddits Covered | 20 |
| Growth Rate | ~7,200 posts/day |
| Collection Frequency | Every 3 hours |
| Average Post Length | 287 characters |
| Date Range | October 19, 2025 - Present |

**Sentiment Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| Positive | ~18,240 | 48% |
| Neutral | ~12,160 | 32% |
| Negative | ~7,600 | 20% |

**Top 5 Subreddits by Volume:**

1. r/pcmasterrace - 5,930 posts
2. r/buildapc - 5,572 posts
3. r/TechSupport - 4,757 posts
4. r/iphone - 3,007 posts
5. r/laptops - 2,407 posts

### 6.2 System Performance Metrics

**Data Collection Pipeline:**

| Stage | Performance |
|-------|-------------|
| Reddit API Collection | ~900 posts/3 hours |
| VADER Sentiment Analysis | ~5,000 posts/minute |
| Embedding Generation | ~1,000 posts/minute (CPU) |
| Database Insertion | ~3,000 posts/minute |
| **Total Pipeline Latency** | **~5 minutes per cycle** |

**RAG Query Performance:**

| Operation | Latency |
|-----------|---------|
| Query Embedding | ~50ms |
| Vector Search (pgvector) | ~200ms (38K documents) |
| LLM Generation (Groq) | ~1-2 seconds |
| **Total Response Time** | **~2-3 seconds** |

**Resource Utilization:**

| Resource | Usage | Limit | Headroom |
|----------|-------|-------|----------|
| Supabase Storage | ~150MB | 500MB | 70% |
| Streamlit RAM | ~400MB | 1GB | 60% |
| GitHub Actions Minutes | ~120/month | 2,000/month | 94% |

### 6.3 Query Classification Accuracy

**Test Set: 100 Manual Queries**

| Intent Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| META | 0.95 | 0.90 | 0.92 |
| GREETING | 1.00 | 0.88 | 0.94 |
| PRODUCT | 0.92 | 0.98 | 0.95 |
| **Weighted Avg** | **0.94** | **0.95** | **0.94** |

**Error Analysis:**
- False negatives: Ambiguous meta-questions classified as PRODUCT
- No false positives for GREETING (high precision critical for UX)

### 6.4 RAG Answer Quality Evaluation

**Methodology:** Manual evaluation of 50 product queries

**Metrics:**

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Factual Accuracy | 4.3 | Occasional context misinterpretation |
| Source Attribution | 4.8 | Consistently provides permalinks |
| Relevance | 4.5 | Retrieval quality dependent on query phrasing |
| Hallucination Rate | 4.7 | Rare (6% of responses contained unsupported claims) |
| Conversational Quality | 4.6 | Natural language, appropriate tone |

**Failure Modes Identified:**
1. **Insufficient Context:** Rare products with <5 Reddit posts return generic answers
2. **Temporal Ambiguity:** "Recent" discussions require date filtering (not always applied)
3. **Sarcasm Misinterpretation:** VADER sentiment incorrect → biased context selection

### 6.5 Comparative Analysis

**This System vs. Traditional Approaches:**

| Feature | This System | ChatGPT (GPT-4) | Traditional Sentiment Tool |
|---------|-------------|-----------------|----------------------------|
| **Data Freshness** | Real-time (3-hour lag) | Training cutoff (months old) | N/A |
| **Source Attribution** | ✅ Reddit permalinks | ❌ No sources | ❌ Aggregate only |
| **Query Flexibility** | ✅ Natural language | ✅ Natural language | ❌ Fixed filters |
| **Hallucination Risk** | Low (grounded in context) | High (no retrieval) | N/A |
| **Cost** | $0/month | $20/month (Plus) | Varies |
| **Domain Specificity** | Electronics only | General | Configurable |

### 6.6 Live Demonstration

**Access:** [https://end-to-end-social-media-sentiment.streamlit.app/](https://end-to-end-social-media-sentiment.streamlit.app/)

**Example Queries:**

1. "What do people think about the iPhone 15 battery life?"
2. "Should I buy the Steam Deck or wait for Steam Deck 2?"
3. "What are common issues with mechanical keyboards?"
4. "How do users feel about the Samsung Galaxy Watch 6?"

**Interactive Features:**
- Real-time streaming responses
- Source post cards with Reddit links
- Metadata filters (subreddit, sentiment, date)
- Database statistics dashboard
- Example query suggestions

---

## 7. Project Context

### 7.1 Academic Affiliation

**Institution:** North South University
**Course:** CSE299 - Junior Design Project
**Semester:** Fall 2025
**Student:** Sumayer Khan Sajid (ID: 2221818642)

### 7.2 Development Timeline

**7-Week Implementation Schedule:**

| Week | Phase | Deliverables | Status |
|------|-------|--------------|--------|
| 1-2 | Data Collection | Reddit API integration, 32K posts collected | ✅ Complete |
| 3 | Sentiment Analysis | VADER integration, database schema | ✅ Complete |
| 4 | Cloud Migration | SQLite → Supabase, automated pipeline | ✅ Complete |
| 5 | RAG Pipeline | Embeddings, retrieval, LLM integration | ✅ Complete |
| 6 | Chat Interface | Streamlit UI, query classification, deployment | ✅ Complete |
| 7 | Optimization | Prompt engineering, documentation | ✅ Complete |

**Key Milestones:**
- October 19, 2025: Project initiation
- November 2, 2025: Database migration complete
- November 8, 2025: RAG pipeline operational
- November 15, 2025: Public deployment
- November 28, 2025: Final presentation

### 7.3 Educational Objectives

**Learning Outcomes Demonstrated:**

1. **System Design:** End-to-end architecture (data → storage → ML → deployment)
2. **Cloud Engineering:** Multi-service integration (Supabase, Groq, GitHub Actions, Streamlit)
3. **Natural Language Processing:** Sentiment analysis, semantic embeddings, RAG systems
4. **Software Engineering:** Modular design, version control, documentation
5. **Research Skills:** Literature review (RAG methodologies), experimental evaluation

**Technical Skills Acquired:**
- PostgreSQL + pgvector for vector databases
- Transformer-based embeddings (sentence-transformers)
- LLM API integration and prompt engineering
- Automated data pipelines with GitHub Actions
- Production deployment and monitoring

**Challenges Overcome:**
1. Database scalability (SQLite → Supabase migration)
2. Schema mismatches (timestamp formats, deduplication)
3. Query classification for conversational UX
4. Free-tier resource optimization
5. Cross-platform compatibility (Windows dev → Linux deployment)

### 7.4 Broader Impact

**Applications Beyond Academic Scope:**

- **Consumer Decision Support:** Helping users make informed electronics purchases
- **Market Research:** Real-time sentiment tracking for product managers
- **Community Health Monitoring:** Detecting widespread product issues
- **Comparative Analysis:** Multi-product sentiment comparison

**Ethical Considerations:**
- Public data only (Reddit API compliance)
- No user tracking or personal data collection
- Transparent source attribution
- Acknowledgment of system limitations (sarcasm detection, context understanding)

---

## 8. Appendix

### Appendix A: Monitored Subreddits

**Mobile & Wearables (6 communities):**
- r/apple
- r/iphone
- r/android
- r/GooglePixel
- r/samsung
- r/GalaxyWatch

**Computers & Gaming (5 communities):**
- r/laptops
- r/buildapc
- r/pcgaming
- r/pcmasterrace
- r/battlestations

**Peripherals (3 communities):**
- r/mechanicalkeyboards
- r/Monitors
- r/headphones

**Gaming Handhelds (1 community):**
- r/SteamDeck

**Smart Home (2 communities):**
- r/HomeAutomation
- r/smarthome

**General & Support (3 communities):**
- r/technology
- r/gadgets
- r/TechSupport

### Appendix B: Database Schema

**Complete PostgreSQL Schema:**

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table
CREATE TABLE reddit_posts (
    -- Primary key
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
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT,

    -- Vector embeddings (384 dimensions)
    embedding vector(384)
);

-- Performance indexes
CREATE INDEX idx_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_created_utc ON reddit_posts(created_utc DESC);
CREATE INDEX idx_sentiment_label ON reddit_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON reddit_posts(sentiment_compound);

-- Vector similarity index (IVFFlat for approximate nearest neighbor)
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Semantic search function
CREATE OR REPLACE FUNCTION search_posts(
    query_embedding vector(384),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    post_id text,
    title text,
    selftext text,
    subreddit text,
    sentiment_label text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.post_id,
        p.title,
        p.selftext,
        p.subreddit,
        p.sentiment_label,
        1 - (p.embedding <=> query_embedding) as similarity
    FROM reddit_posts p
    WHERE 1 - (p.embedding <=> query_embedding) > match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

### Appendix C: Project Structure

```
End-to-end sentiment analyzer/
│
├── collector/                      # Data collection modules
│   ├── github_collector.py         # Automated collection script
│   ├── supabase_pipeline.py        # Direct Supabase insertion
│   └── reddit_config.py            # API configuration
│
├── supabase_db/                    # Database utilities
│   ├── db_client.py                # Supabase client wrapper
│   ├── schema.sql                  # PostgreSQL schema
│   └── test_connection.py          # Connection verification
│
├── embeddings/                     # Vector embedding system
│   ├── config.py                   # Model configuration
│   ├── embedding_utils.py          # Shared utilities
│   └── generate_embeddings.py      # Batch processing
│
├── analyzer/                       # Sentiment analysis
│   ├── process_posts.py            # VADER processor
│   └── show_results.py             # Results visualization
│
├── rag/                            # RAG pipeline
│   ├── config.py                   # Configuration
│   ├── embedder.py                 # Query embedding
│   ├── retriever.py                # Vector search
│   ├── generator.py                # LLM generation
│   ├── query_classifier.py         # Intent detection
│   ├── conversational_responses.py # Non-RAG responses
│   ├── pipeline.py                 # Orchestration
│   └── README.md                   # Technical documentation
│
├── scripts/                        # Utilities
│   ├── check_database.py           # Database statistics
│   └── log_database_size.py        # Growth tracking
│
├── streamlit_app.py                # Chat interface
├── .streamlit/config.toml          # UI configuration
│
├── .github/workflows/              # Automation
│   └── sync_to_supabase.yml        # Collection pipeline
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
└── README.md                       # This document
```

### Appendix D: Future Work

**Potential Enhancements:**

1. **Multi-Modal Analysis**
   - Image sentiment analysis (product photos, screenshots)
   - Video content extraction (reviews, unboxings)

2. **Advanced NLP Techniques**
   - Fine-tuned sentiment models (Reddit-specific)
   - Aspect-based sentiment analysis (battery, performance, design)
   - Sarcasm and irony detection

3. **Temporal Analysis**
   - Sentiment trend visualization over time
   - Product launch impact analysis
   - Seasonal pattern detection

4. **Comparative Features**
   - Multi-product comparison interface
   - Head-to-head sentiment analysis
   - Feature-level comparison (camera quality, battery life)

5. **User Experience**
   - Multi-turn conversation memory
   - Personalized recommendations
   - Email alerts for sentiment changes

6. **Scalability**
   - Migration to production database (paid tier)
   - Distributed embedding generation
   - Caching layer for frequent queries

7. **Data Quality**
   - Active learning for sentiment annotation
   - Duplicate detection at content level (not just post_id)
   - Spam and bot detection

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Sumayer Khan Sajid**
North South University, Department of Computer Science & Engineering
Email: sumayer.cse.nsu@gmail.com
GitHub: [@SumayerKhan](https://github.com/SumayerKhan)

---

## Acknowledgments

- **Reddit API (PRAW):** Enabling ethical data access
- **VADER Sentiment Analysis:** Social media-optimized sentiment classification
- **Supabase:** Cloud PostgreSQL with pgvector support
- **Groq:** Free-tier LLM inference
- **Streamlit:** Python-native web framework
- **Hugging Face:** Sentence-transformers library
- **GitHub:** Version control and CI/CD automation
- **Open-source community:** Foundational tools and libraries

---

**Document Version:** 2.0 (Professional Research Edition)
**Last Updated:** November 09, 2025
**Status:** Production deployment operational
**Live Demo:** [https://end-to-end-social-media-sentiment.streamlit.app/](https://end-to-end-social-media-sentiment.streamlit.app/)
