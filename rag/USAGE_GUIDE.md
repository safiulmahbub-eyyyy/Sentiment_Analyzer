# RAG Pipeline Usage Guide

Quick reference for using the RAG pipeline in different ways.

---

## 🎮 Running the Test Script

The `test_rag.py` script has **3 modes**:

### 1️⃣ Interactive Mode (BEST for experimenting)

**From Terminal:**
```bash
python rag/test_rag.py --mode interactive
```

**From VS Code:**
Just press the "Run" button (▶️) when `test_rag.py` is open - it's now the default!

**What it does:**
- Loads the pipeline once
- Lets you ask unlimited questions
- Type questions naturally
- Type `quit` to exit
- Type `stats` to see database info
- Type `help` for commands

**Example session:**
```
Your question: What do people think about iPhone 15?
Thinking...
ANSWER: Based on Reddit discussions, users generally...

Your question: Are gaming laptops worth it?
Thinking...
ANSWER: ...

Your question: quit
Goodbye!
```

---

### 2️⃣ Quick Mode (Single question)

**From Terminal:**
```bash
python rag/test_rag.py --mode quick --question "What do people think about MacBooks?"
```

**What it does:**
- Asks one question
- Shows formatted answer with sources
- Exits automatically
- Good for testing or scripting

---

### 3️⃣ Test Mode (Full test suite)

**From Terminal:**
```bash
python rag/test_rag.py --mode test
```

**What it does:**
- Runs 13 diverse test queries
- Tests filters (subreddit, sentiment, date)
- Tests response styles (concise, balanced, detailed)
- Generates detailed performance report
- Takes ~5-10 minutes
- Good for validation after changes

---

## 💻 Using RAG in Your Own Code

### Basic Usage

```python
from rag.pipeline import RAGPipeline

# Initialize pipeline (loads models - do this ONCE)
pipeline = RAGPipeline()

# Ask a question
result = pipeline.query("What do people think about gaming laptops?")

# Access the answer
print(result['answer'])

# Access sources
for source in result['sources']:
    print(f"- r/{source['subreddit']}: {source['title']}")

# Access metadata
print(f"Used {result['metadata']['posts_used']} posts")
print(f"Has citations: {result['metadata']['has_citations']}")
```

### Advanced Usage - Filters

```python
# Filter by subreddit (only r/iphone)
result = pipeline.query(
    "What are common iPhone problems?",
    subreddit_filter="iphone"
)

# Filter by sentiment (only positive posts)
result = pipeline.query(
    "What do people love about MacBooks?",
    sentiment_filter="positive"
)

# Filter by date (last 30 days only)
result = pipeline.query(
    "Recent opinions on Samsung phones?",
    days_ago=30
)

# Combine filters
result = pipeline.query(
    "Positive iPhone reviews from last month?",
    subreddit_filter="iphone",
    sentiment_filter="positive",
    days_ago=30
)
```

### Response Styles

```python
# Concise (2-3 sentences)
result = pipeline.query(
    "Are gaming laptops worth it?",
    style="concise"
)

# Balanced (3-5 sentences, default)
result = pipeline.query(
    "Are gaming laptops worth it?",
    style="balanced"
)

# Detailed (comprehensive analysis)
result = pipeline.query(
    "Are gaming laptops worth it?",
    style="detailed"
)
```

### Formatted Output

```python
# Get pre-formatted output (answer + sources + metadata)
formatted = pipeline.query_formatted(
    "What's the best budget smartphone?",
    top_k=10
)

print(formatted)
# Prints nicely formatted output with sources
```

### One-Liner Quick Query

```python
from rag.pipeline import quick_query

# Uses singleton pipeline (loads once, reuses)
answer = quick_query("What do people think about iPhone 15?")
print(answer)
```

---

## 🔧 Testing Individual Components

You can test each component separately:

```bash
# Test embedder
python rag/embedder.py

# Test retriever
python rag/retriever.py

# Test Groq client
python rag/groq_client.py

# Test prompt templates
python rag/prompt_templates.py

# Test generator
python rag/generator.py

# Test full pipeline
python rag/pipeline.py
```

Each component has its own test code at the bottom!

---

## ⚙️ Configuration

Edit `rag/config.py` to change settings:

### Common Settings to Adjust

```python
# Number of posts to retrieve
DEFAULT_TOP_K = 15  # Try 10-30

# Minimum similarity threshold
MIN_SIMILARITY_THRESHOLD = 0.5  # Try 0.3-0.7 (lower = more results)

# Response style
RESPONSE_STYLE = "balanced"  # "concise" / "balanced" / "detailed"

# LLM temperature
TEMPERATURE = 0.3  # 0.1 (focused) - 1.0 (creative)

# LLM model
GROQ_MODEL = "llama-3.3-70b-versatile"  # Current model

# Enable verbose output
VERBOSE = True  # Set to False for quiet mode

# Enable debug mode (shows full prompts)
DEBUG_MODE = False  # Set to True to see prompts
```

---

## 🐛 Troubleshooting

### "No relevant posts found"

Try:
- Lower `similarity_threshold` to 0.3 or 0.4
- Increase `top_k` to 20-30
- Use broader questions
- Check database has embeddings

### "Rate limit exceeded"

- Groq free tier: 30 requests/minute
- Add delays: `time.sleep(2)`
- Wait 1 minute and retry

### Slow first query

- Normal! Model loading takes ~5-15 seconds
- Subsequent queries are fast (~3-5 seconds)
- Use `RAGPipeline()` class to cache the model

### Want to see what's happening

```python
# Enable verbose mode
from rag import config
config.VERBOSE = True

# Enable debug mode (shows full prompts)
config.DEBUG_MODE = True
```

---

## 📊 Understanding Results

### Result Structure

```python
result = {
    'answer': "The generated answer text...",
    'sources': [
        {
            'post_id': '...',
            'title': '...',
            'subreddit': '...',
            'sentiment_label': 'positive',
            'similarity': 0.87,
            'score': 245,
            'permalink': '/r/...'
        },
        # ... more posts
    ],
    'metadata': {
        'question': 'Your question',
        'posts_used': 10,
        'total_posts_retrieved': 15,
        'has_citations': True,
        'citations': ['[r/iphone, Post #1]', ...],
        'style': 'balanced',
        'temperature': 0.3,
        'timing': {
            'embed_time': 0.15,
            'retrieve_time': 0.42,
            'generate_time': 2.3,
            'total_time': 2.87
        }
    }
}
```

### Citation Format

The LLM cites sources like this:
```
[r/subreddit, Post #X]
```

Example:
```
"Users love the battery life [r/iphone, Post #1] but some
report heating issues [r/apple, Post #3]."
```

You can match these citations to `result['sources']` by index.

---

## 🎯 Use Cases

### 1. Product Research
```python
pipeline.query("What do people think about M3 MacBook Pro?")
```

### 2. Issue Investigation
```python
pipeline.query("What are common problems with gaming laptops?")
```

### 3. Feature Comparison
```python
pipeline.query("Is iPhone camera better than Samsung?")
```

### 4. Purchase Decision
```python
pipeline.query("Should I buy a laptop or desktop for gaming?")
```

### 5. Sentiment Analysis
```python
# Positive sentiment only
pipeline.query(
    "What do users love about this product?",
    sentiment_filter="positive"
)

# Negative sentiment only
pipeline.query(
    "What complaints do users have?",
    sentiment_filter="negative"
)
```

---

## 📝 Example Scripts

### Save answers to file

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

questions = [
    "What do people think about iPhone 15?",
    "Are gaming laptops worth it?",
    "Best budget smartphone?"
]

with open('answers.txt', 'w') as f:
    for question in questions:
        result = pipeline.query(question)
        f.write(f"Q: {question}\n")
        f.write(f"A: {result['answer']}\n\n")
```

### Batch processing with filters

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

subreddits = ['iphone', 'android', 'samsung']
question = "What do people think about smartphone cameras?"

for sub in subreddits:
    result = pipeline.query(
        question,
        subreddit_filter=sub
    )
    print(f"\n{'='*60}")
    print(f"r/{sub} perspective:")
    print(f"{'='*60}")
    print(result['answer'])
```

---

## 🚀 Next Steps

- **Week 6:** Build Streamlit chat interface
- **Week 7:** Deploy to Streamlit Cloud

---

**Last Updated:** November 2, 2025
