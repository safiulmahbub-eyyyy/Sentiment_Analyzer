"""
RAG Pipeline Configuration
Central configuration for all RAG components
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

# Use same model as embedding generation for consistency
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Text preparation
MAX_TEXT_LENGTH = 512  # Maximum characters for embedding

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Vector search parameters
DEFAULT_TOP_K = 15  # Number of posts to retrieve
MIN_SIMILARITY_THRESHOLD = 0.5  # Minimum cosine similarity (0-1)

# Metadata filtering defaults
DEFAULT_DATE_RANGE_DAYS = 365  # Look back 1 year by default
ENABLE_SENTIMENT_FILTER = True  # Allow filtering by sentiment
ENABLE_SUBREDDIT_FILTER = True  # Allow filtering by subreddit

# ============================================================
# LLM CONFIGURATION (GROQ)
# ============================================================

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "GROQ_API_KEY not found in environment variables.\n"
        "For local development: Add GROQ_API_KEY to your .env file\n"
        "For Streamlit Cloud: Add GROQ_API_KEY to app secrets in dashboard\n"
        "Get your free API key from https://console.groq.com/"
    )

# Model selection
# Current supported models (as of Nov 2025):
# - llama-3.3-70b-versatile (recommended - fast & high quality)
# - llama-3.1-70b-versatile (alternative 70B model)
# - mixtral-8x7b-32768 (good for longer context)
# - llama-3.1-8b-instant (fastest, smaller model)
GROQ_MODEL = "llama-3.3-70b-versatile"

# Generation parameters
TEMPERATURE = 0.15  # Lower = more focused, Higher = more creative (0-2) - Set low for consistency
MAX_TOKENS = 1024  # Maximum response length
TOP_P = 0.9  # Nucleus sampling (0-1)

# Streaming
ENABLE_STREAMING = False  # Set to True for real-time response streaming

# Rate limiting (Groq free tier: 30 requests/min)
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds between retries

# ============================================================
# CONTEXT BUILDING CONFIGURATION
# ============================================================

# How many posts to include in LLM context
MAX_CONTEXT_POSTS = 10  # Use top 10 most relevant (from top_k retrieved)

# Context formatting
INCLUDE_METADATA = True  # Include subreddit, sentiment, score in context
INCLUDE_POST_SCORES = True  # Include Reddit upvote scores
INCLUDE_SIMILARITY_SCORES = True  # Include vector similarity scores

# Context length management
MAX_CONTEXT_LENGTH = 4000  # Maximum characters for context section

# ============================================================
# PROMPT CONFIGURATION
# ============================================================

# System behavior
REQUIRE_SOURCE_CITATION = True  # Force LLM to cite sources
ALLOW_NO_ANSWER = True  # Allow "I don't have enough information" responses

# Response style
RESPONSE_STYLE = "balanced"  # Options: "concise", "balanced", "detailed"

# ============================================================
# CACHING CONFIGURATION
# ============================================================

# Model caching (keep model loaded in memory)
CACHE_EMBEDDING_MODEL = True
CACHE_GROQ_CLIENT = True

# Query caching (cache results for identical queries)
ENABLE_QUERY_CACHE = False  # Set to True for production
QUERY_CACHE_TTL = 3600  # Cache lifetime in seconds (1 hour)

# ============================================================
# DEBUGGING & LOGGING
# ============================================================

# Verbose output
VERBOSE = True  # Print detailed pipeline steps
DEBUG_MODE = False  # Print full prompts and responses

# Performance tracking
TRACK_QUERY_TIME = True  # Measure and report query execution time

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """
    Validate configuration settings
    Raises ValueError if configuration is invalid
    """
    errors = []

    # Check API key
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is required")

    # Check numeric ranges
    if not 0 <= TEMPERATURE <= 2:
        errors.append(f"TEMPERATURE must be between 0 and 2, got {TEMPERATURE}")

    if not 0 <= TOP_P <= 1:
        errors.append(f"TOP_P must be between 0 and 1, got {TOP_P}")

    if not 0 <= MIN_SIMILARITY_THRESHOLD <= 1:
        errors.append(f"MIN_SIMILARITY_THRESHOLD must be between 0 and 1, got {MIN_SIMILARITY_THRESHOLD}")

    if DEFAULT_TOP_K < 1:
        errors.append(f"DEFAULT_TOP_K must be at least 1, got {DEFAULT_TOP_K}")

    if MAX_CONTEXT_POSTS > DEFAULT_TOP_K:
        errors.append(f"MAX_CONTEXT_POSTS ({MAX_CONTEXT_POSTS}) cannot exceed DEFAULT_TOP_K ({DEFAULT_TOP_K})")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))

    return True


# Validate on import
if __name__ != "__main__":
    validate_config()


if __name__ == "__main__":
    """Print current configuration for debugging"""
    print("="*60)
    print("RAG PIPELINE CONFIGURATION")
    print("="*60)

    print("\nEMBEDDING:")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimension: {EMBEDDING_DIMENSION}")

    print("\nRETRIEVAL:")
    print(f"  Top-k posts: {DEFAULT_TOP_K}")
    print(f"  Min similarity: {MIN_SIMILARITY_THRESHOLD}")

    print("\nLLM:")
    print(f"  Model: {GROQ_MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Streaming: {ENABLE_STREAMING}")

    print("\nCONTEXT:")
    print(f"  Max posts in context: {MAX_CONTEXT_POSTS}")
    print(f"  Include metadata: {INCLUDE_METADATA}")

    print("\nVALIDATION:")
    try:
        validate_config()
        print("  Status: OK")
    except ValueError as e:
        print(f"  Status: FAILED")
        print(f"  Errors: {e}")

    print("="*60)
