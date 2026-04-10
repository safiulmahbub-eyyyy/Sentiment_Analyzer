"""
RAG Pipeline Orchestrator
Main class that coordinates the entire RAG pipeline
"""

import time
from typing import Dict, Any, Optional, List

from rag.embedder import get_embedding_model, embed_query
from rag.retriever import retrieve_similar_posts, rerank_by_relevance, get_diverse_posts
from rag.generator import generate_answer, generate_answer_with_sources_formatted
from rag.groq_client import get_groq_client, test_api_connection
from rag.query_classifier import classify_query, should_use_rag_pipeline
from rag.conversational_responses import generate_conversational_response
from supabase_db.db_client import get_client

from rag.config import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    DEFAULT_DATE_RANGE_DAYS,
    RESPONSE_STYLE,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_CONTEXT_POSTS,
    VERBOSE,
    TRACK_QUERY_TIME
)


class RAGPipeline:
    """
    RAG Pipeline for Consumer Electronics Sentiment Analysis

    This class is the "heated oven" - it loads expensive resources ONCE
    and reuses them for many queries!

    Usage:
        >>> pipeline = RAGPipeline()  # Load model + clients (takes ~15 seconds)
        >>> result = pipeline.query("What do people think about iPhone?")  # Fast!
        >>> result2 = pipeline.query("Are gaming laptops good?")  # Still fast!
        >>> result3 = pipeline.query("Best smartphone camera?")  # Still fast!

    The pipeline orchestrates:
    1. Query embedding (convert question to vector)
    2. Vector search (find similar posts)
    3. Response generation (LLM answer with sources)
    """

    def __init__(self, verbose: bool = VERBOSE):
        """
        Initialize RAG pipeline

        This loads and caches:
        - Embedding model (sentence-transformers)
        - Groq LLM client
        - Supabase database client

        These are expensive operations (~15 seconds total) but only happen ONCE!

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose

        if self.verbose:
            print("="*60)
            print("INITIALIZING RAG PIPELINE")
            print("="*60)

        start_time = time.time()

        # Load embedding model (this is the slow part!)
        if self.verbose:
            print("\n[1/3] Loading embedding model...")
        model_start = time.time()
        self.embedding_model = get_embedding_model()
        model_time = time.time() - model_start
        if self.verbose:
            print(f"[OK] Model loaded in {model_time:.1f}s")

        # Initialize Groq client
        if self.verbose:
            print("\n[2/3] Initializing Groq LLM client...")
        groq_start = time.time()
        self.groq_client = get_groq_client()
        groq_time = time.time() - groq_start
        if self.verbose:
            print(f"[OK] Groq client ready in {groq_time:.1f}s")

        # Initialize Supabase client
        if self.verbose:
            print("\n[3/3] Connecting to Supabase database...")
        db_start = time.time()
        self.supabase_client = get_client()
        db_time = time.time() - db_start
        if self.verbose:
            print(f"[OK] Database connected in {db_time:.1f}s")

        total_time = time.time() - start_time

        if self.verbose:
            print("\n" + "="*60)
            print(f"PIPELINE READY (initialized in {total_time:.1f}s)")
            print("="*60 + "\n")

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
        subreddit_filter: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        days_ago: int = DEFAULT_DATE_RANGE_DAYS,
        style: str = RESPONSE_STYLE,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        rerank: bool = False,
        diversify: bool = False,
        enable_conversational: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question (supports conversation history for follow-ups)

        This is the MAIN method you'll use!

        Args:
            question: User's question
            top_k: Number of posts to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            subreddit_filter: Optional subreddit filter (e.g., "iphone")
            sentiment_filter: Optional sentiment filter ("positive", "negative", "neutral")
            days_ago: Only search posts from last N days
            style: Response style ("concise", "balanced", "detailed")
            temperature: LLM temperature
            max_tokens: Maximum response length
            rerank: Whether to rerank by relevance (combines similarity + engagement)
            diversify: Whether to ensure diversity across subreddits
            enable_conversational: Enable conversational responses for meta/greeting queries
            conversation_history: Optional list of previous conversation turns for context
                                [{"role": "user/assistant", "content": "..."}]
                                Enables Perplexity-style follow-up questions

        Returns:
            Dictionary with:
            - answer: Generated answer
            - sources: Source posts
            - metadata: Query metadata (time, posts used, etc.)

        Example (single turn):
            >>> result = pipeline.query("What do people think about iPhone 15?")
            >>> print(result['answer'])
            Based on Reddit discussions, users generally...

        Example (follow-up with history):
            >>> history = [
            ...     {"role": "user", "content": "Which laptop is best?"},
            ...     {"role": "assistant", "content": "Dell XPS is popular..."}
            ... ]
            >>> result = pipeline.query("What about cheaper ones?", conversation_history=history)

            >>> print(f"Used {len(result['sources'])} sources")
            Used 10 sources
        """
        if TRACK_QUERY_TIME:
            query_start = time.time()

        if self.verbose:
            print(f"\n[QUERY] '{question}'")
            print("-" * 60)

        # Step 0: Classify query intent
        classification = classify_query(question)

        if self.verbose:
            print(f"[CLASSIFY] Type: {classification['type']}, Confidence: {classification['confidence']:.2f}")

        # Handle conversational queries (meta, greeting, out_of_scope)
        if enable_conversational and classification['type'] != 'product_sentiment':
            if self.verbose:
                print(f"[CONVERSATIONAL] Handling {classification['type']} query")

            result = generate_conversational_response(
                query_type=classification['type'],
                question=question,
                classification_confidence=classification['confidence']
            )

            # Add classification to metadata
            result['metadata']['classification'] = classification

            if TRACK_QUERY_TIME:
                total_query_time = time.time() - query_start
                result['metadata']['timing'] = {
                    'total_time': total_query_time
                }

            return result

        # For product sentiment queries with low confidence, warn but proceed
        if classification['confidence'] < 0.4:
            if self.verbose:
                print(f"[WARNING] Low confidence ({classification['confidence']:.2f}) - proceeding anyway")

        # Step 1: Embed query
        if self.verbose:
            print("[1/3] Embedding query...")
        embed_start = time.time()

        query_embedding = embed_query(question)

        embed_time = time.time() - embed_start
        if self.verbose:
            print(f"[OK] Query embedded in {embed_time:.3f}s")

        # Step 2: Retrieve similar posts
        if self.verbose:
            print(f"[2/3] Retrieving similar posts (top_k={top_k})...")
        retrieve_start = time.time()

        posts = retrieve_similar_posts(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            subreddit_filter=subreddit_filter,
            sentiment_filter=sentiment_filter,
            days_ago=days_ago
        )

        retrieve_time = time.time() - retrieve_start
        if self.verbose:
            print(f"[OK] Retrieved {len(posts)} posts in {retrieve_time:.3f}s")

        # Handle case where no posts found - use conversational fallback
        if not posts and enable_conversational:
            if self.verbose:
                print("[NO RESULTS] Using conversational fallback")

            result = generate_conversational_response(
                query_type='no_results',
                question=question,
                additional_context={
                    'filters': {
                        'subreddit_filter': subreddit_filter,
                        'sentiment_filter': sentiment_filter,
                        'days_ago': days_ago
                    }
                }
            )

            # Add classification and timing
            result['metadata']['classification'] = classification

            if TRACK_QUERY_TIME:
                total_query_time = time.time() - query_start
                result['metadata']['timing'] = {
                    'embed_time': embed_time,
                    'retrieve_time': retrieve_time,
                    'total_time': total_query_time
                }

            return result

        # Optional: Re-rank by relevance
        if rerank and posts:
            if self.verbose:
                print("[RERANK] Re-ranking by relevance...")
            posts = rerank_by_relevance(posts)

        # Optional: Diversify results
        if diversify and posts:
            if self.verbose:
                print("[DIVERSIFY] Ensuring subreddit diversity...")
            posts = get_diverse_posts(posts, max_per_subreddit=3)

        # Step 3: Generate answer
        if self.verbose:
            print(f"[3/3] Generating answer (style={style})...")
        generate_start = time.time()

        result = generate_answer(
            question=question,
            retrieved_posts=posts,
            style=style,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_history=conversation_history
        )

        generate_time = time.time() - generate_start
        if self.verbose:
            print(f"[OK] Answer generated in {generate_time:.3f}s")

        # Add classification to metadata
        result['metadata']['classification'] = classification

        # Add timing metadata
        if TRACK_QUERY_TIME:
            total_query_time = time.time() - query_start
            result['metadata']['timing'] = {
                'embed_time': embed_time,
                'retrieve_time': retrieve_time,
                'generate_time': generate_time,
                'total_time': total_query_time
            }

            if self.verbose:
                print(f"\n[TIMING] Total query time: {total_query_time:.3f}s")
                print("-" * 60)

        return result

    def query_formatted(self, question: str, **kwargs) -> str:
        """
        Query and return formatted output with answer + sources

        Args:
            question: User's question
            **kwargs: Additional arguments for query()

        Returns:
            Formatted string with answer and sources

        Example:
            >>> output = pipeline.query_formatted("What's the best laptop?")
            >>> print(output)
            ANSWER:
            ======================================
            Based on Reddit discussions...

            SOURCES:
            ======================================
            1. [r/laptops] ...
        """
        # Get retrieval parameters
        top_k = kwargs.pop('top_k', DEFAULT_TOP_K)
        similarity_threshold = kwargs.pop('similarity_threshold', MIN_SIMILARITY_THRESHOLD)
        subreddit_filter = kwargs.pop('subreddit_filter', None)
        sentiment_filter = kwargs.pop('sentiment_filter', None)
        days_ago = kwargs.pop('days_ago', DEFAULT_DATE_RANGE_DAYS)

        # Embed and retrieve
        query_embedding = embed_query(question)
        posts = retrieve_similar_posts(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            subreddit_filter=subreddit_filter,
            sentiment_filter=sentiment_filter,
            days_ago=days_ago
        )

        # Generate formatted answer
        return generate_answer_with_sources_formatted(question, posts, **kwargs)

    def test_connection(self) -> bool:
        """
        Test all connections (Groq API, Supabase, embedding model)

        Returns:
            True if all tests pass, False otherwise
        """
        print("="*60)
        print("TESTING RAG PIPELINE CONNECTIONS")
        print("="*60)

        all_passed = True

        # Test 1: Groq API
        print("\n[TEST 1] Groq API connection...")
        try:
            if test_api_connection():
                print("Status: PASSED")
            else:
                print("Status: FAILED")
                all_passed = False
        except Exception as e:
            print(f"Status: FAILED - {e}")
            all_passed = False

        # Test 2: Embedding model
        print("\n[TEST 2] Embedding model...")
        try:
            test_embedding = embed_query("test query")
            if len(test_embedding) == 384:
                print("Status: PASSED")
            else:
                print(f"Status: FAILED - Wrong dimension: {len(test_embedding)}")
                all_passed = False
        except Exception as e:
            print(f"Status: FAILED - {e}")
            all_passed = False

        # Test 3: Supabase database
        print("\n[TEST 3] Supabase database...")
        try:
            stats = self.supabase_client.get_stats()
            if stats:
                total_posts = stats.get('total_posts', 0)
                print(f"Status: PASSED (found {total_posts:,} posts)")
            else:
                print("Status: WARNING - No stats returned")
        except Exception as e:
            print(f"Status: FAILED - {e}")
            all_passed = False

        print("\n" + "="*60)
        if all_passed:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED - Check configuration")
        print("="*60)

        return all_passed

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with database stats
        """
        return self.supabase_client.get_stats()


# ============================================================
# CONVENIENCE FUNCTIONS (for quick use without class)
# ============================================================

# Module-level pipeline instance (lazy loaded)
_pipeline_instance = None


def get_pipeline(verbose: bool = VERBOSE) -> RAGPipeline:
    """
    Get or create singleton pipeline instance

    This loads the pipeline ONCE and reuses it.

    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(verbose=verbose)
    return _pipeline_instance


def quick_query(question: str, **kwargs) -> str:
    """
    Quick query function for simple use cases

    Automatically uses singleton pipeline (loads once, reuses forever)

    Args:
        question: User's question
        **kwargs: Additional query parameters

    Returns:
        Formatted answer with sources

    Example:
        >>> answer = quick_query("What do people think about iPhone 15?")
        >>> print(answer)
    """
    pipeline = get_pipeline()
    return pipeline.query_formatted(question, **kwargs)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the RAG pipeline"""
    print("="*60)
    print("RAG PIPELINE TEST")
    print("="*60)

    # Initialize pipeline
    pipeline = RAGPipeline(verbose=True)

    # Test connections
    print("\n" + "="*60)
    pipeline.test_connection()

    # Test query
    print("\n" + "="*60)
    print("TESTING QUERY")
    print("="*60)

    test_question = "What do people think about iPhone 15 battery life?"

    result = pipeline.query(
        question=test_question,
        top_k=5,
        similarity_threshold=0.3,
        style="balanced"
    )

    print("\nRESULT:")
    print("="*60)
    print(f"Question: {test_question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {len(result['sources'])} posts")
    print(f"Has citations: {result['metadata'].get('has_citations', False)}")

    if 'timing' in result['metadata']:
        timing = result['metadata']['timing']
        print(f"\nTiming:")
        print(f"  Embed: {timing['embed_time']:.3f}s")
        print(f"  Retrieve: {timing['retrieve_time']:.3f}s")
        print(f"  Generate: {timing['generate_time']:.3f}s")
        print(f"  Total: {timing['total_time']:.3f}s")

    # Test formatted output
    print("\n" + "="*60)
    print("TESTING FORMATTED OUTPUT")
    print("="*60)

    formatted = pipeline.query_formatted(
        "Are gaming laptops worth it?",
        top_k=3,
        style="concise"
    )

    print(formatted)

    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE")
    print("="*60)
