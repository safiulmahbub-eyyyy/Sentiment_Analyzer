"""
Comprehensive RAG Pipeline Testing Script

Tests all components and validates the complete RAG pipeline with diverse queries
"""

import time

from rag.pipeline import RAGPipeline, quick_query


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    # General product sentiment
    {
        "question": "What do people think about iPhone 15?",
        "category": "General Sentiment",
        "expected": "Should discuss both positive and negative aspects"
    },
    {
        "question": "Are gaming laptops worth it?",
        "category": "Value Assessment",
        "expected": "Should weigh pros/cons and value proposition"
    },

    # Specific features
    {
        "question": "How is the iPhone 15 Pro camera quality?",
        "category": "Feature-Specific",
        "expected": "Should focus on camera-related discussions"
    },
    {
        "question": "What do people say about MacBook battery life?",
        "category": "Feature-Specific",
        "expected": "Should discuss battery performance experiences"
    },

    # Comparisons
    {
        "question": "Is iPhone better than Android?",
        "category": "Comparison",
        "expected": "Should present both sides with user experiences"
    },
    {
        "question": "Should I buy a laptop or desktop for gaming?",
        "category": "Comparison",
        "expected": "Should compare gaming performance and value"
    },

    # Problems/Issues
    {
        "question": "What are common problems with gaming laptops?",
        "category": "Issues",
        "expected": "Should list frequently mentioned issues"
    },
    {
        "question": "Do people have heating issues with MacBooks?",
        "category": "Issues",
        "expected": "Should discuss thermal problems if mentioned"
    },

    # Recommendations
    {
        "question": "What's the best budget smartphone?",
        "category": "Recommendation",
        "expected": "Should cite user recommendations with reasoning"
    },
    {
        "question": "Which laptop is best for college students?",
        "category": "Recommendation",
        "expected": "Should suggest options based on student needs"
    },

    # Specific brands
    {
        "question": "What's the general opinion on Samsung phones?",
        "category": "Brand Sentiment",
        "expected": "Should summarize Samsung sentiment"
    },

    # Edge cases
    {
        "question": "What do people think about quantum computers?",
        "category": "Edge Case",
        "expected": "Should say insufficient information (unlikely in dataset)"
    },

    # Very specific
    {
        "question": "Is the iPhone 15 Pro Max worth the extra cost over the regular iPhone 15?",
        "category": "Specific Comparison",
        "expected": "Should compare value propositions if data exists"
    }
]


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_individual_query(pipeline: RAGPipeline, test_case: dict, test_num: int) -> dict:
    """
    Test a single query and return results

    Args:
        pipeline: RAGPipeline instance
        test_case: Test case dictionary
        test_num: Test number

    Returns:
        Test result dictionary
    """
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_case['category']}")
    print(f"{'='*60}")
    print(f"Question: {test_case['question']}")
    print(f"Expected: {test_case['expected']}")
    print(f"{'-'*60}")

    start_time = time.time()

    try:
        # Query pipeline
        result = pipeline.query(
            question=test_case['question'],
            top_k=10,
            similarity_threshold=0.4,
            style="balanced"
        )

        query_time = time.time() - start_time

        # Display result
        print(f"\nANSWER:")
        print(result['answer'])

        print(f"\nMETADATA:")
        print(f"  Posts retrieved: {result['metadata']['total_posts_retrieved']}")
        print(f"  Posts used: {result['metadata']['posts_used']}")
        print(f"  Has citations: {result['metadata']['has_citations']}")
        print(f"  Query time: {query_time:.2f}s")

        if result['sources']:
            print(f"\nTOP SOURCES:")
            for i, post in enumerate(result['sources'][:3], 1):
                subreddit = post.get('subreddit', 'unknown')
                similarity = post.get('similarity', 0)
                sentiment = post.get('sentiment_label', 'unknown')
                print(f"  {i}. r/{subreddit} (sim: {similarity:.2f}, {sentiment})")

        # Return test result
        return {
            'test_num': test_num,
            'category': test_case['category'],
            'question': test_case['question'],
            'success': True,
            'posts_retrieved': result['metadata']['total_posts_retrieved'],
            'posts_used': result['metadata']['posts_used'],
            'has_citations': result['metadata']['has_citations'],
            'query_time': query_time,
            'answer_length': len(result['answer'])
        }

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()

        return {
            'test_num': test_num,
            'category': test_case['category'],
            'question': test_case['question'],
            'success': False,
            'error': str(e)
        }


def test_filters(pipeline: RAGPipeline):
    """Test retrieval with various filters"""
    print(f"\n{'='*60}")
    print("TESTING FILTERS")
    print(f"{'='*60}")

    test_question = "What do people think about smartphones?"

    # Test 1: Subreddit filter
    print("\n[TEST 1] Subreddit filter (r/iphone only):")
    result1 = pipeline.query(
        question=test_question,
        top_k=5,
        subreddit_filter="iphone"
    )
    posts_retrieved = result1['metadata'].get('total_posts_retrieved', len(result1['sources']))
    print(f"Retrieved {posts_retrieved} posts")
    if result1['sources']:
        subreddits = [p['subreddit'] for p in result1['sources']]
        print(f"Subreddits: {set(subreddits)}")

    # Test 2: Sentiment filter
    print("\n[TEST 2] Sentiment filter (positive only):")
    result2 = pipeline.query(
        question=test_question,
        top_k=5,
        sentiment_filter="positive"
    )
    posts_retrieved = result2['metadata'].get('total_posts_retrieved', len(result2['sources']))
    print(f"Retrieved {posts_retrieved} posts")
    if result2['sources']:
        sentiments = [p['sentiment_label'] for p in result2['sources']]
        print(f"Sentiments: {set(sentiments)}")

    # Test 3: Date filter
    print("\n[TEST 3] Date filter (last 30 days only):")
    result3 = pipeline.query(
        question=test_question,
        top_k=5,
        days_ago=30
    )
    posts_retrieved = result3['metadata'].get('total_posts_retrieved', len(result3['sources']))
    print(f"Retrieved {posts_retrieved} posts")


def test_response_styles(pipeline: RAGPipeline):
    """Test different response styles"""
    print(f"\n{'='*60}")
    print("TESTING RESPONSE STYLES")
    print(f"{'='*60}")

    test_question = "What do people think about gaming laptops?"

    styles = ["concise", "balanced", "detailed"]

    for style in styles:
        print(f"\n[STYLE: {style.upper()}]")
        print("-" * 40)

        result = pipeline.query(
            question=test_question,
            top_k=5,
            style=style
        )

        print(result['answer'])
        print(f"\nLength: {len(result['answer'])} characters")


def run_all_tests():
    """Run complete test suite"""
    print("="*60)
    print("RAG PIPELINE COMPREHENSIVE TEST SUITE")
    print("="*60)

    # Initialize pipeline
    print("\n[SETUP] Initializing RAG pipeline...")
    start_init = time.time()
    pipeline = RAGPipeline(verbose=True)
    init_time = time.time() - start_init
    print(f"[OK] Pipeline initialized in {init_time:.1f}s\n")

    # Test connections
    print("\n[SETUP] Testing connections...")
    if not pipeline.test_connection():
        print("\n[ERROR] Connection tests failed! Check your configuration.")
        return

    # Test individual queries
    print(f"\n{'='*60}")
    print("TESTING DIVERSE QUERIES")
    print(f"{'='*60}")

    results = []
    for i, test_case in enumerate(TEST_QUERIES, 1):
        result = test_individual_query(pipeline, test_case, i)
        results.append(result)

        # Brief pause to avoid rate limiting
        time.sleep(2)

    # Test filters
    test_filters(pipeline)

    # Test response styles
    test_response_styles(pipeline)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_posts_retrieved = sum(r['posts_retrieved'] for r in successful) / len(successful)
        avg_posts_used = sum(r['posts_used'] for r in successful) / len(successful)
        avg_query_time = sum(r['query_time'] for r in successful) / len(successful)
        pct_with_citations = sum(1 for r in successful if r['has_citations']) / len(successful) * 100

        print(f"\nAverage posts retrieved: {avg_posts_retrieved:.1f}")
        print(f"Average posts used: {avg_posts_used:.1f}")
        print(f"Average query time: {avg_query_time:.2f}s")
        print(f"Responses with citations: {pct_with_citations:.1f}%")

    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - Test {r['test_num']}: {r['question']}")
            print(f"    Error: {r.get('error', 'Unknown')}")

    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():
    """
    Interactive mode for manual testing

    Allows you to ask questions and see results in real-time
    """
    print("="*60)
    print("RAG PIPELINE - INTERACTIVE MODE")
    print("="*60)
    print("\nInitializing pipeline...")

    pipeline = RAGPipeline(verbose=False)

    print("\nPipeline ready! Type 'quit' to exit, 'help' for commands.\n")

    while True:
        try:
            # Get question from user
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if question.lower() == 'help':
                print("\nCommands:")
                print("  help     - Show this help")
                print("  quit     - Exit interactive mode")
                print("  stats    - Show database statistics")
                print("\nJust type your question to get an answer!")
                continue

            if question.lower() == 'stats':
                stats = pipeline.get_stats()
                print("\nDatabase Statistics:")
                print(f"  Total posts: {stats.get('total_posts', 0):,}")
                print(f"  Posts with embeddings: {stats.get('posts_with_embeddings', 0):,}")
                continue

            # Process question
            print("\nThinking...")
            result = pipeline.query(question, top_k=10, style="balanced")

            print(f"\n{'-'*60}")
            print("ANSWER:")
            print(f"{'-'*60}")
            print(result['answer'])

            print(f"\n{'-'*60}")
            print(f"Sources: {len(result['sources'])} posts | "
                  f"Citations: {result['metadata']['has_citations']}")
            print(f"{'-'*60}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Testing")
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive', 'quick'],
        default='interactive',  # Changed from 'test' to 'interactive'
        help='Test mode: test=full suite, interactive=manual queries, quick=single query'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question for quick mode'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        # Run full test suite
        run_all_tests()

    elif args.mode == 'interactive':
        # Interactive mode
        interactive_mode()

    elif args.mode == 'quick':
        # Quick single query
        if not args.question:
            print("Please provide a --question for quick mode")
            print("Example: python test_rag.py --mode quick --question 'What do people think about iPhone?'")
        else:
            print("Initializing pipeline...")
            answer = quick_query(args.question)
            print("\n" + answer)
