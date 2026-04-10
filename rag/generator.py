"""
Response Generator Module
Combines retrieved posts with LLM to generate answers
"""

from typing import List, Dict, Any, Optional

from rag.groq_client import generate_completion, generate_completion_streaming
from rag.prompt_templates import (
    get_system_prompt,
    build_context_from_posts,
    format_user_prompt,
    validate_response_has_citations,
    extract_cited_posts
)
from rag.config import (
    RESPONSE_STYLE,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_CONTEXT_POSTS,
    REQUIRE_SOURCE_CITATION,
    VERBOSE,
    DEBUG_MODE
)


def generate_answer(
    question: str,
    retrieved_posts: List[Dict[str, Any]],
    style: str = RESPONSE_STYLE,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_context_posts: int = MAX_CONTEXT_POSTS,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Generate an answer to a question using retrieved posts with optional conversation history

    This is the core function that:
    1. Builds context from retrieved posts
    2. Formats prompts
    3. Calls LLM with conversation history (for follow-up questions)
    4. Returns structured response

    Args:
        question: User's question
        retrieved_posts: Posts from vector search
        style: Response style ("concise", "balanced", "detailed")
        temperature: LLM temperature
        max_tokens: Maximum response length
        max_context_posts: Maximum posts to include in context
        conversation_history: Optional list of previous conversation turns
                            [{"role": "user/assistant", "content": "..."}]
                            Enables Perplexity-style follow-up questions

    Returns:
        Dictionary with:
        - answer: Generated answer text
        - sources: List of source posts used
        - metadata: Additional information (tokens, citations, etc.)

    Example (single turn):
        >>> posts = retrieve_similar_posts(query_embedding, top_k=10)
        >>> result = generate_answer("What do people think about iPhone?", posts)
        >>> print(result['answer'])
        Based on the discussions, users generally...

    Example (follow-up with history):
        >>> history = [
        ...     {"role": "user", "content": "Which laptop is best?"},
        ...     {"role": "assistant", "content": "Dell XPS is popular..."}
        ... ]
        >>> result = generate_answer("What about cheaper ones?", posts, conversation_history=history)
    """
    if VERBOSE:
        print(f"[GENERATOR] Generating answer for: '{question[:80]}...'")
        print(f"[GENERATOR] Using {len(retrieved_posts)} retrieved posts")

    # Handle case with no posts
    if not retrieved_posts:
        return {
            'answer': "I don't have any relevant information to answer this question. "
                     "No Reddit discussions matched your query.",
            'sources': [],
            'metadata': {
                'posts_used': 0,
                'has_citations': False,
                'style': style
            }
        }

    # Build context from posts
    context = build_context_from_posts(retrieved_posts, max_posts=max_context_posts)

    # Get system prompt
    system_prompt = get_system_prompt(style=style)

    # Format user prompt
    user_prompt = format_user_prompt(question, context)

    # Debug output
    if DEBUG_MODE:
        print("\n" + "="*60)
        print("DEBUG: FULL PROMPT")
        print("="*60)
        print("\nSYSTEM PROMPT:")
        print(system_prompt)
        print("\nUSER PROMPT:")
        print(user_prompt)
        print("="*60 + "\n")

    # Generate completion
    if VERBOSE:
        print(f"[GENERATOR] Calling LLM (style: {style}, temp: {temperature})...")

    try:
        answer = generate_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if VERBOSE:
            if conversation_history:
                print(f"[GENERATOR] Answer generated with history ({len(answer)} characters)")
            else:
                print(f"[GENERATOR] Answer generated ({len(answer)} characters)")

    except Exception as e:
        print(f"[GENERATOR] Error generating answer: {e}")
        return {
            'answer': f"Error generating response: {str(e)}",
            'sources': [],
            'metadata': {
                'error': str(e),
                'posts_used': len(retrieved_posts[:max_context_posts])
            }
        }

    # Validate citations
    has_citations = validate_response_has_citations(answer)
    cited_posts = extract_cited_posts(answer) if has_citations else []

    if REQUIRE_SOURCE_CITATION and not has_citations:
        if VERBOSE:
            print("[GENERATOR] Warning: Response lacks citations")

    # Prepare sources
    sources = retrieved_posts[:max_context_posts]

    # Build result
    result = {
        'answer': answer,
        'sources': sources,
        'metadata': {
            'question': question,
            'posts_used': len(sources),
            'total_posts_retrieved': len(retrieved_posts),
            'has_citations': has_citations,
            'citations': cited_posts,
            'style': style,
            'temperature': temperature
        }
    }

    if DEBUG_MODE:
        print("\n" + "="*60)
        print("DEBUG: GENERATED ANSWER")
        print("="*60)
        print(answer)
        print("="*60 + "\n")

    return result


def generate_answer_with_sources_formatted(
    question: str,
    retrieved_posts: List[Dict[str, Any]],
    **kwargs
) -> str:
    """
    Generate answer and format with sources section

    Returns a nicely formatted string with answer + sources

    Args:
        question: User's question
        retrieved_posts: Retrieved posts
        **kwargs: Additional arguments for generate_answer()

    Returns:
        Formatted string with answer and sources

    Example:
        >>> output = generate_answer_with_sources_formatted(question, posts)
        >>> print(output)
        ANSWER:
        Based on Reddit discussions...

        SOURCES:
        1. [r/iphone] iPhone 15 Pro battery life is amazing!
        ...
    """
    # Generate answer
    result = generate_answer(question, retrieved_posts, **kwargs)

    # Format output
    output = "ANSWER:\n"
    output += "="*60 + "\n"
    output += result['answer'] + "\n\n"

    # Add sources
    if result['sources']:
        output += "SOURCES:\n"
        output += "="*60 + "\n"

        for i, post in enumerate(result['sources'], 1):
            subreddit = post.get('subreddit', 'unknown')
            title = post.get('title', 'No title')
            sentiment = post.get('sentiment_label', 'unknown')
            similarity = post.get('similarity', 0)

            output += f"{i}. [r/{subreddit}] ({sentiment.upper()}, relevance: {similarity:.2f})\n"
            output += f"   {title}\n"

            # Add Reddit link if available
            permalink = post.get('permalink')
            if permalink:
                output += f"   https://reddit.com{permalink}\n"

            output += "\n"

    # Add metadata
    metadata = result['metadata']
    output += "METADATA:\n"
    output += "="*60 + "\n"
    output += f"Posts used: {metadata['posts_used']} / {metadata['total_posts_retrieved']}\n"
    output += f"Has citations: {metadata['has_citations']}\n"
    output += f"Style: {metadata['style']}\n"

    return output


def generate_comparison_answer(
    product_a: str,
    product_b: str,
    posts_a: List[Dict[str, Any]],
    posts_b: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a comparison answer between two products

    Args:
        product_a: First product name
        product_b: Second product name
        posts_a: Posts about product A
        posts_b: Posts about product B
        **kwargs: Additional arguments

    Returns:
        Result dictionary with comparison answer
    """
    # Combine posts
    all_posts = posts_a + posts_b

    # Create comparison question
    question = f"Compare {product_a} and {product_b} based on user sentiment"

    # Generate answer
    return generate_answer(question, all_posts, **kwargs)


def generate_multi_aspect_answer(
    question: str,
    retrieved_posts: List[Dict[str, Any]],
    aspects: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Generate answer covering multiple aspects

    Args:
        question: Base question
        retrieved_posts: Retrieved posts
        aspects: List of aspects to cover (e.g., ["battery", "camera", "performance"])
        **kwargs: Additional arguments

    Returns:
        Result dictionary with multi-aspect answer
    """
    # Enhance question with aspects
    enhanced_question = f"{question}\n\nPlease specifically address these aspects:\n"
    for aspect in aspects:
        enhanced_question += f"- {aspect}\n"

    return generate_answer(enhanced_question, retrieved_posts, **kwargs)


# ============================================================
# ANSWER QUALITY VALIDATION
# ============================================================

def validate_answer_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality of a generated answer

    Checks for:
    - Length (not too short or too long)
    - Citations (if required)
    - Relevance signals
    - Error indicators

    Args:
        result: Result from generate_answer()

    Returns:
        Dictionary with validation results:
        - valid: bool
        - issues: List of issues found
        - score: Quality score (0-1)
    """
    issues = []
    score = 1.0

    answer = result['answer']
    metadata = result['metadata']

    # Check length
    if len(answer) < 50:
        issues.append("Answer too short (< 50 characters)")
        score -= 0.3

    if len(answer) > 2000:
        issues.append("Answer very long (> 2000 characters)")
        score -= 0.1

    # Check citations
    if REQUIRE_SOURCE_CITATION and not metadata.get('has_citations', False):
        issues.append("Missing source citations")
        score -= 0.4

    # Check for error indicators
    error_phrases = [
        "i don't have",
        "i cannot",
        "no information",
        "error:",
        "failed to"
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in error_phrases):
        # This might be legitimate (no info available), not always an issue
        if "error:" in answer_lower or "failed to" in answer_lower:
            issues.append("Contains error message")
            score -= 0.5

    # Check if using sources
    if metadata.get('posts_used', 0) == 0:
        issues.append("No posts used in answer")
        score -= 0.6

    # Ensure score doesn't go negative
    score = max(0.0, score)

    return {
        'valid': score >= 0.5,
        'issues': issues,
        'score': score
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the generator with sample posts"""
    print("="*60)
    print("RESPONSE GENERATOR TEST")
    print("="*60)

    # Create sample posts
    sample_posts = [
        {
            'post_id': '1',
            'subreddit': 'iphone',
            'title': 'iPhone 15 Pro battery life is incredible',
            'selftext': 'Upgraded from iPhone 13, battery lasts all day with heavy use.',
            'sentiment_label': 'positive',
            'sentiment_compound': 0.8,
            'similarity': 0.92,
            'score': 245,
            'num_comments': 89,
            'permalink': '/r/iphone/comments/test1'
        },
        {
            'post_id': '2',
            'subreddit': 'apple',
            'title': 'Battery drain issue on iPhone 15',
            'selftext': 'Anyone else experiencing rapid battery drain after iOS update?',
            'sentiment_label': 'negative',
            'sentiment_compound': -0.6,
            'similarity': 0.88,
            'score': 134,
            'num_comments': 56,
            'permalink': '/r/apple/comments/test2'
        },
        {
            'post_id': '3',
            'subreddit': 'iphonehelp',
            'title': 'iPhone 15 battery tips',
            'selftext': 'Turn off background app refresh and it helps a lot!',
            'sentiment_label': 'neutral',
            'sentiment_compound': 0.3,
            'similarity': 0.85,
            'score': 67,
            'num_comments': 23,
            'permalink': '/r/iphonehelp/comments/test3'
        }
    ]

    # Test question
    question = "What do people think about iPhone 15 battery life?"

    print(f"\nQuestion: {question}")
    print(f"Retrieved posts: {len(sample_posts)}")

    # Test 1: Generate answer
    print("\n[TEST 1] Generating answer...")
    result = generate_answer(
        question=question,
        retrieved_posts=sample_posts,
        style="balanced"
    )

    print("\nAnswer:")
    print(result['answer'])

    print(f"\nMetadata:")
    print(f"  Posts used: {result['metadata']['posts_used']}")
    print(f"  Has citations: {result['metadata']['has_citations']}")
    if result['metadata']['has_citations']:
        print(f"  Citations: {result['metadata']['citations']}")

    # Test 2: Validate quality
    print("\n[TEST 2] Validating answer quality...")
    validation = validate_answer_quality(result)
    print(f"Valid: {validation['valid']}")
    print(f"Score: {validation['score']:.2f}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")

    # Test 3: Formatted output
    print("\n[TEST 3] Formatted output with sources:")
    print("-" * 60)
    formatted = generate_answer_with_sources_formatted(question, sample_posts)
    print(formatted)

    print("="*60)
    print("GENERATOR TESTS COMPLETE")
    print("="*60)
