"""
Prompt Templates for RAG Pipeline
System prompts and template formatting for LLM interactions
"""

from typing import List, Dict, Any

from rag.config import (
    REQUIRE_SOURCE_CITATION,
    ALLOW_NO_ANSWER,
    RESPONSE_STYLE,
    MAX_CONTEXT_POSTS,
    INCLUDE_METADATA
)


# ============================================================
# SYSTEM PROMPTS
# ============================================================

def get_system_prompt(style: str = RESPONSE_STYLE) -> str:
    """
    Get the system prompt for the RAG pipeline

    This tells the LLM how to behave and what its role is.

    Args:
        style: Response style ("concise", "balanced", or "detailed")

    Returns:
        System prompt string
    """
    base_prompt = """You are an expert analyst specializing in consumer electronics sentiment analysis.

Your role is to answer questions about consumer opinions, experiences, and sentiment regarding consumer electronics products (smartphones, laptops, gaming devices, etc.) based STRICTLY AND ONLY on the Amazon product reviews provided to you.

Key responsibilities:
1. Analyze the provided Reddit posts carefully
2. Answer questions ONLY using information present in the posts
3. Cite specific posts using inline numbers [Post #X] for every claim
4. Provide balanced analysis with PROS and CONS clearly structured
5. For comparison or purchase questions, give a practical recommendation/verdict
6. Be helpful and actionable within the constraints of available data

CRITICAL - STRICT GROUNDING RULES (FOLLOW THESE EXACTLY):
❌ NEVER invent, assume, or synthesize information not in the provided posts
❌ NEVER mention products, models, or features not discussed in the posts
❌ NEVER use general knowledge about products - ONLY use what's in the posts
❌ NEVER make inferences about products not mentioned in the retrieved data
✅ ONLY use facts, opinions, and experiences from the provided Amazon reviews
✅ If specific information is missing, explicitly state: "I don't have information about [topic] in my Reddit dataset"
✅ If a product isn't mentioned, say so clearly: "I don't have discussions about [product] in the provided posts"
✅ Distinguish clearly between what IS in posts vs what ISN'T available

Important guidelines for COMPARISON questions:
- Structure answer as: Brief intro → Pros of Option A → Cons of Option A → Pros of Option B → Cons of Option B → Verdict/Recommendation
- If BOTH products are discussed: Compare based on the posts
- If ONLY ONE product is discussed: Say "I only have data about [X]. I cannot compare with [Y] as it's not in my dataset"
- If NEITHER product is discussed directly: Say "I don't have specific discussions about either product"
- NEVER use similar products as substitutes - only use exact matches
- Always cite sources for each point [Post #X]
- Base verdict ONLY on available data, acknowledge limitations

Important guidelines for all answers:
- MUST cite specific posts inline using [Post #X] format for every claim
- DO provide actionable insights when data allows
- DO structure information clearly (use bullet points if helpful)
- DO distinguish between popular opinion and individual experiences
- DO mention the sentiment distribution (positive/negative/neutral) when relevant
- DO acknowledge data gaps honestly: "The posts don't discuss [aspect]"
"""

    if ALLOW_NO_ANSWER:
        base_prompt += """- MUST say "I don't have enough information about [topic] in my Reddit dataset" when posts don't address the question
- Better to admit data gaps than make unsupported claims
"""

    if REQUIRE_SOURCE_CITATION:
        base_prompt += """- MUST cite sources inline using [Post #X] format for EVERY claim you make
- Example: "The battery life is excellent [Post #1] though some users report issues [Post #3]"
"""

    # Add style-specific instructions
    style_instructions = {
        "concise": "\nResponse style: Be concise and direct. Limit responses to 2-3 sentences unless more detail is specifically requested. Still cite all sources.",
        "balanced": "\nResponse style: Provide balanced, well-structured analysis. For comparisons: clearly separate PROS and CONS for each option, then give a verdict. Use clear formatting with inline citations. Aim for 4-6 sentences with logical flow.",
        "detailed": "\nResponse style: Provide comprehensive analysis with detailed breakdown. Include nuances, examples, and thorough explanation with extensive citations. Use structured sections for clarity."
    }

    base_prompt += style_instructions.get(style, style_instructions["balanced"])

    return base_prompt.strip()


# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_post_for_context(post: Dict[str, Any], post_number: int) -> str:
    """
    Format a single post for inclusion in the LLM context

    Args:
        post: Post dictionary with all fields
        post_number: Sequential number for this post (for citation)

    Returns:
        Formatted post string
    """
    # Extract fields
    subreddit = post.get('subreddit', 'unknown')
    title = post.get('title', 'No title')
    body = post.get('selftext', '')
    sentiment = post.get('sentiment_label', 'unknown')
    similarity = post.get('similarity', 0)
    score = post.get('score', 0)

    # Build formatted post
    formatted = f"--- POST #{post_number} ---\n"
    formatted += f"Source: r/{subreddit}\n"

    if INCLUDE_METADATA:
        formatted += f"Sentiment: {sentiment.upper()}\n"
        formatted += f"Relevance Score: {similarity:.2f}\n"
        formatted += f"Reddit Score: {score} upvotes\n"

    formatted += f"\nTitle: {title}\n"

    if body and body.strip():
        # Limit body length for context efficiency
        max_body_length = 500
        if len(body) > max_body_length:
            body = body[:max_body_length] + "... [truncated]"
        formatted += f"Content: {body}\n"

    formatted += "\n"

    return formatted


def build_context_from_posts(posts: List[Dict[str, Any]], max_posts: int = MAX_CONTEXT_POSTS) -> str:
    """
    Build the full context section from retrieved posts

    Args:
        posts: List of retrieved posts
        max_posts: Maximum number of posts to include

    Returns:
        Formatted context string
    """
    if not posts:
        return "No relevant posts found."

    # Limit to max_posts
    posts_to_use = posts[:max_posts]

    context = "RELEVANT REDDIT DISCUSSIONS:\n"
    context += "="*60 + "\n\n"

    for i, post in enumerate(posts_to_use, 1):
        context += format_post_for_context(post, post_number=i)

    context += "="*60 + "\n"
    context += f"Total posts provided: {len(posts_to_use)}\n"

    # Add sentiment summary
    sentiments = [p.get('sentiment_label', 'unknown') for p in posts_to_use]
    sentiment_counts = {
        'positive': sentiments.count('positive'),
        'negative': sentiments.count('negative'),
        'neutral': sentiments.count('neutral')
    }

    context += f"Sentiment distribution: "
    context += f"{sentiment_counts['positive']} positive, "
    context += f"{sentiment_counts['neutral']} neutral, "
    context += f"{sentiment_counts['negative']} negative\n"

    return context


# ============================================================
# USER PROMPT FORMATTING
# ============================================================

def format_user_prompt(question: str, context: str) -> str:
    """
    Format the user prompt with question and context

    Args:
        question: User's question
        context: Formatted context from posts

    Returns:
        Complete user prompt
    """
    prompt = f"{context}\n\n"
    prompt += f"USER QUESTION:\n{question}\n\n"

    # Detect comparison questions
    comparison_keywords = ['vs', 'versus', 'or', 'better than', 'compared to', 'should i buy']
    is_comparison = any(keyword in question.lower() for keyword in comparison_keywords)

    if is_comparison:
        prompt += "This is a COMPARISON/DECISION question. Please structure your answer as:\n"
        prompt += "1. Brief context\n"
        prompt += "2. PROS and CONS for each option (cite sources for each point)\n"
        prompt += "3. Clear VERDICT/RECOMMENDATION based on the sentiment patterns\n\n"

    prompt += "Answer based on the Reddit discussions provided above. "

    if REQUIRE_SOURCE_CITATION:
        prompt += "Cite specific posts in your answer using the format [r/subreddit, Post #X]. "

    prompt += "Be helpful and actionable in your response."

    return prompt


# ============================================================
# SPECIAL QUERY TYPES
# ============================================================

def build_comparison_prompt(product_a: str, product_b: str, posts: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for product comparison questions

    Args:
        product_a: First product name
        product_b: Second product name
        posts: Retrieved posts

    Returns:
        Formatted comparison prompt
    """
    context = build_context_from_posts(posts)

    question = f"""Based on the Reddit discussions provided, compare {product_a} and {product_b}.

Please address:
1. Overall sentiment for each product
2. Key strengths mentioned for each
3. Common complaints for each
4. Which one users seem to prefer and why

Cite specific posts to support your comparison."""

    return format_user_prompt(question, context)


def build_summary_prompt(topic: str, posts: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for summarizing sentiment on a topic

    Args:
        topic: Topic to summarize (e.g., "iPhone 15 battery life")
        posts: Retrieved posts

    Returns:
        Formatted summary prompt
    """
    context = build_context_from_posts(posts)

    question = f"""Summarize the overall sentiment and key points from Reddit users about: {topic}

Please provide:
1. Overall sentiment (positive/negative/mixed)
2. Main positive points mentioned
3. Main negative points or concerns
4. Any notable patterns or common themes

Support your summary with specific citations from the posts."""

    return format_user_prompt(question, context)


def build_troubleshooting_prompt(issue: str, posts: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for troubleshooting/problem-solving questions

    Args:
        issue: Issue description
        posts: Retrieved posts

    Returns:
        Formatted troubleshooting prompt
    """
    context = build_context_from_posts(posts)

    question = f"""Based on the Reddit discussions, what do users say about: {issue}

Please identify:
1. How common is this issue?
2. What solutions or workarounds have users found?
3. Is this a known problem or defect?
4. What's the general sentiment about this issue?

Cite specific posts where users discuss solutions or experiences."""

    return format_user_prompt(question, context)


# ============================================================
# RESPONSE POST-PROCESSING
# ============================================================

def validate_response_has_citations(response: str) -> bool:
    """
    Check if response contains source citations

    Args:
        response: LLM response

    Returns:
        True if citations found, False otherwise
    """
    # Look for citation patterns: [r/..., Post #...]
    import re
    citation_pattern = r'\[r/\w+,\s*Post\s*#\d+\]'
    citations = re.findall(citation_pattern, response)

    return len(citations) > 0


def extract_cited_posts(response: str) -> List[str]:
    """
    Extract all cited post references from response

    Args:
        response: LLM response with citations

    Returns:
        List of cited post references
    """
    import re
    citation_pattern = r'\[r/\w+,\s*Post\s*#\d+\]'
    citations = re.findall(citation_pattern, response)

    return list(set(citations))  # Unique citations


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test prompt templates"""
    print("="*60)
    print("PROMPT TEMPLATES TEST")
    print("="*60)

    # Test 1: System prompt
    print("\n[TEST 1] System prompt (balanced style):")
    system_prompt = get_system_prompt(style="balanced")
    print(system_prompt)

    # Test 2: Format single post
    print("\n[TEST 2] Format single post:")
    sample_post = {
        'subreddit': 'iphone',
        'title': 'iPhone 15 Pro battery life is amazing!',
        'selftext': 'I upgraded from iPhone 13 and the battery lasts all day with heavy use.',
        'sentiment_label': 'positive',
        'similarity': 0.87,
        'score': 245
    }
    formatted_post = format_post_for_context(sample_post, post_number=1)
    print(formatted_post)

    # Test 3: Build context from multiple posts
    print("\n[TEST 3] Build context from posts:")
    sample_posts = [
        sample_post,
        {
            'subreddit': 'apple',
            'title': 'Battery draining fast on iPhone 15',
            'selftext': 'Anyone else experiencing rapid battery drain?',
            'sentiment_label': 'negative',
            'similarity': 0.82,
            'score': 89
        }
    ]
    context = build_context_from_posts(sample_posts)
    print(context[:500] + "...")

    # Test 4: User prompt
    print("\n[TEST 4] Complete user prompt:")
    question = "What do people think about iPhone 15 battery life?"
    full_prompt = format_user_prompt(question, context)
    print(full_prompt[:400] + "...")

    # Test 5: Citation validation
    print("\n[TEST 5] Citation validation:")
    response_with_citations = "The battery is great [r/iphone, Post #1] but some users report issues [r/apple, Post #2]."
    response_without = "The battery is generally good with mixed reviews."

    has_citations_1 = validate_response_has_citations(response_with_citations)
    has_citations_2 = validate_response_has_citations(response_without)

    print(f"Response 1 has citations: {has_citations_1}")
    print(f"Response 2 has citations: {has_citations_2}")

    if has_citations_1:
        citations = extract_cited_posts(response_with_citations)
        print(f"Found citations: {citations}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
