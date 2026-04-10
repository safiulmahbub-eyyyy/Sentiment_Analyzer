"""
Query Classifier Module
Classifies user queries by intent before RAG pipeline execution
"""

import re
from typing import Literal, Dict, Any

# Type alias for query types
QueryType = Literal["meta", "greeting", "product_sentiment", "out_of_scope"]


# ============================================================
# KEYWORD PATTERNS
# ============================================================

# Meta questions about the chatbot itself
META_PATTERNS = [
    r'\bwhat can you (do|help|tell|answer|provide)\b',
    r'\bwhat (are|is) (you|your) (capabilities|features|functions|purpose)\b',
    r'\bhow (do|does) (you|this|it) work\b',
    r'\bwhat (do|does) (you|this) (do|provide|offer)\b',
    r'\bwhat kind of questions\b',
    r'\bwhat (are|is) (you|this|it)\b',
    r'\bhelp me\b',
    r'\btell me about (yourself|you|this)\b',
    r'\bwho are you\b',
    r'\bwho made you\b',
    r'\bwhat (is|are) your (purpose|function|goal)\b',
]

# Greetings and social interactions
GREETING_PATTERNS = [
    r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)\b',
    r'\bhow are you\b',
    r'\bhow\'s it going\b',
    r'\bwhat\'s up\b',
    r'\bnice to meet\b',
    r'^thanks?\b',
    r'^thank you\b',
    r'^bye\b',
    r'^goodbye\b',
    r'^see you\b',
]

# Product/tech keywords that indicate sentiment queries
PRODUCT_KEYWORDS = [
    # Brands
    'iphone', 'apple', 'samsung', 'google', 'pixel', 'oneplus', 'xiaomi',
    'macbook', 'dell', 'hp', 'lenovo', 'asus', 'acer', 'msi', 'razer',
    'sony', 'microsoft', 'surface', 'alienware', 'thinkpad',

    # Product categories
    'phone', 'smartphone', 'laptop', 'desktop', 'computer', 'tablet', 'ipad',
    'gaming', 'monitor', 'keyboard', 'mouse', 'headphone', 'earbud', 'airpod',
    'console', 'playstation', 'xbox', 'nintendo', 'switch',
    'processor', 'cpu', 'gpu', 'graphics card', 'ram', 'ssd',

    # Features
    'battery', 'camera', 'screen', 'display', 'performance', 'speed',
    'storage', 'memory', 'charging', 'audio', 'speaker', 'microphone',

    # Operating systems
    'ios', 'android', 'windows', 'macos', 'linux',

    # Sentiment/opinion indicators
    'review', 'opinion', 'experience', 'sentiment', 'think', 'feel',
    'recommend', 'worth', 'buy', 'purchase', 'upgrade', 'switch',
    'problem', 'issue', 'bug', 'complaint', 'praise', 'love', 'hate',
    'better', 'worse', 'best', 'worst', 'good', 'bad', 'vs', 'versus', 'compare',
]

# Out of scope topics
OUT_OF_SCOPE_PATTERNS = [
    r'\b(weather|politics|sports|cooking|recipe|movie|film|book|music|song)\b',
    r'\b(medical|health|doctor|medicine|drug|disease)\b',
    r'\b(legal|law|lawyer|court|crime)\b',
    r'\b(financial|investment|stock|crypto|bitcoin)\b',
    r'\bwhat is \d+\s*[\+\-\*\/]',  # Math calculations
    r'\btranslate\b',
    r'\bwrite (a|an|me|my) (essay|paper|report|code|program)\b',
]


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================

def classify_query(question: str) -> Dict[str, Any]:
    """
    Classify a user query by intent

    Args:
        question: User's question string

    Returns:
        Dictionary with:
        - type: QueryType (meta, greeting, product_sentiment, out_of_scope)
        - confidence: float (0-1)
        - reasoning: str (why this classification)

    Example:
        >>> classify_query("What can you do?")
        {'type': 'meta', 'confidence': 0.95, 'reasoning': 'Meta question about capabilities'}

        >>> classify_query("What do people think about iPhone 15?")
        {'type': 'product_sentiment', 'confidence': 0.9, 'reasoning': 'Contains product keywords'}
    """
    question_lower = question.lower().strip()

    # Check for meta questions (highest priority)
    for pattern in META_PATTERNS:
        if re.search(pattern, question_lower):
            return {
                'type': 'meta',
                'confidence': 0.95,
                'reasoning': 'Meta question about chatbot capabilities'
            }

    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, question_lower):
            return {
                'type': 'greeting',
                'confidence': 0.9,
                'reasoning': 'Social greeting or pleasantry'
            }

    # Check for out of scope topics
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, question_lower):
            return {
                'type': 'out_of_scope',
                'confidence': 0.85,
                'reasoning': 'Query outside consumer electronics domain'
            }

    # Check for product/sentiment keywords
    product_keyword_count = sum(
        1 for keyword in PRODUCT_KEYWORDS
        if keyword in question_lower
    )

    if product_keyword_count > 0:
        # Strong signal for product sentiment query
        confidence = min(0.95, 0.6 + (product_keyword_count * 0.1))
        return {
            'type': 'product_sentiment',
            'confidence': confidence,
            'reasoning': f'Contains {product_keyword_count} product/sentiment keywords'
        }

    # Check for question words indicating information seeking
    question_words = ['what', 'how', 'why', 'which', 'when', 'who', 'where', 'is', 'are', 'do', 'does']
    has_question_word = any(word in question_lower.split()[:3] for word in question_words)

    if has_question_word:
        # Likely a product question but with lower confidence
        return {
            'type': 'product_sentiment',
            'confidence': 0.5,
            'reasoning': 'Question format, assumed to be about products'
        }

    # Default: assume it's a product question but with low confidence
    return {
        'type': 'product_sentiment',
        'confidence': 0.3,
        'reasoning': 'Default classification - no clear signals detected'
    }


def is_meta_question(question: str) -> bool:
    """
    Quick check if query is a meta question

    Args:
        question: User's question

    Returns:
        True if meta question, False otherwise
    """
    result = classify_query(question)
    return result['type'] == 'meta'


def is_greeting(question: str) -> bool:
    """
    Quick check if query is a greeting

    Args:
        question: User's question

    Returns:
        True if greeting, False otherwise
    """
    result = classify_query(question)
    return result['type'] == 'greeting'


def is_product_question(question: str) -> bool:
    """
    Quick check if query is about products

    Args:
        question: User's question

    Returns:
        True if product question, False otherwise
    """
    result = classify_query(question)
    return result['type'] == 'product_sentiment'


def should_use_rag_pipeline(question: str, min_confidence: float = 0.4) -> bool:
    """
    Determine if query should go through RAG pipeline

    Only product_sentiment queries with sufficient confidence should use RAG

    Args:
        question: User's question
        min_confidence: Minimum confidence threshold

    Returns:
        True if should use RAG, False otherwise
    """
    result = classify_query(question)

    # Only product sentiment queries go through RAG
    if result['type'] != 'product_sentiment':
        return False

    # Check confidence threshold
    return result['confidence'] >= min_confidence


# ============================================================
# EXAMPLE QUESTION SUGGESTIONS
# ============================================================

EXAMPLE_QUESTIONS = [
    # General sentiment
    "What do people think about iPhone 15?",
    "Are gaming laptops worth it?",
    "What's the general opinion on Samsung phones?",

    # Feature-specific
    "How is the MacBook battery life?",
    "What do people say about iPhone camera quality?",
    "Are mechanical keyboards better for gaming?",

    # Comparisons
    "Should I buy iPhone or Android?",
    "Which is better: laptop or desktop for gaming?",
    "iPad vs Android tablet for students?",

    # Issues/problems
    "What are common problems with gaming laptops?",
    "Do people have heating issues with MacBooks?",
    "Are there battery drain issues with iPhone 15?",

    # Recommendations
    "What's the best budget smartphone?",
    "Which laptop is best for college students?",
    "What gaming mouse do people recommend?",
]


def get_example_questions(count: int = 5) -> list[str]:
    """
    Get random example questions to show users

    Args:
        count: Number of examples to return

    Returns:
        List of example question strings
    """
    import random
    return random.sample(EXAMPLE_QUESTIONS, min(count, len(EXAMPLE_QUESTIONS)))


def get_similar_example_questions(question: str, count: int = 3) -> list[str]:
    """
    Get example questions similar to the failed query

    Args:
        question: User's question that failed
        count: Number of suggestions

    Returns:
        List of suggested questions
    """
    question_lower = question.lower()

    # Extract keywords from failed question
    keywords = [kw for kw in PRODUCT_KEYWORDS if kw in question_lower]

    # Find example questions with overlapping keywords
    scored_examples = []
    for example in EXAMPLE_QUESTIONS:
        example_lower = example.lower()
        overlap_count = sum(1 for kw in keywords if kw in example_lower)
        if overlap_count > 0:
            scored_examples.append((overlap_count, example))

    # Sort by overlap and return top results
    if scored_examples:
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [ex for _, ex in scored_examples[:count]]

    # Fallback: return random examples
    return get_example_questions(count)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the query classifier"""
    print("="*60)
    print("QUERY CLASSIFIER TEST")
    print("="*60)

    # Test cases
    test_queries = [
        "What can you do?",
        "Hello!",
        "What do people think about iPhone 15?",
        "Are gaming laptops worth it?",
        "What's the weather today?",
        "Tell me about yourself",
        "Should I buy a MacBook or Dell laptop?",
        "How is the battery life on Samsung phones?",
        "Thanks for your help",
        "What is 2 + 2?",
        "smartphone camera quality",
        "best budget phone",
    ]

    print("\nClassifying test queries:\n")

    for i, query in enumerate(test_queries, 1):
        result = classify_query(query)
        use_rag = should_use_rag_pipeline(query)

        print(f"{i}. \"{query}\"")
        print(f"   Type: {result['type']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Use RAG: {use_rag}")
        print()

    # Test example suggestions
    print("\n" + "="*60)
    print("EXAMPLE QUESTIONS")
    print("="*60)
    examples = get_example_questions(5)
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")

    # Test similar suggestions
    print("\n" + "="*60)
    print("SIMILAR QUESTION SUGGESTIONS")
    print("="*60)
    failed_query = "Tell me about quantum computers"
    print(f"Failed query: \"{failed_query}\"")
    print("Suggestions:")
    suggestions = get_similar_example_questions(failed_query, 3)
    for i, sug in enumerate(suggestions, 1):
        print(f"  {i}. {sug}")

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
