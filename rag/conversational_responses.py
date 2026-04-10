"""
Conversational Response Handler
Handles meta-questions, greetings, and non-RAG responses
"""

from typing import Dict, Any, List
from rag.query_classifier import get_example_questions, get_similar_example_questions


# ============================================================
# META QUESTION RESPONSES
# ============================================================

def handle_meta_question(question: str) -> Dict[str, Any]:
    """
    Generate response for meta questions about the chatbot

    Args:
        question: User's meta question

    Returns:
        Response dictionary with answer and metadata
    """
    answer = """I'm a sentiment analysis assistant specialized in consumer electronics!

**What I Can Do:**
- Answer questions about consumer sentiment on electronics products (phones, laptops, gaming devices, etc.)
- Analyze Reddit discussions from 20+ tech subreddits
- Compare products based on real user experiences
- Identify common issues and problems users face
- Provide recommendations based on community sentiment

**How I Work:**
I search through 38,000+ Reddit posts with sentiment analysis, find the most relevant discussions using AI, and synthesize insights to answer your questions.

**Example Questions You Can Ask:**
- "What do people think about iPhone 15?"
- "Are gaming laptops worth it?"
- "Should I buy MacBook or Dell laptop for college?"
- "What are common problems with mechanical keyboards?"
- "How is the Samsung phone battery life?"

**Data Sources:**
r/iphone, r/apple, r/samsung, r/android, r/laptops, r/buildapc, r/pcmasterrace, r/gaming, r/mechanicalkeyboards, and more!

Ask me anything about consumer electronics sentiment!"""

    return {
        'answer': answer,
        'sources': [],
        'metadata': {
            'query_type': 'meta',
            'conversational': True,
            'has_citations': False,
            'example_questions': get_example_questions(5)
        }
    }


# ============================================================
# GREETING RESPONSES
# ============================================================

GREETING_TEMPLATES = [
    "Hello! I'm your consumer electronics sentiment assistant. Ask me about phones, laptops, gaming gear, or any tech product!",
    "Hi there! I analyze Reddit discussions about consumer electronics. What would you like to know about?",
    "Hey! Ready to help you understand what people think about tech products. What's on your mind?",
    "Greetings! I specialize in analyzing consumer sentiment on electronics. How can I help you today?",
]

THANKS_TEMPLATES = [
    "You're welcome! Feel free to ask me anything else about consumer electronics!",
    "Happy to help! Let me know if you have more questions about tech products.",
    "Glad I could assist! Ask me anything about electronics sentiment.",
]

GOODBYE_TEMPLATES = [
    "Goodbye! Come back anytime you need insights on consumer electronics!",
    "See you later! Feel free to return whenever you have tech questions.",
    "Take care! I'll be here whenever you need sentiment analysis on electronics!",
]


def handle_greeting(question: str) -> Dict[str, Any]:
    """
    Generate friendly response for greetings

    Args:
        question: User's greeting

    Returns:
        Response dictionary with answer
    """
    import random

    question_lower = question.lower().strip()

    # Choose appropriate response based on greeting type
    if any(word in question_lower for word in ['thanks', 'thank you']):
        answer = random.choice(THANKS_TEMPLATES)
    elif any(word in question_lower for word in ['bye', 'goodbye', 'see you']):
        answer = random.choice(GOODBYE_TEMPLATES)
    else:
        answer = random.choice(GREETING_TEMPLATES)

    return {
        'answer': answer,
        'sources': [],
        'metadata': {
            'query_type': 'greeting',
            'conversational': True,
            'has_citations': False
        }
    }


# ============================================================
# OUT OF SCOPE RESPONSES
# ============================================================

def handle_out_of_scope(question: str) -> Dict[str, Any]:
    """
    Handle queries outside the system's domain

    Args:
        question: User's out-of-scope question

    Returns:
        Response dictionary explaining limitations
    """
    # Get example questions
    example_questions_text = '\n'.join(f'- "{q}"' for q in get_example_questions(3))

    answer = f"""I'm specialized in analyzing consumer sentiment about electronics products based on Reddit discussions.

Your question seems to be outside my expertise area (consumer electronics).

**I Can Help With:**
- Smartphones (iPhone, Samsung, Google Pixel, etc.)
- Laptops & desktops (MacBook, Dell, HP, gaming rigs, etc.)
- Gaming devices (consoles, keyboards, mice, monitors)
- Audio equipment (headphones, earbuds, speakers)
- Tablets and accessories

**Try Asking:**
{example_questions_text}

Would you like to know about any consumer electronics products?"""

    return {
        'answer': answer,
        'sources': [],
        'metadata': {
            'query_type': 'out_of_scope',
            'conversational': True,
            'has_citations': False,
            'suggestion_questions': get_example_questions(3)
        }
    }


# ============================================================
# NO RESULTS FOUND HANDLER
# ============================================================

def handle_no_results(question: str, attempted_filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate helpful response when no Reddit posts match the query

    Args:
        question: User's question that found no results
        attempted_filters: Filters that were applied during search

    Returns:
        Response dictionary with suggestions
    """
    # Get similar questions that might work
    similar_questions = get_similar_example_questions(question, count=3)

    # Format similar questions
    similar_questions_text = '\n'.join(f'{i+1}. "{q}"' for i, q in enumerate(similar_questions))

    answer = f"""I couldn't find relevant Reddit discussions matching your question: "{question}"

**This Could Mean:**
- The specific product/topic hasn't been discussed much in our database
- Try rephrasing with different keywords
- The topic might be too specific or too new

**Try These Similar Questions Instead:**
{similar_questions_text}

**Or Ask About:**
- General product categories (e.g., "gaming laptops" vs specific model)
- Popular brands (iPhone, Samsung, MacBook, Dell, etc.)
- Common features (battery life, camera quality, performance)
- General sentiment or comparisons

**Database Coverage:**
38,000+ Reddit posts from 20 tech subreddits, focusing on consumer electronics.

Would you like to try a different question?"""

    # Add filter information if provided
    if attempted_filters:
        filter_info = []
        if attempted_filters.get('subreddit_filter'):
            filter_info.append(f"Subreddit: r/{attempted_filters['subreddit_filter']}")
        if attempted_filters.get('sentiment_filter'):
            filter_info.append(f"Sentiment: {attempted_filters['sentiment_filter']}")
        if attempted_filters.get('days_ago'):
            filter_info.append(f"Time range: Last {attempted_filters['days_ago']} days")

        if filter_info:
            answer += f"\n\n**Applied Filters:**\n" + '\n'.join(f'- {f}' for f in filter_info)
            answer += "\n\nTry removing some filters to get more results."

    return {
        'answer': answer,
        'sources': [],
        'metadata': {
            'query_type': 'no_results',
            'conversational': True,
            'has_citations': False,
            'original_question': question,
            'suggestion_questions': similar_questions,
            'filters_applied': attempted_filters or {}
        }
    }


# ============================================================
# LOW CONFIDENCE HANDLER
# ============================================================

def handle_low_confidence_query(question: str, confidence: float) -> Dict[str, Any]:
    """
    Handle queries with low confidence classification

    Args:
        question: User's question
        confidence: Classification confidence score

    Returns:
        Response dictionary with clarification request
    """
    # Get example questions
    example_questions_text = '\n'.join(f'- "{q}"' for q in get_example_questions(3))

    answer = f"""I'm not quite sure what you're asking about: "{question}"

**To Help Me Understand Better:**
Could you rephrase your question to include:
- Specific product names (e.g., iPhone 15, MacBook Pro, Samsung Galaxy)
- Product categories (e.g., gaming laptop, smartphone, mechanical keyboard)
- Features you're interested in (e.g., battery life, camera, performance)

**Example Questions:**
{example_questions_text}

**Or Try:**
- Asking about specific brands or models
- Comparing two products (e.g., "iPhone vs Samsung")
- Asking about common issues or features

I'm here to help with consumer electronics sentiment analysis!"""

    return {
        'answer': answer,
        'sources': [],
        'metadata': {
            'query_type': 'low_confidence',
            'conversational': True,
            'has_citations': False,
            'confidence': confidence,
            'suggestion_questions': get_example_questions(3)
        }
    }


# ============================================================
# MAIN HANDLER
# ============================================================

def generate_conversational_response(
    query_type: str,
    question: str,
    classification_confidence: float = 1.0,
    additional_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Main entry point for generating conversational responses

    Args:
        query_type: Type of query (meta, greeting, out_of_scope, no_results, low_confidence)
        question: User's question
        classification_confidence: Confidence score from classifier
        additional_context: Any additional context (filters, etc.)

    Returns:
        Response dictionary with answer and metadata
    """
    additional_context = additional_context or {}

    # Route to appropriate handler
    if query_type == 'meta':
        return handle_meta_question(question)

    elif query_type == 'greeting':
        return handle_greeting(question)

    elif query_type == 'out_of_scope':
        return handle_out_of_scope(question)

    elif query_type == 'no_results':
        return handle_no_results(question, additional_context.get('filters'))

    elif query_type == 'low_confidence':
        return handle_low_confidence_query(question, classification_confidence)

    else:
        # Fallback
        return handle_meta_question(question)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test conversational responses"""
    print("="*60)
    print("CONVERSATIONAL RESPONSES TEST")
    print("="*60)

    # Test cases
    test_cases = [
        ('meta', "What can you do?"),
        ('greeting', "Hello!"),
        ('greeting', "Thanks for your help!"),
        ('greeting', "Goodbye!"),
        ('out_of_scope', "What's the weather today?"),
        ('no_results', "What do people think about quantum computers?"),
        ('low_confidence', "stuff things products"),
    ]

    for query_type, question in test_cases:
        print(f"\n{'='*60}")
        print(f"Type: {query_type}")
        print(f"Question: \"{question}\"")
        print(f"{'='*60}")

        result = generate_conversational_response(
            query_type=query_type,
            question=question,
            classification_confidence=0.85
        )

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nMetadata: {result['metadata']}")

    # Test no results with filters
    print(f"\n{'='*60}")
    print("Testing no_results with filters")
    print(f"{'='*60}")

    result = generate_conversational_response(
        query_type='no_results',
        question="What do people think about Nokia phones?",
        additional_context={
            'filters': {
                'subreddit_filter': 'iphone',
                'sentiment_filter': 'positive',
                'days_ago': 30
            }
        }
    )

    print(f"\nAnswer:\n{result['answer']}")

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
