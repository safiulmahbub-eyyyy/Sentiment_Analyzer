"""
Groq API Client Module
Handles communication with Groq LLM API for response generation
"""

import time
from typing import Optional, Generator, List, Dict
from functools import lru_cache

from groq import Groq

from rag.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    ENABLE_STREAMING,
    MAX_RETRIES,
    RETRY_DELAY,
    CACHE_GROQ_CLIENT,
    VERBOSE
)


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Get or create Groq API client (cached)

    Uses lru_cache to create client once and reuse it
    (Same "heat oven once" pattern as the embedding model!)

    Returns:
        Groq client instance
    """
    if VERBOSE:
        print("[GROQ] Initializing Groq API client...")

    client = Groq(api_key=GROQ_API_KEY)

    if VERBOSE:
        print(f"[GROQ] Client ready! Model: {GROQ_MODEL}")

    return client


def generate_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    top_p: float = TOP_P,
    model: str = GROQ_MODEL
) -> str:
    """
    Generate a completion from Groq API with optional conversation history

    This is the main function for getting LLM responses. Supports multi-turn conversations
    by accepting conversation history (Perplexity-style follow-ups).

    Args:
        prompt: User prompt/question (current turn)
        system_prompt: Optional system instructions
        conversation_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]
                            Should NOT include system prompt (added separately)
                            Example: [{"role": "user", "content": "Which laptop?"},
                                     {"role": "assistant", "content": "Based on..."}]
        temperature: Sampling temperature (0-2, lower = more focused)
        max_tokens: Maximum response length
        top_p: Nucleus sampling parameter (0-1)
        model: Groq model to use

    Returns:
        Generated text response

    Example (single turn):
        >>> response = generate_completion(
        ...     prompt="What is sentiment analysis?",
        ...     system_prompt="You are a helpful AI assistant."
        ... )
        >>> print(response)
        Sentiment analysis is...

    Example (multi-turn with history):
        >>> history = [
        ...     {"role": "user", "content": "Which laptop is best?"},
        ...     {"role": "assistant", "content": "Based on posts, the Dell XPS..."}
        ... ]
        >>> response = generate_completion(
        ...     prompt="What about cheaper ones?",
        ...     system_prompt="You are a helpful AI assistant.",
        ...     conversation_history=history
        ... )
    """
    # Get cached client
    client = get_groq_client()

    # Build messages
    messages = []

    # 1. Add system prompt (always first)
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # 2. Add conversation history (previous turns)
    if conversation_history:
        messages.extend(conversation_history)

    # 3. Add current user prompt
    messages.append({
        "role": "user",
        "content": prompt
    })

    # Make API call with retries
    for attempt in range(MAX_RETRIES):
        try:
            if VERBOSE:
                print(f"[GROQ] Generating completion (attempt {attempt + 1}/{MAX_RETRIES})...")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False  # Non-streaming for simplicity
            )

            # Extract response text
            answer = response.choices[0].message.content

            if VERBOSE:
                tokens = response.usage.total_tokens
                print(f"[GROQ] Completion received ({tokens} tokens)")

            return answer

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"[GROQ] Error: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[GROQ] Failed after {MAX_RETRIES} attempts: {e}")
                raise


def generate_completion_streaming(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    top_p: float = TOP_P,
    model: str = GROQ_MODEL
) -> Generator[str, None, None]:
    """
    Generate a streaming completion from Groq API

    Yields response chunks as they arrive (better UX for chat interfaces!)

    Args:
        prompt: User prompt/question
        system_prompt: Optional system instructions
        temperature: Sampling temperature
        max_tokens: Maximum response length
        top_p: Nucleus sampling parameter
        model: Groq model to use

    Yields:
        Text chunks as they arrive

    Example:
        >>> for chunk in generate_completion_streaming("Tell me about AI"):
        ...     print(chunk, end='', flush=True)
        AI is a branch of computer science...
    """
    # Get cached client
    client = get_groq_client()

    # Build messages
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    # Make streaming API call
    try:
        if VERBOSE:
            print("[GROQ] Generating streaming completion...")

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True  # Enable streaming
        )

        # Yield chunks as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        if VERBOSE:
            print("\n[GROQ] Streaming complete")

    except Exception as e:
        print(f"[GROQ] Streaming error: {e}")
        raise


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text

    Rough approximation: ~4 characters per token
    (Good enough for checking if we're within limits)

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def validate_prompt_length(prompt: str, system_prompt: Optional[str] = None, max_tokens: int = MAX_TOKENS) -> bool:
    """
    Validate that prompt + completion will fit within model context

    Args:
        prompt: User prompt
        system_prompt: System prompt
        max_tokens: Maximum tokens for completion

    Returns:
        True if valid, raises ValueError if too long
    """
    # Most Groq models have 8k-32k context windows
    # We'll use conservative 6k for safety
    MAX_CONTEXT = 6000

    prompt_tokens = count_tokens_estimate(prompt)
    system_tokens = count_tokens_estimate(system_prompt) if system_prompt else 0

    total_input_tokens = prompt_tokens + system_tokens
    total_tokens = total_input_tokens + max_tokens

    if total_tokens > MAX_CONTEXT:
        raise ValueError(
            f"Prompt too long: {total_tokens} tokens (max: {MAX_CONTEXT}). "
            f"Input: {total_input_tokens}, Output: {max_tokens}"
        )

    return True


def test_api_connection() -> bool:
    """
    Test Groq API connection with a simple query

    Returns:
        True if connection successful, False otherwise
    """
    try:
        print("[GROQ] Testing API connection...")

        response = generate_completion(
            prompt="Say 'Hello' if you can hear me.",
            system_prompt="You are a test assistant. Respond with exactly 'Hello'.",
            max_tokens=10
        )

        if "hello" in response.lower():
            print("[GROQ] API connection test PASSED")
            return True
        else:
            print(f"[GROQ] API connection test FAILED - Unexpected response: {response}")
            return False

    except Exception as e:
        print(f"[GROQ] API connection test FAILED - Error: {e}")
        return False


def clear_client_cache():
    """
    Clear the cached Groq client

    Useful for resetting connection or changing API keys
    """
    get_groq_client.cache_clear()
    if VERBOSE:
        print("[GROQ] Client cache cleared")


# ============================================================
# CHAT CONVERSATION SUPPORT
# ============================================================

class ChatSession:
    """
    Maintains a conversation session with Groq

    Useful for multi-turn conversations (Week 6 - Streamlit chat)
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize chat session

        Args:
            system_prompt: System instructions for the conversation
        """
        self.messages = []
        self.client = get_groq_client()

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

    def send_message(
        self,
        user_message: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Send a message and get response

        Args:
            user_message: User's message
            temperature: Sampling temperature
            max_tokens: Maximum response length

        Returns:
            Assistant's response
        """
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Get response
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=TOP_P
        )

        # Extract and store assistant response
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self, keep_system: bool = True):
        """
        Clear conversation history

        Args:
            keep_system: Whether to keep the system prompt
        """
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test the Groq client"""
    print("="*60)
    print("GROQ CLIENT TEST")
    print("="*60)

    # Test 1: API connection
    print("\n[TEST 1] Testing API connection...")
    if test_api_connection():
        print("Status: PASSED")
    else:
        print("Status: FAILED - Check your GROQ_API_KEY in .env")
        exit(1)

    # Test 2: Simple completion
    print("\n[TEST 2] Simple completion...")
    response = generate_completion(
        prompt="What is sentiment analysis in one sentence?",
        system_prompt="You are a helpful AI assistant. Be concise.",
        max_tokens=100
    )
    print(f"Response: {response}")

    # Test 3: Token counting
    print("\n[TEST 3] Token estimation...")
    text = "This is a test message for token counting"
    tokens = count_tokens_estimate(text)
    print(f"Text: '{text}'")
    print(f"Estimated tokens: {tokens}")

    # Test 4: Prompt validation
    print("\n[TEST 4] Prompt validation...")
    try:
        validate_prompt_length("Short prompt", max_tokens=100)
        print("Validation: PASSED (short prompt)")
    except ValueError as e:
        print(f"Validation: FAILED - {e}")

    # Test 5: Chat session
    print("\n[TEST 5] Chat session...")
    chat = ChatSession(system_prompt="You are a helpful assistant. Be brief.")
    response1 = chat.send_message("Hi, what's your purpose?", max_tokens=50)
    print(f"User: Hi, what's your purpose?")
    print(f"Assistant: {response1}")

    response2 = chat.send_message("Thanks!", max_tokens=20)
    print(f"User: Thanks!")
    print(f"Assistant: {response2}")
    print(f"Conversation history: {len(chat.messages)} messages")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
