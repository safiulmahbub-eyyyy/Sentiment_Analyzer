"""
Reddit API Configuration
Loads credentials from .env and creates Reddit client
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

try:
    import praw
except ImportError as e:
    raise ImportError("praw is not installed. Run: pip install praw") from e


def get_reddit_client() -> "praw.Reddit":
    """
    Create and return a Reddit API client.
    
    Returns:
        praw.Reddit: Authenticated Reddit client
        
    Raises:
        RuntimeError: If credentials are missing from .env
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    # Validate credentials exist
    if not all([client_id, client_secret, user_agent]):
        raise RuntimeError(
            "Missing Reddit credentials. "
            "Ensure .env has REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT"
        )
    
    # Create Reddit client
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    return reddit


# Test when run directly
if __name__ == "__main__":
    try:
        reddit = get_reddit_client()
        print(f" Reddit client initialized successfully!")
        print(f"   Read-only mode: {reddit.read_only}")
        
        # Quick test: Get one post from r/python
        test_post = next(reddit.subreddit('python').hot(limit=1))
        print(f" API connection working!")
        print(f"   Test post: {test_post.title[:50]}...")
        
    except Exception as e:
        print(f" Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file exists")
        print("2. Verify credentials at https://www.reddit.com/prefs/apps")
        print("3. Make sure .env has all three variables")