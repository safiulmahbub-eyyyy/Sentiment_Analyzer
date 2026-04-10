"""
Supabase client wrapper for database operations
"""

import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SupabaseClient:
    """Wrapper for Supabase client with utility methods"""

    def __init__(self):
        """Initialize Supabase client"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for admin operations

        if not self.url or not self.key:
            raise ValueError(
                "Missing Supabase credentials. "
                "Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file"
            )

        self.client: Client = create_client(self.url, self.key)

    def insert_posts(self, posts: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """
        Insert posts in batches

        Args:
            posts: List of post dictionaries
            batch_size: Number of posts per batch

        Returns:
            Dictionary with success and error counts
        """
        success_count = 0
        error_count = 0

        # Process in batches
        for i in range(0, len(posts), batch_size):
            batch = posts[i:i + batch_size]

            try:
                response = self.client.table('reddit_posts').upsert(batch).execute()
                success_count += len(batch)
            except Exception as e:
                print(f"Error inserting batch {i//batch_size + 1}: {e}")
                error_count += len(batch)

        return {
            'success': success_count,
            'errors': error_count
        }

    def get_posts_without_embeddings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch posts that don't have embeddings yet

        Args:
            limit: Maximum number of posts to fetch (default: 1000 due to Supabase limits)

        Returns:
            List of posts
        """
        if limit is None:
            limit = 1000

        query = self.client.table('reddit_posts').select('*').is_('embedding', 'null').limit(limit)

        response = query.execute()
        return response.data

    def update_embeddings(self, updates: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """
        Update embeddings for posts

        Args:
            updates: List of {post_id, embedding} dictionaries
            batch_size: Number of updates per batch

        Returns:
            Dictionary with success and error counts
        """
        success_count = 0
        error_count = 0

        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]

            try:
                for update in batch:
                    self.client.table('reddit_posts').update({
                        'embedding': update['embedding']
                    }).eq('post_id', update['post_id']).execute()

                success_count += len(batch)
            except Exception as e:
                print(f"Error updating batch {i//batch_size + 1}: {e}")
                error_count += len(batch)

        return {
            'success': success_count,
            'errors': error_count
        }

    def update_sentiment(self, updates: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """
        Update sentiment scores for posts

        Args:
            updates: List of dictionaries with post_id and sentiment fields
                     {post_id, sentiment_pos, sentiment_neg, sentiment_neu,
                      sentiment_compound, sentiment_label}
            batch_size: Number of updates per batch

        Returns:
            Dictionary with success and error counts
        """
        success_count = 0
        error_count = 0

        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]

            try:
                for update in batch:
                    self.client.table('reddit_posts').update({
                        'sentiment_pos': update['sentiment_pos'],
                        'sentiment_neg': update['sentiment_neg'],
                        'sentiment_neu': update['sentiment_neu'],
                        'sentiment_compound': update['sentiment_compound'],
                        'sentiment_label': update['sentiment_label']
                    }).eq('post_id', update['post_id']).execute()

                success_count += len(batch)
            except Exception as e:
                print(f"Error updating sentiment batch {i//batch_size + 1}: {e}")
                error_count += len(batch)

        return {
            'success': success_count,
            'errors': error_count
        }

    def get_posts_without_sentiment(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch posts that don't have sentiment scores yet

        Args:
            limit: Maximum number of posts to fetch

        Returns:
            List of posts
        """
        query = self.client.table('reddit_posts').select('*').is_('sentiment_label', 'null')

        if limit:
            query = query.limit(limit)

        response = query.execute()
        return response.data

    def search_similar_posts(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 20,
        filter_subreddit: Optional[str] = None,
        filter_sentiment: Optional[str] = None,
        days_ago: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search for similar posts using vector similarity

        Args:
            query_embedding: Query vector (384 dimensions)
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Number of results to return
            filter_subreddit: Optional subreddit filter
            filter_sentiment: Optional sentiment filter
            days_ago: Only search posts from last N days

        Returns:
            List of similar posts with similarity scores
        """
        try:
            response = self.client.rpc(
                'search_similar_posts',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': match_count,
                    'filter_subreddit': filter_subreddit,
                    'filter_sentiment': filter_sentiment,
                    'days_ago': days_ago
                }
            ).execute()

            return response.data
        except Exception as e:
            print(f"Error searching similar posts: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with database stats
        """
        try:
            response = self.client.rpc('get_database_stats', {}).execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def get_post_count(self) -> int:
        """Get total number of posts"""
        try:
            response = self.client.table('reddit_posts').select('post_id', count='exact').execute()
            return response.count
        except Exception as e:
            print(f"Error getting post count: {e}")
            return 0


# Singleton instance
_client = None

def get_client() -> SupabaseClient:
    """Get or create Supabase client singleton"""
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client
