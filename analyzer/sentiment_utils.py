"""
Sentiment Analysis Utilities
Shared functions for VADER sentiment analysis across the project
"""

from typing import Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def calculate_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, Any]:
    """
    Calculate sentiment scores using VADER
    
    Shared utility used by:
    - analyzer/process_posts.py (SQLite batch processing)
    - supabase_pipeline.py (inline sentiment enrichment)
    
    Args:
        text: Text to analyze (title + body recommended)
        analyzer: VADER analyzer instance

    Returns:
        Dictionary with sentiment scores and label:
        {
            'sentiment_pos': float,      # Positive score [0-1]
            'sentiment_neg': float,      # Negative score [0-1]
            'sentiment_neu': float,      # Neutral score [0-1]
            'sentiment_compound': float, # Compound score [-1 to 1]
            'sentiment_label': str       # 'positive', 'negative', or 'neutral'
        }
    """
    scores = analyzer.polarity_scores(text)

    # Determine label based on compound score
    # VADER thresholds: ≥0.05 = positive, ≤-0.05 = negative, else neutral
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'sentiment_pos': scores['pos'],
        'sentiment_neg': scores['neg'],
        'sentiment_neu': scores['neu'],
        'sentiment_compound': compound,
        'sentiment_label': label
    }


def prepare_text_for_sentiment(title: str, body: str = None) -> str:
    """
    Prepare post text for sentiment analysis
    
    Args:
        title: Post title (required)
        body: Post body/selftext (optional)
    
    Returns:
        Combined text ready for analysis
    """
    text = title or ""
    
    if body and body.strip():
        text += " " + body
    
    return text.strip()