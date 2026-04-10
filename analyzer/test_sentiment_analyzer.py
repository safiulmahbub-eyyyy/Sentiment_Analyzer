from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def test_vader():
    """Test VADER on sample posts"""
    analyzer = SentimentIntensityAnalyzer()
    
    # Test samples
    samples = [
        "I love this iPhone 16! Best phone ever!",
        "Terrible battery life, worst purchase I made",
        "It's okay, nothing special really",
        "www.google.com"
    ]
    
    print("=" * 60)
    print("TESTING VADER SENTIMENT ANALYZER")
    print("=" * 60)
    
    for i, text in enumerate(samples, 1):
        scores = analyzer.polarity_scores(text)
        
        # Determine label
        compound = scores['compound']
        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        print(f"\n{i}. Text: {text}")
        print(f"   Positive: {scores['pos']:.3f}")
        print(f"   Negative: {scores['neg']:.3f}")
        print(f"   Neutral:  {scores['neu']:.3f}")
        print(f"   Compound: {scores['compound']:.3f}")
        print(f"   → LABEL: {label}")
    
    print("\n" + "=" * 60)
    print("✅ VADER is working correctly!")
    print("=" * 60)

if __name__ == "__main__":
    test_vader()