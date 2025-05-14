import pytest
from finagentx.nlp.sentiment_rag import MarketSentimentAnalyzer

def test_sentiment_analysis():
    analyzer = MarketSentimentAnalyzer()
    
    # Test positive sentiment
    sentiment, score = analyzer.analyze_sentiment("Apple Inc. beats earnings expectations!")
    assert sentiment == 'POSITIVE'
    assert score > 0.7  # Confidence should be high for positive sentiment
    
    # Test negative sentiment
    sentiment, score = analyzer.analyze_sentiment("Apple Inc. faces supply chain disruptions!")
    assert sentiment == 'NEGATIVE'
    assert score > 0.7  # Confidence should be high for negative sentiment

def test_sentiment_invalid_input():
    analyzer = MarketSentimentAnalyzer()
    
    # Test empty input (edge case)
    sentiment, score = analyzer.analyze_sentiment("")
    assert sentiment == 'NEGATIVE'  # Assuming empty input returns negative sentiment
    assert score > 0.0  # Confidence should be above zero
