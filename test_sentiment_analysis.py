#!/usr/bin/env python3
"""
Test sentiment analysis with actual news data structure
"""

import json
from agents.sentiment_agent import analyze_news_sentiment_correlation

def test_sentiment_analysis():
    """Test sentiment analysis with the actual news data structure"""
    
    # Simulate the news data structure from the logs
    news_data = {
        "recent_headlines": [
            {
                "headline": "Apple Reports Strong Q2 Earnings, Beats Analyst Expectations",
                "date": "2025-06-02",
                "source": "financial_news"
            },
            {
                "headline": "Apple Stock Rises on Positive Outlook for iPhone Sales",
                "date": "2025-06-05",
                "source": "market_watch"
            },
            {
                "headline": "Apple CEO Discusses Innovation Strategy at Tech Conference",
                "date": "2025-06-08",
                "source": "tech_news"
            }
        ],
        "similar_headlines": [
            {
                "headline": "Apple Maintains Strong Position in Smartphone Market",
                "date": "2025-06-01",
                "source": "industry_report"
            },
            {
                "headline": "Analysts Upgrade Apple Stock Rating Following Results",
                "date": "2025-06-03",
                "source": "analyst_report"
            }
        ]
    }
    
    # Sample stock data
    stock_data = {
        "ticker": "AAPL",
        "prices": [
            {"date": "2025-06-01", "close": 150.25},
            {"date": "2025-06-02", "close": 152.80},
            {"date": "2025-06-03", "close": 155.40}
        ]
    }
    
    print("🧪 Testing Sentiment Analysis with News Data...")
    print("=" * 60)
    
    try:
        result = analyze_news_sentiment_correlation(news_data, stock_data)
        
        print("✅ Sentiment Analysis Results:")
        print(f"📊 Overall Sentiment: {result['sentiment_trend']}")
        print(f"🔢 Sentiment Score: {result['avg_sentiment_score']}")
        print(f"📈 Total Headlines: {result['total_headlines']}")
        print(f"🏷️ Top Topics: {result['top_topics']}")
        print(f"💰 Financial Terms: {result['financial_mentions']}")
        print(f"🔗 Correlation Insights: {result['correlation_insights']}")
        
        print("\n" + "=" * 60)
        print("✅ Sentiment analysis test completed successfully!")
        
        # Test with JSON string format
        print("\n🧪 Testing with JSON string format...")
        json_string = json.dumps(news_data)
        result2 = analyze_news_sentiment_correlation(json_string, stock_data)
        print(f"✅ JSON string format also works: {result2['total_headlines']} headlines")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sentiment_analysis()