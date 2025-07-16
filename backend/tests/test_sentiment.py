#!/usr/bin/env python3
"""
Test script for sentiment analysis integration
"""

import asyncio
import json
from graph import run_query_with_persistence, compile_workflow_with_persistence

async def test_sentiment_integration():
    """Test sentiment analysis with correlation functionality"""
    
    print("🧪 Testing Sentiment Analysis Integration...")
    
    # Initialize workflow
    await compile_workflow_with_persistence()
    
    # Test sentiment analysis queries
    test_queries = [
        "What is the sentiment analysis of AAPL news?",
        "How does news sentiment correlate with MSFT price?",
        "Show me AAPL price and news sentiment analysis",
        "What is the correlation between news and price for TSLA?",
        "Analyze the market sentiment for GOOGL"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        try:
            response = await run_query_with_persistence(query, f"sentiment-test-{i}")
            print(f"AI: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Sentiment analysis integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_sentiment_integration())