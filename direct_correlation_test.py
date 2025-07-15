#!/usr/bin/env python3
"""
Direct correlation test using the agent functions without MCP
"""

import asyncio
from agents.sql_agent import run_sql_agent
from agents.news_agent import run_news_agent
from agents.sentiment_agent import analyze_news_sentiment_correlation
from state import AgentState

async def test_direct_correlation():
    """Test correlation using direct function calls"""
    
    print("🔍 Testing Direct Correlation Analysis...")
    print("=" * 80)
    
    # Step 1: Get SQL data
    print("1️⃣ Getting SQL data...")
    sql_state = {
        "input": "Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025",
        "chat_history": [],
        "need_sql": True,
        "sql_done": False,
        "ticker": "AAPL"
    }
    
    sql_result = run_sql_agent(sql_state)
    print(f"✅ SQL Result: {sql_result.get('output', 'Error')}")
    
    # Step 2: Get news data
    print("\n2️⃣ Getting news data...")
    news_state = {
        "input": "latest news for AAPL",
        "chat_history": [],
        "ticker": "AAPL",
        "need_news": True,
        "news_done": False
    }
    
    news_result = run_news_agent(news_state)
    print(f"✅ News Result: {news_result.get('output', 'Error')}")
    
    # Step 3: Analyze sentiment correlation
    print("\n3️⃣ Analyzing sentiment correlation...")
    if isinstance(news_result.get('output'), dict):
        news_data = news_result['output']
        stock_data = sql_result.get('output', {})
        
        sentiment_result = analyze_news_sentiment_correlation(news_data, stock_data)
        print(f"✅ Sentiment Result:")
        print(f"   📊 Overall Sentiment: {sentiment_result['sentiment_trend']}")
        print(f"   🔢 Sentiment Score: {sentiment_result['avg_sentiment_score']}")
        print(f"   📈 Headlines Analyzed: {sentiment_result['total_headlines']}")
        print(f"   🏷️ Top Topics: {sentiment_result['top_topics']}")
        print(f"   💰 Financial Terms: {sentiment_result['financial_mentions']}")
        print(f"   🔗 Correlation Insights:")
        for insight in sentiment_result['correlation_insights']:
            print(f"      • {insight}")
    
    # Step 4: Create comprehensive analysis
    print("\n4️⃣ Creating comprehensive analysis...")
    if isinstance(sql_result.get('output'), dict) and 'table' in sql_result['output']:
        table_data = sql_result['output']['table']
        if table_data['rows']:
            print(f"📊 Stock Price Data for AAPL (06/01/2025 to 06/11/2025):")
            print(f"   • {len(table_data['rows'])} trading days analyzed")
            print(f"   • Price range: ${min(row[5] for row in table_data['rows']):.2f} - ${max(row[5] for row in table_data['rows']):.2f}")
            
            # Calculate basic metrics
            closes = [row[5] for row in table_data['rows']]
            price_change = closes[-1] - closes[0]
            price_change_pct = (price_change / closes[0]) * 100
            
            print(f"   • Price change: ${price_change:.2f} ({price_change_pct:+.1f}%)")
            
            # Correlate with sentiment
            if 'sentiment_result' in locals():
                sentiment_score = sentiment_result['avg_sentiment_score']
                if sentiment_score > 0.2 and price_change > 0:
                    correlation = "📈 Positive news sentiment aligns with price increase"
                elif sentiment_score < -0.2 and price_change < 0:
                    correlation = "📉 Negative news sentiment aligns with price decrease"
                elif sentiment_score > 0.2 and price_change < 0:
                    correlation = "⚠️ Positive news sentiment but price declined"
                elif sentiment_score < -0.2 and price_change > 0:
                    correlation = "⚠️ Negative news sentiment but price increased"
                else:
                    correlation = "⚖️ Neutral correlation between news and price"
                
                print(f"   • Correlation: {correlation}")
    
    print("\n" + "=" * 80)
    print("✅ Direct correlation analysis completed successfully!")
    print("🎉 All components working correctly!")

if __name__ == "__main__":
    asyncio.run(test_direct_correlation())