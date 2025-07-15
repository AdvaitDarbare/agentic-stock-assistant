#!/usr/bin/env python3
"""
Test MCP sentiment tool directly
"""

import asyncio
import json
import httpx

async def test_mcp_sentiment():
    """Test the MCP sentiment correlation tool"""
    
    print("🧪 Testing MCP Sentiment Tool...")
    print("=" * 60)
    
    # Test data matching the actual flow
    test_data = {
        "news_data": {
            "recent_headlines": [
                {
                    "headline": "Apple Reports Strong Q2 Earnings, Beats Analyst Expectations",
                    "date": "2025-06-02"
                },
                {
                    "headline": "Apple Stock Rises on Positive Outlook for iPhone Sales",
                    "date": "2025-06-05"
                }
            ],
            "similar_headlines": [
                {
                    "headline": "Apple Maintains Strong Position in Smartphone Market",
                    "date": "2025-06-01"
                }
            ]
        },
        "stock_data": {
            "ticker": "AAPL",
            "date_range": "2025-06-01 to 2025-06-11",
            "price_data": "Available"
        },
        "ticker": "AAPL"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8040/tools/run_sentiment_correlation",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ MCP Sentiment Tool Response:")
                print(json.dumps(result, indent=2))
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ MCP sentiment tool test completed!")

if __name__ == "__main__":
    asyncio.run(test_mcp_sentiment())