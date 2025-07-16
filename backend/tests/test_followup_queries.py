#!/usr/bin/env python3
"""
Test follow-up queries with chat history to verify JSON serialization fix
"""

import asyncio
from graph import run_query_with_persistence

async def test_followup_queries():
    """Test follow-up queries with chat history"""
    
    print("🔍 Testing Follow-up Queries with Chat History...")
    print("=" * 80)
    
    # Use the same thread ID to maintain chat history
    thread_id = "test-followup-conversation"
    
    # First query - correlation analysis
    print("1️⃣ First query: Correlation analysis")
    try:
        response1 = await run_query_with_persistence(
            "Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025",
            thread_id
        )
        print(f"✅ First query successful: {len(response1)} characters")
    except Exception as e:
        print(f"❌ First query failed: {e}")
        return
    
    print("\n" + "-" * 80)
    
    # Second query - News query (should work with chat history)
    print("2️⃣ Second query: Latest news on MSFT")
    try:
        response2 = await run_query_with_persistence(
            "Latest news on MSFT",
            thread_id
        )
        print(f"✅ Second query successful: {len(response2)} characters")
        print(f"📰 Response preview: {response2[:200]}...")
    except Exception as e:
        print(f"❌ Second query failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n" + "-" * 80)
    
    # Third query - SQL query (should work with chat history)
    print("3️⃣ Third query: MSFT open price on specific date")
    try:
        response3 = await run_query_with_persistence(
            "Can you tell open price of MSFT on 06/11/2025",
            thread_id
        )
        print(f"✅ Third query successful: {len(response3)} characters")
        print(f"💰 Response preview: {response3[:200]}...")
    except Exception as e:
        print(f"❌ Third query failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("✅ All follow-up queries completed successfully!")
    print("🎉 Chat history serialization is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_followup_queries())