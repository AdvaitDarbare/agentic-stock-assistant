#!/usr/bin/env python3
"""
Test script for LangGraph persistence functionality
"""

import asyncio
import json
from graph import run_query_with_persistence, compile_workflow_with_persistence

async def test_persistence():
    """Test conversation persistence across multiple queries"""
    
    print("🧪 Testing LangGraph Persistence...")
    
    # Initialize workflow
    await compile_workflow_with_persistence()
    
    # Test conversation thread
    thread_id = "test-thread-123"
    
    print(f"\n📝 Starting conversation thread: {thread_id}")
    
    # Query 1: Ask about AAPL stock
    print("\n1️⃣ Query: 'What is the latest close price of AAPL?'")
    response1 = await run_query_with_persistence(
        "What is the latest close price of AAPL?", 
        thread_id
    )
    print(f"AI: {response1}")
    
    # Query 2: Follow-up question (should remember AAPL context)
    print("\n2️⃣ Query: 'What about the open price?'")
    response2 = await run_query_with_persistence(
        "What about the open price?", 
        thread_id
    )
    print(f"AI: {response2}")
    
    # Query 3: Ask about news (should remember we're talking about AAPL)
    print("\n3️⃣ Query: 'Any recent news about it?'")
    response3 = await run_query_with_persistence(
        "Any recent news about it?", 
        thread_id
    )
    print(f"AI: {response3}")
    
    # Query 4: Switch to different stock
    print("\n4️⃣ Query: 'Now tell me about MSFT open price'")
    response4 = await run_query_with_persistence(
        "Now tell me about MSFT open price", 
        thread_id
    )
    print(f"AI: {response4}")
    
    # Query 5: Follow-up (should remember MSFT context)
    print("\n5️⃣ Query: 'What was the high price?'")
    response5 = await run_query_with_persistence(
        "What was the high price?", 
        thread_id
    )
    print(f"AI: {response5}")
    
    print("\n✅ Persistence test completed!")
    print("💾 All conversation context should be preserved across queries")

async def test_multiple_threads():
    """Test multiple conversation threads"""
    
    print("\n🧪 Testing Multiple Conversation Threads...")
    
    # Thread 1: Focus on AAPL
    print("\n📝 Thread 1: AAPL focused")
    response1 = await run_query_with_persistence(
        "Tell me about AAPL stock price", 
        "thread-aapl"
    )
    print(f"Thread 1: {response1}")
    
    # Thread 2: Focus on MSFT
    print("\n📝 Thread 2: MSFT focused")
    response2 = await run_query_with_persistence(
        "Tell me about MSFT stock price", 
        "thread-msft"
    )
    print(f"Thread 2: {response2}")
    
    # Continue Thread 1
    print("\n📝 Continue Thread 1: Follow-up on AAPL")
    response3 = await run_query_with_persistence(
        "What about the news?", 
        "thread-aapl"
    )
    print(f"Thread 1: {response3}")
    
    # Continue Thread 2
    print("\n📝 Continue Thread 2: Follow-up on MSFT")
    response4 = await run_query_with_persistence(
        "What about the news?", 
        "thread-msft"
    )
    print(f"Thread 2: {response4}")
    
    print("\n✅ Multiple threads test completed!")

if __name__ == "__main__":
    asyncio.run(test_persistence())
    asyncio.run(test_multiple_threads())