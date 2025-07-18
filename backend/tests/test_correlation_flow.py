#!/usr/bin/env python3
"""
Test the complete correlation flow from query to sentiment analysis
"""

import asyncio
import json
from graph import run_query_with_persistence, compile_workflow_with_persistence

async def test_correlation_flow():
    """Test the complete correlation analysis flow"""
    
    print("🧪 Testing Complete Correlation Flow...")
    print("=" * 60)
    
    # Initialize workflow
    await compile_workflow_with_persistence()
    
    # Test specific correlation query that was causing issues
    test_query = "Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025"
    
    print(f"🔍 Testing Query: {test_query}")
    print("-" * 60)
    
    try:
        response = await run_query_with_persistence(test_query, "correlation-test-apple")
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Correlation flow test completed!")

if __name__ == "__main__":
    asyncio.run(test_correlation_flow())