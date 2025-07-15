#!/usr/bin/env python3
"""
Simple test of correlation query without persistence
"""

import asyncio
from graph import run_query_once

def test_correlation():
    """Test correlation query with memory checkpointer"""
    
    query = "Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025"
    print(f"🔍 Testing: {query}")
    print("=" * 80)
    
    try:
        result = run_query_once(query)
        print(f"✅ Result:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correlation()