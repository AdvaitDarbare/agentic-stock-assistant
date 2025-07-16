#!/usr/bin/env python3
"""
Test MCP endpoints directly
"""

import asyncio
import httpx
import json

async def test_mcp_endpoints():
    """Test various MCP endpoint formats"""
    
    servers = {
        "sql": "http://localhost:8010",
        "news": "http://localhost:8020",
        "sentiment": "http://localhost:8040"
    }
    
    test_data = {
        "input": "AAPL price",
        "chat_history": []
    }
    
    print("🔍 Testing MCP Endpoints...")
    print("=" * 60)
    
    for name, base_url in servers.items():
        print(f"\n📡 Testing {name} server at {base_url}")
        
        # Test various endpoint formats
        endpoints_to_test = [
            f"{base_url}/",
            f"{base_url}/run_{name}_agent",
            f"{base_url}/tools/run_{name}_agent",
            f"{base_url}/call/run_{name}_agent",
            f"{base_url}/mcp",
            f"{base_url}/health"
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints_to_test:
                try:
                    # Try GET first
                    response = await client.get(endpoint)
                    if response.status_code != 404:
                        print(f"  ✅ GET {endpoint}: {response.status_code}")
                        if response.text and len(response.text) < 200:
                            print(f"     Response: {response.text[:100]}")
                    
                    # Try POST with JSON
                    response = await client.post(endpoint, json=test_data)
                    if response.status_code != 404:
                        print(f"  ✅ POST {endpoint}: {response.status_code}")
                        if response.text and len(response.text) < 200:
                            print(f"     Response: {response.text[:100]}")
                    
                except Exception as e:
                    print(f"  ❌ {endpoint}: {e}")
        
        print(f"  📊 {name} server test complete")
    
    print("\n" + "=" * 60)
    print("🔍 MCP endpoint testing completed!")

if __name__ == "__main__":
    asyncio.run(test_mcp_endpoints())