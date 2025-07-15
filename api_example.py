#!/usr/bin/env python3
"""
Example API usage with persistence
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api_persistence():
    """Test API persistence functionality"""
    
    print("🌐 Testing API Persistence...")
    
    # Start new conversation
    print("\n1️⃣ Starting new conversation...")
    response = requests.post(f"{API_BASE}/chat/new", json={
        "query": "What is the close price of AAPL?"
    })
    
    if response.status_code == 200:
        data = response.json()
        thread_id = data["thread_id"]
        print(f"Thread ID: {thread_id}")
        print(f"AI: {data['answer']}")
        
        # Continue conversation with same thread
        print("\n2️⃣ Continuing conversation...")
        response2 = requests.post(f"{API_BASE}/chat", json={
            "query": "What about the open price?",
            "thread_id": thread_id
        })
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"AI: {data2['answer']}")
            
            # Another follow-up
            print("\n3️⃣ Another follow-up...")
            response3 = requests.post(f"{API_BASE}/chat", json={
                "query": "Any news about it?",
                "thread_id": thread_id
            })
            
            if response3.status_code == 200:
                data3 = response3.json()
                print(f"AI: {data3['answer']}")
            else:
                print(f"Error: {response3.status_code}")
        else:
            print(f"Error: {response2.status_code}")
    else:
        print(f"Error: {response.status_code}")

def test_multiple_api_threads():
    """Test multiple API conversation threads"""
    
    print("\n🌐 Testing Multiple API Threads...")
    
    # Start two different conversations
    conversations = []
    
    # Conversation 1: AAPL
    print("\n📝 Starting AAPL conversation...")
    response1 = requests.post(f"{API_BASE}/chat/new", json={
        "query": "Tell me about AAPL stock"
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        conversations.append({
            "thread_id": data1["thread_id"],
            "topic": "AAPL"
        })
        print(f"AAPL Thread: {data1['thread_id']}")
        print(f"AI: {data1['answer']}")
    
    # Conversation 2: MSFT
    print("\n📝 Starting MSFT conversation...")
    response2 = requests.post(f"{API_BASE}/chat/new", json={
        "query": "Tell me about MSFT stock"
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        conversations.append({
            "thread_id": data2["thread_id"],
            "topic": "MSFT"
        })
        print(f"MSFT Thread: {data2['thread_id']}")
        print(f"AI: {data2['answer']}")
    
    # Continue both conversations
    for conv in conversations:
        print(f"\n📝 Continuing {conv['topic']} conversation...")
        response = requests.post(f"{API_BASE}/chat", json={
            "query": "What about recent news?",
            "thread_id": conv["thread_id"]
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"AI: {data['answer']}")

if __name__ == "__main__":
    print("🚀 Make sure to start the API server first:")
    print("uvicorn main:app --reload --port 8000")
    print("\nPress Enter to continue...")
    input()
    
    try:
        test_api_persistence()
        test_multiple_api_threads()
        print("\n✅ API persistence tests completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")