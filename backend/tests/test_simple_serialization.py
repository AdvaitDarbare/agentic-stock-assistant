#!/usr/bin/env python3
"""
Simple test to check if chat history serialization works
"""
import asyncio
from graph import _serialize_chat_history
from langchain_core.messages import HumanMessage, AIMessage

async def test_serialization():
    """Test chat history serialization directly"""
    
    print("🔍 Testing Chat History Serialization...")
    print("=" * 50)
    
    # Create sample chat history with LangChain message objects
    test_history = [
        HumanMessage(content="Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025"),
        AIMessage(content="I'll help you analyze the correlation between news sentiment and Apple stock price for the specified date range."),
        HumanMessage(content="Latest news on MSFT"),
        AIMessage(content="Here's the latest news on Microsoft..."),
        HumanMessage(content="Can you tell open price of MSFT on 06/11/2025")
    ]
    
    print("✅ Original chat history (LangChain messages):")
    for i, msg in enumerate(test_history):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content[:50]}...")
    
    print("\n🔄 Serializing chat history...")
    try:
        serialized = _serialize_chat_history(test_history)
        print("✅ Serialization successful!")
        
        print(f"\n📊 Serialized format:")
        for i, msg in enumerate(serialized):
            print(f"  {i+1}. {msg}")
            
        # Test JSON serialization
        import json
        json_str = json.dumps(serialized, indent=2)
        print(f"\n✅ JSON serialization successful! ({len(json_str)} characters)")
        
        # Test deserialization
        deserialized = json.loads(json_str)
        print(f"✅ JSON deserialization successful! ({len(deserialized)} messages)")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_serialization())
    if result:
        print("\n🎉 Chat history serialization is working correctly!")
        print("✅ The 'Object of type HumanMessage is not JSON serializable' error should be fixed.")
    else:
        print("\n❌ Chat history serialization test failed.")