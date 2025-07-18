#!/usr/bin/env python3
"""
Test script to validate the query handling improvements
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import the necessary modules
from agents.sql_agent import run_sql_agent
from state import AgentState

def test_single_date_query():
    """Test: Can you tell me the open price of AAPL on 2025-06-06"""
    print("\n=== Testing Single Date Query ===")
    
    state = AgentState(
        input="Can you tell me the open price of AAPL on 2025-06-06",
        chat_history=[],
        current_date="2025-07-18"
    )
    
    try:
        result = run_sql_agent(state)
        print(f"Query successful: {result.get('output', {}).get('sql', 'No SQL found')}")
        
        # Check if we got data
        if isinstance(result.get('output'), dict) and 'table' in result['output']:
            table = result['output']['table']
            rows = table.get('rows', [])
            if rows:
                print(f"Results: {len(rows)} rows found")
                print(f"First row: {rows[0] if rows else 'No rows'}")
                return True
            else:
                print("No data returned")
                return False
        else:
            print("Error in result format")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_range_query():
    """Test: Can you tell me the open price, close price of AAPL from 2025-06-06 to 2025-06-11"""
    print("\n=== Testing Range Query ===")
    
    state = AgentState(
        input="Can you tell me the open price, close price of AAPL from 2025-06-06 to 2025-06-11",
        chat_history=[],
        current_date="2025-07-18"
    )
    
    try:
        result = run_sql_agent(state)
        print(f"Query successful: {result.get('output', {}).get('sql', 'No SQL found')}")
        
        # Check if we got data
        if isinstance(result.get('output'), dict) and 'table' in result['output']:
            table = result['output']['table']
            rows = table.get('rows', [])
            if rows:
                print(f"Results: {len(rows)} rows found")
                print(f"Sample rows: {rows[:3] if len(rows) >= 3 else rows}")
                return True
            else:
                print("No data returned")
                return False
        else:
            print("Error in result format")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing SQL Agent Query Improvements")
    print("====================================")
    
    # Test the queries
    test1_success = test_single_date_query()
    test2_success = test_range_query()
    
    print(f"\n=== Test Results ===")
    print(f"Single date query: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Range query: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed - check the logs above")