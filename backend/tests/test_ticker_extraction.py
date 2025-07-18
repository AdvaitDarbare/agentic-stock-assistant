#!/usr/bin/env python3
"""
Test ticker extraction for correlation queries
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ticker_map import ticker_map
import re

def _extract_tickers_from_query(query: str):
    """
    Return a list of all tickers (normalized to uppercase) mentioned in the query.
    """
    tickers = []

    # 1) $TICKER syntax
    for m in re.finditer(r"\$([A-Za-z]{1,5})\b", query):
        t = m.group(1).upper()
        if t.lower() in ticker_map and t not in tickers:
            tickers.append(t)

    # 2) ALL-CAPS words
    for w in re.findall(r"\b([A-Z]{2,5})\b", query):
        if w.lower() in ticker_map:
            T = ticker_map[w.lower()]
            if T not in tickers:
                tickers.append(T)

    # 3) Company names and ticker symbols (case-insensitive)
    words = query.lower().split()
    for word in words:
        # Clean punctuation from word
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in ticker_map:
            T = ticker_map[clean_word]
            if T not in tickers:
                tickers.append(T)

    # 4) Multi-word company names (check common patterns)
    query_lower = query.lower()
    multi_word_companies = [
        name for name in ticker_map.keys() if ' ' in name
    ]
    for company in multi_word_companies:
        if company in query_lower:
            T = ticker_map[company]
            if T not in tickers:
                tickers.append(T)

    return tickers

def test_ticker_extraction():
    """Test various ticker extraction scenarios"""
    
    test_cases = [
        "Can you tell me the correlation between news and Apple stock price from 06/01/2025 to 06/11/2025",
        "What is the sentiment for Microsoft stock?",
        "Show me AAPL price data",
        "Tesla vs Amazon correlation",
        "Meta platforms earnings",
        "Google stock performance",
        "NVDA price analysis",
        "$TSLA stock news",
        "How does news affect Apple stock price?",
        "Microsoft vs Apple correlation analysis"
    ]
    
    print("🧪 Testing Ticker Extraction...")
    print("=" * 60)
    
    for query in test_cases:
        tickers = _extract_tickers_from_query(query)
        print(f"Query: {query}")
        print(f"Extracted tickers: {tickers}")
        print("-" * 60)
    
    print("✅ Ticker extraction test completed!")

if __name__ == "__main__":
    test_ticker_extraction()