import sys, json
from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP
from transformers import pipeline
import re
from datetime import datetime
from collections import Counter

# ─── Setup ────────────────────────────────────────────────────────────────
sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert") # HuggingFace
# You could also use BERTopic / scikit-learn for clustering.

def analyze_headlines(headlines: List[str]) -> Dict[str, any]:
    """Basic sentiment analysis of headlines"""
    if not headlines:
        return {
            "sentiment_counts": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "top_topics": [],
            "avg_sentiment_score": 0.0,
            "sentiment_trend": "neutral"
        }
    
    # 1. Sentiment analysis
    scores = [sentiment(h)[0] for h in headlines]
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total_score = 0
    
    for s in scores:
        label = s["label"]
        score = s["score"]
        counts[label] += 1
        
        # Convert to numerical score (-1 to 1)
        if label == "POSITIVE":
            total_score += score
        elif label == "NEGATIVE":
            total_score -= score
        # NEUTRAL contributes 0
    
    avg_sentiment = total_score / len(headlines) if headlines else 0
    
    # 2. Extract key financial terms and topics
    financial_terms = [
        "earnings", "revenue", "profit", "loss", "growth", "decline", 
        "bullish", "bearish", "upgrade", "downgrade", "target", "analyst",
        "beat", "miss", "guidance", "outlook", "dividend", "split"
    ]
    
    words = Counter()
    financial_mentions = Counter()
    
    for headline in headlines:
        headline_lower = headline.lower()
        # Count all words
        words.update(w for w in re.findall(r'\b\w+\b', headline_lower) if len(w) > 2)
        # Count financial terms
        for term in financial_terms:
            if term in headline_lower:
                financial_mentions[term] += 1
    
    # 3. Determine overall sentiment trend
    if avg_sentiment > 0.1:
        sentiment_trend = "positive"
    elif avg_sentiment < -0.1:
        sentiment_trend = "negative"
    else:
        sentiment_trend = "neutral"
    
    return {
        "sentiment_counts": counts,
        "top_topics": [w for w, _ in words.most_common(5)],
        "financial_mentions": dict(financial_mentions.most_common(3)),
        "avg_sentiment_score": round(avg_sentiment, 3),
        "sentiment_trend": sentiment_trend,
        "total_headlines": len(headlines)
    }

def analyze_news_sentiment_correlation(news_data: Dict, stock_data: Optional[Dict] = None) -> Dict[str, any]:
    """Enhanced sentiment analysis with stock correlation insights"""
    
    try:
        # Extract headlines from news data
        headlines = []
        
        # Handle different news data formats
        if isinstance(news_data, str):
            # Try to parse JSON string
            try:
                import json
                news_data = json.loads(news_data)
            except:
                # If parsing fails, treat as single headline
                headlines = [news_data]
        
        if isinstance(news_data, dict):
            # Extract headlines from different fields
            if "recent_headlines" in news_data:
                recent = news_data["recent_headlines"]
                if isinstance(recent, list):
                    for item in recent:
                        if isinstance(item, dict) and "headline" in item:
                            headlines.append(item["headline"])
                        elif isinstance(item, str):
                            headlines.append(item)
            
            if "similar_headlines" in news_data:
                similar = news_data["similar_headlines"]
                if isinstance(similar, list):
                    for item in similar:
                        if isinstance(item, dict) and "headline" in item:
                            headlines.append(item["headline"])
                        elif isinstance(item, str):
                            headlines.append(item)
        
        elif isinstance(news_data, list):
            headlines = news_data
        
        # Perform sentiment analysis
        sentiment_analysis = analyze_headlines(headlines)
        
        # Ensure sentiment_analysis is a dictionary
        if not isinstance(sentiment_analysis, dict):
            print(f"Warning: analyze_headlines returned {type(sentiment_analysis)}: {sentiment_analysis}", file=sys.stderr)
            sentiment_analysis = {
                "sentiment_counts": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
                "top_topics": [],
                "financial_mentions": {},
                "avg_sentiment_score": 0.0,
                "sentiment_trend": "neutral",
                "total_headlines": 0
            }
        
        # Add correlation insights if stock data is available
        correlation_insights = []
        if stock_data and isinstance(stock_data, dict):
            sentiment_score = sentiment_analysis.get("avg_sentiment_score", 0.0)
            
            # Try to extract stock price info
            if "close" in str(stock_data).lower():
                if sentiment_score > 0.2:
                    correlation_insights.append("📈 Strong positive news sentiment may support stock price")
                elif sentiment_score < -0.2:
                    correlation_insights.append("📉 Negative news sentiment may pressure stock price")
                else:
                    correlation_insights.append("⚖️ Neutral sentiment suggests limited news impact on price")
            
            # Check for specific financial terms impact
            financial_terms = sentiment_analysis.get("financial_mentions", {})
            if "earnings" in financial_terms:
                correlation_insights.append("💰 Earnings-related news detected - high price impact potential")
            if "analyst" in financial_terms:
                correlation_insights.append("👨‍💼 Analyst activity detected - may influence investor sentiment")
        
        return {
            **sentiment_analysis,
            "correlation_insights": correlation_insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in analyze_news_sentiment_correlation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Return safe default response
        return {
            "sentiment_counts": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "top_topics": [],
            "financial_mentions": {},
            "avg_sentiment_score": 0.0,
            "sentiment_trend": "neutral",
            "total_headlines": 0,
            "correlation_insights": [f"Error in sentiment analysis: {str(e)}"],
            "analysis_timestamp": datetime.now().isoformat()
        }

# ─── MCP tool ─────────────────────────────────────────────────────────────
mcp = FastMCP("SentimentAgent", port=8040, stateless_http=True, json_response=True)

@mcp.tool(name="run_sentiment_trend")
def _tool(state: dict) -> dict:
    # Expect state["headlines"] = List[str]
    headlines = state.get("headlines", [])
    result = analyze_headlines(headlines)
    return {"output": result}

@mcp.tool(name="run_sentiment_correlation")
def _correlation_tool(state: dict) -> dict:
    """
    Analyze sentiment correlation between news and stock data
    Expected state keys:
    - news_data: Dict or List containing news headlines
    - stock_data: Optional Dict containing stock price information
    - ticker: String ticker symbol for context
    """
    try:
        news_data = state.get("news_data", {})
        stock_data = state.get("stock_data", None)
        ticker = state.get("ticker", "")
        
        # Perform correlation analysis
        result = analyze_news_sentiment_correlation(news_data, stock_data)
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"Warning: analyze_news_sentiment_correlation returned {type(result)}: {result}", file=sys.stderr)
            result = {
                "sentiment_trend": "neutral",
                "avg_sentiment_score": 0.0,
                "sentiment_counts": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
                "total_headlines": 0,
                "top_topics": [],
                "financial_mentions": {},
                "correlation_insights": ["Unable to analyze sentiment - data format issue"],
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        # Add ticker context
        if ticker:
            result["ticker"] = ticker
        
        # Format output for LLM consumption
        formatted_result = {
            "sentiment_analysis": {
                "overall_sentiment": result.get("sentiment_trend", "neutral"),
                "sentiment_score": result.get("avg_sentiment_score", 0.0),
                "sentiment_breakdown": result.get("sentiment_counts", {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}),
                "total_headlines_analyzed": result.get("total_headlines", 0)
            },
            "key_insights": {
                "top_topics": result.get("top_topics", []),
                "financial_terms": result.get("financial_mentions", {}),
                "correlation_insights": result.get("correlation_insights", [])
            },
            "analysis_meta": {
                "timestamp": result.get("analysis_timestamp", datetime.now().isoformat()),
                "ticker": ticker
            }
        }
        
        return {"output": formatted_result}
        
    except Exception as e:
        print(f"Error in sentiment correlation tool: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Return error response in proper format
        error_result = {
            "sentiment_analysis": {
                "overall_sentiment": "error",
                "sentiment_score": 0.0,
                "sentiment_breakdown": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
                "total_headlines_analyzed": 0
            },
            "key_insights": {
                "top_topics": [],
                "financial_terms": {},
                "correlation_insights": [f"Error in sentiment analysis: {str(e)}"]
            },
            "analysis_meta": {
                "timestamp": datetime.now().isoformat(),
                "ticker": state.get("ticker", ""),
                "error": str(e)
            }
        }
        
        return {"output": error_result}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")