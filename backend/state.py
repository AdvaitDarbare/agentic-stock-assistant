"""
state.py - Shared state definition for the multi-agent system
"""

from typing import Optional, TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    query: str
    need_sql: bool
    need_news: bool
    need_sentiment: bool  # NEW: Whether sentiment analysis is needed
    sql_done: bool
    news_done: bool
    sentiment_done: bool  # NEW: Whether sentiment analysis is complete
    sql_result: Optional[str]
    news_result: Optional[str]
    sentiment_result: Optional[str]  # NEW: Sentiment analysis results
    answer: Optional[str]
    error: Optional[str]
    chat_history: List[HumanMessage | AIMessage]
    last_ticker: Optional[str]
    last_date: Optional[str]
    last_query: Optional[str]
    is_range_query: bool
    input: Optional[str]  # For MCP compatibility
    output: Optional[str]  # For MCP compatibility
    current_date: Optional[str]  # For SQL agent compatibility
    next_node: Optional[str]  # For MCP compatibility