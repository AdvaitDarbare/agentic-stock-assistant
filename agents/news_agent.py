"""
news_agent.py – MCP Server Version (Fully MCP Compliant)
────────────────────────────────────────────────────────────────
Handles questions about company news by searching the `news_articles` table.
Enhanced with better LLM-driven decision making and MCP compliance.
"""

import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import psycopg2
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pgvector.psycopg2 import register_vector
from state import AgentState

# ─── DEBUG LOGGER ────────────────────────────────────────────────────────
def _dbg(tag: str, txt):
    print(f"\n### NEWS-{tag}\n{txt}\n", file=sys.stderr, flush=True)

# ─── ENV / CONFIG ────────────────────────────────────────────────────────
load_dotenv()
DB_NAME  = os.getenv("DB_NAME")
DB_USER  = os.getenv("DB_USER")
DB_PASS  = os.getenv("DB_PASS")
DB_HOST  = os.getenv("DB_HOST")
DB_PORT  = os.getenv("DB_PORT")

TABLE_NAME  = "news_articles"
CONTENT_COL = "headline"
DATE_COL    = "date"
VECTOR_COL  = "embedding"
STOCK_COL   = "stock"

# Company‑name → ticker shortcuts
NAME_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
}

OLLAMA_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
embedder = OllamaEmbeddings(model=OLLAMA_MODEL)

# LLM for intelligent processing
_LLM = ChatOllama(model=os.getenv("LLM_MODEL", "gemma3n:e2b"), temperature=0)

# ─── DB CONNECTION ────────────────────────────────────────────────────────
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        connect_timeout=10,
    )
    register_vector(conn)
    return conn

# ─── LOAD TICKERS AT STARTUP ─────────────────────────────────────────────
def _load_known_tickers() -> set[str]:
    try:
        sql = f"SELECT DISTINCT {STOCK_COL} FROM {TABLE_NAME};"
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return {row[0].upper() for row in cur.fetchall()}
    except Exception as e:
        _dbg("TICKER-LOAD-ERROR", f"Failed to load tickers: {e}")
        # Fallback to known tickers
        return {"AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META"}

KNOWN_TICKERS = _load_known_tickers()

# ─── ENHANCED TICKER EXTRACTION ───────────────────────────────────────────
_DOLLAR_RE = re.compile(r"\$([A-Za-z]{1,5})")
_UPPER_RE  = re.compile(r"\b[A-Z]{1,5}\b")

def extract_ticker_with_llm(query: str) -> str | None:
    """Use LLM to intelligently extract ticker from query"""
    
    # First try simple regex approaches
    q_low = query.lower()
    
    # Check for $TICKER format
    m = _DOLLAR_RE.search(query)
    if m and (sym := m.group(1).upper()) in KNOWN_TICKERS:
        return sym
    
    # Check for uppercase ticker symbols
    for sym in _UPPER_RE.findall(query):
        if sym in KNOWN_TICKERS:
            return sym
    
    # Check company name mapping
    for name, sym in NAME_MAP.items():
        if name in q_low and sym in KNOWN_TICKERS:
            return sym
    
    # If simple methods fail, use LLM
    llm_prompt = f"""
Extract the stock ticker symbol from this query. Only respond with the ticker symbol (like AAPL, MSFT, etc.) or "NONE" if no ticker is found.

Available tickers: {', '.join(sorted(KNOWN_TICKERS))}

Query: "{query}"

Rules:
- Look for company names (Apple -> AAPL, Microsoft -> MSFT, etc.)
- Look for ticker symbols mentioned directly
- Look for $TICKER format
- If multiple tickers, choose the primary one
- If no clear ticker, respond with "NONE"

Ticker:"""

    try:
        response = _LLM.invoke(llm_prompt).content.strip().upper()
        _dbg("LLM-TICKER-EXTRACTION", f"Query: {query} -> Response: {response}")
        
        if response and response != "NONE" and response in KNOWN_TICKERS:
            return response
    except Exception as e:
        _dbg("LLM-TICKER-ERROR", f"LLM ticker extraction failed: {e}")
    
    return None

# ─── DB QUERIES ──────────────────────────────────────────────────────────
def fetch_raw_articles(ticker: str, limit: int = 5) -> List[Tuple[str, datetime]]:
    try:
        sql = f"""
            SELECT {CONTENT_COL}, {DATE_COL}
              FROM {TABLE_NAME}
             WHERE {STOCK_COL} = %s
          ORDER BY {DATE_COL} DESC
             LIMIT %s
        """
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (ticker, limit))
            return cur.fetchall()
    except Exception as e:
        _dbg("FETCH-ERROR", f"Failed to fetch articles for {ticker}: {e}")
        return []

def similarity_search(query: str, ticker: str, k: int = 5):
    try:
        vec = embedder.embed_query(query)
        sql = f"""
            SELECT {CONTENT_COL}, {DATE_COL},
                   1 - ({VECTOR_COL} <=> %s::vector) AS similarity
              FROM {TABLE_NAME}
             WHERE {STOCK_COL} = %s
          ORDER BY similarity DESC
             LIMIT %s
        """
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (vec, ticker, k))
            return cur.fetchall()
    except Exception as e:
        _dbg("SIMILARITY-ERROR", f"Similarity search failed for {ticker}: {e}")
        return []

# ─── INTELLIGENT NEWS FORMATTING ─────────────────────────────────────────
def format_news_response(ticker: str, query: str, recent: List, similar: List) -> dict:
    """Format news response in a structured way for the LLM"""
    
    # Structure the response data
    response_data = {
        "ticker": ticker,
        "query": query,
        "recent_headlines": [
            {
                "headline": headline.replace('\n', ' ').strip(),
                "date": str(pub_date)
            }
            for headline, pub_date in recent
        ],
        "similar_headlines": [
            {
                "headline": headline.replace('\n', ' ').strip(),
                "date": str(pub_date),
                "similarity": float(score)
            }
            for headline, pub_date, score in similar
        ]
    }
    
    _dbg("FORMATTED-RESPONSE", response_data)
    return response_data

# ─── MAIN AGENT FUNCTION ─────────────────────────────────────────────────
def run_news_agent(state: AgentState) -> AgentState:
    _dbg("INPUT-STATE", state)
    
    query = state.get("input", "").strip()
    if not query:
        query = "latest news"
    
    # Extract ticker using enhanced method
    ticker = extract_ticker_with_llm(query)
    
    # Also check if ticker was provided explicitly in state
    if not ticker and state.get("ticker"):
        provided_ticker = state.get("ticker", "").upper()
        if provided_ticker in KNOWN_TICKERS:
            ticker = provided_ticker
    
    if not ticker:
        error_msg = (
            f"Sorry—couldn't recognize a valid ticker in your question: '{query}'. "
            f"Please phrase it like 'latest news for $TSLA' or 'Tesla news'. "
            f"Available tickers: {', '.join(sorted(KNOWN_TICKERS))}"
        )
        _dbg("NO-TICKER-ERROR", error_msg)
        return {
            **state,
            "output": error_msg,
            "next_node": None,
        }
    
    _dbg("TICKER-EXTRACTED", f"Query: '{query}' -> Ticker: {ticker}")
    
    # Fetch news data
    recent = fetch_raw_articles(ticker, limit=5)
    similar = similarity_search(query, ticker, k=5)
    
    _dbg("RAW-DATA", {
        "recent_count": len(recent),
        "similar_count": len(similar),
        "recent_sample": recent[:2] if recent else "None",
        "similar_sample": similar[:2] if similar else "None"
    })
    
    if not recent and not similar:
        no_data_msg = f"No news articles found for {ticker}. The news database may be empty or the ticker may not have recent coverage."
        _dbg("NO-DATA", no_data_msg)
        return {
            **state,
            "output": no_data_msg,
            "next_node": None,
        }
    
    # Format response
    formatted_response = format_news_response(ticker, query, recent, similar)
    
    # Convert to string for compatibility with the synthesis layer
    output_str = str(formatted_response)
    
    _dbg("FINAL-OUTPUT", output_str)
    
    return {
        **state,
        "output": output_str,
        "next_node": None,
    }

# ─── MCP Server Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "NewsAgent",
        port=8020,
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool(
        name="run_news_agent",
        description="Search for news articles about a specific stock ticker"
    )
    def _tool(state: dict) -> dict:
        """
        MCP Tool: Search for news articles about stocks
        
        Args:
            state: Dict containing:
                - input: User query about news
                - ticker: Optional explicit ticker symbol
                - chat_history: Previous conversation context
        
        Returns:
            Dict with news results in structured format
        """
        return run_news_agent(state)

    _dbg("SERVER-STARTING", f"News Agent MCP Server starting on port 8020")
    mcp.run(transport="streamable-http")