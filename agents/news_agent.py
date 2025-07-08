"""
news_agent.py – MCP Server Version
───────────────────────────────────────────────
Handles questions about company news by searching the `news_articles` table.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import psycopg2
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg2 import register_vector
from state import AgentState

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

# ─── DB CONNECTION ────────────────────────────────────────────────────────
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
    )
    register_vector(conn)
    return conn

# ─── LOAD TICKERS AT STARTUP ─────────────────────────────────────────────
def _load_known_tickers() -> set[str]:
    sql = f"SELECT DISTINCT {STOCK_COL} FROM {TABLE_NAME};"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return {row[0].upper() for row in cur.fetchall()}

KNOWN_TICKERS = _load_known_tickers()

# ─── TICKER EXTRACTION ───────────────────────────────────────────────────
_DOLLAR_RE = re.compile(r"\$([A-Za-z]{1,5})")
_UPPER_RE  = re.compile(r"\b[A-Z]{1,5}\b")

def extract_ticker(q: str) -> str | None:
    q_low = q.lower()

    m = _DOLLAR_RE.search(q)
    if m and (sym := m.group(1).upper()) in KNOWN_TICKERS:
        return sym

    for sym in _UPPER_RE.findall(q):
        if sym in KNOWN_TICKERS:
            return sym

    for name, sym in NAME_MAP.items():
        if name in q_low and sym in KNOWN_TICKERS:
            return sym

    return None

# ─── DB QUERIES ──────────────────────────────────────────────────────────
def fetch_raw_articles(ticker: str, limit: int = 5) -> List[Tuple[str, datetime]]:
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

def similarity_search(query: str, ticker: str, k: int = 5):
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

# ─── Agent Function ──────────────────────────────────────────────────────
def run_news_agent(state: AgentState) -> AgentState:
    q = state.get("input", "").strip() or "latest news"

    ticker = extract_ticker(q)
    if not ticker:
        return {
            **state,
            "output": (
                "Sorry—couldn’t recognise a valid ticker in your question. "
                "Please phrase it like “latest news for $TSLA” or “Tesla news”."
            ),
            "next_node": None,
        }

    recent = fetch_raw_articles(ticker, limit=5)
    sim    = similarity_search(q, ticker, k=5)

    lines: List[str] = [f"✅ **Latest 5 headlines for {ticker}:**"]
    for i, (headline, pub_date) in enumerate(recent, 1):
        lines.append(f"{i}. [{pub_date}] {headline.replace('\n', ' ')[:200]}")

    lines.append(f"\n🎯 **Top‑5 similar to “{q}” ({ticker}):**")
    for i, (headline, pub_date, score) in enumerate(sim, 1):
        lines.append(f"{i}. [{pub_date}] (sim={score:.3f}) {headline.replace('\n', ' ')[:200]}")

    return {
        **state,
        "output": "\n".join(lines),
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

    @mcp.tool(name="run_news_agent")
    def _tool(state: dict) -> dict:
        return run_news_agent(state)

    mcp.run(transport="streamable-http")




