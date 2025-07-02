"""
 Multi-Agent Stock & News Assistant – LangGraph
 ------------------------------------------------
 • router        – decides which specialist to run next
 • agent_sql     – fetches numbers from Postgres
 • agent_news    – fetches recent headlines
 • agent_fallback– handles chit-chat/unknowns
 • synth         – writes the final answer and updates "memory"
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional, TypedDict, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# ─── Import specialist agents ────────────────────────────────────────────
from agents.sql_agent import run_sql_agent
from agents.news_agent import run_news_agent
from agents.fallback_agent import run_fallback_agent

# ─── Env & LLM ───────────────────────────────────────────────────────────
load_dotenv()
_LLM = ChatOllama(model=os.getenv("LLM_MODEL", "gemma2b:latest"), temperature=0)

# ─── Helpers ──────────────────────────────────────────────────────────────
_US_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](20\d{2})\b")


def _normalize_dates(text: str) -> str:
    """Turn 06/11/2025 → 2025-06-11 so Postgres is happy."""
    def _fix(m):
        mon, day, yr = m.group(1), m.group(2), m.group(3)
        return f"{yr}-{int(mon):02d}-{int(day):02d}"

    return _US_DATE_RE.sub(_fix, text)


TICKER_MAP = {
    # aliases and symbols
    "aapl": "AAPL", "apple": "AAPL",
    "msft": "MSFT", "microsoft": "MSFT",
    "googl": "GOOGL", "google": "GOOGL", "alphabet": "GOOGL",
    "tsla": "TSLA", "tesla": "TSLA",
    "amzn": "AMZN", "amazon": "AMZN",
}

_TICKER_RE = re.compile(r"\$?[A-Za-z]{1,5}")


def _extract_ticker(q: str) -> str:
    """Extract an explicit $TICKER, alias, or bare symbol; return "" if none."""
    dollar = re.search(r"\$([A-Za-z]{1,5})\b", q)
    if dollar:
        return dollar.group(1).upper()

    for word in re.findall(_TICKER_RE, q.lower()):
        if word in TICKER_MAP:
            return TICKER_MAP[word]

    upper = re.findall(r"\b([A-Z]{1,5})\b", q)
    if upper:
        return upper[0]
    return ""


# ─── State type ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # per-turn input
    query: str
    # router-derived flags
    need_sql: bool
    need_news: bool
    sql_done: bool
    news_done: bool
    sql_result: Optional[str]
    news_result: Optional[str]
    answer: Optional[str]
    error: Optional[str]
    # long-term memory
    chat_history: List[HumanMessage | AIMessage]
    last_ticker: Optional[str]
    last_date: Optional[str]
    last_query: Optional[str]         # ← keeps track of previous question


# ─── Router ───────────────────────────────────────────────────────────────
_ROUTER_PROMPT = PromptTemplate.from_template(
    """
Analyze this question and return JSON exactly like:
  {{ "need_sql": true|false, "need_news": true|false }}

Rules:
- need_sql  = true if the user asks about price / open / close / high / low / volume or any financial data.
- need_news = true if the user asks for news, headlines, articles, or updates.
- If the question asks for BOTH data & news, set both to true.
- If the question is ambiguous or general, default both to false and let the fallback agent handle it.

Question: "{q}"
"""
)


def router_node(s: AgentState) -> AgentState:
    """
    Decide what the user needs and produce a per-turn state.

    • sql_done/news_done flags set by specialist nodes are preserved so that
      the same agent is not called twice in a single turn.
    • If the user sends a new question (query ≠ last_query), wipe old flags
      and results so they don’t bleed into the next turn.
    """
    print("\n--- ENTERING ROUTER ---")
    print(f"Incoming state: {s}")

    is_new_query = s.get("last_query") != s["query"]

    # carry over previous per-turn info
    new_state: AgentState = {
        "query": s["query"],
        "chat_history": s.get("chat_history", []),
        "last_ticker": s.get("last_ticker"),
        "last_date": s.get("last_date"),
        "last_query": s["query"],               # update memory
        "need_sql": s.get("need_sql", False),
        "need_news": s.get("need_news", False),
        "sql_done": s.get("sql_done", False),
        "news_done": s.get("news_done", False),
        "sql_result": s.get("sql_result"),
        "news_result": s.get("news_result"),
        "answer": None,
        "error": None,
    }

    # clear turn-scoped keys for a brand-new question
    if is_new_query:
        new_state.update(
            need_sql=False,
            need_news=False,
            sql_done=False,
            news_done=False,
            sql_result=None,
            news_result=None,
            error=None,
        )

    # ticker extraction
    current_ticker = _extract_ticker(new_state["query"]) or new_state.get("last_ticker", "")
    print(f"Previous Ticker: '{s.get('last_ticker')}', Current Ticker: '{current_ticker}'")
    new_state["last_ticker"] = current_ticker

    # heuristics
    q_low = new_state["query"].lower()
    need_sql_heuristic = bool(re.search(r"\b(price|open|close|volume|high|low|financial data|stock data)\b", q_low))
    need_news_heuristic = bool(re.search(r"\b(news|headline|article|update|latest updates|related news)\b", q_low))

    # LLM confirmation
    try:
        raw = (_ROUTER_PROMPT | _LLM).invoke({"q": new_state["query"]}).content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "").strip()
        flags = json.loads(raw)
        need_sql_llm = bool(flags.get("need_sql"))
        need_news_llm = bool(flags.get("need_news"))
    except Exception as exc:
        print("Router fallback to heuristics:", exc)
        need_sql_llm = need_news_llm = False

    new_state["need_sql"] = new_state["need_sql"] or need_sql_heuristic or need_sql_llm
    new_state["need_news"] = new_state["need_news"] or need_news_heuristic or need_news_llm

    print(f"Determined needs: SQL={new_state['need_sql']}, News={new_state['need_news']}")
    print(f"Outgoing state: {new_state}")
    print("--- EXITING ROUTER ---\n")
    return new_state


# ─── Decide edge ─────────────────────────────────────────────────────────
def decide_next(s: AgentState) -> Literal["agent_sql", "agent_news", "agent_fallback", "synth"]:
    if s.get("error"):
        return "synth"
    if s.get("need_sql") and not s.get("sql_done"):
        return "agent_sql"
    if s.get("need_news") and not s.get("news_done"):
        return "agent_news"
    if not (s.get("need_sql") or s.get("need_news")):
        return "agent_fallback"
    if (s.get("sql_done") or not s.get("need_sql")) and (
        s.get("news_done") or not s.get("need_news")
    ):
        return "synth"
    return "synth"  # safeguard


# ─── Specialist nodes ────────────────────────────────────────────────────
def agent_sql_node(s: AgentState) -> AgentState:
    print("\n>>> EXECUTING SQL AGENT NODE <<<")
    try:
        cleaned_q = _normalize_dates(s["query"])
        payload = {
            "input": cleaned_q,
            "last_ticker": s.get("last_ticker"),
            "last_date": s.get("last_date"),
            "chat_history": s.get("chat_history", []),
        }
        result_state = run_sql_agent(payload)
        result = result_state.get("output", "")
        print(f"SQL agent result: {result}\n")
        return {**s, "sql_result": result, "sql_done": True}
    except Exception as exc:
        print(f"SQL agent error: {exc}\n")
        return {**s, "error": f"SQL agent error: {exc}"}


def agent_news_node(s: AgentState) -> AgentState:
    print("\n>>> EXECUTING NEWS AGENT NODE <<<")
    try:
        symbol = s.get("last_ticker")
        if not symbol:
            return {**s, "error": "News agent error: Could not identify a ticker."}
        result = run_news_agent({"input": f"latest news for {symbol}"}).get("output", "")
        print(f"News agent result: {result[:100]}...\n")
        return {**s, "news_result": result, "news_done": True}
    except Exception as exc:
        print(f"News agent error: {exc}\n")
        return {**s, "error": f"News agent error: {exc}"}


def agent_fallback_node(s: AgentState) -> AgentState:
    print("\n>>> EXECUTING FALLBACK AGENT NODE <<<")
    try:
        result = run_fallback_agent({"input": s["query"]}).get("output", "")
        return {**s, "sql_result": result, "sql_done": True, "news_done": True}
    except Exception as exc:
        return {**s, "error": f"Fallback agent error: {exc}"}


# ─── Synth node ──────────────────────────────────────────────────────────
_SYNTH_PROMPT = PromptTemplate.from_template("""
You are compiling the final answer for the user.
The variables below already contain the fetched data. **Use them verbatim.**

User question: {q}

Stock data returned (Python repr):
{sql}

News data returned (markdown):
{news}

== Formatting rules (MUST follow) ==
1. Begin with the **ticker, date, and requested field/value** on its own line:
   `AMZN close on 2025-06-13  →  212.10`
   • Round prices to 2 decimals.
2. Add a blank line, then the header `### Latest headlines` (level-3 markdown).
3. Show the 5 headlines as a numbered list **exactly as they appear** (no extra summarising).
4. If news is empty, omit the header entirely.
""")


def synth_node(s: AgentState) -> AgentState:
    print("\n--- ENTERING SYNTH NODE ---")
    if s.get("error"):
        answer = f"Sorry – {s['error']}"
    else:
        stock = s.get("sql_result") or "No stock info found."
        news = s.get("news_result") or ""
        print(f"Synthesizing final answer with stock: '{stock}' and news: '{news[:100]}...'")
        answer = (_SYNTH_PROMPT | _LLM).invoke(
            {"q": s["query"], "sql": stock, "news": news}
        ).content.strip()

    print("[DEBUG] Final synthesized answer:", answer)


    # update memory
    chat = s.get("chat_history", [])
    chat += [HumanMessage(content=s["query"]), AIMessage(content=answer)]

    # remember latest date, if any
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", _normalize_dates(s["query"]))
    last_date = date_match.group() if date_match else s.get("last_date")

    return {
        **s,
        "answer": answer,
        "chat_history": chat,
        "last_date": last_date,
        "sql_done": True,
        "news_done": True,
    }


# ─── Build graph ─────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("agent_sql", agent_sql_node)
workflow.add_node("agent_news", agent_news_node)
workflow.add_node("agent_fallback", agent_fallback_node)
workflow.add_node("synth", synth_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    decide_next,
    {
        "agent_sql": "agent_sql",
        "agent_news": "agent_news",
        "agent_fallback": "agent_fallback",
        "synth": "synth",
    },
)
for leaf in ("agent_sql", "agent_news", "agent_fallback"):
    workflow.add_edge(leaf, "router")
workflow.add_edge("synth", END)
workflow = workflow.compile()


# ─── CLI runner for quick tests ─────────────────────────────────────────
def run_query_once(question: str):
    init: AgentState = {
        "query": question,
        "chat_history": [],
        "last_ticker": None,
        "last_date": None,
        "last_query": None,   # ← initialise
        "need_sql": False,
        "need_news": False,
        "sql_done": False,
        "news_done": False,
        "sql_result": None,
        "news_result": None,
        "answer": None,
        "error": None,
    }
    return workflow.invoke(init)["answer"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(run_query_once(q))
    else:
        print("Interactive chat – type 'quit' to exit")
        state = None
        while True:
            q = input("You: ")
            if q.lower() in {"quit", "exit"}:
                break
            state = workflow.invoke({"query": q, **(state or {})})
            print("AI:", state.get("answer"))
