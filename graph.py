"""
 Multi-Agent Stock & News Assistant – LangGraph (MCP edition)
 -----------------------------------------------------------
 • router        – decides which specialist to run next
 • agent_sql     – calls the SQL MCP server (:8010/mcp)
 • agent_news    – calls the News MCP server (:8020/mcp)
 • agent_fallback– calls the Fallback MCP server (:8030/mcp)
 • synth         – writes the final answer and updates "memory"
"""

from __future__ import annotations

import asyncio, json, os, re, datetime, pprint, sys
from typing import Literal, Optional, TypedDict, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── Debug helper ────────────────────────────────────────────────────────
def _dbg(tag: str, obj) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === {tag} ===", file=sys.stderr, flush=True)
    pprint.pprint(obj, width=110, stream=sys.stderr)
    print("-" * 40, file=sys.stderr, flush=True)

def _extract_output(res):
    """Return tool output for both {output: ...} and raw-string shapes."""
    if isinstance(res, dict) and "output" in res:
        return res["output"]
    return res

# ─── Env & LLM ───────────────────────────────────────────────────────────
load_dotenv()
_LLM = ChatOllama(model=os.getenv("LLM_MODEL", "gemma2b:latest"), temperature=0)

# ─── MCP tool bootstrap ─────────────────────────────────────────────────
mcp_client: MultiServerMCPClient | None = None
sql_tool = news_tool = fb_tool = None

async def _init_mcp_tools():
    global mcp_client, sql_tool, news_tool, fb_tool
    if mcp_client is None:
        mcp_client = MultiServerMCPClient(
            {
                "sql":  {"url": "http://localhost:8010/mcp", "transport": "streamable_http"},
                "news": {"url": "http://localhost:8020/mcp", "transport": "streamable_http"},
                "fb":   {"url": "http://localhost:8030/mcp", "transport": "streamable_http"},
            }
        )
        for t in await mcp_client.get_tools():          # adapter returns list
            if t.name.endswith("run_sql_agent"):
                sql_tool = t
            elif t.name.endswith("run_news_agent"):
                news_tool = t
            elif t.name.endswith("run_fallback_agent"):
                fb_tool = t

        assert sql_tool and news_tool and fb_tool, "Missing MCP tools!"

asyncio.run(_init_mcp_tools())

# ─── Helpers ────────────────────────────────────────────────────────────
_US_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](20\d{2})\b")

def _normalize_dates(text: str) -> str:
    def _fix(m): return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return _US_DATE_RE.sub(_fix, text)

TICKER_MAP = {
    "aapl": "AAPL", "apple": "AAPL",
    "msft": "MSFT", "microsoft": "MSFT",
    "googl": "GOOGL", "google": "GOOGL", "alphabet": "GOOGL",
    "tsla": "TSLA", "tesla": "TSLA",
    "amzn": "AMZN", "amazon": "AMZN",
}
_TICKER_RE = re.compile(r"\$?[A-Za-z]{1,5}")

def _extract_ticker(q: str) -> str:
    if m := re.search(r"\$([A-Za-z]{1,5})\b", q): return m.group(1).upper()
    for w in re.findall(_TICKER_RE, q.lower()):
        if w in TICKER_MAP: return TICKER_MAP[w]
    upp = re.findall(r"\b([A-Z]{1,5})\b", q)
    return upp[0] if upp else ""

# ─── State schema ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    need_sql: bool
    need_news: bool
    sql_done: bool
    news_done: bool
    sql_result: Optional[str]
    news_result: Optional[str]
    answer: Optional[str]
    error: Optional[str]
    chat_history: List[HumanMessage | AIMessage]
    last_ticker: Optional[str]
    last_date: Optional[str]
    last_query: Optional[str]

# ─── Router ─────────────────────────────────────────────────────────────
_ROUTER_PROMPT = PromptTemplate.from_template("""
Return JSON: {{ "need_sql": true|false, "need_news": true|false }}
Rules:
- need_sql  = true if user asks about price/open/close/high/low/volume.
- need_news = true if user asks for news/headlines/articles/updates.
If ambiguous, default both to false.
Question: "{q}"
""")

def router_node(s: AgentState) -> AgentState:
    _dbg("ROUTER-IN", s)
    is_new = s.get("last_query") != s["query"]
    st: AgentState = {
        "query": s["query"],
        "chat_history": s.get("chat_history", []),
        "last_ticker": s.get("last_ticker"),
        "last_date": s.get("last_date"),
        "last_query": s["query"],
        "need_sql": False if is_new else s.get("need_sql", False),
        "need_news": False if is_new else s.get("need_news", False),
        "sql_done": False if is_new else s.get("sql_done", False),
        "news_done": False if is_new else s.get("news_done", False),
        "sql_result": None if is_new else s.get("sql_result"),
        "news_result": None if is_new else s.get("news_result"),
        "answer": None,
        "error": None,
    }
    if st["sql_done"]:  st["need_sql"]  = False
    if st["news_done"]: st["need_news"] = False
    st["last_ticker"] = _extract_ticker(st["query"]) or st.get("last_ticker", "")
    ql = st["query"].lower()
    st["need_sql"]  |= bool(re.search(r"\b(price|open|close|volume|high|low)\b", ql))
    st["need_news"] |= bool(re.search(r"\b(news|headline|article|update)\b", ql))

    try:
        raw = (_ROUTER_PROMPT | _LLM).invoke({"q": st["query"]}).content.strip()
        if raw.startswith("```"): raw = raw.strip("`").replace("json", "").strip()
        flags = json.loads(raw)
        st["need_sql"]  |= bool(flags.get("need_sql"))
        st["need_news"] |= bool(flags.get("need_news"))
    except Exception: pass
    _dbg("ROUTER-OUT", st)
    return st

# ─── Edge decision ──────────────────────────────────────────────────────
def decide_next(s: AgentState) -> Literal["agent_sql","agent_news","agent_fallback","synth"]:
    if s.get("error"): return "synth"
    if s["need_sql"]  and not s["sql_done"]:  return "agent_sql"
    if s["need_news"] and not s["news_done"]: return "agent_news"
    if not (s["need_sql"] or s["need_news"]): return "agent_fallback"
    return "synth"

# ─── Specialist nodes ───────────────────────────────────────────────────
async def agent_sql_node(s: AgentState) -> AgentState:
    _dbg("SQL-IN", s)
    try:
        res = await sql_tool.ainvoke({"state": {**s, "input": _normalize_dates(s["query"])}})
        _dbg("SQL-RAW", res)
        return {**s, "sql_result": _extract_output(res), "sql_done": True}
    except Exception as e:
        _dbg("SQL-ERR", e)
        return {**s, "error": f"SQL agent error: {e}", "sql_done": True, "need_sql": False}

async def agent_news_node(s: AgentState) -> AgentState:
    _dbg("NEWS-IN", s)
    try:
        ticker = s.get("last_ticker") or "AAPL"
        res = await news_tool.ainvoke({"state": {"input": f"latest news for {ticker}", **s}})
        _dbg("NEWS-RAW", res)
        return {**s, "news_result": _extract_output(res), "news_done": True}
    except Exception as e:
        _dbg("NEWS-ERR", e)
        return {**s, "error": f"News agent error: {e}", "news_done": True, "need_news": False}

async def agent_fallback_node(s: AgentState) -> AgentState:
    res = await fb_tool.ainvoke({"state": {"input": s["query"]}})
    return {**s, "sql_result": _extract_output(res), "sql_done": True, "news_done": True}

# ─── Synth node ─────────────────────────────────────────────────────────
_SYNTH_PROMPT = PromptTemplate.from_template("""
You are compiling the final answer for the user.
The variables below already contain the fetched data. **Use them verbatim.**

User question: {q}

Stock data returned (Python repr):
{sql}

News data returned (markdown):
{news}

== Formatting rules (MUST follow) ==

1. For the stock data, write a natural-language answer summarizing **all the available fields**.
   - Always include the ticker and date.
   - Mention each field/value clearly in a sentence.
   - Round all numeric prices to 2 decimals.

   Example for one field:
   "On June 12, 2025, AAPL's close price was 212.10."

   Example for multiple fields:
   "On June 12, 2025, AAPL opened at 190.23 and closed at 195.44."

2. Include the news section **only if the USER QUESTION explicitly asks about news, headlines, articles, or updates**.
   - If the user question does not mention these, completely omit the news section—even if news data is present.
   - If the user question does mention news, and news data is available, then:
     - Add a blank line.
     - Add the header: ### Latest headlines
     - List the 5 news headlines as a numbered list exactly as they appear. No extra summarizing.
   - If news data is empty, omit the header entirely.
""")

def synth_node(s: AgentState) -> AgentState:
    _dbg("SYNTH-IN", s)

    def _pretty_stock(val):
        if isinstance(val, str) and val.startswith("[("):   # "[(199.175,)]"
            try:
                num = float(val.strip("[() ,]"))
                return f"On {s['last_date']}, {s['last_ticker']} closed at ${num:.2f}."
            except Exception:
                pass
        return val or "No stock info."
    
    if s.get("error"):
        answer = f"Sorry – {s['error']}"
    else:
        answer = (_SYNTH_PROMPT | _LLM).invoke(
            {
                "q": s["query"],
                "sql": _pretty_stock(s.get("sql_result")),   # ← use helper
                "news": s.get("news_result") or "",
            }
        ).content.strip()
    _dbg("SYNTH-OUT", answer)
    chat = s.get("chat_history", []) + [HumanMessage(content=s["query"]), AIMessage(content=answer)]
    last_date = re.search(r"\d{4}-\d{2}-\d{2}", _normalize_dates(s["query"]))
    return {**s, "answer": answer, "chat_history": chat,
            "last_date": last_date.group() if last_date else s.get("last_date"),
            "sql_done": True, "news_done": True}

# ─── Build graph ────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("agent_sql", agent_sql_node, is_async=True)
workflow.add_node("agent_news", agent_news_node, is_async=True)
workflow.add_node("agent_fallback", agent_fallback_node, is_async=True)
workflow.add_node("synth", synth_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next,
    {"agent_sql":"agent_sql","agent_news":"agent_news",
     "agent_fallback":"agent_fallback","synth":"synth"})
for leaf in ("agent_sql","agent_news","agent_fallback"):
    workflow.add_edge(leaf, "router")
workflow.add_edge("synth", END)
workflow = workflow.compile()

# ─── CLI helper ─────────────────────────────────────────────────────────
def run_query_once(question: str) -> str:
    init: AgentState = {"query": question, "chat_history": [],
        "last_ticker": None, "last_date": None, "last_query": None,
        "need_sql": False, "need_news": False, "sql_done": False, "news_done": False,
        "sql_result": None, "news_result": None, "answer": None, "error": None}
    return workflow.invoke(init)["answer"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(run_query_once(" ".join(sys.argv[1:])))
    else:
        print("Interactive chat – type 'quit' to exit")
        state = None
        while True:
            q = input("You: ")
            if q.lower() in {"quit", "exit"}:
                break
            state = workflow.invoke({"query": q, **(state or {})})
            print("AI:", state.get("answer"))
