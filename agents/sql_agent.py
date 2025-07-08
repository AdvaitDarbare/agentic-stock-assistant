"""
sql_agent.py – MCP Server Version
────────────────────────────────────────────────────────
Generates and executes SQL queries against the `stock_data` table.
"""

import re
import sys
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import mlflow.langchain  # keeps LangChain autolog active

from tools.db import get_sql_agent_chain
from state import AgentState

# ─── DEBUG LOGGER ────────────────────────────────────────────────────────
def _dbg(tag: str, txt):
    print(f"\n### {tag}\n{txt}\n", file=sys.stderr, flush=True)

# ─── Helpers ─────────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL)

def _clean_sql(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip("`").replace("sql", "").strip()

def _validate(sql: str) -> None:
    lowered = sql.lower()
    if "stock_data" not in lowered:
        raise ValueError("query must reference only stock_data")
    banned = {"news", "headline", "url"}
    if any(word in lowered for word in banned):
        raise ValueError("query mentions invalid column")

# ─── Agent Function ──────────────────────────────────────────────────────
def run_sql_agent(state: AgentState) -> AgentState:
    sql_agent_chain = get_sql_agent_chain()
    llm = ChatOllama(
        model=sql_agent_chain.llm.model,
        temperature=sql_agent_chain.llm.temperature,
        verbose=False,               # keep prompt quiet; we print manually
    )

    history: List = state.get("chat_history", []) or []
    question: str = state.get("input", "").strip()

    # — date extraction for prompt completeness —
    current_date: str = state.get("current_date", "")
    iso = re.search(r"\d{4}-\d{2}-\d{2}", question)
    slash = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", question)
    if iso:
        current_date = iso.group()
    elif slash:
        m, d, y = slash.groups()
        if len(y) == 2:
            y = "20" + y
        current_date = f"{y}-{int(m):02d}-{int(d):02d}"

    # prompt construction
    try:
        table_info = sql_agent_chain.input_schema.db.get_table_info()
    except Exception:
        table_info = (
            "stock_data(ticker TEXT, date DATE, "
            "open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC)"
        )

    system_prompt = f"""
You have exactly ONE table:

  {table_info.strip()}

✅ Never reference any other table or column.
✅ If the user also mentions news / headlines / articles / updates,
   IGNORE that part and answer only the price query.
✅ Return a single
     SELECT … FROM stock_data WHERE ticker = '<TICKER>' AND date = '<YYYY-MM-DD>';
   Wrap the SQL in ```sql … ```.

Now answer: {{input}}
""".strip()

    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=question)]

    cleaned_sql = ""
    try:
        # ----- LLM call -----
        response = llm.invoke(messages).content
        _dbg("RAW LLM RESPONSE", response)

        cleaned_sql = _clean_sql(response)
        _dbg("CLEANED SQL", cleaned_sql)

        _validate(cleaned_sql)

        # ----- DB query -----
        raw_answer = sql_agent_chain.db.run(cleaned_sql)
        _dbg("DB RESULT", raw_answer)

        answer_text = str(raw_answer)
    except Exception as err:
        _dbg("ERROR", err)
        answer_text = "Sorry, I couldn't answer that due to a database error."

    if not answer_text:
        answer_text = "I couldn't find an answer."

    new_history = history + [HumanMessage(content=question),
                             AIMessage(content=answer_text)]

    return {
        **state,
        "chat_history": new_history,
        "input": question,
        "output": answer_text,
        "current_date": current_date,
        "next_node": None,
    }

# ─── MCP Server Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "SqlAgent",
        port=8010,
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool(name="run_sql_agent")
    def _tool(state: dict) -> dict:
        return run_sql_agent(state)

    mcp.run(transport="streamable-http")
