
"""
agents/sql_agent.py
Generates and executes SQL queries against the `stock_data` table.
Ensures the LLM never references other tables and ignores any news-related text.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import mlflow.langchain  # keeps LangChain autolog active

from tools.db import get_sql_agent_chain          # factory → chain with .db & .llm
from state import AgentState                      # shared TypedDict used by graph

# ─── helpers ──────────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL)


def _clean_sql(text: str) -> str:
    """Extract SQL from ```sql … ``` or fallback to stripping backticks."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip("`").replace("sql", "").strip()


def _validate(sql: str) -> None:
    """Reject any query that touches unknown tables/columns."""
    lowered = sql.lower()
    if "stock_data" not in lowered:
        raise ValueError("query must reference only stock_data")
    banned = {"news", "headline", "url"}
    if any(word in lowered for word in banned):
        raise ValueError("query mentions invalid column")


# ─── main entry point called by the LangGraph node ───────────────────────
def run_sql_agent(state: AgentState) -> AgentState:
    """
    Build a schema-aware prompt, ask the LLM for SQL, execute it,
    and return a partial AgentState.
    """
    sql_agent_chain = get_sql_agent_chain()       # provides .db and .llm
    llm = ChatOllama(
        model=sql_agent_chain.llm.model,
        temperature=sql_agent_chain.llm.temperature,
    )

    history: List = state.get("chat_history", []) or []
    question: str = state.get("input", "").strip()

    # —— best-guess current date (used in few-shot example) ——
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

    # —— fetch schema for grounding ——
    try:
        table_info = sql_agent_chain.input_schema.db.get_table_info()
    except Exception:
        table_info = (
            "stock_data("
            "ticker TEXT, date DATE, "
            "open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC)"
        )

    # —— system prompt with rules + few-shot ——
    system_prompt = f"""You have exactly ONE table:

  {table_info.strip()}

✅ Never reference any other table or column.
✅ If the user also mentions news / headlines / articles / updates,
   IGNORE that part and answer only the price query.
✅ Return a single
     SELECT … FROM stock_data WHERE ticker = '<TICKER>' AND date = '<YYYY-MM-DD>';
   Wrap the SQL in ```sql … ```.

Examples you MUST follow:

Q1: “What was TSLA’s close on 2025-06-16?”
A1:
```sql
SELECT close
  FROM stock_data
 WHERE ticker = 'TSLA' AND date = '2025-06-16';
```

Q2: “Open price of MSFT on 06/11/2025 and today’s headlines on Microsoft”
A2:
```sql
SELECT open
  FROM stock_data
 WHERE ticker = 'MSFT' AND date = '2025-06-11';
```
-- (notice: ignored the news clause)

Q3: “High and low for AMZN yesterday”
A3:
```sql
SELECT high, low
  FROM stock_data
 WHERE ticker = 'AMZN' AND date = {current_date or 'YYYY-MM-DD'};
```

Now answer: {{input}}
""".strip()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    cleaned_sql = ""
    try:
        # ask the LLM for SQL
        response = llm.invoke(messages).content
        cleaned_sql = _clean_sql(response)
        print("[DEBUG] Generated SQL:", cleaned_sql)

        _validate(cleaned_sql)                     # belt-and-suspenders

        raw_answer = sql_agent_chain.db.run(cleaned_sql)
        answer_text = str(raw_answer)
    except Exception as exc:
        print("[ERROR] SQL chain or DB run failed:", exc)
        if cleaned_sql:
            print("[ERROR] SQL that failed:", cleaned_sql)
        answer_text = "Sorry, I couldn't answer that due to a database error."

    if not answer_text:
        answer_text = "I couldn't find an answer."

    new_history = history + [
        HumanMessage(content=question),
        AIMessage(content=answer_text),
    ]

    return {
        "chat_history": new_history,
        "input": question,
        "output": answer_text,
        "current_date": current_date,
        "next_node": None,
    }
