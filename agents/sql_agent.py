import re
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from tools.db import get_sql_agent_chain
from state import AgentState

# Initialize the SQL agent chain once (singleton)
sql_agent_chain = get_sql_agent_chain()


def run_sql_agent(state: AgentState) -> AgentState:
    """LangGraph node that powers stock-price queries via SQL."""

    # ------------------------------------------------------------------
    # 1. Unpack state & extract date tokens
    # ------------------------------------------------------------------
    history = state["chat_history"]
    question = state["input"]
    current_date = state.get("current_date", "")

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", question)          # YYYY-MM-DD
    slash_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", question)  # MM/DD/YY(YY)
    if iso_match:
        current_date = iso_match.group()
    elif slash_match:
        m, d, y = slash_match.groups()
        if len(y) == 2:
            y = "20" + y  # two-digit year → 20YY
        current_date = f"{y}-{int(m):02d}-{int(d):02d}"

    # Replace pronoun "that day"
    if "that day" in question.lower() and current_date:
        question = re.sub(r"\bthat day\b", current_date, question, flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # 2. Invoke the SQL agent
    # ------------------------------------------------------------------
    result: Any = sql_agent_chain.invoke({
        "input": question,
        "chat_history": history,
    })

    # DEBUG prints — comment out when satisfied
    print("DEBUG raw result ->", result, type(result))

    # ------------------------------------------------------------------
    # 3. Robust extraction of answer text
    # ------------------------------------------------------------------
    if isinstance(result, dict):
        answer_text = (
            result.get("output")   # LangChain 0.2 default
            or result.get("answer")
            or result.get("result")
            or ""
        )
    else:
        answer_text = str(result)

    print("DEBUG answer_text ->", repr(answer_text))

    if not answer_text:
        answer_text = "I couldn't find an answer."

    # ------------------------------------------------------------------
    # 4. Append exchange to chat history and return new state
    # ------------------------------------------------------------------
    messages = history + [HumanMessage(content=question), AIMessage(content=answer_text)]

    return {
    "chat_history": messages,
    "input": question,
    "output": answer_text,
    "current_date": current_date,
    "next_node": None,              # ← preserve key
}