import re
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from tools.db import get_sql_agent_chain
from state import AgentState

def run_sql_agent(state: AgentState) -> AgentState:
    # Create a fresh agent each time for hot-reload compatibility
    sql_agent_chain = get_sql_agent_chain()

    # Get input state
    history = state.get("chat_history", [])
    if not isinstance(history, list):
        history = []

    question = state.get("input", "").strip()
    current_date = state.get("current_date", "")

    # Try to extract a date from the question
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", question)
    slash_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", question)

    if iso_match:
        current_date = iso_match.group()
    elif slash_match:
        m, d, y = slash_match.groups()
        if len(y) == 2:
            y = "20" + y
        current_date = f"{y}-{int(m):02d}-{int(d):02d}"

    # Run the SQL agent with chat history and question
    result: Any = sql_agent_chain.invoke({
        "input": question,
        "chat_history": history,
    })

    print("DEBUG raw result ->", result, type(result))

    if isinstance(result, dict):
        answer_text = (
            result.get("output")
            or result.get("answer")
            or result.get("result")
            or ""
        )
    else:
        answer_text = str(result)

    print("DEBUG answer_text ->", repr(answer_text))

    if not answer_text:
        print(f"[WARN] No answer found for: {question}")
        answer_text = "I couldn't find an answer."

    # Append new exchange to chat history
    messages = history + [
        HumanMessage(content=question),
        AIMessage(content=answer_text)
    ]

    return {
        "chat_history": messages,
        "input": question,
        "output": answer_text,
        "current_date": current_date,
        "next_node": None,
    }