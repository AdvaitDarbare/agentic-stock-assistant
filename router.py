# agents/router.py
"""
LLM-powered router node for the LangGraph workflow.

It inspects the user’s message and decides which downstream
agent should handle the request:

• "sql_agent"   – stock prices, volumes, dates, financial info
• "news_agent"  – recent headlines, articles, company news
• "fallback"    – anything unrelated or ambiguous
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from route_schema import RouteDecision
from state import AgentState

import json

###############################################################################
# 1. LLM with structured-output via *function-calling* (needed for gpt-3.5)
###############################################################################
llm = ChatOllama(model="gemma3n:e2b", temperature=0)

# Use Ollama’s function-calling interface so the response
# can be parsed directly into the RouteDecision Pydantic model.
router_llm = llm

###############################################################################
# 2. Router node
###############################################################################
def run_llm_router(state: AgentState) -> AgentState:
    try:
        response = router_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a routing assistant for a stock-and-news chatbot.\n\n"
                        "Choose only one of the following routes:\n"
                        "- 'sql_agent' for stock prices, volumes, and dates.\n"
                        "- 'news_agent' for news, headlines, and company updates.\n"
                        "- 'fallback' for unrelated or unclear queries.\n\n"
                        "Just reply with one of: sql_agent, news_agent, or fallback."
                    )
                ),
                HumanMessage(content=state.get("input", "")),
            ]
        )

        # Clean raw string response
        choice = response.content.strip().lower()

        if choice in {"sql_agent", "news_agent", "fallback"}:
            state["next_node"] = choice
        else:
            print("⚠️ Unexpected router output →", choice)
            state["next_node"] = "fallback"

        print("🧭 Router decision →", state["next_node"])

    except Exception as err:
        print("[Router] error → fallback:", err)
        state["next_node"] = "fallback"

    return state