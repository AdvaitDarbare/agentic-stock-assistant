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
from langchain_openai import ChatOpenAI
from route_schema import RouteDecision
from state import AgentState

###############################################################################
# 1. LLM with structured-output via *function-calling* (needed for gpt-3.5)
###############################################################################
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Use OpenAI’s function-calling interface so the response
# can be parsed directly into the RouteDecision Pydantic model.
router_llm = llm.with_structured_output(
    RouteDecision,
    method="function_calling",          # <- critical for gpt-3.5
)

###############################################################################
# 2. Router node
###############################################################################
def run_llm_router(state: AgentState) -> AgentState:
    """
    Decide which downstream node to run next and store that choice
    in state["next_node"]. Falls back gracefully if the LLM response
    can’t be parsed.
    """
    try:
        decision = router_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a routing assistant for a stock-and-news chatbot.\n\n"
                        "Choose one:\n"
                        "• If the user asks about stock prices, volumes, dates or other "
                        "numeric financial info → return \"sql_agent\".\n"
                        "• If the user asks for recent news, headlines, announcements, "
                        "or company events → return \"news_agent\".\n"
                        "• If the request is unrelated or unclear → return \"fallback\".\n\n"
                        "Respond ONLY with the value in the `step` field."
                    )
                ),
                HumanMessage(content=state.get("input", "")),
            ]
        )
        state["next_node"] = decision.step      # "sql_agent" | "news_agent" | "fallback"
        print("🧭 Router decision →", decision.step)

    except Exception as err:
        # Any parsing failure or LLM issue routes to fallback
        print("[Router] structured output failed → fallback:", err)
        state["next_node"] = "fallback"

    return state