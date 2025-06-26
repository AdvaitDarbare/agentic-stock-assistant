# graph.py
from langgraph.graph import StateGraph
from state import AgentState
from router import run_llm_router  # <- NEW
from agents.sql_agent import run_sql_agent
from agents.news_agent import run_news_agent
from agents.fallback_agent import run_fallback_agent

def decide_next_node(state: AgentState) -> str:
    return state["next_node"]

workflow = StateGraph(AgentState)
workflow.add_node("router", run_llm_router)  # <- LLM-powered
workflow.add_node("sql_agent", run_sql_agent)
workflow.add_node("news_agent", run_news_agent)
workflow.add_node("fallback", run_fallback_agent)


workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next_node, ["sql_agent", "news_agent", "fallback"])
workflow.set_finish_point("sql_agent")
workflow.set_finish_point("news_agent")
workflow.set_finish_point("fallback")


graph = workflow.compile()