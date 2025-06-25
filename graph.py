from langgraph.graph import StateGraph
from state import AgentState
from router import run_router
from agents.sql_agent import run_sql_agent
from agents.news_agent import run_news_agent

def decide_next_node(state: AgentState) -> str:
    """
    Conditional edge function that returns the name of the next node.
    LangGraph will use this return value to route to the appropriate node.
    """
    return state["next_node"]

# Build the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("router", run_router)
workflow.add_node("sql_agent", run_sql_agent)
workflow.add_node("news_agent", run_news_agent)

# Entry point
workflow.set_entry_point("router")

# Conditional routing: the function returns which node name to go to
workflow.add_conditional_edges(
    "router",                    # from this node
    decide_next_node,           # this function decides where to go
    ["sql_agent", "news_agent"] # possible destination nodes
)

# Both agents are end points
workflow.set_finish_point("sql_agent")
workflow.set_finish_point("news_agent")

graph = workflow.compile()