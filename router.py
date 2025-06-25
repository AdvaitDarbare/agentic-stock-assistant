# router.py
from state import AgentState

NEWS_WORDS = {"news", "headline", "article", "update"}

def run_router(state: AgentState) -> AgentState:
    """
    Decide WHICH node to call next, then stash that choice in state["next_node"].
    Return only the state dict (LangGraph requirement).
    """
    q = state["input"].lower()
    state["next_node"] = (
        "news_agent"
        if any(w in q for w in NEWS_WORDS)
        else "sql_agent"
    )
    return state           # <- single dict, not a tuple