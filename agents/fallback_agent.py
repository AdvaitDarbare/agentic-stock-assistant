from state import AgentState

def run_fallback_agent(state: AgentState) -> AgentState:
    query = state.get("input", "")
    answer = (
        "I’m not sure I can help with that. "
        "Try asking about stock prices or stock-related news."
    )
    state["output"] = answer
    state["next_node"] = None   # finished
    return state