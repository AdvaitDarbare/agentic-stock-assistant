from state import AgentState

def run_fallback_agent(state: AgentState) -> AgentState:
    query = state.get("input", "")
    answer = (
        "I’m not sure I can help with that. "
        "Try asking about Tesla stock prices or Tesla-related news."
    )
    state["output"] = answer
    state["next_node"] = None   # finished
    return state