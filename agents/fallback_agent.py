"""
fallback_agent.py – MCP Server Version
───────────────────────────────────────────────
Handles questions that don't match SQL or news.
Returns a polite fallback message.
"""

from state import AgentState

def run_fallback_agent(state: AgentState) -> AgentState:
    query = state.get("input", "")
    answer = (
        "I’m not sure I can help with that. "
        "Try asking about stock prices or stock-related news."
    )
    return {
        **state,
        "output": answer,
        "next_node": None
    }

# ─── MCP Server Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "FallbackAgent",
        port=8030,
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool(name="run_fallback_agent")
    def _tool(state: dict) -> dict:
        return run_fallback_agent(state)

    mcp.run(transport="streamable-http")
