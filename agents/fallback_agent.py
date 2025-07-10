"""
fallback_agent.py – MCP Server Version (Fully MCP Compliant)
───────────────────────────────────────────────────────────────────────
Handles questions that don't match SQL or news patterns.
Enhanced with intelligent responses and helpful guidance.
"""

import sys
from langchain_ollama import ChatOllama
from state import AgentState
import os
# ─── DEBUG LOGGER ────────────────────────────────────────────────────────
def _dbg(tag: str, txt):
    print(f"\n### FALLBACK-{tag}\n{txt}\n", file=sys.stderr, flush=True)

# ─── LLM for intelligent responses ────────────────────────────────────────
llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma:2b"), temperature=0.3)

def generate_helpful_response(query: str) -> str:
    """Generate a helpful response for non-stock queries"""
    
    fallback_prompt = f"""
You are a helpful financial assistant. The user asked a question that doesn't seem to be about stock prices or stock news.

User query: "{query}"

Provide a brief, helpful response that:
1. Acknowledges their question politely
2. Explains what you can help with (stock prices and stock news)
3. Gives 2-3 specific examples of questions you can answer
4. Keeps the tone friendly and professional

Examples of what you CAN help with:
- "What's Apple's stock price today?"
- "Show me MSFT news from this week"
- "TSLA open price from 2025-06-01 to 2025-06-10"
- "Latest headlines about Amazon"

Keep your response concise (2-3 sentences) and helpful.
"""

    try:
        response = _LLM.invoke(fallback_prompt).content.strip()
        _dbg("LLM-RESPONSE", response)
        return response
    except Exception as e:
        _dbg("LLM-ERROR", f"Failed to generate response: {e}")
        # Fallback to static message
        return (
            "I'm not sure I can help with that specific question. "
            "I specialize in stock prices and stock-related news. "
            "Try asking about stock prices (like 'AAPL close price') or stock news (like 'Tesla latest news')."
        )

def run_fallback_agent(state: AgentState) -> AgentState:
    _dbg("INPUT-STATE", state)
    
    query = state.get("input", "").strip()
    
    if not query:
        answer = "Hello! I'm here to help with stock prices and financial news. What would you like to know?"
    else:
        # Check for common patterns that might need redirection
        query_lower = query.lower()
        
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            answer = (
                "Hello! I'm your financial assistant. I can help you with stock prices and stock news. "
                "Try asking something like 'AAPL stock price' or 'latest Microsoft news'."
            )
        elif any(question in query_lower for question in ["what can you do", "help", "how do you work"]):
            answer = (
                "I can help you with two main things:\n"
                "1. **Stock Prices**: Ask about open, close, high, low prices for any ticker (e.g., 'TSLA close price')\n"
                "2. **Stock News**: Get latest headlines and news for companies (e.g., 'Apple news today')\n\n"
                "Just mention a company name or ticker symbol and what you'd like to know!"
            )
        else:
            # Use LLM to generate a contextual response
            answer = generate_helpful_response(query)
    
    _dbg("FINAL-OUTPUT", answer)
    
    return {
        **state,
        "output": answer,
        "next_node": None,
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

    @mcp.tool(
        name="run_fallback_agent",
        description="Handle general questions not related to stock prices or news"
    )
    def _tool(state: dict) -> dict:
        """
        MCP Tool: Handle fallback cases for non-stock queries
        
        Args:
            state: Dict containing:
                - input: User query that doesn't match stock/news patterns
                - chat_history: Previous conversation context
        
        Returns:
            Dict with helpful fallback response
        """
        return run_fallback_agent(state)

    _dbg("SERVER-STARTING", "Fallback Agent MCP Server starting on port 8030")
    mcp.run(transport="streamable-http")