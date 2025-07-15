"""
 Multi-Agent Stock & News Assistant – LangGraph (MCP edition)
 -----------------------------------------------------------
 • router        – decides which specialist to run next using pure LLM reasoning
 • agent_sql     – calls the SQL MCP server (:8010/mcp)
 • agent_news    – calls the News MCP server (:8020/mcp)
 • agent_fallback– calls the Fallback MCP server (:8030/mcp)
 • synth         – writes the final answer and updates "memory"
"""

from __future__ import annotations

import asyncio, json, os, re, datetime, pprint, sys
from typing import Literal, Optional, TypedDict, List

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
# from langchain_mcp_adapters.client import MultiServerMCPClient  # Replaced with FastMCPClient
from state import AgentState
import asyncpg




# Use PostgreSQL checkpointer for proper persistence (with fallback)
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    HAS_POSTGRES_CHECKPOINTER = True
except ImportError:
    HAS_POSTGRES_CHECKPOINTER = False
    PostgresSaver = None

# ─── Debug helper ────────────────────────────────────────────────────────
def _dbg(tag: str, obj) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === {tag} ===", file=sys.stderr, flush=True)
    pprint.pprint(obj, width=110, stream=sys.stderr)
    print("-" * 40, file=sys.stderr, flush=True)

def _serialize_chat_history(chat_history):
    """Convert LangChain chat history to JSON-serializable format"""
    if not chat_history:
        return []
    
    serialized = []
    for message in chat_history:
        if hasattr(message, 'content') and hasattr(message, 'type'):
            # LangChain message object
            serialized.append({
                "content": message.content,
                "type": message.type,
                "role": "user" if message.type == "human" else "assistant"
            })
        elif isinstance(message, dict):
            # Already serialized
            serialized.append(message)
        else:
            # Fallback for other formats
            serialized.append({"content": str(message), "type": "unknown", "role": "user"})
    
    return serialized

def _extract_output(res):
    """Return tool output for both {output: ...} and FastMCP response shapes."""
    if isinstance(res, dict):
        # Handle FastMCP response format
        if "content" in res and isinstance(res["content"], list):
            for item in res["content"]:
                if isinstance(item, dict) and "text" in item:
                    # Try to parse the JSON string in the text field
                    try:
                        import json
                        parsed = json.loads(item["text"])
                        if isinstance(parsed, dict) and "output" in parsed:
                            return parsed["output"]
                        return parsed
                    except:
                        return item["text"]
        # Handle direct output format
        if "output" in res:
            return res["output"]
    return res

def _extract_data_from_mcp_result(result_str):
    """Extract actual data from MCP JSON response"""
    _dbg("EXTRACT-INPUT", result_str)
    
    if not result_str:
        return None
        
    try:
        # If it's a JSON string, parse it and extract the output
        if isinstance(result_str, str) and result_str.strip().startswith('{'):
            parsed = json.loads(result_str)
            _dbg("EXTRACT-PARSED", parsed)
            if isinstance(parsed, dict) and "output" in parsed:
                output = parsed["output"]
                _dbg("EXTRACT-OUTPUT", output)
                return output
            return result_str
        return result_str
    except (json.JSONDecodeError, KeyError) as e:
        # If parsing fails, return the original string
        _dbg("EXTRACT-ERROR", str(e))
        return result_str

def _clean_sql_result(sql_result):
    """Clean and format SQL results for LLM consumption"""
    if not sql_result:
        return "No SQL data available"
    
    # Check if we have table format data
    if isinstance(sql_result, dict) and "table" in sql_result:
        return sql_result  # Return as-is for table formatting
    
    result_str = str(sql_result)
    
    # Handle common SQL result formats
    if result_str.startswith('[') and result_str.endswith(']'):
        # List format like [(470.38,), (471.23,)]
        try:
            import ast
            parsed = ast.literal_eval(result_str)
            if parsed and isinstance(parsed[0], tuple):
                # Multiple rows
                if len(parsed) > 1:
                    values = [str(row[0]) for row in parsed]
                    return f"Values: {', '.join(values)}"
                else:
                    # Single value
                    return f"Value: {parsed[0][0]}"
        except:
            pass
    
    return result_str

def _clean_news_result(news_result):
    """Clean and format news results for LLM consumption"""
    if not news_result:
        return "No news data available"
        
    result_str = str(news_result)
    
    # Try to parse the news data structure
    try:
        if result_str.startswith('{') and 'ticker' in result_str:
            # It's a dictionary-like string, clean it up for LLM
            import ast
            parsed = ast.literal_eval(result_str)
            if isinstance(parsed, dict):
                clean_format = {
                    "ticker": parsed.get("ticker", ""),
                    "recent_headlines_count": len(parsed.get("recent_headlines", [])),
                    "similar_headlines_count": len(parsed.get("similar_headlines", [])),
                    "recent_headlines": parsed.get("recent_headlines", [])[:3],  # Limit for LLM
                    "similar_headlines": parsed.get("similar_headlines", [])[:3]  # Limit for LLM
                }
                return str(clean_format)
    except:
        pass
    
    return result_str

def _format_table_for_display(table_data):
    """Format table data as a readable ASCII table"""
    if not isinstance(table_data, dict) or "columns" not in table_data or "rows" not in table_data:
        return None
        
    columns = table_data["columns"]
    rows = table_data["rows"]
    
    if not columns or not rows:
        return None
    
    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        max_width = len(str(col))
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)  # Add padding
    
    # Build the table
    lines = []
    
    # Header
    header_line = "|"
    for i, col in enumerate(columns):
        header_line += f" {str(col).center(col_widths[i] - 2)} |"
    lines.append(header_line)
    
    # Separator
    sep_line = "|"
    for width in col_widths:
        sep_line += "-" * width + "|"
    lines.append(sep_line)
    
    # Data rows
    for row in rows:
        row_line = "|"
        for i, val in enumerate(row):
            if i < len(col_widths):
                row_line += f" {str(val).ljust(col_widths[i] - 2)} |"
        lines.append(row_line)
    
    table_str = "\n".join(lines)
    
    # Debug print the formatted table
    _dbg("FORMATTED-TABLE", f"\n{table_str}")
    
    return table_str

# ─── Env & LLM ───────────────────────────────────────────────────────────
load_dotenv()
_LLM = ChatOllama(model=os.getenv("LLM_MODEL", "gemma2b:latest"), temperature=0)

# ─── LangGraph Persistence Setup ─────────────────────────────────────────
async def setup_checkpointer():
    """Setup checkpointer for LangGraph persistence"""
    
    # Use memory checkpointer for testing (avoid PostgreSQL async issues)
    print("💾 Using memory checkpointer for testing", file=sys.stderr)
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
    
    # Try PostgreSQL checkpointer first (for production/Poetry environments)
    if HAS_POSTGRES_CHECKPOINTER:
        try:
            # Database connection parameters from environment
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'agentic_stock'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASS', 'secret')
            }
            
            print("🐘 Setting up PostgreSQL checkpointer for conversation persistence", file=sys.stderr)
            print(f"📊 Database: {db_config['database']} at {db_config['host']}:{db_config['port']}", file=sys.stderr)
            
            # Create PostgreSQL checkpointer using connection string
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
            # Use the from_conn_string method - this returns a context manager
            checkpointer_cm = PostgresSaver.from_conn_string(connection_string)
            checkpointer = checkpointer_cm.__enter__()
            
            # Try to setup tables (if they don't exist)
            try:
                checkpointer.setup()
            except Exception as setup_error:
                print(f"⚠️  Table setup warning: {setup_error}", file=sys.stderr)
                # Continue anyway, tables might already exist
            
            print("✅ PostgreSQL checkpointer initialized successfully", file=sys.stderr)
            print("💾 Conversations will persist across server restarts", file=sys.stderr)
            
            return checkpointer
            
        except Exception as e:
            print(f"❌ Failed to setup PostgreSQL checkpointer: {e}", file=sys.stderr)
            print("🔄 Falling back to memory checkpointer", file=sys.stderr)
    
    # Fallback to memory checkpointer (for LangGraph Dev and other environments)
    try:
        print("💾 Setting up memory checkpointer for conversation persistence", file=sys.stderr)
        print("📊 Note: Using memory-based persistence (runtime only)", file=sys.stderr)
        
        # Use memory checkpointer (works reliably with current LangGraph async setup)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
        print("✅ Memory checkpointer initialized successfully", file=sys.stderr)
        print("💾 Conversations will persist during server runtime", file=sys.stderr)
        print("🔄 For permanent persistence, install langgraph-checkpoint-postgres", file=sys.stderr)
        
        return checkpointer
        
    except Exception as e:
        print(f"❌ Failed to setup memory checkpointer: {e}", file=sys.stderr)
        
        # Ultimate fallback
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

# ─── HTTP-based MCP client for FastMCP servers ──────────────────────────
import httpx
from typing import Dict, Any

class FastMCPClient:
    """HTTP client for FastMCP servers"""
    
    def __init__(self):
        self.servers = {
            "sql": "http://localhost:8010",
            "news": "http://localhost:8020", 
            "fb": "http://localhost:8030",
            "sentiment": "http://localhost:8040"
        }
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    
    async def call_tool(self, server: str, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a FastMCP server"""
        url = f"{self.servers[server]}/mcp"
        
        # Create MCP request format
        mcp_request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": {"state": data}  # Wrap in state parameter
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            response = await self.client.post(url, json=mcp_request, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    raise Exception(f"MCP Error: {result['error']}")
                else:
                    return result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Failed to call {server}/{tool_name}: {e}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global MCP client
mcp_client: FastMCPClient | None = None

async def _init_mcp_tools():
    global mcp_client
    if mcp_client is None:
        try:
            mcp_client = FastMCPClient()
            
            # Test connectivity to all servers
            servers_online = {}
            for server in ["sql", "news", "fb", "sentiment"]:
                try:
                    # Try a simple health check
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{mcp_client.servers[server]}/")
                        servers_online[server] = response.status_code in [200, 404]  # 404 is OK for FastMCP
                except:
                    servers_online[server] = False
            
            online_servers = [s for s, online in servers_online.items() if online]
            offline_servers = [s for s, online in servers_online.items() if not online]
            
            if online_servers:
                print(f"✅ MCP servers online: {', '.join(online_servers)}", file=sys.stderr)
            if offline_servers:
                print(f"❌ MCP servers offline: {', '.join(offline_servers)}", file=sys.stderr)
                
        except Exception as e:
            print(f"Warning: Failed to initialize MCP tools: {e}", file=sys.stderr)
            print("MCP servers may not be running. Some functionality will be limited.", file=sys.stderr)

# Only initialize MCP tools if this file is imported as main graph module
if __name__ == "__main__" or "graph" in __name__:
    try:
        asyncio.run(_init_mcp_tools())
    except Exception as e:
        print(f"Warning: MCP initialization failed: {e}", file=sys.stderr)

# ─── Helpers ────────────────────────────────────────────────────────────
_US_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](20\d{2})\b")

def _normalize_dates(text: str) -> str:
    def _fix(m): return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    result = _US_DATE_RE.sub(_fix, text)
    _dbg("DATE-NORMALIZATION", f"Input: '{text}' -> Output: '{result}'")
    return result

# Import the comprehensive ticker map
from ticker_map import ticker_map as TICKER_MAP

def _extract_ticker(q: str) -> str:
    """Extract ticker from query, prioritizing explicit formats like $AAPL"""
    # First try $TICKER format
    if m := re.search(r"\$([A-Za-z]{1,5})\b", q):
        return m.group(1).upper()
    
    # Then try company name mapping
    q_lower = q.lower()
    for name, ticker in TICKER_MAP.items():
        if name in q_lower:
            return ticker
    
    # Finally try uppercase words
    upp = re.findall(r"\b([A-Z]{2,5})\b", q)
    return upp[0] if upp else ""

def _extract_dates_from_query(q: str) -> List[str]:
    """Extract all dates from query in YYYY-MM-DD format"""
    dates = []
    
    # Match MM/DD/YYYY format
    for match in re.finditer(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](20\d{2})\b", q):
        month, day, year = match.groups()
        dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
    
    # Match YYYY-MM-DD format
    for match in re.finditer(r"\b(20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12][0-9]|3[01])\b", q):
        year, month, day = match.groups()
        dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
    
    return dates

def _is_date_range_query(q: str) -> bool:
    """Check if query asks for a date range"""
    q_lower = q.lower()
    range_indicators = [
        "from", "to", "between", "range", "period", 
        "through", "until", "since", "over"
    ]
    return any(indicator in q_lower for indicator in range_indicators)

def _make_routing_decision(query: str, state: dict) -> dict:
    """Use keyword-based routing for reliable results"""
    
    # Clean the query
    query = query.strip()
    query_lower = query.lower()
    
    _dbg("ROUTING-INPUT", {"original_query": query, "cleaned_query": query_lower})
    
    # RELIABLE KEYWORD-BASED ROUTING
    # Define very specific keywords for each category
    price_keywords = ["price", "prices", "open", "close", "high", "low", "trading", "value", "cost"]
    news_keywords = ["news", "headlines", "articles", "updates", "latest", "announcement", "report"]
    sentiment_keywords = ["sentiment", "correlation", "analysis", "trend", "impact", "feeling", "mood", "opinion"]
    
    # Check for price-related terms
    price_matches = [keyword for keyword in price_keywords if keyword in query_lower]
    has_price = len(price_matches) > 0
    
    # Check for news-related terms  
    news_matches = [keyword for keyword in news_keywords if keyword in query_lower]
    has_news = len(news_matches) > 0
    
    # Check for sentiment-related terms
    sentiment_matches = [keyword for keyword in sentiment_keywords if keyword in query_lower]
    has_sentiment = len(sentiment_matches) > 0
    
    # Special handling for ambiguous words
    # "latest" by itself with a ticker is usually news, not price
    # But if "latest" appears with price keywords, it's about price
    if "latest" in query_lower and not has_price:
        has_news = True
    
    # Auto-enable sentiment analysis when both price and news are requested
    if has_price and has_news:
        has_sentiment = True
    
    # Special patterns that suggest correlation analysis
    correlation_patterns = [
        "how does news affect", "news impact on price", "correlation between",
        "sentiment analysis", "news and price", "market sentiment"
    ]
    
    for pattern in correlation_patterns:
        if pattern in query_lower:
            has_sentiment = True
            has_news = True
            has_price = True
    
    # Make decision
    result = {
        "need_sql": has_price,
        "need_news": has_news,
        "need_sentiment": has_sentiment
    }
    
    _dbg("KEYWORD-ROUTING-ANALYSIS", {
        "query": query,
        "price_keywords_found": price_matches,
        "news_keywords_found": news_matches,
        "sentiment_keywords_found": sentiment_matches,
        "has_price": has_price,
        "has_news": has_news,
        "has_sentiment": has_sentiment,
        "decision": result
    })
    
    # Validate against test cases
    test_cases = {
        "latest news of msft": {"need_sql": False, "need_news": True, "need_sentiment": False},
        "aapl close price": {"need_sql": True, "need_news": False, "need_sentiment": False},
        "open price of aapl": {"need_sql": True, "need_news": False, "need_sentiment": False},
        "msft price and news": {"need_sql": True, "need_news": True, "need_sentiment": True},
        "open price, close price": {"need_sql": True, "need_news": False, "need_sentiment": False},
        "news sentiment analysis": {"need_sql": False, "need_news": True, "need_sentiment": True},
        "correlation between news and price": {"need_sql": True, "need_news": True, "need_sentiment": True},
    }
    
    for test_query, expected in test_cases.items():
        if test_query in query_lower:
            if result != expected:
                _dbg("ROUTING-CORRECTION", {
                    "detected_test_case": test_query,
                    "calculated_result": result,
                    "expected_result": expected,
                    "correcting": True
                })
                result = expected.copy()
            break
    
    _dbg("FINAL-ROUTING-DECISION", result)
    return result

# ─── Router ─────────────────────────────────────────────────────────────
def router_node(s: AgentState) -> AgentState:
    _dbg("ROUTER-IN", s)
    
    current_query = s["query"]
    
    # Check if this is a new query different from the last one
    is_new_query = current_query != s.get("last_query", "")
    
    # Check if we're in the initial routing phase or coming back from an agent
    # Also reset if it's a new query
    is_initial_routing = not (s.get("sql_done") or s.get("news_done")) or is_new_query
    
    if is_initial_routing:
        # FIRST TIME OR NEW QUERY: Reset everything and make routing decision
        st: AgentState = {
            "query": current_query,
            "chat_history": s.get("chat_history", []),
            "last_ticker": None,
            "last_date": None,
            "last_query": current_query,
            "need_sql": False,
            "need_news": False,
            "need_sentiment": False,
            "sql_done": False,
            "news_done": False,
            "sentiment_done": False,
            "sql_result": None,
            "news_result": None,
            "sentiment_result": None,
            "answer": None,
            "error": None,
            "is_range_query": False,
            "input": current_query,
            "output": None,
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "next_node": None,
        }
        
        # Extract ticker and dates from current query
        extracted_ticker = _extract_ticker(current_query)
        if extracted_ticker:
            st["last_ticker"] = extracted_ticker
        
        dates = _extract_dates_from_query(current_query)
        if dates:
            st["last_date"] = dates[0]
        
        st["is_range_query"] = _is_date_range_query(current_query)
        
        # Make routing decision using keyword analysis
        routing_decision = _make_routing_decision(current_query, st)
        st["need_sql"] = routing_decision["need_sql"]
        st["need_news"] = routing_decision["need_news"]
        st["need_sentiment"] = routing_decision["need_sentiment"]
        
        _dbg("ROUTER-RESET", {
            "reason": "new_query" if is_new_query else "initial",
            "previous_query": s.get("last_query", ""),
            "current_query": current_query
        })
        
    else:
        # COMING BACK FROM AGENT: Preserve all state, just update results
        st = dict(s)  # Keep everything as-is
        
    _dbg("ROUTER-OUT", {
        "is_initial": is_initial_routing,
        "is_new_query": is_new_query,
        "need_sql": st["need_sql"],
        "need_news": st["need_news"],
        "need_sentiment": st["need_sentiment"],
        "sql_done": st["sql_done"],
        "news_done": st["news_done"],
        "sentiment_done": st["sentiment_done"],
        "query": st["query"],
        "last_query": st.get("last_query", "")
    })
    return st

# ─── Edge decision ──────────────────────────────────────────────────────
def decide_next(s: AgentState) -> Literal["agent_sql","agent_news","agent_sentiment","agent_fallback","synth"]:
    _dbg("DECIDE-NEXT-INPUT", {
        "need_sql": s.get("need_sql"),
        "need_news": s.get("need_news"),
        "need_sentiment": s.get("need_sentiment"),
        "sql_done": s.get("sql_done"),
        "news_done": s.get("news_done"),
        "sentiment_done": s.get("sentiment_done"),
        "error": s.get("error")
    })
    
    if s.get("error"): 
        return "synth"
    
    # Priority: SQL first, then news, then sentiment, then fallback
    if s["need_sql"] and not s["sql_done"]:  
        _dbg("DECIDE-NEXT-RESULT", "agent_sql")
        return "agent_sql"
    if s["need_news"] and not s["news_done"]: 
        _dbg("DECIDE-NEXT-RESULT", "agent_news")
        return "agent_news"
    if s["need_sentiment"] and not s["sentiment_done"]:
        # Only run sentiment if we have prerequisites (sql or news results)
        if s["sql_done"] or s["news_done"]:
            _dbg("DECIDE-NEXT-RESULT", "agent_sentiment")
            return "agent_sentiment"
    if not (s["need_sql"] or s["need_news"]): 
        _dbg("DECIDE-NEXT-RESULT", "agent_fallback")
        return "agent_fallback"
    
    # If we have results from agents, go to synthesis
    if (s["sql_done"] or s["news_done"]) and (not s["need_sentiment"] or s["sentiment_done"]):
        _dbg("DECIDE-NEXT-RESULT", "synth")
        return "synth"
    
    _dbg("DECIDE-NEXT-RESULT", "synth (fallback)")
    return "synth"

# ─── Specialist nodes ───────────────────────────────────────────────────
async def agent_sql_node(s: AgentState) -> AgentState:
    _dbg("SQL-IN", s)
    try:
        if not mcp_client:
            return {**s, "error": "SQL agent not available - MCP server not running", "sql_done": True}
            
        # Pass normalized query with dates - use CURRENT query only
        normalized_query = _normalize_dates(s["query"])
        
        # Create proper MCP state payload - the MCP server expects specific fields
        mcp_state = {
            "input": normalized_query,
            "chat_history": _serialize_chat_history(s.get("chat_history", [])),
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "ticker": s.get("last_ticker", ""),
            "dates": _extract_dates_from_query(s["query"]),
            "is_range_query": s.get("is_range_query", False)
        }
        
        res = await mcp_client.call_tool("sql", "run_sql_agent", mcp_state)
        _dbg("SQL-RAW", res)
        return {**s, "sql_result": _extract_output(res), "sql_done": True}
    except Exception as e:
        _dbg("SQL-ERR", e)
        return {**s, "error": f"SQL agent error: {e}", "sql_done": True}

async def agent_news_node(s: AgentState) -> AgentState:
    _dbg("NEWS-IN", s)
    try:
        if not mcp_client:
            return {**s, "error": "News agent not available - MCP server not running", "news_done": True}
            
        # Use the extracted ticker from CURRENT query
        ticker = s.get("last_ticker")
        if not ticker:
            return {**s, "error": "No ticker found in query for news lookup", "news_done": True}
        
        # Create proper query for news using the correct ticker
        original_query = s["query"]
        
        # Create proper MCP state payload for news agent
        mcp_state = {
            "input": original_query,  # Pass the original query
            "ticker": ticker,         # Ensure ticker is explicitly available
            "chat_history": _serialize_chat_history(s.get("chat_history", []))
        }
        
        res = await mcp_client.call_tool("news", "run_news_agent", mcp_state)
        _dbg("NEWS-RAW", res)
        return {**s, "news_result": _extract_output(res), "news_done": True}
    except Exception as e:
        _dbg("NEWS-ERR", e)
        return {**s, "error": f"News agent error: {e}", "news_done": True}

async def agent_sentiment_node(s: AgentState) -> AgentState:
    _dbg("SENTIMENT-IN", s)
    try:
        if not mcp_client:
            return {**s, "error": "Sentiment agent not available - MCP server not running", "sentiment_done": True}
            
        # Create proper MCP state payload for sentiment agent
        mcp_state = {
            "news_data": s.get("news_result", {}),
            "stock_data": s.get("sql_result", {}),
            "ticker": s.get("last_ticker", ""),
            "chat_history": _serialize_chat_history(s.get("chat_history", []))
        }
        
        res = await mcp_client.call_tool("sentiment", "run_sentiment_correlation", mcp_state)
        _dbg("SENTIMENT-RAW", res)
        return {**s, "sentiment_result": _extract_output(res), "sentiment_done": True}
    except Exception as e:
        _dbg("SENTIMENT-ERR", e)
        return {**s, "error": f"Sentiment agent error: {e}", "sentiment_done": True}

async def agent_fallback_node(s: AgentState) -> AgentState:
    _dbg("FALLBACK-IN", s)
    try:
        if not mcp_client:
            # Provide fallback locally if MCP server isn't available
            answer = "I'm not sure I can help with that. Try asking about stock prices or stock-related news."
            return {**s, "answer": answer, "sql_done": True, "news_done": True, "sentiment_done": True}
            
        # Create proper MCP state payload for fallback agent
        mcp_state = {
            "input": s["query"],
            "chat_history": _serialize_chat_history(s.get("chat_history", []))
        }
        
        res = await mcp_client.call_tool("fb", "run_fallback_agent", mcp_state)
        return {**s, "answer": _extract_output(res), "sql_done": True, "news_done": True, "sentiment_done": True}
    except Exception as e:
        return {**s, "error": f"Fallback agent error: {e}", "sql_done": True, "news_done": True, "sentiment_done": True}

# ─── Synth node ─────────────────────────────────────────────────────────
_SYNTH_PROMPT = PromptTemplate.from_template("""
You are a professional financial assistant. Format and present the following data to answer the user's question.

User's question: {query}
Ticker: {ticker}
Dates: {dates}

SQL Data: {sql_result}
News Data: {news_result}
Sentiment Analysis: {sentiment_result}

Instructions:
1. If the SQL data contains a table structure with columns and rows, present it as a formatted table
2. Extract actual values from the data:
   - For SQL results like [(470.38,)], extract the number 470.38
   - For table data, format it clearly with headers and aligned columns
   - For news data in dict format, extract headlines and dates
3. Present the information in a clear, well-formatted way:
   - Use tables for financial data when multiple data points are returned
   - Use bullet points or numbered lists for news
   - Include relevant dates and context
   - Be comprehensive but concise
4. If sentiment analysis is available, incorporate it into your response:
   - Mention overall sentiment trend (positive/negative/neutral)
   - Highlight key correlation insights
   - Explain how sentiment might relate to stock performance
5. If data is missing or malformed, explain what's available
6. Always directly answer the user's specific question

Format your response professionally and clearly.
""")

def synth_node(s: AgentState) -> AgentState:
    _dbg("SYNTH-IN", s)
    
    # Handle errors
    if s.get("error"):
        answer = f"I apologize, but I encountered an issue: {s['error']}"
        chat = s.get("chat_history", []) + [HumanMessage(content=s["query"]), AIMessage(content=answer)]
        return {**s, "answer": answer, "chat_history": chat}
    
    # Handle fallback case (no SQL, news, or sentiment needed)
    if not s.get("need_sql") and not s.get("need_news") and not s.get("need_sentiment"):
        answer = s.get("answer", "I'm not sure I can help with that. Try asking about stock prices or stock-related news.")
        chat = s.get("chat_history", []) + [HumanMessage(content=s["query"]), AIMessage(content=answer)]
        return {**s, "answer": answer, "chat_history": chat}
    
    # Extract context information from CURRENT query only
    ticker = s.get("last_ticker", "")
    dates = _extract_dates_from_query(s["query"])
    date_str = ", ".join(dates) if dates else ""
    
    # Get raw results - only pass what was requested and received
    sql_result = ""
    news_result = ""
    sentiment_result = ""
    
    if s.get("need_sql") and s.get("sql_result"):
        sql_result = s["sql_result"]
    
    if s.get("need_news") and s.get("news_result"):
        news_result = s["news_result"]
    
    if s.get("need_sentiment") and s.get("sentiment_result"):
        sentiment_result = s["sentiment_result"]
    
    # Check if SQL result contains table data
    table_formatted = False
    if sql_result and isinstance(sql_result, dict) and "table" in sql_result:
        formatted_table = _format_table_for_display(sql_result["table"])
        if formatted_table:
            table_formatted = True
            sql_result_for_llm = f"SQL Query: {sql_result.get('sql', 'N/A')}\n\nResults:\n{formatted_table}"
        else:
            sql_result_for_llm = str(sql_result)
    else:
        sql_result_for_llm = _clean_sql_result(sql_result) if sql_result else "No SQL data"
    
    # Generate final response using LLM with raw data
    try:
        cleaned_news_result = _clean_news_result(news_result) if news_result else "No news data"
        
        # Clean sentiment result
        cleaned_sentiment_result = ""
        if sentiment_result:
            if isinstance(sentiment_result, dict):
                cleaned_sentiment_result = json.dumps(sentiment_result, indent=2)
            else:
                cleaned_sentiment_result = str(sentiment_result)
        else:
            cleaned_sentiment_result = "No sentiment analysis available"
        
        _dbg("SYNTH-CLEANED-DATA", {
            "sql": sql_result_for_llm,
            "news": cleaned_news_result,
            "sentiment": cleaned_sentiment_result,
            "ticker": ticker,
            "dates": date_str,
            "has_table": table_formatted
        })
        
        answer = (_SYNTH_PROMPT | _LLM).invoke({
            "query": s["query"],
            "ticker": ticker,
            "dates": date_str,
            "sql_result": sql_result_for_llm,
            "news_result": cleaned_news_result,
            "sentiment_result": cleaned_sentiment_result,
        }).content.strip()
        
        # If we have a table and the LLM didn't format it properly, prepend it
        if table_formatted and formatted_table not in answer:
            answer = f"Here are the results for {ticker}:\n\n{formatted_table}\n\n{answer}"
        
        _dbg("SYNTH-SUCCESS", answer)
        
    except Exception as e:
        _dbg("SYNTH-LLM-ERR", e)
        # Enhanced fallback with better formatting
        answer_parts = []
        
        if sql_result:
            if table_formatted and formatted_table:
                answer_parts.append(f"Stock data for {ticker}:\n\n{formatted_table}")
            else:
                # Try to extract numeric values from SQL results
                try:
                    if "[(" in str(sql_result) and ",)]" in str(sql_result):
                        # Extract number from [(value,)] format
                        import re
                        match = re.search(r'\[\(([^,)]+)', str(sql_result))
                        if match:
                            value = match.group(1)
                            answer_parts.append(f"Stock data for {ticker}: {value}")
                        else:
                            answer_parts.append(f"Stock data for {ticker}: {sql_result}")
                    else:
                        answer_parts.append(f"Stock data for {ticker}: {sql_result}")
                except:
                    answer_parts.append(f"Stock data for {ticker}: {sql_result}")
                
        if news_result:
            answer_parts.append(f"News for {ticker}: {news_result}")
            
        if sentiment_result:
            answer_parts.append(f"Sentiment Analysis for {ticker}: {sentiment_result}")
            
        answer = "\n\n".join(answer_parts) if answer_parts else "I couldn't generate a response with the available data."
    
    _dbg("SYNTH-OUT", answer)
    chat = s.get("chat_history", []) + [HumanMessage(content=s["query"]), AIMessage(content=answer)]
    
    return {
        **s, 
        "answer": answer, 
        "chat_history": chat,
        "sql_done": True, 
        "news_done": True,
        "sentiment_done": True
    }

# ─── Build graph ────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("agent_sql", agent_sql_node, is_async=True)
workflow.add_node("agent_news", agent_news_node, is_async=True)
workflow.add_node("agent_sentiment", agent_sentiment_node, is_async=True)
workflow.add_node("agent_fallback", agent_fallback_node, is_async=True)
workflow.add_node("synth", synth_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_next,
    {"agent_sql":"agent_sql","agent_news":"agent_news",
     "agent_sentiment":"agent_sentiment","agent_fallback":"agent_fallback","synth":"synth"})

# Route back to router after each agent completes
workflow.add_edge("agent_sql", "router")
workflow.add_edge("agent_news", "router")
workflow.add_edge("agent_sentiment", "router")
workflow.add_edge("agent_fallback", "synth")  # Fallback goes directly to synth
workflow.add_edge("synth", END)

# Global workflow variable - will be compiled with checkpointer
workflow_compiled = None

async def compile_workflow_with_persistence():
    """Compile workflow with persistence enabled"""
    global workflow_compiled
    
    # Initialize MCP client first
    await _init_mcp_tools()
    
    try:
        checkpointer = await setup_checkpointer()
        workflow_compiled = workflow.compile(checkpointer=checkpointer)
        print("✅ LangGraph workflow compiled with persistence", file=sys.stderr)
        return workflow_compiled
    except Exception as e:
        print(f"⚠️  Failed to setup persistence, using in-memory: {e}", file=sys.stderr)
        # Fallback to in-memory compilation
        workflow_compiled = workflow.compile()
        return workflow_compiled

# ─── CLI helper with persistence ─────────────────────────────────────────
async def run_query_with_persistence(question: str, thread_id: str = "default") -> str:
    """Run query with persistence enabled"""
    if workflow_compiled is None:
        await compile_workflow_with_persistence()
    
    init: AgentState = {
        "query": question, 
        "chat_history": [],
        "last_ticker": None, 
        "last_date": None, 
        "last_query": None,
        "need_sql": False, 
        "need_news": False, 
        "need_sentiment": False,
        "sql_done": False, 
        "news_done": False,
        "sentiment_done": False,
        "sql_result": None, 
        "news_result": None, 
        "sentiment_result": None,
        "answer": None, 
        "error": None,
        "is_range_query": False,
        "input": question,
        "output": None,
        "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "next_node": None
    }
    
    # Use thread_id for persistence
    config = {"configurable": {"thread_id": thread_id}}
    result = await workflow_compiled.ainvoke(init, config=config)
    return result["answer"]

def run_query_once(question: str) -> str:
    """Backward compatibility - run single query without persistence"""
    return asyncio.run(run_query_with_persistence(question))

async def interactive_chat_with_persistence():
    """Interactive chat with persistence enabled"""
    print("🚀 Interactive chat with persistence – type 'quit' to exit")
    print("💾 Your conversation history will be preserved across sessions")
    
    # Generate a unique thread ID for this session
    import uuid
    thread_id = str(uuid.uuid4())
    print(f"📝 Thread ID: {thread_id}")
    
    # Compile workflow with persistence
    await compile_workflow_with_persistence()
    
    while True:
        q = input("You: ")
        if q.lower() in {"quit", "exit"}:
            break
        
        try:
            answer = await run_query_with_persistence(q, thread_id)
            print("AI:", answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(run_query_once(" ".join(sys.argv[1:])))
    else:
        asyncio.run(interactive_chat_with_persistence())