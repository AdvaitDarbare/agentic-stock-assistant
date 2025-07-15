"""
sql_agent.py – MCP Server Version (Fully MCP Compliant)
────────────────────────────────────────────────────────
Generates and executes SQL queries against the `stock_data` table.
Enhanced for better LLM-driven decision making and strict MCP compliance.
"""

import json
import os
import re
import sys
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import mlflow.langchain  # keeps LangChain autolog active

from tools.db import get_sql_agent_chain
from state import AgentState
from ticker_map import ticker_map


# ─── DEBUG LOGGER ────────────────────────────────────────────────────────
def _dbg(tag: str, txt):
    print(f"\n### SQL-{tag}\n{txt}\n", file=sys.stderr, flush=True)

# ─── Helpers ─────────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL)

def _force_fix_table_name(sql: str) -> str:
    """
    Force-correct any hallucinated table name to exactly 'stock_data'.
    """
    # Fix common misspellings like "sstock_data"
    sql = re.sub(r"\bs+tock[_]*data\b", "stock_data", sql, flags=re.IGNORECASE)
    # Fix any variation of stock_data
    sql = re.sub(r"\b[sS]*tock[_]*[dD]ata\b", "stock_data", sql)
    return sql


def _clean_sql(text: str) -> str:
    """Extract SQL from LLM response, handling various formats"""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    cleaned = text.strip()
    cleaned = cleaned.replace("```sql", "").replace("```", "")
    cleaned = cleaned.replace("sql", "", 1).strip()
    return cleaned

def _validate_sql(sql: str) -> None:
    """Strict validation to ensure SQL agent only handles stock data"""
    lowered = sql.lower()
    if "stock_data" not in lowered:
        raise ValueError("SQL agent must query the stock_data table")
    forbidden_terms = [
        "news", "headline", "article", "news_articles", 
        "langchain_id", "embedding", "similarity"
    ]
    for term in forbidden_terms:
        if term in lowered:
            raise ValueError(f"SQL agent must not query news data. Found forbidden term: {term}")
    dangerous_ops = ["drop", "delete", "truncate", "insert", "update", "alter"]
    for op in dangerous_ops:
        if f" {op} " in f" {lowered} " or lowered.startswith(f"{op} "):
            raise ValueError(f"SQL agent only performs SELECT queries. Found: {op}")
    if sql.count(';') > 1:
        raise ValueError("SQL agent should generate exactly one SELECT query")

def _extract_dates_from_query(query: str) -> List[str]:
    dates = []
    
    # Match MM/DD/YYYY format
    for match in re.finditer(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](20\d{2})\b", query):
        month, day, year = match.groups()
        dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
    
    # Match YYYY-MM-DD format
    for match in re.finditer(r"\b(20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12][0-9]|3[01])\b", query):
        year, month, day = match.groups()
        dates.append(f"{year}-{int(month):02d}-{int(day):02d}")
    
    # Sort dates to ensure proper order for range queries
    dates.sort()
    return dates

def _is_date_range_query(query: str) -> bool:
    query_lower = query.lower()
    range_indicators = [
        "from", "to", "between", "range", "period", 
        "through", "until", "since", "over"
    ]
    return any(indicator in query_lower for indicator in range_indicators)

def _extract_tickers_from_query(query: str) -> List[str]:
    """
    Return a list of all tickers (normalized to uppercase) mentioned in the query.
    """
    tickers = []

    # 1) $TICKER syntax
    for m in re.finditer(r"\$([A-Za-z]{1,5})\b", query):
        t = m.group(1).upper()
        if t.lower() in ticker_map and t not in tickers:
            tickers.append(t)

    # 2) ALL-CAPS words
    for w in re.findall(r"\b([A-Z]{2,5})\b", query):
        if w.lower() in ticker_map:
            T = ticker_map[w.lower()]
            if T not in tickers:
                tickers.append(T)

    # 3) Company names and ticker symbols (case-insensitive)
    words = query.lower().split()
    for word in words:
        # Clean punctuation from word
        clean_word = re.sub(r'[^a-z]', '', word)
        if clean_word in ticker_map:
            T = ticker_map[clean_word]
            if T not in tickers:
                tickers.append(T)

    # 4) Multi-word company names (check common patterns)
    query_lower = query.lower()
    multi_word_companies = [
        name for name in ticker_map.keys() if ' ' in name
    ]
    for company in multi_word_companies:
        if company in query_lower:
            T = ticker_map[company]
            if T not in tickers:
                tickers.append(T)

    return tickers



def _analyze_query_intent(query: str) -> dict:
    analysis_prompt = f"""
Analyze this stock data query and determine what information the user wants.

Query: "{query}"

Respond with a JSON object containing:
- "columns": list of stock_data columns needed ["date", "open", "close", "high", "low"]
- "date_filter": "single", "range", or "recent" 
- "ticker_required": true/false
- "intent": brief description of what user wants

Stock data table has columns: ticker, date, open, high, low, close

Examples:
- "AAPL open price" -> {{"columns": ["date", "open"], "date_filter": "recent", "ticker_required": true, "intent": "recent open price"}}
- "Tesla close from 2025-06-06 to 2025-06-11" -> {{"columns": ["date", "close"], "date_filter": "range", "ticker_required": true, "intent": "close prices in date range"}}
- "MSFT high and low yesterday" -> {{"columns": ["date", "high", "low"], "date_filter": "single", "ticker_required": true, "intent": "high/low for specific date"}}
- "correlation between news and Apple stock price" -> {{"columns": ["date", "open", "high", "low", "close"], "date_filter": "range", "ticker_required": true, "intent": "comprehensive price data for correlation analysis"}}
- "Apple stock price" -> {{"columns": ["date", "open", "high", "low", "close"], "date_filter": "recent", "ticker_required": true, "intent": "full price data"}}

For correlation queries, return ALL price columns (open, high, low, close) for comprehensive analysis.

Respond only with the JSON object:
"""
    try:
        llm = ChatOllama(model=os.getenv("LLM_MODEL", "gemma:2b"), temperature=0)
        response = llm.invoke(analysis_prompt).content.strip()
        _dbg("QUERY-ANALYSIS-RAW", response)
        json_match = re.search(r'\{[^}]*\}', response)
        if json_match:
            analysis = json.loads(json_match.group())
            _dbg("QUERY-ANALYSIS-PARSED", analysis)
            return analysis
    except Exception as e:
        _dbg("QUERY-ANALYSIS-ERROR", f"Analysis failed: {e}")
    return {
        "columns": ["date", "open", "close"],
        "date_filter": "recent",
        "ticker_required": True,
        "intent": "general stock data"
    }

# ─── ENHANCED SQL GENERATION ─────────────────────────────────────────────
def generate_sql_with_context(
    query: str,
    tickers: List[str],
    dates: List[str],
    is_range: bool,
    table_info: str
) -> str:
    intent = _analyze_query_intent(query)
    _dbg("QUERY-INTENT", intent)

    system_prompt = f"""
You are a SQL expert for stock price data ONLY. Generate a SELECT query that **must** use the table name exactly `stock_data`.

TABLE SCHEMA:
{table_info.strip()}

QUERY ANALYSIS:
- User question: {query}
- Detected tickers: {tickers or "Must be provided"}
- Detected dates: {dates or "None - use recent data"}
- Is date range: {is_range}
- Intent: {intent.get('intent')}
- Needed columns: {intent.get('columns')}

STRICT REQUIREMENTS:
1. ONLY query the stock_data table.
2. ALWAYS include 'ticker' and 'date' in SELECT.
3. If multiple tickers, use `WHERE ticker IN ('T1','T2',…)`; otherwise `WHERE ticker = 'T1'`.
4. For ranges: `WHERE date BETWEEN 'start_date' AND 'end_date'`.
5. For recent: `ORDER BY date DESC LIMIT 10`.
6. Use proper 'YYYY-MM-DD'.
7. Select only requested columns.

Generate the SQL wrapped in ```sql ```:
"""
    llm = ChatOllama(model=os.getenv("LLM_MODEL"), temperature=0)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate SQL for: {query}")
    ]).content
    _dbg("SQL-GENERATION-RAW", response)

    # 1) Clean and force correct table name
    sql = _clean_sql(response)
    sql = _force_fix_table_name(sql).strip().rstrip(";")

    # 2) Strip out any existing WHERE… clause entirely
    sql = re.sub(r"(?is)\bWHERE\b.*$", "", sql).strip()

    # 3) Build a canonical WHERE clause
    if tickers:
        if len(tickers) == 1:
            ticker_filter = f"ticker = '{tickers[0]}'"
        else:
            vals = ", ".join(f"'{t}'" for t in tickers)
            ticker_filter = f"ticker IN ({vals})"

        if is_range and len(dates) >= 2:
            date_filter = f"date BETWEEN '{dates[0]}' AND '{dates[1]}'"
            where_clause = f"WHERE {ticker_filter} AND {date_filter}"
        else:
            where_clause = f"WHERE {ticker_filter}"

        sql = f"{sql}\n{where_clause}"

    # 4) Ensure ticker & date columns are selected
    m = re.search(r"select\s+(.*?)\s+from", sql, flags=re.IGNORECASE)
    if m:
        cols = [c.strip().lower() for c in m.group(1).split(",")]
        missing = [c for c in ("ticker", "date") if c not in cols]
        if missing:
            sql = re.sub(
                r"(?i)select\s+",
                f"SELECT {', '.join(missing)}, ",
                sql,
                count=1
            )

    _dbg("SQL-GENERATION-CLEANED", sql)
    return sql


def _format_results_as_table(columns: List[str], rows: List[Any]) -> Dict[str, Any]:
    """Convert database results to table format"""
    formatted_rows = []
    
    # Handle different row formats
    for row in rows:
        if isinstance(row, dict):
            # If row is a dictionary, extract values in column order
            row_values = []
            for col in columns:
                value = row.get(col, '')
                # Convert datetime objects to strings
                if hasattr(value, 'strftime'):
                    value = value.strftime('%Y-%m-%d')
                row_values.append(value)
            formatted_rows.append(row_values)
        elif isinstance(row, (list, tuple)):
            # If row is already a list/tuple, just convert to list
            row_values = []
            for value in row:
                # Convert datetime objects to strings
                if hasattr(value, 'strftime'):
                    value = value.strftime('%Y-%m-%d')
                row_values.append(value)
            formatted_rows.append(row_values)
        else:
            # Fallback for other formats
            formatted_rows.append([str(row)])
    
    return {
        "columns": columns,
        "rows": formatted_rows
    }

# ─── MAIN AGENT FUNCTION ─────────────────────────────────────────────────
def run_sql_agent(state: AgentState) -> AgentState:
    _dbg("INPUT-STATE", state)
    try:
        sql_agent_chain = get_sql_agent_chain()
    except Exception as e:
        error_msg = f"Failed to initialize SQL agent chain: {e}"
        _dbg("INIT-ERROR", error_msg)
        return {
            **state,
            "output": error_msg,
            "next_node": None,
        }

    history: List = state.get("chat_history", []) or []
    question: str = state.get("input", "").strip()
    if not question:
        return {
            **state,
            "output": "No query provided for SQL agent",
            "next_node": None,
        }

    dates   = _extract_dates_from_query(question)
    is_range = _is_date_range_query(question)
    tickers = _extract_tickers_from_query(question)
    
    # Enhanced ticker extraction for correlation queries
    if not tickers and ("correlation" in question.lower() or "sentiment" in question.lower()):
        # Try to extract ticker from state context if available
        if state.get("ticker"):
            tickers = [state["ticker"]]
        elif state.get("last_ticker"):
            tickers = [state["last_ticker"]]
    
    # fallback to previous state if none found
    if not tickers and state.get("last_tickers"):
        tickers = state["last_tickers"]
    state["last_tickers"] = tickers

    _dbg("EXTRACTED-CONTEXT", {
        "query":   question,
        "tickers": tickers,
        "dates":   dates,
        "is_range": is_range,
        "is_correlation_query": "correlation" in question.lower()
    })

    # Validate we have a ticker for correlation queries
    if not tickers and ("correlation" in question.lower() or "sentiment" in question.lower()):
        return {
            **state,
            "output": "Error: No ticker found for correlation analysis. Please specify a stock symbol (e.g., AAPL, MSFT).",
            "next_node": None,
        }



    try:
        # Get database connection from the chain
        db = sql_agent_chain.db
        table_info = db.get_table_info()
    except Exception:
        table_info = (
            "stock_data(ticker TEXT, date DATE, "
            "open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC)"
        )

    try:
        sql_query = generate_sql_with_context(question, tickers, dates, is_range, table_info)
        _dbg("GENERATED-SQL", sql_query)
        _validate_sql(sql_query)
        
        # Execute the SQL query using the database's _execute method directly
        try:
            # Use the database connection to execute
            result = db._execute(sql_query)
            _dbg("DB-EXECUTE-RESULT", result)
            
            # Handle different result formats
            if hasattr(result, 'fetchall'):
                # Traditional cursor result
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description] if hasattr(result, 'description') else []
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                # List of dictionaries (common in modern ORMs)
                rows = result
                columns = list(result[0].keys()) if result else []
            else:
                # If result is already a list of tuples
                rows = result
                # Try to extract columns from the SQL query
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    columns_str = select_match.group(1)
                    columns = [col.strip() for col in columns_str.split(',')]
                else:
                    columns = []
            
            _dbg("DB-RESULT-ROWS", rows)
            _dbg("DB-RESULT-COLUMNS", columns)
            
        except AttributeError:
            # Fallback to using run method if _execute is not available
            _dbg("FALLBACK-TO-RUN", "Using db.run() method")
            raw_result = db.run(sql_query)
            _dbg("DB-RUN-RESULT", raw_result)
            
            # Parse the result if it's a string representation of a list
            if isinstance(raw_result, str) and raw_result.startswith('['):
                import ast
                try:
                    rows = ast.literal_eval(raw_result)
                    # Extract columns from SQL
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        columns_str = select_match.group(1)
                        columns = [col.strip() for col in columns_str.split(',')]
                    else:
                        columns = ['ticker', 'date', 'value']  # Default columns
                except:
                    rows = []
                    columns = []
            else:
                rows = raw_result if isinstance(raw_result, list) else []
                columns = []

        if rows:
            # Format as table
            table_data = _format_results_as_table(columns, rows)
            
            # Debug print the table
            _dbg("FORMATTED-COLUMNS", table_data["columns"])
            _dbg("FORMATTED-ROWS", table_data["rows"])
            
            # Print a pretty table to logs
            try:
                if table_data["columns"] and table_data["rows"]:
                    table_widths = []
                    for i, col in enumerate(table_data["columns"]):
                        max_width = len(str(col))
                        for row in table_data["rows"]:
                            if i < len(row):
                                max_width = max(max_width, len(str(row[i])))
                        table_widths.append(max_width)
                    
                    header_line = " | ".join(f"{col:<{table_widths[i]}}" for i, col in enumerate(table_data["columns"]))
                    separator = "-+-".join('-' * w for w in table_widths)
                    data_lines = []
                    for row in table_data["rows"]:
                        row_values = []
                        for i in range(len(table_data["columns"])):
                            if i < len(row):
                                row_values.append(f"{str(row[i]):<{table_widths[i]}}")
                        data_lines.append(" | ".join(row_values))
                    
                    table_str = "\n" + header_line + "\n" + separator + "\n" + "\n".join(data_lines)
                    _dbg("RESULT-TABLE", table_str)
            except Exception as e:
                _dbg("RESULT-TABLE-PRINT-ERROR", str(e))
            
            output = {
                "sql": sql_query,
                "table": table_data
            }
        else:
            output = {
                "sql": sql_query,
                "table": {
                    "columns": [],
                    "rows": []
                }
            }

        new_history = history + [
            HumanMessage(content=question),
            AIMessage(content=f"Executed SQL query and returned {len(rows) if rows else 0} rows")
        ]

        return {
            **state,
            "chat_history": new_history,
            "input": question,
            "output": output,
            "next_node": None,
        }

    except Exception as err:
        error_msg = f"SQL execution error: {str(err)}"
        _dbg("EXECUTION-ERROR", error_msg)
        new_history = history + [
            HumanMessage(content=question),
            AIMessage(content=error_msg)
        ]
        return {
            **state,
            "chat_history": new_history,
            "input": question,
            "output": error_msg,
            "next_node": None,
        }

# ─── MCP Server Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "SqlAgent",
        port=8010,
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool(
        name="run_sql_agent",
        description="Execute SQL queries against stock price data"
    )
    def _tool(state: dict) -> dict:
        return run_sql_agent(state)

    _dbg("SERVER-STARTING", "SQL Agent MCP Server starting on port 8010")
    mcp.run(transport="streamable-http")