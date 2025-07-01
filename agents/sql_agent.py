import re
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import mlflow.langchain
from tools.db import get_sql_agent_chain
from state import AgentState
from langchain_ollama import ChatOllama


def clean_sql_output(text: str) -> str:
    # This regex will now look for content within ```sql ... ``` or just ``` ... ```
    # If no such block is found, it will try to clean up leading/trailing backticks and 'sql' keyword
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip("`").replace("sql", "").strip() # Fallback for no code blocks

def run_sql_agent(state: AgentState) -> AgentState:
    sql_agent_chain = get_sql_agent_chain()

    llm = ChatOllama(model=sql_agent_chain.llm.model, temperature=sql_agent_chain.llm.temperature)  # reinstantiate LLM

    history = state.get("chat_history", [])
    if not isinstance(history, list):
        history = []

    question = state.get("input", "").strip()
    current_date = state.get("current_date", "")

    # Extract date from input
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", question)
    slash_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", question)
    if iso_match:
        current_date = iso_match.group()
    elif slash_match:
        m, d, y = slash_match.groups()
        if len(y) == 2:
            y = "20" + y
        current_date = f"{y}-{int(m):02d}-{int(d):02d}"

    # 🧠 Inject schema into the prompt
    try:
        table_info = sql_agent_chain.input_schema.db.get_table_info()
    except Exception as e:
        print("[ERROR] Could not get schema info:", e)
        table_info = "stock_prices(symbol TEXT, date DATE, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT, pct_change FLOAT)"

    prompt = f"""
You are a helpful assistant that generates PostgreSQL queries.

Today's date: {current_date}

Schema:
{table_info}

Instructions:
- Only return valid SQL.
- Wrap column names in double quotes.
- **Always wrap the SQL query in a ```sql ... ``` block.**
""".strip()

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ]
    cleaned_sql = "" # Initialize here for error logging
    try:
        response = llm.invoke(messages).content
        cleaned_sql = clean_sql_output(response)
        print("[DEBUG] Generated SQL:", cleaned_sql)
        raw_answer = sql_agent_chain.db.run(cleaned_sql)
        answer_text = str(raw_answer)
    except Exception as e:
        print("[ERROR] SQL chain or database run failed:", e)
        print(f"[ERROR] SQL that failed: {cleaned_sql}") # Log the problematic SQL
        answer_text = "Sorry, I couldn't answer that due to a database error."

    if not answer_text:
        answer_text = "I couldn't find an answer."

    messages = history + [
        HumanMessage(content=question),
        AIMessage(content=answer_text)
    ]

    return {
        "chat_history": messages,
        "input": question,
        "output": answer_text,
        "current_date": current_date,
        "next_node": None,
    }