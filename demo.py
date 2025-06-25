import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI  # Updated import
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # Updated import
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit  # Updated import
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    output: str
    current_date: str  # 👈 Track most recent date referenced

# Load environment variables
load_dotenv()
user, pwd, host, port, db = (
    os.getenv("DB_USER"),
    os.getenv("DB_PASS"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("DB_NAME"),
)
uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

# Set up database and agent
sql_db = SQLDatabase.from_uri(uri, include_tables=["stock_prices"])
llm = ChatOpenAI(temperature=0)
toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

# SQL agent (chain)
sql_agent_chain = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True
)

# LangGraph state
def run_agent(state):
    import re

    history = state["chat_history"]
    question = state["input"]

    # --- Extract date from the question (YYYY-MM-DD OR MM/DD/YYYY) ---
    current_date = state.get("current_date", "")

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", question)
    slash_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", question)

    if iso_match:
        current_date = iso_match.group()
    elif slash_match:
        m, d, y = slash_match.groups()
        if len(y) == 2:            # handle YY → 20YY
            y = '20' + y
        current_date = f"{y}-{int(m):02d}-{int(d):02d}"

    # If the user says "that day", replace it with the current_date
    if "that day" in question.lower() and current_date:
        question = re.sub(r"\bthat day\b", current_date, question, flags=re.IGNORECASE)

    print(f"\n> User: {question}")
    messages = history + [HumanMessage(content=question)]
    result = sql_agent_chain.invoke({
        "input": question,
        "chat_history": history
    })
    answer_text = result.get("output", str(result))
    messages.append(AIMessage(content=answer_text))

    return {
        "chat_history": messages,
        "input": question,
        "output": answer_text,
        "current_date": current_date
    }

# Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("sql_agent", run_agent)
workflow.set_entry_point("sql_agent")
workflow.set_finish_point("sql_agent")
graph = workflow.compile()

# Run loop
if __name__ == "__main__":
    print("SQL Agent Demo - LangGraph with LangSmith Tracing")
    print("=" * 50)
    
    # Initialize state
    state = {
        "input": "",
        "chat_history": [],
        "output": "",
        "current_date": ""
    }
    
    print("\nLangSmith tracing is enabled if you have LANGCHAIN_API_KEY set.")
    print("Type 'quit' to exit")
    print("\nExample queries:")
    print("- 'What was the stock price on 2024-01-15?'")
    print("- 'Show me prices for AAPL on 12/25/2023'")
    print("- 'What happened that day?' (after mentioning a date)")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif not user_input:
                continue
            
            # Update state with user input
            state["input"] = user_input
            
            # Run the graph with LangSmith tracing
            result = graph.invoke(state)
            
            # Update state with result
            state = result
            
            print(f"\nAssistant: {result['output']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\nCheck your LangSmith dashboard for execution traces!")