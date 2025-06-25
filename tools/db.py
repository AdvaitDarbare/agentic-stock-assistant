import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent


def get_sql_agent_chain():
    """
    Initializes and returns a SQL agent chain connected to the stock_prices table.
    """
    # Load DB credentials from environment
    load_dotenv()
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")
    uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

    # Set up SQLDatabase and agent toolkit
    sql_db = SQLDatabase.from_uri(uri, include_tables=["stock_prices"])
    llm = ChatOpenAI(temperature=0)
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    # Create and return the SQL agent
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=False
    )
