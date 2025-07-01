import os
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama import ChatOllama

load_dotenv()

def get_sql_agent_chain():
    DB_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    db = SQLDatabase.from_uri(DB_URI)

    llm = ChatOllama(model=os.getenv("LLM_MODEL"), temperature=0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(llm=llm, toolkit=toolkit)

    # We'll return an object with the LLM, db, and schema info.
    class SqlAgentChain:
        def __init__(self, llm, db, input_schema):
            self.llm = llm
            self.db = db
            self.input_schema = input_schema

    return SqlAgentChain(llm=llm, db=db, input_schema=toolkit)
