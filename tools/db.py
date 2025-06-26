import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_sql_agent_chain() -> AgentExecutor:
    load_dotenv()
    uri = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    sql_db = SQLDatabase.from_uri(uri, include_tables=["stock_prices"])
    llm = ChatOpenAI(temperature=0)
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    # Prompt for OpenAI tools agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful SQL assistant for the `stock_prices` table.\n"
                "When the user refers to a prior date (e.g. 'that day', 'same day'), "
                "infer it from chat_history.\n\n"
                "Use the provided tools to query the database and provide accurate information."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(
        llm=llm,
        tools=toolkit.get_tools(),
        prompt=prompt,
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )