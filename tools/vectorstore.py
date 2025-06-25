# vector_store.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_postgres.vectorstores import PGVector
from langchain_openai.embeddings import OpenAIEmbeddings

def get_vectorstore():
    # Load credentials
    load_dotenv()
    user, pwd, host, port, db = map(os.getenv,
        ("DB_USER","DB_PASS","DB_HOST","DB_PORT","DB_NAME")
    )
    # Use psycopg3 driver
    uri = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db}"
    engine = create_engine(uri)

    # Embedding client
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize PGVector in JSONB mode
    vectorstore = PGVector(
        embeddings,
        connection=engine,
        collection_name="news_embeddings",
        use_jsonb=True,
    )
    return vectorstore