# news_agent.py
import os
import psycopg2
import openai
from datetime import date, timedelta
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import AsIs

# Load ENV & API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# DB config
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

COMPANY = "TSLA"

def get_db_connection():
    """
    Returns a psycopg2 connection with pgvector support registered.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    # register pgvector adapter so psycopg knows how to cast Vector columns
    try:
        register_vector(conn)
    except Exception:
        # if register_vector isn’t available, assume vector already works
        pass
    return conn

def fetch_raw_articles(limit=5):
    """
    Fetch the most recent <limit> raw TSLA articles from news_embeddings.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, publish_date
                  FROM news_embeddings
                 WHERE company = %s
              ORDER BY publish_date DESC
                 LIMIT %s
            """, (COMPANY, limit))
            return cur.fetchall()

def similarity_search(query: str, k=5):
    """
    Generate an embedding for `query` and then run a
    pgvector <=> distance search to return top-k articles.
    """
    # 1) create the query embedding
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    q_emb = resp.data[0].embedding

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 2) order by cosine similarity (1 − cosine_distance)
            #    you can also order by embedding <=> %s for raw distance
            cur.execute(f"""
                SELECT content, publish_date,
                       1 - (embedding <=> %s::vector) AS similarity
                  FROM news_embeddings
                 WHERE company = %s
              ORDER BY similarity DESC
                 LIMIT %s
            """, (q_emb, COMPANY, k))
            return cur.fetchall()

def run_news_agent(state, k: int = 5):
    """
    LangGraph entrypoint. Uses state['input'] as the query.
    Returns the updated state with state['output'] filled.
    """
    # extract the user’s query (fallback if missing)
    query = (
        state.get("input")
        or f"latest news on {COMPANY} stock"
    )

    # 1) show raw latest
    raw = fetch_raw_articles(limit=5)

    # 2) run vector‐similarity search
    sim = similarity_search(query, k=k)

    # 3) format both blocks into a single output
    lines = ["✅ **Recent 5 raw articles:**"]
    for i, (content, pub_date) in enumerate(raw, 1):
        snippet = content.replace("\n"," ")[:200]
        lines.append(f"{i}. [{pub_date}] {snippet}")

    lines.append("\n🎯 **Top-{k} similar articles for:** “{query}”")
    for i, (content, pub_date, score) in enumerate(sim, 1):
        snippet = content.replace("\n"," ")[:200]
        lines.append(f"{i}. [{pub_date}] (sim={score:.3f}) {snippet}")

    # stash back into state and return
    state["output"] = "\n".join(lines)
    return state

# keep your old main() for standalone runs
def main():
    print("=== RAW NEWS ===")
    for row in fetch_raw_articles():
        print(row)
    print("\n=== SIMILARITY SEARCH ===")
    for row in similarity_search("Tesla stock price earnings revenue"):
        print(row)

if __name__ == "__main__":
    main()