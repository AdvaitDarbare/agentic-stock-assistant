import os
import asyncio
import aiohttp
from datetime import date, timedelta, datetime
from dotenv import load_dotenv

from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_postgres.v2.engine import PGEngine
from langchain_ollama import OllamaEmbeddings

from langchain.schema import Document
from sqlalchemy import text

# ------------------ CONFIG & ENV ------------------
load_dotenv()
DATABASE_URL = (
    f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME')}"
)
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Just grab a handful of tickers to test
ALL_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","ADBE","PEP",
    # …etc, up to 100
]
TICKERS = ALL_TICKERS[:5]  # <- change slice as desired

# ------------------ Dynamic 7-day window ------------------
TODAY     = date.today()
FROM_DATE = (TODAY - timedelta(days=7)).isoformat()
TO_DATE   = TODAY.isoformat()

# ------------------ Fetch from Finnhub ------------------
async def fetch_company_news(session, symbol, from_date, to_date):
    url = (
        "https://finnhub.io/api/v1/company-news"
        f"?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
    )
    async with session.get(url) as resp:
        if resp.status != 200:
            print(f"❌ Finnhub error for {symbol}: {resp.status}")
            return []
        return await resp.json()

# ------------------ Main ingestion ------------------
async def main():
    # 1) Connect & (re)create table with VECTOR(768)
    engine = PGEngine.from_connection_string(DATABASE_URL)
    async with engine._pool.connect() as conn:
        # Drop old
        await conn.execute(text('DROP TABLE IF EXISTS news_articles CASCADE;'))
        # Ensure extensions
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        # Create table
        await conn.execute(text("""
            CREATE TABLE news_articles (
              langchain_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
              headline    TEXT,
              embedding   VECTOR(768),
              url         TEXT,
              publisher   TEXT,
              date        TIMESTAMP,
              stock       TEXT
            );
        """))
        await conn.commit()
        print("✅ Table news_articles ready (VECTOR(768))")

    # 2) Init local Ollama embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    print(f"✅ Ollama embeddings ready ({EMBED_MODEL})")

    # 3) Create PGVectorStore wrapper
    vs = await PGVectorStore.create(
        engine=engine,
        embedding_service=embeddings,
        table_name="news_articles",
        id_column="langchain_id",
        content_column="headline",
        embedding_column="embedding",
        metadata_columns=["url", "publisher", "date", "stock"],
    )
    print("✅ PGVectorStore initialized")

    total = 0
    async with aiohttp.ClientSession() as session:
        for ticker in TICKERS:
            print(f"📈 Fetching {ticker} news ({FROM_DATE} → {TO_DATE})")
            items = await fetch_company_news(session, ticker, FROM_DATE, TO_DATE)
            docs = []
            for art in items:
                h = art.get("headline") or ""
                s = art.get("summary")  or ""
                if not h: continue

                ts = art.get("datetime")
                dt = datetime.fromtimestamp(ts) if ts else datetime.now()

                txt = h + (". " + s if s else "")
                docs.append(Document(
                    page_content=txt,
                    metadata={
                        "url":       art.get("url",""),
                        "publisher": art.get("source",""),
                        "date":      dt.isoformat(),
                        "stock":     ticker
                    },
                ))

            if docs:
                await vs.aadd_documents(docs)
                total += len(docs)
                print(f"   ✅ Stored {len(docs)} docs for {ticker}")
            else:
                print(f"   ⚠️ No docs for {ticker}")

    print(f"\n🎉 Done — {total} total documents ingested.")
    await engine.close()

if __name__ == "__main__":
    asyncio.run(main())
