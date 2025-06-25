import os
from datetime import date, timedelta

from dotenv import load_dotenv
import requests
import openai
import psycopg2

# ------------------ CONFIG & ENV ------------------
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_NAME  = os.getenv("DB_NAME")
DB_USER  = os.getenv("DB_USER")
DB_PASS  = os.getenv("DB_PASS")
DB_HOST  = os.getenv("DB_HOST")
DB_PORT  = os.getenv("DB_PORT")

openai.api_key = OPENAI_API_KEY
COMPANY = "TSLA"        # change to any ticker or company name you like

# ------------------ FETCH NEWS --------------------
today = date.today()
one_week_ago = today - timedelta(days=7)

url = "https://finnhub.io/api/v1/company-news"
params = {
    "symbol": COMPANY,
    "from": one_week_ago.isoformat(),
    "to": today.isoformat(),
    "token": FINNHUB_API_KEY
}

res = requests.get(url, params=params)
print(f"Finnhub status: {res.status_code}")

if res.status_code != 200:
    print("Finnhub error:", res.text)
    exit(1)

articles = res.json()
if not articles:
    print("No articles returned—try a different ticker or date range.")
    exit(0)

# ------------------ DB CONNECT --------------------
print("Connecting to DB with:")
print(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT)

conn = psycopg2.connect(
    dbname   = DB_NAME,
    user     = DB_USER,
    password = DB_PASS,
    host     = DB_HOST,
    port     = DB_PORT
)
cursor = conn.cursor()

# ------------------ EMBED & INSERT ---------------
for art in articles:
    title = art.get("headline", "")
    desc  = art.get("summary") or ""
    text  = f"{title}\n{desc}".strip()
    pub_date = art.get("datetime")
    if pub_date:
        pub_date = date.fromtimestamp(pub_date).isoformat()
    else:
        pub_date = today.isoformat()

    # get embedding
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = resp.data[0].embedding  # list of floats

    # insert row
    cursor.execute(
        """
        INSERT INTO news_embeddings (content, embedding, publish_date, company)
        VALUES (%s, %s, %s, %s)
        """,
        (text, embedding, pub_date, COMPANY)
    )
    print(f"✅ Inserted article: {title[:80]}")

conn.commit()
cursor.close()
conn.close()
print("🎉 All done!")