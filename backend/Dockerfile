FROM python:3.11-slim

# install pg_isready and suppress apt warnings
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client \
 && rm -rf /var/lib/apt/lists/*

# upgrade pip and install Poetry
RUN pip install --upgrade pip --no-cache-dir --disable-pip-version-check \
 && pip install poetry --no-cache-dir

WORKDIR /app

# Configure Poetry to install into global env, then install only main deps
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
 && (poetry install --without dev --no-cache || \
     pip install --no-cache-dir psycopg2-binary pandas pgvector langchain langchain-core langchain-ollama langgraph fastapi uvicorn yfinance kaggle mlflow langchain-postgres asyncpg mcp mcp-server langchain-mcp-adapters xxhash ormsgpack python-dotenv)

# copy your app code
COPY . .

# wait for Postgres to be ready, then launch the news_agent by default
CMD sh -c "\
    until pg_isready -h ${POSTGRES_HOST:-db} -U ${POSTGRES_USER:-postgres}; do \
      echo 'Waiting for Postgres…' && sleep 1; \
    done; \
    python agents/news_agent.py \
"