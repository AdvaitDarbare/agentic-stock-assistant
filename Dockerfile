FROM python:3.12.3-slim-bullseye

# install libpq-dev & build-essential so psycopg can compile its C extension
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libpq-dev \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml poetry.lock ./

# Install Poetry and project dependencies (skip installing the root project)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --without dev --no-interaction --no-ansi

# Copy project code
COPY . .

# Expose application port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
