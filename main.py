from fastapi import FastAPI
from langgraph.fastapi import add_routes

# 1️⃣  import the compiled LangGraph object from graph.py
from graph import workflow

app = FastAPI(
    title="SQL Agent Demo (Gemma)",
    description="Minimal FastAPI wrapper around a multi-agent Stock & News assistant",
    version="0.1.0",
)

# 2️⃣  Mount the LangGraph routes:
#     /chat  – POST {"input": "..."} → assistant response
#     /graph – interactive graph+console UI
add_routes(app, workflow)          # path_prefix defaults to "/"

# 3️⃣  Your own sanity-check endpoints
@app.get("/")
def read_root():
    return {"message": "Hello, your API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
