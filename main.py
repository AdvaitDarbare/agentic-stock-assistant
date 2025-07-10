# main.py
from fastapi import FastAPI, Request
from graph import workflow   # ← your compiled StateGraph

app = FastAPI(
    title="SQL Agent Demo (Gemma)",
    description="Minimal FastAPI wrapper around a multi-agent Stock & News assistant",
    version="0.1.0",
)

@app.post("/chat")
async def chat(request: Request):
    # 1️⃣ Read the incoming JSON body
    payload = await request.json()                          
    query   = payload.get("input") or payload.get("query") or ""
    
    # 2️⃣ Create complete fresh state for the workflow
    init_state = {
        "query": query,
        "chat_history": [],  # Fresh start for each API call
        "last_ticker": None,
        "last_date": None,
        "last_query": None,
        "need_sql": False,
        "need_news": False,
        "sql_done": False,
        "news_done": False,
        "sql_result": None,
        "news_result": None,
        "answer": None,
        "error": None,
        "is_range_query": False,
        "input": query,
        "output": None,
        "current_date": None,
        "next_node": None
    }
    
    # 3️⃣ Invoke your LangGraph workflow
    result = workflow.invoke(init_state)
    # 4️⃣ Return exactly what the graph produced
    return {"answer": result["answer"]}

@app.get("/")
def read_root():
    return {"message": "Hello, your API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
