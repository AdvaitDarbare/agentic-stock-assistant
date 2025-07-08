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
    # 2️⃣ Invoke your LangGraph workflow
    result  = workflow.invoke({"query": query})
    # 3️⃣ Return exactly what the graph produced
    return {"answer": result["answer"]}

@app.get("/")
def read_root():
    return {"message": "Hello, your API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
