from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

# Import MCP initializer alongside your workflow functions
from graph import (
    run_query_with_persistence,
    compile_workflow_with_persistence,
    _init_mcp_tools,
)

app = FastAPI(
    title="SQL Agent Demo (Gemma) with Persistence",
    description="Multi-agent Stock & News assistant with LangGraph persistence",
    version="0.2.0",
)

class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    thread_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize MCP tools and compile the workflow with persistence on startup"""
    try:
        # 1) Initialize MCP clients (SQL, News, Fallback)
        await _init_mcp_tools()
        # 2) Compile the LangGraph workflow (with Postgres persistence)
        await compile_workflow_with_persistence()
        print("✅ FastAPI server started with LangGraph persistence")
    except Exception as e:
        print(f"⚠️  Failed to initialize persistence: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with persistence support
    
    - If thread_id is provided, continues existing conversation
    - If thread_id is None, starts new conversation thread
    """
    try:
        # Generate new thread ID if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        # Run query with persistence
        answer = await run_query_with_persistence(request.query, thread_id)
        return ChatResponse(answer=answer, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/new")
async def new_chat_thread(request: Request):
    """Start a new conversation thread"""
    payload = await request.json()
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    thread_id = str(uuid.uuid4())
    answer = await run_query_with_persistence(query, thread_id)
    return ChatResponse(answer=answer, thread_id=thread_id)

@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """Get conversation history for a specific thread"""
    # TODO: implement retrieval from persistence tables
    return {"thread_id": thread_id, "message": "History retrieval not yet implemented"}

@app.get("/")
def read_root():
    return {
        "message": "SQL Agent Demo with LangGraph Persistence",
        "features": [
            "Multi-agent MCP architecture",
            "PostgreSQL persistence",
            "Thread-based conversations",
            "Stock price queries",
            "News headline queries"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "persistence": "enabled"}