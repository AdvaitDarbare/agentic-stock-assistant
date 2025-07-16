from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

# Import MCP initializer alongside your workflow functions
try:
    from graph import (
        run_query_with_persistence,
        compile_workflow_with_persistence,
        _init_mcp_tools,
    )
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"MCP imports failed: {e}")
    MCP_AVAILABLE = False

app = FastAPI(
    title="FinanceScope - AI Stock & Market Analysis",
    description="Intelligent financial analysis with real-time stock data, news insights, and sentiment analysis powered by multi-agent LangGraph workflows",
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
    if not MCP_AVAILABLE:
        print("⚠️  MCP not available, running in fallback mode")
        return
        
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
    thread_id = request.thread_id or str(uuid.uuid4())
    
    if not MCP_AVAILABLE:
        # Fallback mode - simple responses
        query_lower = request.query.lower()
        if any(word in query_lower for word in ['stock', 'price', 'aapl', 'msft', 'googl', 'tesla', 'nvda']):
            answer = "I'm unable to access real-time stock data at the moment. The MCP servers are not properly connected. Please check the server configuration."
        elif any(word in query_lower for word in ['news', 'headlines', 'article']):
            answer = "I'm unable to access news data at the moment. The MCP servers are not properly connected. Please check the server configuration."
        else:
            answer = f"Hello! I received your message: '{request.query}'. I'm currently running in fallback mode because the MCP servers are not properly connected. You can try asking about stock prices or news headlines, but I won't be able to provide real data."
        
        return ChatResponse(answer=answer, thread_id=thread_id)
    
    try:
        # Run query with persistence
        answer = await run_query_with_persistence(request.query, thread_id)
        return ChatResponse(answer=answer, thread_id=thread_id)
    except Exception as e:
        # Fallback response if MCP/persistence fails
        print(f"Error in chat endpoint: {e}")
        
        # Simple fallback responses based on query content
        query_lower = request.query.lower()
        if any(word in query_lower for word in ['stock', 'price', 'aapl', 'msft', 'googl', 'tesla', 'nvda']):
            answer = "I'm unable to access real-time stock data at the moment. Please try again later or check your connection to the MCP servers."
        elif any(word in query_lower for word in ['news', 'headlines', 'article']):
            answer = "I'm unable to access news data at the moment. Please try again later or check your connection to the MCP servers."
        else:
            answer = f"Hello! I received your message: '{request.query}'. I'm currently having trouble connecting to my data sources, but I'm working on it. You can try asking about stock prices or news headlines."
        
        return ChatResponse(answer=answer, thread_id=thread_id)

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
    return {"message": "Hello, your API is running!"}

@app.post("/chat/test")
async def chat_test(request: ChatRequest):
    """Simple test endpoint that bypasses MCP"""
    return ChatResponse(
        answer=f"Hello! You said: '{request.query}'. This is a test response to verify the frontend-backend connection is working.",
        thread_id=request.thread_id or str(uuid.uuid4())
    )

@app.post("/chat/simple")
async def chat_simple(request: dict):
    """Ultra simple test endpoint"""
    query = request.get('query', 'No query provided')
    return {
        "answer": f"Simple response: You said '{query}'. The backend is working!",
        "thread_id": str(uuid.uuid4())
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "persistence": "enabled"}