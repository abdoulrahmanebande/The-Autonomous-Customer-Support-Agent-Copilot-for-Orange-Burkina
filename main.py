import uvicorn
from contextlib import asynccontextmanager # Imported for safe lifecycle handling
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from src.retrieval_pipeline.rag_search import RAGSearch

app_state = {"rag_search": None}

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        print("[SERVER] Initializing RAG Pipeline and Vector DB connection...")
        app_state["rag_search"] = RAGSearch()
        print("[SERVER] RAG Pipeline loaded successfully and ready.")
    except Exception as e:
        print("\n" + "="*50)
        print(f"[SERVER CRITICAL ERROR] RAG failed to initialize on startup!")
        print(f"Reason: {str(e)}")
        print("="*50 + "\n")

        app_state["rag_search"] = None
        
    yield # Control is handed over to handle incoming API traffic
    
    print("[SERVER] Cleaning up pipeline resources...")
    app_state["rag_search"] = None


# Initialize FastAPI App and register the lifespan manager
app = FastAPI(
    title="Customer Support Agent Copilot API",
    description="Backend API for real-time telecom knowledge retrieval and synthesis.",
    version="1.0.0",
    lifespan=lifespan # <--- Registered here
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=3)

class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    title: str
    url: str
    text: str

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_201_CREATED)
async def process_copilot_query(payload: QueryRequest):

    rag_search_instance = app_state.get("rag_search")
    
    if rag_search_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="The RAG engine failed to initialize on startup. Please check backend infrastructure or logs."
        )
        
    try:
        print(f"[API] Processing query: {payload.query} (top_k={payload.top_k})")
        result = rag_search_instance.search_and_summarize(query=payload.query, top_k=payload.top_k)
        
        return QueryResponse(
            title=result.title,
            url=result.url,
            text=result.text
        )
        
    except Exception as e:
        print(f"[API ERROR] Exception occurred during generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal Server Generation Error: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if app_state.get("rag_search") is dict or app_state.get("rag_search") is None:
        return {"status": "unhealthy", "database_connected": False}
    return {"status": "healthy", "database_connected": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)