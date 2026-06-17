import uvicorn
from typing import List
import shutil
import uuid
from pathlib import Path
import os
from contextlib import asynccontextmanager # Imported for safe lifecycle handling
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field, ConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from src.data_ingestion_pipeline.vector_store import QdrantVectorStore
from src.retrieval_pipeline.rag_search import RAGSearch
from src.logger import logging
from src.retrieval_pipeline.rag_search import GuardrailRAGEngine

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
    
llm_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
vector_store = QdrantVectorStore()

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

UPLOAD_DIR = os.path.join(os.getcwd(), 'temp_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
class IngestionResponse(BaseModel):
    status: str
    message: str
    filename: str 

class ChatMessage(BaseModel):
    role: str=Field(description='Whether it is a human or an AI')
    content: str=Field(description='The actual message.')

class QueryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=3)
    chat_history: List[ChatMessage]=[]

class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    title: str
    url: str
    text: str


    
def rewrite_query_with_history(user_query: str, chat_history: List[ChatMessage]) -> str:
    if not chat_history:
        return user_query
    
    messages = [
        AIMessage(
            content=(
                "You are an internal query contextualizer for a telecom support system.\n"
                "Review the following conversation history and the latest user question.\n"
                "Reformulate the question into a single standalone query that contains all necessary context "
                "to search a technical documentation database. Do NOT answer the question. Just output the rewritten query string."
            )
        )
    ]
    
    for message in chat_history:
        if message.role == 'human':
            messages.append(HumanMessage(content=message.content))
        else:
            messages.append(AIMessage(content=message.content))
            
    messages.append(HumanMessage(content=f"Latest Question: {user_query}\n\nStandalone Query:"))
    
    try:
        response = llm_model.invoke(messages)
        rewritten_text = response.content.strip()
        logging.info(f"[MEMORY LOG] Original: '{user_query}' -> Contextualized: '{rewritten_text}'")
        return rewritten_text
    except Exception as e:
        logging.info(f"[MEMORY WARNING] Query rewrite failed: {e}. Falling back to raw text input.")
        return user_query
    

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_copilot_query(payload: QueryRequest):

    rag_search_instance = app_state.get("rag_search")
    
    if rag_search_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="The RAG engine failed to initialize on startup. Please check backend infrastructure or logs."
        )
        
    try:
        search_term = rewrite_query_with_history(payload.query, payload.chat_history)
        print(f"[API] Processing query: {search_term} (top_k={payload.top_k})")
        # Run the guardrail engine
        guardrail_engine = GuardrailRAGEngine(vector_store=vector_store, threshold=0.35)
        secure_context = guardrail_engine.get_secure_context(search_term)
    
        # Safe Exit if information is missing or untrusted
        if secure_context == "TRIGGER_FALLBACK_SIGNAL":
            return {
                "title": "Information Not Found",
                "url": "N/A",
                "text": "I am sorry, but the official internal documentation does not contain verified details to accurately answer this query. Please consult an administrator."
            }
            
        result = rag_search_instance.search_and_summarize(query=search_term, top_k=payload.top_k)
        
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

def background_vector_ingestion(file_path: str, collection_name: str):
    try:
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()
        
        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            return
        raw_documents = loader.load()  

        vector_store.build_from_documents(raw_documents)
    except Exception as e:
        logging.info(f"Backgound Vector Ingestion Failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
@app.post('/api/v1/ingest', response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_telecom_document(
    background_tasks: BackgroundTasks,
    file: UploadFile=File(...)
):
    # 1. Extract and sanitize the filename string cleanly
    filename_str = file.filename if file.filename else f"uploaded_document_{uuid.uuid4().hex}.pdf"
    if not filename_str.endswith(('.pdf', '.json', '.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Unsupported file extension. Please upload a .pdf, .txt or .json file.'
        )
    temp_file_path = os.path.join(os.getcwd(), UPLOAD_DIR, filename_str)
    
    try:
        with open(temp_file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file to storage: {str(e)}"
        )
    
    background_tasks.add_task(
        background_vector_ingestion,
        file_path=str(temp_file_path),
        collection_name='orange_telecom_store'
    )
    
    return IngestionResponse(
        status='Processing',
        message='Document Uploaded successfully. Vector tracking computation'
                'is executing in the background.',
        filename=filename_str
    )
        
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if app_state.get("rag_search") is dict or app_state.get("rag_search") is None:
        return {"status": "unhealthy", "database_connected": False}
    return {"status": "healthy", "database_connected": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)