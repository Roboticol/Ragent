from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel
from api.logger import log_query
from agent.orchestrator import run_agent

# ---------- App ----------
app = FastAPI(title="Research RAG API")

# ---------- Auth ----------
load_dotenv()
API_KEY = os.getenv("API_KEY")
print(API_KEY)

def verify_api_key(x_api_key: str):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- Load once ----------
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(
        anonymized_telemetry=False
    )
)
collection = client.get_or_create_collection("research_rag")

print("Final count:", collection.count())
print("Chroma dir exists:", CHROMA_DIR.exists())
print("Chroma files:", list(CHROMA_DIR.iterdir()))

    
vector_store = VectorStore(collection)
embedder = EmbeddingModel()

# ---------- Schemas ----------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)                 
    start = time.time()                       

    # Embed query
    query_embedding = embedder.embed_texts([req.query])

    # Retrieve
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=req.top_k
    )

    print(results)

    documents = results["documents"][0]

    # Run agent
    answer = run_agent(
        query=req.query,
        sources=documents
    )

    latency_ms = int((time.time() - start) * 1000)

    # Log request
    log_query(
        query=req.query,
        top_k=req.top_k,
        retrieved=len(documents),
        latency_ms=latency_ms
    )

    return QueryResponse(
        answer=answer,
        sources=documents
    )
