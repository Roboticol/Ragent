from fastapi import FastAPI
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings

from core.vector_store import VectorStore
from core.embeddings import Embedder
from agent.orchestrator import run_agent

app = FastAPI(title="Research RAG API")

chroma_client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = chroma_client.get_or_create_collection("research_rag")

vector_store = VectorStore(collection)
embedder = Embedder()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    query_embedding = embedder.embed_texts([req.query])

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=req.top_k
    )

    documents = results["documents"][0]

    answer = run_agent(
        question=req.query,
        context=documents
    )

    return QueryResponse(
        answer=answer,
        sources=documents
    )