from core.chroma_client import get_chroma_collection
from core.vector_store import VectorStore
from agent.orchestrator import run_agent

collection = get_chroma_collection()
vector_store = VectorStore(collection)

if __name__ == "__main__":
    run_agent(vector_store)
