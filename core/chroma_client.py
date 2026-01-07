import chromadb
from chromadb.config import Settings

def get_chroma_collection():
    client = chromadb.Client(
        Settings(
            persist_directory="/data/chroma",
            anonymized_telemetry=False
        )
    )

    return client.get_or_create_collection(
        name="research_rag"
    )
