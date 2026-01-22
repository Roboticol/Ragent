from pathlib import Path
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import semantic_chunk_text
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from chromadb import PersistentClient
from chromadb.config import Settings
from tools.tools import hash_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

def ingest_directory(data_dir: str):
    client = PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    collection = client.get_or_create_collection("research_rag")

    store = VectorStore(collection)
    embedder = EmbeddingModel()

    for pdf_path in Path(data_dir).glob("*.pdf"):
        file_hash = hash_file(pdf_path)

        text = load_pdf(pdf_path)
        chunks = semantic_chunk_text(embedder, 0.75, 8, text)

        embeddings = embedder.embed_texts(chunks)

        metadatas = [
            {
                "source": pdf_path.name,
                "hash": file_hash,
                "chunk_id": i
            }
            for i in range(len(chunks))
        ]

        store.add(chunks, embeddings, metadatas)

        print(f"Ingested {pdf_path.name} ({len(chunks)} chunks)")
    print("Collection count:", collection.count())

if __name__ == "__main__":
    ingest_directory("./data/papers")

    client = PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    collection = client.get_or_create_collection("research_rag")
    store = VectorStore(collection)

    embedder = EmbeddingModel()

    query_embedding = embedder.embed_texts(["Agents"])

    print(store.search(query_embedding))
    print("Final count:", collection.count())
    print("Chroma dir exists:", CHROMA_DIR.exists())
    print("Chroma files:", list(CHROMA_DIR.iterdir()))

    