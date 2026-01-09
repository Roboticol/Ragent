from pathlib import Path
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import semantic_chunk_text
from core.embeddings import Embedder
from core.vector_store import VectorStore
from chromadb import Client
from chromadb.config import Settings
from tools.tools import hash_file

def ingest_directory(data_dir: str):
    client = Client(Settings(persist_directory="./chroma_db"))
    collection = client.get_or_create_collection("research_rag")

    store = VectorStore(collection)
    embedder = Embedder()

    for pdf_path in Path(data_dir).glob("*.pdf"):
        file_hash = hash_file(pdf_path)

        text = load_pdf(pdf_path)
        chunks = semantic_chunk_text(text)

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

if __name__ == "__main__":
    ingest_directory("data/papers")
