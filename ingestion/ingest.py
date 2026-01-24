from pathlib import Path
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import semantic_chunk_text
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from chromadb import PersistentClient
from chromadb.config import Settings
from tools.tools import hash_file
from dotenv import load_dotenv
import os

load_dotenv()
INGEST_DIR = os.getenv("INGEST_DIR")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

def is_already_ingested(collection, file_hash: str) -> bool:
    result = collection.get(
        where={"hash": file_hash},
        limit=1
    )
    return len(result["ids"]) > 0

def ingest_directory(data_dir: str):
    client = PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    collection = client.get_or_create_collection("research_rag")

    # metadata for response
    pdfs_ingested = []
    txts_ingested = []
    prev_colcount = collection.count()
    
    store = VectorStore(collection)
    embedder = EmbeddingModel()

    # iterate through text files
    for txt_path in Path(data_dir).glob("*.txt"):
        file_hash = hash_file(txt_path)

        # check if file_hash already exists in collection, if it does, then skip current iteration
        if is_already_ingested(collection, file_hash):
            print(f"Skipping {txt_path.name} (already ingested)")
            continue
        
        textf = open(txt_path, "r")
        text = list(textf)
        chunks = semantic_chunk_text(embedder, 0.75, 8, text)

        embeddings = embedder.embed_texts(chunks)

        metadatas = [
            {
                "source": txt_path.name,
                "hash": file_hash,
                "chunk_id": i
            }
            for i in range(len(chunks))
        ]

        store.add(chunks, embeddings, metadatas)
        txts_ingested.append(txt_path.name)
        print(f"Ingested {txt_path.name} ({len(chunks)} chunks)")
        textf.close()


    # iterate through pdf files
    for pdf_path in Path(data_dir).glob("*.pdf"):
        file_hash = hash_file(pdf_path)

        # check if file_hash already exists in collection, if it does, then skip current iteration
        if is_already_ingested(collection, file_hash):
            print(f"Skipping {pdf_path.name} (already ingested)")
            continue
        
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
        pdfs_ingested.append(pdf_path.name)
        print(f"Ingested {pdf_path.name} ({len(chunks)} chunks)")

    colcount = collection.count()
    print("Collection count:", collection.count())
    return {
        "txts_ingested": txts_ingested,
        "pdfs_ingested": pdfs_ingested,
        "previous_collection_count": prev_colcount,
        "final_collection_count": colcount,
        "collection_count_change": colcount - prev_colcount
    }

if __name__ == "__main__":
    ingest_directory(INGEST_DIR)

    client = PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    collection = client.get_or_create_collection("research_rag")
    # store = VectorStore(collection)

    # embedder = EmbeddingModel()

    # query_embedding = embedder.embed_texts(["Agents"])

    # print(store.search(query_embedding))
    print("Final collection count:", collection.count())
    print("Collect")
    print("Chroma dir exists:", CHROMA_DIR.exists())
    print("Chroma files:", list(CHROMA_DIR.iterdir()))