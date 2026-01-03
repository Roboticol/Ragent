# rag/ingestion/ingest.py
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from ingestion.chunker import semantic_chunk_text
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore

def ingest_pdf(
    pdf_path: str,
    vector_store: VectorStore,
    embedder: EmbeddingModel,
    semantic_chunker: bool
):
    pages = load_pdf(pdf_path)

    all_chunks = []
    metadatas = []

    for page in pages:
        chunks = []
        if semantic_chunker:
            chunks = semantic_chunk_text(embedder, 0.75, 8)
        else:
            chunks = chunk_text(page["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "source": pdf_path,
                "page": page["page"],
                "chunk_id": i,
                "text": chunk
            })

    embeddings = embedder.embed_texts(all_chunks)
    vector_store.add(embeddings, metadatas)
