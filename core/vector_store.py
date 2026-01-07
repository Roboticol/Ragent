import numpy as np
import chromadb

class VectorStore:
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def add(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict]
    ):
        ids = [
            f"{meta['hash']}_{i}"
            for i, meta in enumerate(metadatas)
        ]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
