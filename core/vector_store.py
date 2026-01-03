import chromadb
import numpy as np

class VectorStore:
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def add(self, chunks: list[str], embeddings: np.ndarray, metadatas: list[dict]):
        self.collection.add(documents = chunks,
                            embeddings = embeddings,
                            metadatas = metadatas,
                            ids = [f"{metadatas["hash"]}_{i}" for i in range(len(chunks))]
                        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        return self.collection.query(query_embeddings = query_embedding.tolist(),
                                     n_results = top_k)
