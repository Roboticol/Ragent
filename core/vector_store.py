import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]):
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results
