from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # Converts a list of texts into embeddings

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
