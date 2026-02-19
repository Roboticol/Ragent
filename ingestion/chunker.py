from core.embeddings import EmbeddingModel
from tools.tools import split_sentences
import numpy as np

def chunk_text(text: str, chunk_size=800, overlap=100):
    # fixed-length chunker for simpler texts
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def semantic_chunk_text(model: EmbeddingModel, similarity_threshold: int, language: str, max_sentences: int, text: str):
    # semantic chunker for technical/academic documents
    sentences = split_sentences(text, language)
    embeddings = model.embed_texts(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i])

        if similarity < similarity_threshold or len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks