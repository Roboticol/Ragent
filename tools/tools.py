from pathlib import Path
import spacy
import hashlib
from core.retriever import Retriever

nlp = spacy.load("en_core_web_sm")

def load_codebase(path):
    code = []
    for file in Path(path).rglob("*.py"):
        code.append((file.name, file.read_text()))
    return code

def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_text_files(path):
    texts = []
    for file in Path(path).rglob("*.txt"):
        texts.append((file.name, file.read_text()))
    return texts


def parse_response(response):
    return "\n".join(response)


def retrieve_context(retriever: Retriever, query: str) -> str:
    results = retriever.retrieve(query, top_k=5)

    context_blocks = []
    for r in results:
        context_blocks.append(
            f"[Source: {r['metadata']['id']}]\n{r['metadata']['text']}"
        )

    return "\n\n".join(context_blocks)




def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]