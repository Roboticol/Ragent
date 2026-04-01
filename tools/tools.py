from pathlib import Path
import spacy
import hashlib
from core.retriever import Retriever
from indicnlp.tokenize import sentence_tokenize
import json

nlp = None

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




def split_sentences(text, language):
    with open("./languages.json", "r") as f:
        data = json.load(f)
        for i in data["languages"]:
            if i["name"] == language:
                if i["embed"] == "indicnlp":
                    return sentence_tokenize.sentence_split(
                        " ".join(text), lang=i["code"]
                    )
                
                nlp = spacy.load(i["embed"])
                doc = nlp(" ".join(text))

                sentences = []

                for sent in doc.sents:
                    if sent.text.strip():
                        # print("---" + sent.text.strip())
                        sentences.append(sent.text.strip())
                
                return sentences