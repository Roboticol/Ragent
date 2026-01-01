from pathlib import Path


def read_text_files(path):
    texts = []
    for file in Path(path).rglob("*.txt"):
        texts.append((file.name, file.read_text()))
    return texts
