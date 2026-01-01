from pathlib import Path


def load_codebase(path):
    code = []
    for file in Path(path).rglob("*.py"):
        code.append((file.name, file.read_text()))
    return code
