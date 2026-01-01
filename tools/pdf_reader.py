import pdfplumber
from pathlib import Path


def read_pdfs(path):
    documents = []

    for pdf_file in Path(path).rglob("*.pdf"):
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        documents.append((pdf_file.name, text))

    return documents
