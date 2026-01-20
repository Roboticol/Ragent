import fitz  # PyMuPDF

def load_pdf(path: str) -> list[str]:
    doc = fitz.open(path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(text)

    return pages
