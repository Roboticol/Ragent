import fitz  # PyMuPDF

def load_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text
            })

    return pages
