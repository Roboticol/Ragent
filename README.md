# Ragent

**Ragent** is a research-focused Retrieval-Augmented Generation (RAG) system designed to ingest, index, and query large document collections of documents in english and other indic languages using semantic chunking, semantic search and generative AI.

---

## 🚀 Key Features

* 📄 **PDF Ingestion Pipeline**

  * Robust PDF loading
  * Semantic sentence-based chunking
  * Can be used to ingest documents in english and indic languages
  * Hash-based document identification

* 🧠 **Semantic Search with Embeddings**

  * Sentence-transformer–based embeddings
  * Vector storage using **ChromaDB**
  * Metadata-aware retrieval

* 🤖 **Agentic Reasoning Layer**

  * Designed for research-style question answering

* 🌐 **FastAPI-based API**

  * Query endpoint with API-key authentication
  * Request logging and latency tracking
  * Health check endpoint

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **FastAPI** – API layer
* **ChromaDB** – Vector database (persistent storage)
* **Sentence-Transformers** – Embeddings
* **spaCy** – Sentence segmentation
* **PyMuPDF (fitz)** – PDF parsing

---
## ✅ Prerequisites
Make sure [ollama](https://ollama.com/) is installed on your system. Then,
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull mashriram/sarvam-1
```
Create a `.env` file in your root folder, containing:
```
API_KEY="SECRET"
INGEST_DIR = "./your/path"
```
`API_KEY` is the API KEY of the RAG.
`INGEST_DIR` is where the files you want to ingest are stored. Currently only PDFs and .txts are supported.

## 📥 Ingestion Workflow

1. Place PDF files inside the `data/` directory
2. Run the ingestion script:

```bash
python -m ingestion.ingest
```

3. The pipeline will:

   * Load PDFs
   * Semantically chunk text
   * Generate embeddings based on the language of the text
   * Store chunks + metadata in ChromaDB

Embeddings are persisted to disk using `PersistentClient`.

---

## 🔎 Querying the RAG API

### Start the API

```bash
uvicorn main:app --reload
```

### Example POST Request

```http
POST /query
Content-Type: application/json
X-API-Key: YOUR_API_KEY

{
  "query": "Explain transformer attention",
  "top_k": 5
}
```

### Example Response

```json
{
  "answer": "...",
  "sources": ["paper1.pdf", "paper2.pdf"]
}
```