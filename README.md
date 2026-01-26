# Ragent

**Ragent** is a production-oriented, research-focused Retrieval-Augmented Generation (RAG) system designed to ingest, index, and query large document collections (such as PDFs) using semantic search and generative AI.

---

## 🚀 Key Features

* 📄 **PDF Ingestion Pipeline**

  * Robust PDF loading
  * Semantic sentence-based chunking
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

## 📥 Ingestion Workflow

1. Place PDF files inside the `data/` directory
2. Run the ingestion script:

```bash
python -m ingestion.ingest
```

3. The pipeline will:

   * Load PDFs
   * Semantically chunk text
   * Generate embeddings
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

---

## 🔐 Authentication

* API access is protected using an **API key**
* Key is loaded from environment variables (`.env` file)

---

## 📊 Logging & Observability

Each request logs:

* Query text
* Number of retrieved chunks
* Latency (ms)

This lays the foundation for:

* Monitoring
* Analytics dashboards
* Future observability integrations

---

## 🧪 Development Notes

* Absolute paths are used for ChromaDB to avoid multi-process issues
* ChromaDB persistence is handled via `PersistentClient`
* Architecture is modular and extensible (reranking, hybrid search, tools)

---

## 🛣️ Roadmap

* [ ] Reranking (cross-encoders)
* [ ] Hybrid search (BM25 + embeddings)
* [ ] Streaming responses
* [ ] Document-level access control
* [ ] Evaluation metrics (Recall@K, MRR)

---

## 🎯 Why This Project Matters

Ragent demonstrates:

* Real-world RAG engineering challenges
* Production debugging (persistence, paths, clients)
* Clean separation of concerns
* Interview-ready system design

---

## 📄 License

MIT License (or specify your own)

---

**Author:** You
**Project Name:** Ragent
