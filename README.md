# RAG Starter (Local, Minimal, Student‑Friendly)

A tiny, reliable pipeline to turn your course notes (IPYNB + PDF) into a local retrieval‑augmented chat app using **Ollama**, **ChromaDB**, and **Streamlit**.

## Quickstart

1) **Install prerequisites**
```bash
pip install -r requirements.txt
# Optional: if you hit torch install issues, try: pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

2) **Run a local model with Ollama**
```bash
# Install from https://ollama.com
ollama run mistral    # downloads and verifies the model
```

3) **Add documents**
- Put your `.ipynb` and `.pdf` files into `data/notes/` (and `data/extras/` if you want).
- Example: drop *Elements of Statistical Learning* PDF in `data/extras/`.

4) **Ingest (build the index)**
```bash
python ingest.py
```

5) **Serve the app**
```bash
streamlit run serve.py
```

Open the printed local URL in your browser.

---

## Features

- **ipynb + PDF ingestion** (markdown cells from notebooks; page‑preserving PDF chunks)
- **Page & source metadata** preserved → answers can cite **pages**
- **Hybrid retrieval**: semantic (embeddings) + lexical BM25 re‑rank
- **Page Finder**: ask “What pages cover cross‑validation?” and get page ranges with snippets
- **Completely local**: small model via Ollama; Chroma uses SQLite

## Notes

- Default embedding model: `all-MiniLM-L6-v2` (sentence-transformers). You can switch to Ollama embeddings later.
- If PDFs have a table of contents, we map pages to sections when possible.
- Keep chunks around ~1000 words with overlap ~200 to balance recall/precision; you can tune in `ingest.py`.

## Safety & Academic Use

Use this for **study**. If you reference answers in graded work, **cite pages/sections** shown in the app.
