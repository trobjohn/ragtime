import os, json, glob
from typing import List, Tuple
import fitz  # pymupdf
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

DATA_DIR = "data"
INDEX_DIR = "index"
COLLECTION_NAME = "course"

def chunk_text(text: str, n_words: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    if not words:
        return []
    i, out = 0, []
    step = max(1, n_words - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+n_words]))
        i += step
    return out

def read_ipynb_markdown(path: str) -> str:
    try:
        nb = json.load(open(path, "r"))
        parts = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type", "") == "markdown":
                parts.append("".join(cell.get("source", [])))
        return "\\n\\n".join(parts)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return ""

def read_pdf_pages_with_sections(path: str) -> List[Tuple[int, str, str]]:
    doc = fitz.open(path)
    toc = doc.get_toc(simple=True) or []
    page_to_section = {}
    for _, title, page in toc:
        page_to_section.setdefault(page, title)
    expanded = [""] * (len(doc) + 1)
    current = ""
    for p in range(1, len(doc)+1):
        if p in page_to_section:
            current = page_to_section[p]
        expanded[p] = current
    out = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        out.append((i, expanded[i], text))
    return out

def read_python_comments(path: str) -> str:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        content = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                content.append(stripped)
            elif in_docstring or stripped.startswith("#"):
                content.append(stripped)
        return "\\n".join(content)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return ""

def ensure_dirs():
    os.makedirs(os.path.join(DATA_DIR, "notes"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "extras"), exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def main():
    ensure_dirs()
    client = chromadb.Client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    coll = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    all_docs, all_metas, all_ids = [], [], []

    for p in tqdm(glob.glob(os.path.join(DATA_DIR, "notes", "*.ipynb"))):
        text = read_ipynb_markdown(p)
        chunks = chunk_text(text, n_words=1000, overlap=200)
        for j, ch in enumerate(chunks):
            all_docs.append(ch)
            all_metas.append({"source": os.path.basename(p), "type": "ipynb"})
            all_ids.append(f"{os.path.basename(p)}::cell{j}")

    for p in tqdm(glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)):
        pages = read_pdf_pages_with_sections(p)
        for page_num, section, page_text in pages:
            chunks = chunk_text(page_text, n_words=900, overlap=180)
            for j, ch in enumerate(chunks):
                all_docs.append(ch)
                all_metas.append({
                    "source": os.path.basename(p),
                    "type": "pdf",
                    "page": page_num,
                    "section": section
                })
                all_ids.append(f"{os.path.basename(p)}::p{page_num}::{j}")

    for p in tqdm(glob.glob(os.path.join(DATA_DIR, "**", "*.py"), recursive=True)):
        text = read_python_comments(p)
        if text.strip():
            chunks = chunk_text(text, n_words=800, overlap=150)
            for j, ch in enumerate(chunks):
                all_docs.append(ch)
                all_metas.append({
                    "source": os.path.basename(p),
                    "type": "py",
                    "path": p
                })
                all_ids.append(f"{os.path.basename(p)}::pyblock{j}")

    if not all_docs:
        print("No documents found. Add .ipynb, .py, or .pdf files to data/ and rerun.")
        return

    print(f"Upserting {len(all_docs)} chunks into Chroma...")
    B = 500
    for i in range(0, len(all_docs), B):
        coll.add(
            documents=all_docs[i:i+B],
            metadatas=all_metas[i:i+B],
            ids=all_ids[i:i+B]
        )

    jpath = os.path.join(INDEX_DIR, "docs.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        for _id, doc, meta in zip(all_ids, all_docs, all_metas):
            rec = {"id": _id, "text": doc, "meta": meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
    print(f"Wrote lexical corpus mirror: {jpath}")
    print("Done.")

if __name__ == "__main__":
    main()
