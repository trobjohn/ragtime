import os, json
from typing import List, Dict, Any
import streamlit as st
import chromadb
from rank_bm25 import BM25Okapi
import requests
import pandas as pd

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL  = os.environ.get("LLM_MODEL", "gemma:2b")
EXTRAS_DIR = os.path.join("data", "extras")  # change to "books" later if you rename
COLLECTION = "course"

st.set_page_config(page_title="Course RAG (Local)", layout="wide")
st.title("Course RAG (Local)")

@st.cache_resource
def get_chroma():
    return chromadb.Client()

# @st.cache_resource
# def get_collection(_client):
#     return _client.get_or_create_collection(COLLECTION)

@st.cache_resource
def get_chroma():
    from chromadb.config import Settings
    return chromadb.Client(Settings(anonymized_telemetry=False))

@st.cache_resource
def get_collection():
    client = get_chroma()
    return client.get_or_create_collection(COLLECTION)


# @st.cache_resource
# def load_bm25_corpus():
#     path = os.path.join("index", "docs.jsonl")
#     if not os.path.exists(path):
#         return None, None, None
#     texts, metas, ids = [], [], []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             rec = json.loads(line)
#             texts.append(rec["text"])
#             metas.append(rec["meta"])
#             ids.append(rec["id"])
#     tokenized = [t.split() for t in texts]
#     bm25 = BM25Okapi(tokenized)
#     return bm25, texts, metas

@st.cache_resource
def load_bm25_corpus():
    path = os.path.join("index", "docs.jsonl")
    if not os.path.exists(path):
        return None, None, None

    # Read whole file; handle both JSONL and concatenated JSON objects.
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read().strip()

    texts, metas, ids = [], [], []
    dec = json.JSONDecoder()
    i, n = 0, len(blob)
    while i < n:
        # skip whitespace/newlines between objects
        while i < n and blob[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(blob, i)   # robust: works even without newlines
        i = j
        texts.append(obj["text"])
        metas.append(obj["meta"])
        ids.append(obj["id"])

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts, metas

client = get_chroma()
coll   = get_collection() # edited for client error
bm25, bm_texts, bm_metas = load_bm25_corpus()


# --- Sidebar source controls ---
book_files = []
if os.path.exists(EXTRAS_DIR):
    book_files = sorted([f for f in os.listdir(EXTRAS_DIR) if f.lower().endswith(".pdf")])
# after bm25, bm_texts, bm_metas = load_bm25_corpus()
sources_by_type = {"pdf": set(), "ipynb": set(), "py": set()}
for m in (bm_metas or []):
    t = m.get("type")
    s = m.get("source", "")
    if t in sources_by_type and s:
        sources_by_type[t].add(s)

books_indexed = sorted(sources_by_type["pdf"])

with st.sidebar:
    st.markdown("### Reference pool")
    pool = st.radio("Choose books:", ["All books", "Custom"], horizontal=True)
    selected_books = books_indexed if pool == "All books" else st.multiselect(
        "Books (PDFs in data/extras/)", options=books_indexed, default=books_indexed
    )
    include_notes = st.checkbox("Include notebooks (.ipynb)", value=True)
    include_py    = st.checkbox("Include .py comments", value=False)

def current_allowed_sources():
    allowed = set(selected_books)
    if include_notes:
        allowed |= sources_by_type["ipynb"]
    if include_py:
        allowed |= sources_by_type["py"]
    # If absolutely nothing selected, don't filter at all:
    return allowed or None

####################################################




def bm25_query(q: str, topk: int = 20):
    if bm25 is None:
        return []
    scores = bm25.get_scores(q.split())
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
    return [(float(scores[i]), bm_texts[i], bm_metas[i]) for i in idx]

def semantic_query(q: str, topk: int = 10):
    res = coll.query(query_texts=[q], n_results=topk)
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for d, m, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist)  # invert cosine distance
        out.append((sim, d, m))
    return out
def hybrid_query(q: str, k_sem: int = 10, k_bm: int = 20, w_sem: float = 0.6,
                 allowed_sources: set | None = None):
    sem = semantic_query(q, k_sem)
    bm  = bm25_query(q, k_bm)

    def norm(items):
        if not items:
            return items
        vals = [s for s,_,_ in items]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return [((s - lo)/rng, txt, meta) for s, txt, meta in items]

    sem_n = norm(sem)
    bm_n  = norm(bm)

    from collections import defaultdict
    agg = defaultdict(lambda: {"sem":0.0,"bm":0.0,"txt":None,"meta":None})
    def key(meta, txt):
        return (meta.get("source"), meta.get("page"), txt[:80])

    for s, txt, m in sem_n:
        k = key(m, txt)
        agg[k]["sem"] = max(agg[k]["sem"], s)
        agg[k]["txt"] = txt
        agg[k]["meta"] = m
    for s, txt, m in bm_n:
        k = key(m, txt)
        agg[k]["bm"] = max(agg[k]["bm"], s)
        if agg[k]["txt"] is None:
            agg[k]["txt"] = txt
            agg[k]["meta"] = m

    scored = []
    for _, v in agg.items():
        score = w_sem*v["sem"] + (1-w_sem)*v["bm"]
        scored.append((score, v["txt"], v["meta"]))

    scored.sort(key=lambda t: -t[0])

    if allowed_sources:  # filter AFTER scored exists
        scored = [(s, txt, m) for (s, txt, m) in scored
                  if m.get("source","") in allowed_sources]

    return scored

def page_finder(q: str, topn: int = 5, w_sem: float = 0.6,
                allowed_sources: set | None = None):
    hits = hybrid_query(q, k_sem=15, k_bm=50, w_sem=w_sem,
                        allowed_sources=allowed_sources)
    from collections import defaultdict
    bucket = defaultdict(list)
    for s, txt, m in hits:
        page = m.get("page", None)
        src  = m.get("source", "")
        sec  = m.get("section", "")
        bucket[(src, page, sec)].append((s, txt))
    rows = []
    for (src, page, sec), chunks in bucket.items():
        best = max(chunks, key=lambda t: t[0])
        snippet = best[1][:240].replace("\n", " ") + ("…" if len(best[1])>240 else "")
        rows.append({"source": src, "page": page, "section": sec,
                     "score": best[0], "snippet": snippet})
    rows.sort(key=lambda r: -r["score"])
    return rows[:topn]

with st.sidebar:
    st.markdown("### Tools")
    mode = st.radio("Mode", ["Regex Mentions", "Corpus Search", "Page Finder", "Ask"], horizontal=True)
    w_sem = st.slider("Hybrid weight (semantic)", 0.0, 1.0, 0.6, 0.05)
    topk  = st.slider("Top‑k context chunks", 3, 25, 5, 1)

    q = st.text_input("Query")

# if st.button("Go") and q:
#     allowed = current_allowed_sources()    
#     if mode == "Page Finder":
#         rows = page_finder(q, topn=8, w_sem=w_sem, allowed_sources=allowed)
#         st.subheader("Likely relevant pages")
#         st.dataframe(pd.DataFrame(rows))
#     else:
#         hits = hybrid_query(q, k_sem=20, k_bm=60, w_sem=w_sem, allowed_sources=allowed)[:topk]
#         context = []
#         cites = []
#         for i, (score, txt, meta) in enumerate(hits, start=1):
#             src = meta.get("source", "")
#             page = meta.get("page", "?")
#             cites.append(f"[{i}] {src} p{page}")
#             context.append(f"[{i}] ({src} p{page}) {txt}")

    if st.button("Go") and q:
        allowed = current_allowed_sources()

        if mode == "Regex Mentions":
            # needs: find_mentions(...) helper + sidebar toggles whole_word, case_sensitive, use_regex
            rows = find_mentions(
                q,
                allowed_sources=allowed,
                whole_word=whole_word,
                case_sensitive=case_sensitive,
                use_regex=use_regex,
                topn=100,
            )
            st.subheader(f"Occurrences of “{q}”")
            st.dataframe(pd.DataFrame(rows))

        elif mode == "Corpus Search":
            # needs: corpus_search(...) helper + sidebar slider topn_search
            rows = corpus_search(q, allowed_sources=allowed, topn=topn_search)
            st.subheader("Top matches (BM25, corpus-only)")
            st.dataframe(pd.DataFrame(rows))

        elif mode == "Page Finder":
            rows = page_finder(q, topn=8, w_sem=w_sem, allowed_sources=allowed)
            st.subheader("Likely relevant pages")
            st.dataframe(pd.DataFrame(rows))

        else:  # "Ask" (RAG + LLM)
            hits = hybrid_query(q, k_sem=20, k_bm=60, w_sem=w_sem, allowed_sources=allowed)[:topk]
            if not hits:
                st.warning("No context found. Try widening your source selection or switching modes.")
            else:
                context, cites = [], []
                for i, (score, txt, meta) in enumerate(hits, start=1):
                    src  = meta.get("source", "")
                    page = meta.get("page", "?")
                    cites.append(f"[{i}] {src} p{page}")
                    context.append(f"[{i}] ({src} p{page}) {txt}")

                prompt = (
                    "Use the context to answer. FIRST cite snippet IDs like [1],[2]. "
                    "If unsure, say 'Not in corpus.'\n\n"
                    "CONTEXT:\n" + "\n\n".join(context) + f"\n\nQUESTION: {q}\nANSWER:"
                )

                try:
                    r = requests.post(
                        OLLAMA_URL,
                        json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
                        timeout=120
                    )
                    ans = r.json().get("response", "(no response)")
                except Exception as e:
                    ans = f"(LLM error: {e})"

                st.markdown("### Answer")
                st.write(ans)
                with st.expander("Context & Citations"):
                    st.markdown("\n".join(cites))
                    for i, (score, txt, meta) in enumerate(hits, start=1):
                        st.markdown(f"**[{i}]** {meta.get('source')} p{meta.get('page','?')} — score {score:.3f}")
                        st.write(txt[:1200] + ("…" if len(txt) > 1200 else ""))

        ## Prompting and context:
                
        # prompt = (
        #     "Use the context to answer. FIRST cite snippet IDs like [1],[2] before any claims. "
        #     "If unsure, say what is missing. Keep it concise.\n\n"
        #     "CONTEXT:\n" + "\n\n".join(context) + f"\n\nQUESTION: {q}\nANSWER:"
        #     )
        

        system = (
            "You are a tutor answering STRICTLY from provided context. "
            "Rules: (1) Start with citations like [1][2] before claims. "
            "(2) Use ONLY the snippets; do not invent facts; do not add external links or resources. "
            "(3) If the context is insufficient, say: 'Not in corpus' and suggest what to search next. "
            "(4) Keep it concise and student-friendly."
        )

        prompt = (
            "Use the context to answer. FIRST cite snippet IDs like [1],[2]. "
            "If unsure, say 'Not in corpus.'\n\n"
            "CONTEXT:\n" + "\n\n".join(context) + f"\n\nQUESTION: {q}\nANSWER:"
        )

        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": os.environ.get("LLM_MODEL","mistral"),
                    "system": system,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,     # less 'creative'
                        "top_p": 0.9,
                        "num_ctx": 4096
                    }
                },
                timeout=120
            )
            ans = r.json().get("response", "(no response)")






            # r = requests.post(OLLAMA_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=120)
            # ans = r.json().get("response", "(no response)")
        except Exception as e:
            ans = f"(LLM error: {e})"
        st.markdown("### Answer")
        st.write(ans)
        with st.expander("Context & Citations"):
            st.markdown("\n".join(cites))
            for i, (score, txt, meta) in enumerate(hits, start=1):
                st.markdown(f"**[{i}]** {meta.get('source')} p{meta.get('page','?')} — score {score:.3f}")
                st.write(txt[:1200] + ("…" if len(txt)>1200 else ""))
