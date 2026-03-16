"""
RAG module for Elon Voice Bot
Fetches Elon Musk interview transcripts + biography excerpts,
builds a FAISS index, and retrieves relevant context at query time.
"""
import re
import pickle
import threading
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAG_DIR  = BASE_DIR / "assets" / "rag"

_embedder   = None
_index      = None
_chunks     = []      # list of (text, source_label)
_rag_lock   = threading.Lock()
_rag_ready  = False

# ── Sources ───────────────────────────────────────────────────────────────────

SOURCES = [
    # Lex Fridman #400 – wide-ranging conversation
    ("https://lexfridman.com/elon-musk-4-transcript/",          "lex_400"),
    # Lex Fridman #252 – SpaceX, Tesla, AI
    ("https://lexfridman.com/elon-musk-3-transcript/",          "lex_252"),
    # TED 2022 – future vision
    ("https://www.ted.com/talks/elon_musk_a_future_worth_getting_excited_about/transcript", "ted_2022"),
    # CBS News – Isaacson biography excerpt (childhood)
    ("https://www.cbsnews.com/news/book-excerpt-elon-musk-by-walter-isaacson/", "isaacson_excerpt"),
    # Comprehensive Isaacson book notes
    ("https://sameerbajaj.com/musk/",                           "isaacson_notes"),
]

CHUNK_SIZE    = 150   # words
CHUNK_OVERLAP = 30    # words
SIM_THRESHOLD = 0.35  # higher threshold = only genuinely relevant passages

# ── Text helpers ──────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\[\d+:\d+:\d+\])', '', text)   # timestamps
    return text.strip()

def _chunk(text: str):
    words = text.split()
    step  = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        if len(chunk.split()) >= 40:
            yield chunk

def _fetch_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, timeout=30, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    for tag in soup(['script','style','nav','header','footer','aside']):
        tag.decompose()
    return _clean(soup.get_text(separator=' ', strip=True))

def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    pages = PdfReader(path).pages
    return _clean(' '.join(p.extract_text() or '' for p in pages))

def _load_txt(path: str) -> str:
    return _clean(open(path, encoding='utf-8', errors='ignore').read())

# ── Index build / load ────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def _build_and_save(texts_labels):
    global _index, _chunks
    import faiss

    all_chunks = []
    for text, label in texts_labels:
        for chunk in _chunk(text):
            all_chunks.append((chunk, label))

    print(f"[RAG] Embedding {len(all_chunks)} chunks...")
    emb = _get_embedder().encode(
        [c[0] for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
    ).astype(np.float32)
    faiss.normalize_L2(emb)

    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    RAG_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(RAG_DIR / "rag.index"))
    with open(RAG_DIR / "rag_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    _index  = idx
    _chunks = all_chunks
    print(f"[RAG] Index saved: {idx.ntotal} vectors from {len(set(l for _,l in all_chunks))} sources.")

def _load_saved():
    global _index, _chunks
    import faiss
    _index  = faiss.read_index(str(RAG_DIR / "rag.index"))
    with open(RAG_DIR / "rag_chunks.pkl", "rb") as f:
        _chunks = pickle.load(f)
    print(f"[RAG] Index loaded: {_index.ntotal} vectors.")

# ── Public API ────────────────────────────────────────────────────────────────

def init(extra_files: list[str] = None):
    """Build or load the RAG index. Call once at startup."""
    global _rag_ready
    with _rag_lock:
        index_path  = RAG_DIR / "rag.index"
        chunks_path = RAG_DIR / "rag_chunks.pkl"

        if index_path.exists() and chunks_path.exists():
            _load_saved()
            _rag_ready = True
            return

        print("[RAG] Building knowledge base from web sources...")
        texts_labels = []

        for url, label in SOURCES:
            try:
                print(f"[RAG] Fetching {label}...")
                texts_labels.append((_fetch_url(url), label))
            except Exception as e:
                print(f"[RAG] Failed {url}: {e}")

        if extra_files:
            for path in extra_files:
                label = Path(path).stem
                try:
                    text = _load_pdf(path) if path.endswith('.pdf') else _load_txt(path)
                    texts_labels.append((text, label))
                    print(f"[RAG] Loaded {label} ({len(text.split())} words)")
                except Exception as e:
                    print(f"[RAG] Failed {path}: {e}")

        if texts_labels:
            _build_and_save(texts_labels)
            _rag_ready = True
        else:
            print("[RAG] No sources loaded, RAG disabled.")

def add_document(path: str):
    """Add a new document (PDF or TXT) and rebuild the index."""
    global _chunks
    label = Path(path).stem
    try:
        text = _load_pdf(path) if path.endswith('.pdf') else _load_txt(path)
    except Exception as e:
        print(f"[RAG] Failed to load {path}: {e}")
        return

    new_chunks = [(chunk, label) for chunk in _chunk(text)]
    existing   = list(_chunks)

    # Rebuild with combined corpus
    _build_and_save([(c, l) for c, l in existing + new_chunks])
    print(f"[RAG] Added '{label}': {len(new_chunks)} chunks.")

def query(question: str, k: int = 3) -> list[str]:
    """Return top-k relevant passages for the question."""
    if not _rag_ready or _index is None:
        return []
    import faiss
    emb = _get_embedder().encode([question]).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, idxs = _index.search(emb, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i >= 0 and score >= SIM_THRESHOLD:
            results.append(_chunks[i][0])
    return results
