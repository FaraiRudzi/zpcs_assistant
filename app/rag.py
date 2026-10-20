"""Objectives 2 and 3: embeddings + semantic search, then cited answer generation."""
import re
from pathlib import Path
import numpy as np
from . import db, generator
from .config import ANSWER_MAX_TOKENS, EMBED_MODEL, EMBED_SEQ_LEN, RETRIEVER_PATH, TOP_K, MIN_SCORE
from .prompts import NOT_FOUND, messages as build_messages

_model = None
_cache = None  # in-memory index of all approved chunks; rebuilt after any upload or delete

def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(str(RETRIEVER_PATH) if Path(RETRIEVER_PATH).exists() else EMBED_MODEL)
        _model.max_seq_length = EMBED_SEQ_LEN
    return _model

def embed(texts):
    v = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)

def reload_model():
    """Forget the loaded embedding model so the freshly trained one is used."""
    global _model
    _model = None
    invalidate()

def reindex_all():
    """Re-embed every stored passage with the current (newly trained) model."""
    rows = db.q("SELECT id, text FROM chunks")
    if not rows:
        return
    vecs = embed([r["text"] for r in rows])
    c = db.conn()
    c.executemany("UPDATE chunks SET embedding=? WHERE id=?", [(vecs[i].tobytes(), rows[i]["id"]) for i in range(len(rows))])
    c.commit()
    c.close()
    invalidate()

def invalidate():
    global _cache
    _cache = None

def index_document(doc_id: int, chunks: list):
    vecs = embed([c["text"] for c in chunks])
    c = db.conn()
    c.executemany(
        "INSERT INTO chunks(doc_id,page,section,text,embedding) VALUES(?,?,?,?,?)",
        [(doc_id, ch["page"], ch["section"], ch["text"], vecs[i].tobytes()) for i, ch in enumerate(chunks)])
    c.commit()
    c.close()
    invalidate()

def _load():
    global _cache
    if _cache is None:
        rows = db.q("""SELECT c.id, c.doc_id, c.page, c.section, c.text, c.embedding, d.title, d.access_level
                       FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.status='approved'""")
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else None
        levels = np.array([r["access_level"] for r in rows])
        _cache = (rows, matrix, levels)
    return _cache

def retrieve(query: str, user_level: int, k: int = TOP_K):
    """Access control is applied here, BEFORE ranking, so restricted text never reaches the model."""
    rows, matrix, levels = _load()
    if matrix is None:
        return []
    allowed = np.where(levels <= user_level)[0]
    if allowed.size == 0:
        return []
    sem = matrix[allowed] @ embed([query])[0]
    qtokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
    hits = []
    for pos in np.argsort(-sem)[: k * 3]:
        r = rows[int(allowed[pos])]
        if sem[pos] < MIN_SCORE:
            continue
        words = set(re.findall(r"[a-z0-9]{4,}", r["text"].lower()))
        bonus = 0.1 * (len(qtokens & words) / len(qtokens)) if qtokens else 0.0
        hits.append({"score": float(sem[pos]) + bonus, "title": r["title"], "page": r["page"],
                     "section": r["section"], "text": r["text"], "doc_id": r["doc_id"]})
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]

def _source(n, h):
    return {"n": n, "title": h["title"], "page": h["page"] or None, "section": h["section"],
            "excerpt": h["text"][:300] + ("..." if len(h["text"]) > 300 else "")}

def _extractive(hits, note="Here are the most relevant passages from the approved documents."):
    return {"answer": note, "sources": [_source(i, h) for i, h in enumerate(hits, 1)], "mode": "extractive"}

def _finalize(answer, hits, mode):
    if answer.startswith(NOT_FOUND[:30]):
        return {"answer": NOT_FOUND, "sources": [], "mode": "refused"}
    # Verify citations: keep only numbers that map to retrieved passages.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)})
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", answer)
    use = cited or list(range(1, len(hits) + 1))
    return {"answer": answer, "sources": [_source(n, hits[n - 1]) for n in use], "mode": mode if cited else "uncited"}

def _mode():
    return "trained-model" if generator.is_trained() else "base-model"

def ask(question: str, user_level: int, quick: bool = False) -> dict:
    hits = retrieve(question, user_level)
    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "mode": "refused"}
    if quick:
        return _extractive(hits)
    try:
        answer = generator.chat(build_messages(hits, question), ANSWER_MAX_TOKENS)
    except Exception:
        return _extractive(hits, "The answer model could not be loaded, so here are the most relevant passages instead.")
    return _finalize(answer, hits, _mode())

def ask_stream(question: str, user_level: int, quick: bool = False):
    """Yields {'t':'tok','v':...} events while the answer is written, then one {'t':'done',...} event."""
    hits = retrieve(question, user_level)
    if not hits:
        yield {"t": "done", "answer": NOT_FOUND, "sources": [], "mode": "refused"}
        return
    if quick:
        yield {"t": "done", **_extractive(hits)}
        return
    acc = ""
    try:
        for piece in generator.stream(build_messages(hits, question), ANSWER_MAX_TOKENS):
            acc += piece
            yield {"t": "tok", "v": piece}
    except Exception:
        if not acc:
            yield {"t": "done", **_extractive(hits, "The answer model could not be loaded, so here are the most relevant passages instead.")}
            return
    yield {"t": "done", **_finalize(acc.strip(), hits, _mode())}

def warm_up():
    """Load the models once at start-up so the first question is not slow."""
    try:
        _embedder()
        _load()
        generator._load()
    except Exception as e:
        print("[ZPCS] Warm-up skipped:", e)

def summarise(doc_id: int) -> str:
    rows = db.q("SELECT text FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,))
    text = " ".join(r["text"] for r in rows)
    words = text.split()[:5000]
    body = " ".join(words)
    try:
        return generator.chat([{"role": "system", "content": "Summarise the document: title, key areas, important provisions. Use only the text given."},
                               {"role": "user", "content": body}], max_new_tokens=350)
    except Exception:
        return "Summary unavailable (language model offline). Opening text:\n\n" + " ".join(words[:200]) + "..."
