"""The 'Train model' button: fine-tunes the retrieval model on questions generated from the uploaded documents,
and measures precision, recall and F1 before and after.

Training pairs: (section heading as a question -> section text) and (a sentence taken out of a passage -> the rest of it).
Testing: questions built from headings of sections held out of training."""
import json, random, re, threading, time
import numpy as np
from . import db, rag
from .config import EMBED_MODEL, EMBED_SEQ_LEN, MAX_PAIRS, MIN_SCORE, RETRIEVER_PATH, TOP_K, TRAIN_EPOCHS

_lock = threading.Lock()
state = {"state": "idle", "stage": "", "progress": 0, "epoch": 0, "epochs": TRAIN_EPOCHS, "error": None, "run": None}

def _set(**kw):
    state.update(kw)

def _split(text):
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if 8 <= len(p.split()) <= 70 and len(re.findall(r"\b\d{1,3}[A-Z]?\.", p)) < 2]  # skip contents-style lines

def _body(c):
    h, t = c["section"] or "", c["text"]
    return (t[len(h) + 2:] if h and t.startswith(h + ". ") else t), h

def _compose(h, body):
    return (h + ". " if h else "") + re.sub(r"\s+", " ", body).strip()

def _heading_question(h, rng):
    """Turn a section heading into a question an officer might ask."""
    x = re.sub(r"^\d+[A-Z]{0,2}\.?\s+", "", h).strip().rstrip(".")
    x = x[:1].lower() + x[1:]
    return rng.choice(["What does the law say about {x}?", "What are the rules on {x}?", "Explain {x}.", "{X}?"]).format(x=x, X=x[:1].upper() + x[1:])

def build_sets(chunks, rng):
    """Returns (train_pairs, eval_queries, corpus).
    Test questions come from section headings of sections the model NEVER trains on (20%), so the score shows
    whether it finds unseen sections. eval_queries = [(question, set_of_relevant_corpus_indexes)]."""
    from collections import defaultdict
    bodies = [_body(c) for c in chunks]
    corpus = [re.sub(r"\s+", " ", b).strip() for b, _ in bodies]
    by_sec = defaultdict(list)
    for i, (_, h) in enumerate(bodies):
        if h:
            by_sec[h].append(i)
    secs = sorted(by_sec)
    rng.shuffle(secs)
    held = set(secs[:max(1, len(secs) // 5)])
    queries = [(_heading_question(h, rng), set(by_sec[h])) for h in sorted(held)]
    train = []
    for i, (body, h) in enumerate(bodies):
        if h in held:
            continue
        if h:
            for _ in range(2):
                train.append((_heading_question(h, rng), corpus[i]))
        for s in rng.sample(_split(body), min(2, len(_split(body)))):
            train.append((s, re.sub(r"\s+", " ", body.replace(s, "")).strip()))
    if len(train) > MAX_PAIRS:
        train = rng.sample(train, MAX_PAIRS)
    return train, queries, corpus

def _evaluate(model, corpus, queries):
    c = model.encode(corpus, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    q = model.encode([s for s, _ in queries], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    sims = q @ c.T
    rel_hit = retrieved_total = rel_total = top1 = 0
    mrr = 0.0
    for i, (_, gold) in enumerate(queries):
        row = sims[i]
        order = np.argsort(-row)[:TOP_K]
        got = {int(j) for j in order if row[j] >= MIN_SCORE}   # what the live system would return
        rel_hit += len(got & gold)
        retrieved_total += len(got)
        rel_total += len(gold)
        top1 += int(order[0]) in gold
        mrr += 1.0 / (int((row > max(row[g] for g in gold)).sum()) + 1)
    n = len(queries)
    p, r = rel_hit / max(retrieved_total, 1), rel_hit / max(rel_total, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "top1": round(top1 / n, 4), "mrr": round(mrr / n, 4)}

def _run():
    t0 = time.time()
    try:
        _set(state="running", stage="Preparing training data from your documents", progress=5, epoch=0, error=None)
        rows = db.q("SELECT c.id, c.section, c.text FROM chunks c JOIN documents d ON d.id=c.doc_id "
                    "WHERE d.status='approved' ORDER BY c.id")
        chunks = [dict(r) for r in rows]
        n_docs = db.q("SELECT COUNT(*) n FROM documents WHERE status='approved'", one=True)["n"]
        train, queries, corpus = build_sets(chunks, random.Random(42))
        if len(queries) < 10 or len(train) < 50:
            raise ValueError(f"Not enough structured text to train on ({len(queries)} test sections, {len(train)} training pairs; need at least 10 and 50). Upload more documents with numbered section headings.")
        import importlib.util
        missing = [m for m in ("datasets", "accelerate", "torch") if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError("Missing packages: " + ", ".join(missing) + ". Run: pip install -U datasets accelerate torch  (in the same environment as the server), then restart the server.")
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        model = SentenceTransformer(EMBED_MODEL)
        model.max_seq_length = EMBED_SEQ_LEN
        _set(stage="Measuring the model before training", progress=10)
        baseline = _evaluate(model, corpus, queries)
        history = [dict(epoch=0, **baseline)]
        loader = DataLoader([InputExample(texts=[q, p]) for q, p in train], shuffle=True, batch_size=32)
        loss = losses.MultipleNegativesRankingLoss(model)
        E = TRAIN_EPOCHS
        for ep in range(1, E + 1):
            _set(stage=f"Training: epoch {ep} of {E}", epoch=ep, progress=15 + int(70 * (ep - 1) / E))
            model.fit(train_objectives=[(loader, loss)], epochs=1, show_progress_bar=False,
                      warmup_steps=int(0.1 * len(loader)) if ep == 1 else 0)
            history.append(dict(epoch=ep, **_evaluate(model, corpus, queries)))
            _set(progress=15 + int(70 * ep / E))
        _set(stage="Saving the model and re-indexing documents", progress=90)
        RETRIEVER_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(RETRIEVER_PATH))
        rag.reload_model()
        rag.reindex_all()
        final = history[-1]
        secs = round(time.time() - t0, 1)
        db.q("INSERT INTO training_runs(n_docs,n_chunks,n_train,n_eval,baseline,final,history,seconds) VALUES(?,?,?,?,?,?,?,?)",
             (n_docs, len(chunks), len(train), len(queries), json.dumps(baseline), json.dumps(
                 {k: v for k, v in final.items() if k != "epoch"}), json.dumps(history), secs), write=True)
        load_latest()
        _set(state="done", stage="Training complete", progress=100)
    except Exception as e:
        _set(state="error", stage="Training failed", error=str(e), progress=0)
    finally:
        if _lock.locked():
            _lock.release()

def load_latest():
    r = db.q("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1", one=True)
    state["run"] = None if not r else {
        "finished_at": r["finished_at"], "n_docs": r["n_docs"], "n_chunks": r["n_chunks"], "n_train": r["n_train"],
        "n_eval": r["n_eval"], "baseline": json.loads(r["baseline"]), "final": json.loads(r["final"]),
        "history": json.loads(r["history"]), "seconds": r["seconds"], "k": TOP_K, "threshold": MIN_SCORE}

def start() -> bool:
    if not _lock.acquire(blocking=False):
        return False
    threading.Thread(target=_run, daemon=True).start()
    return True

def snapshot():
    return dict(state)
