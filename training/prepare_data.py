"""Step 1: export every indexed passage to a labelling file.
Run after uploading your documents:  python -m training.prepare_data
Then open training/data/to_label.jsonl and fill in 1-3 question/answer pairs per passage."""
import json
from pathlib import Path
from app import db

OUT = Path("training/data/to_label.jsonl")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = db.q("SELECT c.id, c.page, c.section, c.text, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id ORDER BY c.id")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {"id": r["id"], "title": r["title"], "page": r["page"] or None, "section": r["section"],
                   "text": r["text"], "qa": [{"question": "", "answer": ""}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} passages to {OUT}. Fill in the question/answer pairs, then run build_dataset.")

if __name__ == "__main__":
    main()
