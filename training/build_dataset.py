"""Step 2: turn the labelled passages into training data.
Positive examples: question + the gold passage mixed with distractor passages -> answer with a [n] citation.
Negative examples: question + unrelated passages -> the refusal sentence (teaches the model to say 'not found').
Run:  python -m training.build_dataset"""
import json, random
from pathlib import Path
from app.prompts import NOT_FOUND, messages

SRC = Path("training/data/to_label.jsonl")
OUT = Path("training/data")
random.seed(42)

def passage(rec):
    return {"title": rec["title"], "page": rec["page"], "text": rec["text"], "id": rec["id"]}

def make_example(question, shown, answer):
    msgs = messages(shown, question) + [{"role": "assistant", "content": answer}]
    return {"messages": msgs}

def build(records, pool, n_distractors=2, neg_ratio=0.3):
    out = []
    for rec in records:
        for qa in rec["qa"]:
            q, a = qa["question"].strip(), qa["answer"].strip()
            if not q or not a:
                continue
            others = [passage(p) for p in pool if p["id"] != rec["id"]]
            shown = random.sample(others, min(n_distractors, len(others))) + [passage(rec)]
            random.shuffle(shown)
            n = [i for i, p in enumerate(shown, 1) if p["id"] == rec["id"]][0]
            out.append(make_example(q, shown, f"{a} [{n}]"))
            if random.random() < neg_ratio and len(others) >= n_distractors + 1:   # same question, gold passage removed
                out.append(make_example(q, random.sample(others, n_distractors + 1), NOT_FOUND))
    return out

def main():
    records = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    labelled = [r for r in records if any(x["question"].strip() and x["answer"].strip() for x in r["qa"])]
    random.shuffle(labelled)
    cut = max(1, int(len(labelled) * 0.9))
    train, val = labelled[:cut], labelled[cut:] or labelled[:1]
    train_ex, val_ex = build(train, records), build(val, records)
    for name, data in (("train", train_ex), ("val", val_ex)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(labelled)} labelled passages -> {len(train_ex)} train and {len(val_ex)} validation examples.")
    if len(train_ex) < 300:
        print("Warning: fewer than ~300 training examples. Label more passages for a reliable model.")

if __name__ == "__main__":
    main()
