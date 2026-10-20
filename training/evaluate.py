"""Step 4: measure your model on the validation set (use these numbers in your report).
Run:  python -m training.evaluate
Reports: citation accuracy, correct refusals, and false refusals."""
import json, re
from pathlib import Path
from app import generator
from app.prompts import NOT_FOUND

def main(path="training/data/val.jsonl"):
    rows = [json.loads(l)["messages"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = neg = cite_ok = ref_ok = false_ref = 0
    for m in rows:
        gold_refuse = m[-1]["content"].startswith(NOT_FOUND[:30])
        gold_cite = re.findall(r"\[(\d+)\]", m[-1]["content"])
        pred = generator.chat(m[:-1])
        refused = pred.startswith(NOT_FOUND[:30])
        if gold_refuse:
            neg += 1; ref_ok += refused
        else:
            pos += 1; false_ref += refused
            cite_ok += bool(gold_cite) and gold_cite[-1] in re.findall(r"\[(\d+)\]", pred)
    print(f"Answerable questions: {pos}  | correct citation: {cite_ok/max(pos,1):.0%}  | wrongly refused: {false_ref/max(pos,1):.0%}")
    print(f"Unanswerable questions: {neg} | correctly refused: {ref_ok/max(neg,1):.0%}")
    print("Also read some answers yourself: automatic scores do not check the wording.")

if __name__ == "__main__":
    main()
