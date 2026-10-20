"""Step 3: fine-tune a small open model (LoRA) on your dataset and save it as YOUR model.
Needs a GPU (a free Google Colab T4 is enough for the 0.5B or 1.5B model).
Run:  python -m training.train --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3"""
import argparse, json
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def load(path):
    return [json.loads(l)["messages"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

class Data(torch.utils.data.Dataset):
    def __init__(self, convs, tok, max_len):
        self.items = []
        for m in convs:
            prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(m[-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]   # learn only the answer, not the prompt
            self.items.append({"input_ids": ids, "labels": labels})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": (ids != pad_id).long() | (lab != -100).long()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="training/data/train.jsonl")
    ap.add_argument("--val", default="training/data/val.jsonl")
    ap.add_argument("--out", default="models/zpcs-generator")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=1536)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    train_ds, val_ds = Data(load(a.train), tok, a.max_len), Data(load(a.val), tok, a.max_len)
    args = TrainingArguments(output_dir="training/checkpoints", num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                             eval_strategy="epoch", save_strategy="no", logging_steps=10, warmup_ratio=0.05,
                             bf16=torch.cuda.is_available(), report_to="none", remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=lambda b: collate(b, tok.pad_token_id)).train()
    merged = model.merge_and_unload()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Saved your model to {a.out}. Restart the web app and it will use it automatically.")

if __name__ == "__main__":
    main()
