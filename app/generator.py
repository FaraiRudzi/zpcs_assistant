"""Answer generator: loads YOUR fine-tuned model (models/zpcs-generator), or a base model before training."""
import os
from pathlib import Path
from threading import Thread
from .config import GENERATOR_PATH, BASE_MODEL

_tok = _model = None

def is_trained() -> bool:
    return Path(GENERATOR_PATH).exists()

def _load():
    global _tok, _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        gpu = torch.cuda.is_available()
        base = BASE_MODEL or ("Qwen/Qwen2.5-1.5B-Instruct" if gpu else "Qwen/Qwen2.5-0.5B-Instruct")
        src = str(GENERATOR_PATH) if is_trained() else base
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _tok = AutoTokenizer.from_pretrained(src)
        if gpu:
            _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16).to("cuda")
        else:
            _model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
        _model.eval()
    return _tok, _model

def _inputs(messages):
    tok, model = _load()
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    return tok, model, enc.to(model.device)

def chat(messages, max_new_tokens: int = 200) -> str:
    import torch
    tok, model, enc = _inputs(messages)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def stream(messages, max_new_tokens: int = 200):
    """Yield the answer piece by piece so the user sees words appear immediately."""
    import torch
    from transformers import TextIteratorStreamer
    tok, model, enc = _inputs(messages)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    def work():
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    Thread(target=work, daemon=True).start()
    for piece in streamer:
        yield piece
