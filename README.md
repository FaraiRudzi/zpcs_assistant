# ZPCS Intelligent Knowledge Assistant (prototype)

A Retrieval-Augmented Generation (RAG) system that answers officers' questions from authorised ZPCS documents and cites its sources.

| Objective | Where it is implemented |
|---|---|
| 1. Extract and clean text (OCR if needed) | `app/ingest.py` |
| 2. Embeddings stored for semantic search | `app/rag.py` (`index_document`, `retrieve`), tables in `app/db.py` |
| 3. Retrieve passages and generate cited answers | `app/rag.py` (`ask`) |
| 4. Web interface with role-based access | `app/main.py`, `app/auth.py`, `static/index.html` |

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000 and sign in as `admin` / `
` (set `ZPCS_ADMIN_PASSWORD` and `ZPCS_SECRET_KEY` first; see `.env.example`).
The first run downloads the embedding model, so it needs internet once. After that everything runs locally.

**Answer model:** the system uses *your own fine-tuned model* from `models/zpcs-generator`. Until you train it, the base model (`Qwen/Qwen2.5-1.5B-Instruct`) is downloaded and used so you can test the pipeline. See "Train your own model" below.

**Scanned documents:** `pip install pytesseract pdf2image`, install Tesseract and Poppler on the machine, and set `ZPCS_OCR=1`.

## Admin login and accounts
- First start: the admin is created from `ZPCS_ADMIN_USERNAME` and `ZPCS_ADMIN_PASSWORD`. These variables only matter the first time, when the database is empty.
- Change your own password any time: **Change password** in the top bar.
- Reset any user's password: Users tab, **Reset password**.
- Locked out, or want a different admin name: stop the server and run `python -m app.set_admin`.
- **Self-registration:** users click "Create an account" on the login page. Their request stays *pending* until an administrator approves it in the Users tab, choosing their role. Rejected requests are deleted.

## Speed
Answers stream word by word. To make them faster: keep the default small model on CPU-only machines (the app picks `Qwen2.5-0.5B` automatically without a GPU), use a GPU if you have one, tick **Quick answers** in the chat to show passages only, or lower `ZPCS_ANSWER_TOKENS` (default 200). Models load at start-up so the first question is not slow; set `ZPCS_WARMUP=0` to turn that off.

## Workflow in the web app
1. **Administrator** signs in and uploads documents (Documents tab).
2. In **Train model**, the administrator clicks the button. The system generates test questions from the documents (Inverse Cloze Task), fine-tunes the retrieval model, re-indexes everything, and shows precision, recall, F1, top-1 accuracy and MRR before and after, with charts. Gears turn while it trains.
3. **Officers** sign in and use the chat window. Answers show citations and the sources used.

The button trains the *retriever* (the part that finds the right passage) and needs no labelled data. Fine-tuning the *answer-writing* model needs a hand-checked question-and-answer set and a GPU, so it stays a command-line step (see "Train your own model" below).

## Try it
1. Sign in as admin, open Admin, upload `sample_docs/sample_rehabilitation_policy.txt`.
2. Create an officer user, sign in as that user, and ask: *How can an inmate get vocational skills?*
3. Upload a document with level "Administrators only" and confirm the officer cannot find it.

Run the tests: `pip install pytest && pytest`

## Access levels
Officer = 1, Senior officer = 2, Administrator = 3. A user can only retrieve chunks from documents whose level is at or below theirs. The filter runs **before** ranking, so restricted text never reaches the model.

## Known limits and next steps
- Vectors are held in memory and searched with NumPy. This suits hundreds of documents; move to pgvector or Qdrant for more.
- No approval workflow, document versioning, reranker or Shona/Ndebele tuning yet.
- Run behind HTTPS on the ZPCS network, use a strong secret key, and keep inmate case data out of this system.

## Train your own model
Training a language model from scratch needs billions of words, so this project **fine-tunes** a small open model (LoRA) on a ZPCS question-and-answer dataset that you build. The result is saved as your own model and the web app loads it automatically.

1. Upload your documents in the web app (Admin tab) so they are indexed.
2. `python -m training.prepare_data` exports every passage to `training/data/to_label.jsonl`.
3. Fill in 1 to 3 question/answer pairs per passage (a short, correct answer taken from the passage). Aim for 300+ pairs. You may draft pairs with an AI tool, but a person must check every one against the document.
4. `python -m training.build_dataset` creates `train.jsonl` and `val.jsonl`. It adds citations and also builds "not found" examples so the model learns to refuse instead of guessing.
5. `python -m training.train --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3` trains and saves `models/zpcs-generator`. Use a GPU (free Google Colab T4 works). For weak hardware use `Qwen/Qwen2.5-0.5B-Instruct`.
6. `python -m training.evaluate` reports citation accuracy and refusal accuracy on unseen questions. Put these numbers in your report.
7. Restart the server. Answers now come from your model (the API reports `trained-model` in the `mode` field).

Prompts are defined once in `app/prompts.py`, so training and the live system always use the same format.
