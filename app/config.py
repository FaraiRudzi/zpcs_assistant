import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ZPCS_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "zpcs.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("ZPCS_SECRET_KEY", "dev-only-change-me")
TOKEN_HOURS = 8
ADMIN_USERNAME = os.getenv("ZPCS_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ZPCS_ADMIN_PASSWORD", "ChangeMe123!")

EMBED_MODEL = os.getenv("ZPCS_EMBED_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
CHUNK_WORDS = 120
CHUNK_OVERLAP = 20
TOP_K = 5
MIN_SCORE = float(os.getenv("ZPCS_MIN_SCORE", "0.30"))  # below this the system refuses to answer

# Your own fine-tuned model is loaded from GENERATOR_PATH; before training, the base model is used.
GENERATOR_PATH = Path(os.getenv("ZPCS_GENERATOR_PATH", BASE_DIR / "models" / "zpcs-generator"))
# Empty = automatic: 1.5B model on a GPU, the faster 0.5B model on CPU-only machines.
BASE_MODEL = os.getenv("ZPCS_BASE_MODEL", "")
OCR_ENABLED = os.getenv("ZPCS_OCR", "0") == "1"
MAX_UPLOAD_MB = 50
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt"}

ROLE_LEVEL = {"officer": 1, "senior_officer": 2, "administrator": 3}

# Retriever fine-tuning (the "Train model" button)
RETRIEVER_PATH = Path(os.getenv("ZPCS_RETRIEVER_PATH", BASE_DIR / "models" / "zpcs-retriever"))
TRAIN_EPOCHS = int(os.getenv("ZPCS_TRAIN_EPOCHS", "2"))
EMBED_SEQ_LEN = 256
MAX_PAIRS = int(os.getenv("ZPCS_MAX_PAIRS", "800"))
ANSWER_MAX_TOKENS = int(os.getenv("ZPCS_ANSWER_TOKENS", "200"))
WARMUP = os.getenv("ZPCS_WARMUP", "1") == "1"
