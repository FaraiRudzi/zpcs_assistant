import json, re, threading, time, uuid
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import db, rag, trainer
from .auth import admin_only, current_user, hash_password, make_token, verify_password
from .config import (ADMIN_PASSWORD, ADMIN_USERNAME, ALLOWED_EXT, MAX_UPLOAD_MB, ROLE_LEVEL, UPLOAD_DIR, BASE_DIR, WARMUP)
from .ingest import chunk_pages, extract_pages

USERNAME_RE = re.compile(r"[A-Za-z0-9_.@-]{3,64}")
USERNAME_MSG = "Username must be 3-64 characters: letters, numbers, dots, dashes, underscores or @ (no spaces)"

app = FastAPI(title="ZPCS Intelligent Knowledge Assistant")

class Login(BaseModel):
    username: str
    password: str

class Question(BaseModel):
    question: str
    quick: bool = False

class Register(BaseModel):
    full_name: str
    service_number: str = ""
    username: str
    password: str

class Profile(BaseModel):
    full_name: str
    service_number: str = ""
    username: str
    current_password: str = ""
    new_password: str = ""

class PwChange(BaseModel):
    current: str
    new: str

class PwReset(BaseModel):
    password: str

class Approve(BaseModel):
    role: str = "officer"

class NewUser(BaseModel):
    username: str
    password: str
    role: str
    full_name: str = ""

@app.on_event("startup")
def startup():
    db.init()
    trainer.load_latest()
    if not db.q("SELECT 1 FROM users LIMIT 1", one=True):
        db.q("INSERT INTO users(username,password_hash,role,full_name) VALUES(?,?,?,?)",
             (ADMIN_USERNAME, hash_password(ADMIN_PASSWORD), "administrator", "System Administrator"), write=True)
        print(f"[ZPCS] Created admin user '{ADMIN_USERNAME}'. Change its password before real use.")
    if WARMUP:
        threading.Thread(target=rag.warm_up, daemon=True).start()

@app.post("/api/login")
def login(body: Login):
    u = db.q("SELECT * FROM users WHERE username=?", (body.username,), one=True)
    if not u or not verify_password(body.password, u["password_hash"]):
        db.log(body.username, "login_failed")
        raise HTTPException(401, "Wrong username or password")
    if u["status"] != "active":
        raise HTTPException(403, "Your account is waiting for administrator approval.")
    db.log(u["username"], "login")
    return {"token": make_token(u["username"], u["role"]), "role": u["role"], "username": u["username"]}

_reg_log = {}

@app.post("/api/register")
def register(body: Register, request: Request):
    ip = request.client.host if request.client else "?"
    now = time.time()
    recent = [t for t in _reg_log.get(ip, []) if now - t < 3600]
    if len(recent) >= 5:
        raise HTTPException(429, "Too many registration attempts. Try again later.")
    username = body.username.strip()
    if not body.full_name.strip():
        raise HTTPException(400, "Full name is required")
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(400, USERNAME_MSG)
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.q("SELECT 1 FROM users WHERE username=?", (username,), one=True):
        raise HTTPException(409, "That username is already taken")
    db.q("INSERT INTO users(username,password_hash,role,full_name,service_number,status) VALUES(?,?,?,?,?,'pending')",
         (username, hash_password(body.password), "officer", body.full_name.strip()[:100], body.service_number.strip()[:30]), write=True)
    _reg_log[ip] = recent + [now]
    db.log(username, "register_request", body.full_name.strip()[:100])
    return {"status": "pending"}

@app.post("/api/me/password")
def change_password(body: PwChange, user=Depends(current_user)):
    if not verify_password(body.current, user["password_hash"]):
        raise HTTPException(400, "Current password is wrong")
    if len(body.new) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")
    db.q("UPDATE users SET password_hash=? WHERE id=?", (hash_password(body.new), user["id"]), write=True)
    db.log(user["username"], "change_password")
    return {"ok": True}

@app.get("/api/me")
def me(user=Depends(current_user)):
    return {"username": user["username"], "role": user["role"], "full_name": user["full_name"] or "",
            "service_number": user.get("service_number") or ""}

@app.put("/api/me")
def update_me(body: Profile, user=Depends(current_user)):
    name, username = body.full_name.strip(), body.username.strip()
    if not name:
        raise HTTPException(400, "Full name is required")
    if username != user["username"]:    # existing usernames are never re-validated, only a changed one
        if not USERNAME_RE.fullmatch(username):
            raise HTTPException(400, USERNAME_MSG)
    if username != user["username"] and db.q("SELECT 1 FROM users WHERE username=?", (username,), one=True):
        raise HTTPException(409, "That username is already taken")
    sn = body.service_number.strip()[:30]
    new_hash = None
    if body.new_password or body.current_password:          # password is optional, but checked before anything is saved
        if not verify_password(body.current_password, user["password_hash"]):
            raise HTTPException(400, "Current password is wrong")
        if len(body.new_password) < 8:
            raise HTTPException(400, "New password must be at least 8 characters")
        new_hash = hash_password(body.new_password)
    db.q("UPDATE users SET full_name=?, service_number=?, username=? WHERE id=?", (name[:100], sn, username, user["id"]), write=True)
    if new_hash:
        db.q("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user["id"]), write=True)
    db.log(username, "update_profile", ("username changed from " + user["username"] + "; " if username != user["username"] else "") + ("password changed" if new_hash else ""))
    return {"username": username, "role": user["role"], "full_name": name[:100], "service_number": sn,
            "token": make_token(username, user["role"])}

@app.post("/api/ask")
def ask(body: Question, user=Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    result = rag.ask(question[:1000], user["level"], body.quick)
    db.log(user["username"], "ask", f"{question[:200]} -> {result['mode']}, {len(result['sources'])} sources")
    return result

@app.post("/api/ask/stream")
def ask_stream(body: Question, user=Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")

    def events():
        mode, n = "error", 0
        for ev in rag.ask_stream(question[:1000], user["level"], body.quick):
            if ev["t"] == "done":
                mode, n = ev["mode"], len(ev["sources"])
            yield json.dumps(ev) + "\n"
        db.log(user["username"], "ask", f"{question[:200]} -> {mode}, {n} sources")
    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"})

@app.get("/api/documents")
def list_documents(user=Depends(current_user)):
    rows = db.q("SELECT id,title,access_level,uploaded_at,(SELECT COUNT(*) FROM chunks WHERE doc_id=documents.id) AS chunks "
                "FROM documents WHERE access_level<=? ORDER BY id DESC", (user["level"],))
    return [dict(r) for r in rows]

@app.get("/api/documents/{doc_id}/summary")
def summary(doc_id: int, user=Depends(current_user)):
    d = db.q("SELECT * FROM documents WHERE id=? AND access_level<=?", (doc_id, user["level"]), one=True)
    if not d:
        raise HTTPException(404, "Document not found")
    db.log(user["username"], "summarise", d["title"])
    return {"title": d["title"], "summary": rag.summarise(doc_id)}

@app.post("/api/documents")
def upload(file: UploadFile = File(...), title: str = Form(""), access_level: int = Form(1), user=Depends(admin_only)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Allowed types: {', '.join(sorted(ALLOWED_EXT))}")
    if access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level must be 1, 2 or 3")
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    chunks = chunk_pages(extract_pages(path))
    if not chunks:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "No text could be extracted. For scanned files enable OCR (ZPCS_OCR=1).")
    doc_id = db.q("INSERT INTO documents(title,filename,access_level,uploaded_by) VALUES(?,?,?,?)",
                  (title.strip() or Path(file.filename).stem, path.name, access_level, user["username"]), write=True)
    rag.index_document(doc_id, chunks)
    db.log(user["username"], "upload", f"{file.filename} ({len(chunks)} chunks, level {access_level})")
    return {"id": doc_id, "chunks": len(chunks)}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, user=Depends(admin_only)):
    d = db.q("SELECT * FROM documents WHERE id=?", (doc_id,), one=True)
    if not d:
        raise HTTPException(404, "Not found")
    db.q("DELETE FROM documents WHERE id=?", (doc_id,), write=True)
    (UPLOAD_DIR / d["filename"]).unlink(missing_ok=True)
    rag.invalidate()
    db.log(user["username"], "delete", d["title"])
    return {"deleted": doc_id}

@app.post("/api/train")
def train(user=Depends(admin_only)):
    if not trainer.start():
        raise HTTPException(409, "Training is already running")
    db.log(user["username"], "train_start")
    return {"started": True}

@app.get("/api/train/status")
def train_status(user=Depends(admin_only)):
    return trainer.snapshot()

@app.get("/api/users")
def users(user=Depends(admin_only)):
    return [dict(r) for r in db.q("SELECT id,username,role,full_name,service_number,status FROM users ORDER BY status DESC, id")]

@app.post("/api/users/{uid}/approve")
def approve_user(uid: int, body: Approve, user=Depends(admin_only)):
    if body.role not in ROLE_LEVEL:
        raise HTTPException(400, f"role must be one of {list(ROLE_LEVEL)}")
    row = db.q("SELECT username FROM users WHERE id=? AND status='pending'", (uid,), one=True)
    if not row:
        raise HTTPException(404, "No pending request with that id")
    db.q("UPDATE users SET status='active', role=? WHERE id=?", (body.role, uid), write=True)
    db.log(user["username"], "approve_user", f"{row['username']} as {body.role}")
    return {"approved": row["username"]}

@app.post("/api/users/{uid}/reject")
def reject_user(uid: int, user=Depends(admin_only)):
    row = db.q("SELECT username FROM users WHERE id=? AND status='pending'", (uid,), one=True)
    if not row:
        raise HTTPException(404, "No pending request with that id")
    db.q("DELETE FROM users WHERE id=?", (uid,), write=True)
    db.log(user["username"], "reject_user", row["username"])
    return {"rejected": row["username"]}

@app.post("/api/users/{uid}/password")
def reset_password(uid: int, body: PwReset, user=Depends(admin_only)):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    row = db.q("SELECT username FROM users WHERE id=?", (uid,), one=True)
    if not row:
        raise HTTPException(404, "User not found")
    db.q("UPDATE users SET password_hash=? WHERE id=?", (hash_password(body.password), uid), write=True)
    db.log(user["username"], "reset_password", row["username"])
    return {"ok": True}

@app.post("/api/users")
def create_user(body: NewUser, user=Depends(admin_only)):
    if body.role not in ROLE_LEVEL:
        raise HTTPException(400, f"role must be one of {list(ROLE_LEVEL)}")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if db.q("SELECT 1 FROM users WHERE username=?", (body.username,), one=True):
        raise HTTPException(409, "Username already exists")
    db.q("INSERT INTO users(username,password_hash,role,full_name) VALUES(?,?,?,?)",
         (body.username, hash_password(body.password), body.role, body.full_name), write=True)
    db.log(user["username"], "create_user", f"{body.username} ({body.role})")
    return {"created": body.username}

@app.get("/api/logs")
def logs(user=Depends(admin_only)):
    return [dict(r) for r in db.q("SELECT ts,username,action,detail FROM logs ORDER BY id DESC LIMIT 200")]

app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
