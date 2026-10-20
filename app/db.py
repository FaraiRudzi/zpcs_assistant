import sqlite3
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, role TEXT NOT NULL, full_name TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, filename TEXT NOT NULL,
  access_level INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'approved',
  uploaded_by TEXT, uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER, section TEXT, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS training_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, finished_at TEXT DEFAULT CURRENT_TIMESTAMP,
  n_docs INTEGER, n_chunks INTEGER, n_train INTEGER, n_eval INTEGER,
  baseline TEXT, final TEXT, history TEXT, seconds REAL);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP,
  username TEXT, action TEXT, detail TEXT);
"""

def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c

def init():
    c = conn()
    c.executescript(SCHEMA)
    cols = {r[1] for r in c.execute("PRAGMA table_info(users)")}   # upgrade older databases
    if "status" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    if "service_number" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN service_number TEXT DEFAULT ''")
    c.commit()
    c.close()

def q(sql, params=(), one=False, write=False):
    c = conn()
    try:
        cur = c.execute(sql, params)
        if write:
            c.commit()
            return cur.lastrowid
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        c.close()

def log(username, action, detail=""):
    q("INSERT INTO logs(username, action, detail) VALUES(?,?,?)", (username, action, detail[:500]), write=True)
