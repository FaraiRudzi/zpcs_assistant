import hashlib, hmac, os, datetime as dt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from . import db
from .config import SECRET_KEY, TOKEN_HOURS, ROLE_LEVEL

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + h.hex()

def verify_password(password: str, stored: str) -> bool:
    salt_hex, h_hex = stored.split("$")
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(h.hex(), h_hex)

def make_token(username: str, role: str) -> str:
    exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": username, "role": role, "exp": exp}, SECRET_KEY, algorithm="HS256")

bearer = HTTPBearer(auto_error=False)

def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    if not creds:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    row = db.q("SELECT * FROM users WHERE username=?", (payload["sub"],), one=True)
    if not row:
        raise HTTPException(401, "Unknown user")
    if row["status"] != "active":
        raise HTTPException(401, "Account not active")
    user = dict(row)
    user["level"] = ROLE_LEVEL[user["role"]]
    return user

def admin_only(user: dict = Depends(current_user)) -> dict:
    if user["level"] < 3:
        raise HTTPException(403, "Administrator only")
    return user
