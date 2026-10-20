"""Set or reset the administrator login from the command line:  python -m app.set_admin"""
import getpass
from . import db
from .auth import hash_password

def main():
    db.init()
    username = input("Admin username [admin]: ").strip() or "admin"
    pw = getpass.getpass("New password (8+ characters): ")
    if len(pw) < 8 or pw != getpass.getpass("Repeat password: "):
        print("Passwords must match and be at least 8 characters. Nothing changed.")
        return
    if db.q("SELECT 1 FROM users WHERE username=?", (username,), one=True):
        db.q("UPDATE users SET password_hash=?, role='administrator', status='active' WHERE username=?",
             (hash_password(pw), username), write=True)
        print(f"Password updated for administrator '{username}'.")
    else:
        db.q("INSERT INTO users(username,password_hash,role,full_name,status) VALUES(?,?,'administrator','Administrator','active')",
             (username, hash_password(pw)), write=True)
        print(f"Administrator '{username}' created.")

if __name__ == "__main__":
    main()
