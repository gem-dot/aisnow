import sqlite3
import hashlib
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / 'users.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        quota INTEGER DEFAULT 3,
        referral_code TEXT UNIQUE,
        referred_by TEXT,
        referral_count INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, referred_by=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        referral_code = str(uuid.uuid4())[:8]
        c.execute('INSERT INTO users VALUES (?, ?, 3, ?, ?, 0)',
                  (username, hash_password(password), referral_code, referred_by))
        if referred_by:
            c.execute('UPDATE users SET referral_count = referral_count + 1 WHERE referral_code = ?', (referred_by,))
        conn.commit()
        return True, referral_code
    except sqlite3.IntegrityError:
        return False, None
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return row and row[0] == hash_password(password)

def get_user_info(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT quota, referral_code, referral_count FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return {'quota': row[0], 'referral_code': row[1], 'referral_count': row[2]} if row else None

def update_quota(username, delta):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET quota = quota + ? WHERE username = ?', (delta, username))
    conn.commit()
    conn.close()

def get_referral_rate(referral_count):
    return 0.4 if referral_count >= 5 else 0.3

init_db()
