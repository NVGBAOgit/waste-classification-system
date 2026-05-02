import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "trash_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            scan_time TEXT,
            trash_type TEXT,
            confidence REAL,
            category TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(username, trash_type, confidence):
    if trash_type in ['cardboard', 'glass', 'metal', 'paper', 'plastic']: category = "Tái Chế"
    elif trash_type in ['organic']: category = "Hữu Cơ"
    elif trash_type in ['battery']: category = "Nguy Hại"
    else: category = "Vô Cơ"
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO history (username, scan_time, trash_type, confidence, category) VALUES (?, ?, ?, ?, ?)',
              (username, now, trash_type, confidence, category))
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and check_password(password, result[0]):
        return True
    return False

def init_feedback_db():
    """Hàm tạo bảng phản hồi nếu chưa có"""
    conn = sqlite3.connect('trash_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            predicted_class TEXT,
            true_class TEXT,
            is_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_feedback(username, predicted_class, true_class, is_correct):
    """Hàm lưu đánh giá của người dùng vào cơ sở dữ liệu"""
    init_feedback_db() # Luôn kiểm tra và tạo bảng trước khi lưu
    conn = sqlite3.connect('trash_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO feedback (username, predicted_class, true_class, is_correct) VALUES (?, ?, ?, ?)',
              (username, predicted_class, true_class, is_correct))
    conn.commit()
    conn.close()