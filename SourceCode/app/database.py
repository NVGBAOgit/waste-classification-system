import sqlite3
import bcrypt
import os
import shutil
from datetime import datetime

DB_NAME = "trash_history.db"

# AUTO BACKUP 
def auto_backup_db():
    if not os.path.exists(DB_NAME):
        return
    
    backup_dir = "backup_db"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Lấy ngày hiện tại để làm tên file backup (VD: trash_history_20260530.db)
    today_str = datetime.now().strftime("%Y%m%d")
    backup_file = os.path.join(backup_dir, f"trash_history_{today_str}.db")
    
    # Chỉ backup 1 lần mỗi ngày để tránh ghi đè liên tục gây nặng máy
    if not os.path.exists(backup_file):
        shutil.copy2(DB_NAME, backup_file)

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
            image_path TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    # Nâng cấp bảng history cũ: Thêm cột image_path
    try:
        c.execute("ALTER TABLE history ADD COLUMN image_path TEXT")
    except:
        pass
        
    conn.commit()
    conn.close()

def save_to_db(username, trash_type, confidence, image_path=None):
    if trash_type in ['cardboard', 'glass', 'metal', 'paper', 'plastic']: category = "Tái Chế"
    elif trash_type in ['organic']: category = "Hữu Cơ"
    elif trash_type in ['battery']: category = "Nguy Hại"
    else: category = "Vô Cơ"
    
    init_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO history (username, scan_time, trash_type, confidence, category, image_path) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, now, trash_type, confidence, category, image_path))
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, password):
    init_db()
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
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            predicted_class TEXT,
            true_class TEXT,
            is_correct BOOLEAN,
            image_path TEXT,
            timestamp TEXT
        )
    ''')
    # Nâng cấp bảng feedback cũ: Thêm cột image_path
    try:
        c.execute("ALTER TABLE feedback ADD COLUMN image_path TEXT")
    except:
        pass
        
    try:
        c.execute('''
            UPDATE feedback 
            SET timestamp = (
                SELECT scan_time FROM history WHERE history.image_path = feedback.image_path
            )
            WHERE image_path IS NOT NULL 
              AND EXISTS (SELECT 1 FROM history WHERE history.image_path = feedback.image_path)
        ''')
    except:
        pass
    
    conn.commit()
    conn.close()

def save_feedback(username, predicted_class, true_class, is_correct, image_path=None):
    init_feedback_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO feedback (username, predicted_class, true_class, is_correct, image_path, timestamp) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, predicted_class, true_class, is_correct, image_path, now))
    conn.commit()
    conn.close()

def delete_history_and_feedback(record_id, image_path=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 1. Xóa trong bảng lịch sử quét chính
    c.execute("DELETE FROM history WHERE id = ?", (record_id,))
    # 2. Xóa luôn phản hồi của bản ghi đó dựa trên đường dẫn ảnh (bảo toàn dữ liệu)
    if image_path and str(image_path) != 'nan':
        c.execute("DELETE FROM feedback WHERE image_path = ?", (str(image_path),))
    conn.commit()
    conn.close()

def delete_all_user_data(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Xóa toàn bộ bản ghi lịch sử quét
    c.execute("DELETE FROM history WHERE username = ?", (username,))
    # Xóa toàn bộ bản ghi phản hồi (feedback)
    c.execute("DELETE FROM feedback WHERE username = ?", (username,))
    conn.commit()
    conn.close()