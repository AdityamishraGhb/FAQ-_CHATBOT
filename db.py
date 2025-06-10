import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json

DATABASE_PATH = "logs.db"

def init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create conversation logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT DEFAULT NULL
        )
    """)
    
    # Create training data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def log_conversation(question: str, answer: str, session_id: Optional[str] = None) -> int:
    """Log a conversation to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversation_logs (question, answer, session_id)
        VALUES (?, ?, ?)
    """, (question, answer, session_id))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return log_id

def get_all_logs() -> List[Dict]:
    """Retrieve all conversation logs from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, question, answer, timestamp, session_id
        FROM conversation_logs
        ORDER BY timestamp DESC
    """)
    
    logs = []
    for row in cursor.fetchall():
        logs.append({
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "timestamp": row[3],
            "session_id": row[4]
        })
    
    conn.close()
    return logs

def add_training_data(question: str, answer: str) -> int:
    """Add training data to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO training_data (question, answer)
        VALUES (?, ?)
    """, (question, answer))
    
    training_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return training_id

def get_training_data() -> List[Dict]:
    """Retrieve all training data from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, question, answer, created_at
        FROM training_data
        ORDER BY created_at DESC
    """)
    
    training_data = []
    for row in cursor.fetchall():
        training_data.append({
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "created_at": row[3]
        })
    
    conn.close()
    return training_data

def get_recent_conversations(limit: int = 10) -> List[Dict]:
    """Get recent conversations for context."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT question, answer, timestamp
        FROM conversation_logs
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            "question": row[0],
            "answer": row[1],
            "timestamp": row[2]
        })
    
    conn.close()
    return conversations


