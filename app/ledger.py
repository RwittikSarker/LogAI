# app/ledger.py
import sqlite3
import json
import os
from app import config

class AnalysisLedger:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(config.LOGS_DIR, "analysis.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attack_type TEXT,
                    severity TEXT,
                    target_id TEXT,
                    evidence TEXT,
                    reason TEXT,
                    action TEXT,
                    chunk_index INTEGER
                )
            """)
            conn.commit()

    def set_context(self, context_text: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO state (key, value) VALUES ('rolling_context', ?)", (context_text,))
            conn.commit()

    def get_context(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM state WHERE key='rolling_context'")
            row = cursor.fetchone()
            return row[0] if row else "No prior context established."

    def add_finding(self, finding: dict, chunk_index: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO findings (attack_type, severity, target_id, evidence, reason, action, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.get("attack_type"),
                finding.get("severity", "Medium"),
                finding.get("target_id", "Unknown"),
                str(finding.get("evidence", [])),
                finding.get("reason"),
                finding.get("action"),
                chunk_index
            ))
            conn.commit()

    def get_all_findings(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM findings")
            return [dict(row) for row in cursor.fetchall()]

    def clear(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._init_db()