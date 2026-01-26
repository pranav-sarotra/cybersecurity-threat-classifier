"""
Database module for scan history.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class ScanDatabase:
    def __init__(self, db_path: str = "data/scan_history.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    threat_score REAL NOT NULL,
                    threat_level TEXT NOT NULL,
                    indicators TEXT,
                    ml_scores TEXT,
                    rule_scores TEXT,
                    full_result TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ip_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    country TEXT,
                    threat_score REAL,
                    is_vpn INTEGER,
                    is_proxy INTEGER,
                    full_result TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS url_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT,
                    is_suspicious INTEGER,
                    threat_score REAL,
                    full_result TEXT
                )
            """)
            conn.commit()

    def save_scan(self, result: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO scans (timestamp, input_text, classification, confidence, threat_score, threat_level, indicators, ml_scores, rule_scores, full_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                result.get('input_text', '')[:1000],
                result.get('classification', ''),
                result.get('confidence', 0),
                result.get('threat_score', 0),
                result.get('threat_level', ''),
                json.dumps(result.get('indicators', [])),
                json.dumps(result.get('ml_scores', {})),
                json.dumps(result.get('rule_scores', {})),
                json.dumps(result)
            ))
            conn.commit()
            return cursor.lastrowid

    def save_ip_check(self, result: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO ip_checks (timestamp, ip_address, country, threat_score, is_vpn, is_proxy, full_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                result.get('ip', ''),
                result.get('country', ''),
                result.get('threat_score', 0),
                1 if result.get('is_vpn') else 0,
                1 if result.get('is_proxy') else 0,
                json.dumps(result)
            ))
            conn.commit()
            return cursor.lastrowid

    def save_url_check(self, result: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO url_checks (timestamp, url, domain, is_suspicious, threat_score, full_result)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                result.get('url', ''),
                result.get('domain', ''),
                1 if result.get('is_suspicious') else 0,
                result.get('threat_score', 0),
                json.dumps(result)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_scans(self, limit: int = 50) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM scans ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            cursor = conn.execute("SELECT COUNT(*) FROM scans")
            stats['total_scans'] = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM ip_checks")
            stats['total_ip_checks'] = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM url_checks")
            stats['total_url_checks'] = cursor.fetchone()[0]
            return stats

    def clear_history(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM scans")
            conn.execute("DELETE FROM ip_checks")
            conn.execute("DELETE FROM url_checks")
            conn.commit()