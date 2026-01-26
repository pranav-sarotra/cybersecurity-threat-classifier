"""
File Hash Analysis Module.
"""

import re
import hashlib
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FileHashInfo:
    """File hash analysis results."""
    hash_value: str
    hash_type: str = ""
    is_valid: bool = False
    is_malware: bool = False
    malware_names: List[str] = field(default_factory=list)
    threat_score: float = 0.0
    threat_level: str = "UNKNOWN"
    check_timestamp: str = ""
    errors: List[str] = field(default_factory=list)


class FileAnalyzer:
    """File hash analysis for malware detection."""

    KNOWN_MALICIOUS_HASHES = {
        '44d88612fea8a8f36de82e1278abb02f': 'EICAR-Test-File',
    }

    def __init__(self, virustotal_api_key: str = "", timeout: int = 30):
        self.vt_api_key = virustotal_api_key
        self.timeout = timeout

    def analyze_hash(self, hash_value: str) -> FileHashInfo:
        info = FileHashInfo(hash_value=hash_value.lower().strip(), check_timestamp=datetime.now().isoformat())

        if not self._validate_hash(hash_value, info):
            return info

        self._check_local_database(info)
        self._calculate_threat_score(info)
        return info

    def _validate_hash(self, hash_value: str, info: FileHashInfo) -> bool:
        hash_value = hash_value.lower().strip()
        if not re.match(r'^[a-f0-9]+$', hash_value):
            info.is_valid = False
            info.errors.append("Invalid hash: contains non-hex characters")
            return False

        hash_types = {32: 'MD5', 40: 'SHA-1', 64: 'SHA-256', 128: 'SHA-512'}
        hash_type = hash_types.get(len(hash_value))
        if not hash_type:
            info.is_valid = False
            info.errors.append(f"Unrecognized hash length: {len(hash_value)}")
            return False

        info.is_valid = True
        info.hash_type = hash_type
        return True

    def _check_local_database(self, info: FileHashInfo) -> None:
        if info.hash_value in self.KNOWN_MALICIOUS_HASHES:
            info.is_malware = True
            info.malware_names.append(self.KNOWN_MALICIOUS_HASHES[info.hash_value])
            info.threat_score = 100

    def _calculate_threat_score(self, info: FileHashInfo) -> None:
        if info.is_malware:
            info.threat_score = max(info.threat_score, 75)
        info.threat_score = min(info.threat_score, 100)

        if info.threat_score >= 75:
            info.threat_level = "CRITICAL"
        elif info.threat_score >= 50:
            info.threat_level = "HIGH"
        elif info.threat_score >= 25:
            info.threat_level = "MEDIUM"
        elif info.threat_score > 0:
            info.threat_level = "LOW"
        else:
            info.threat_level = "NONE"

    @staticmethod
    def compute_hash(data: bytes, hash_type: str = 'sha256') -> str:
        hash_funcs = {'md5': hashlib.md5, 'sha1': hashlib.sha1, 'sha256': hashlib.sha256}
        func = hash_funcs.get(hash_type.lower())
        if not func:
            raise ValueError(f"Unsupported hash type: {hash_type}")
        return func(data).hexdigest()

    def get_threat_summary(self, info: FileHashInfo) -> Dict:
        if not info.is_valid:
            return {"status": "INVALID", "threat_level": "UNKNOWN"}

        colors = {"CRITICAL": "#ff0000", "HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#ffcc00", "NONE": "#00ff88"}
        return {
            "status": "ANALYZED", "hash_type": info.hash_type, "threat_level": info.threat_level,
            "threat_level_color": colors.get(info.threat_level, "#888888"),
            "threat_score": info.threat_score, "is_malware": info.is_malware, "malware_names": info.malware_names
        }