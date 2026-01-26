"""
Domain Analysis Module.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DomainInfo:
    """Domain analysis results."""
    domain: str
    is_valid: bool = False
    subdomain: str = ""
    root_domain: str = ""
    tld: str = ""
    is_suspicious: bool = False
    has_suspicious_tld: bool = False
    threat_score: float = 0.0
    reputation_score: float = 100.0
    indicators: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    check_timestamp: str = ""


class DomainChecker:
    """Domain security analysis tool."""

    SUSPICIOUS_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'club', 'online', 'site'}
    FREE_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq'}

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def check_domain(self, domain: str, deep_scan: bool = False) -> DomainInfo:
        info = DomainInfo(domain=domain.lower().strip(), check_timestamp=datetime.now().isoformat())

        if not self._parse_domain(domain, info):
            return info

        self._check_tld(info)
        self._calculate_threat_score(info)
        return info

    def _parse_domain(self, domain: str, info: DomainInfo) -> bool:
        domain = domain.lower().strip()
        if '://' in domain:
            domain = domain.split('://')[1]
        domain = domain.split('/')[0].split('?')[0].split(':')[0]
        info.domain = domain

        domain_pattern = r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,}$'
        if not re.match(domain_pattern, domain):
            info.is_valid = False
            info.errors.append(f"Invalid domain format: {domain}")
            return False

        info.is_valid = True
        parts = domain.split('.')
        info.tld = parts[-1]
        if len(parts) >= 2:
            info.root_domain = '.'.join(parts[-2:])
            if len(parts) > 2:
                info.subdomain = '.'.join(parts[:-2])
        return True

    def _check_tld(self, info: DomainInfo) -> None:
        tld = info.tld.lower()
        if tld in self.SUSPICIOUS_TLDS:
            info.has_suspicious_tld = True
            info.is_suspicious = True
            info.threat_score += 15
            info.indicators.append(f"⚠️ Suspicious TLD: .{tld}")
        if tld in self.FREE_TLDS:
            info.threat_score += 10
            info.indicators.append(f"⚠️ Free TLD commonly abused: .{tld}")

    def _calculate_threat_score(self, info: DomainInfo) -> None:
        info.threat_score = min(info.threat_score, 100)
        info.reputation_score = max(0, 100 - info.threat_score)
        if info.threat_score >= 25:
            info.is_suspicious = True

    def get_threat_summary(self, info: DomainInfo) -> Dict:
        if not info.is_valid:
            return {"status": "INVALID", "threat_level": "UNKNOWN"}

        if info.threat_score >= 75:
            level, color = "CRITICAL", "#ff0000"
        elif info.threat_score >= 50:
            level, color = "HIGH", "#ff4444"
        elif info.threat_score >= 25:
            level, color = "MEDIUM", "#ffaa00"
        elif info.threat_score > 0:
            level, color = "LOW", "#ffcc00"
        else:
            level, color = "NONE", "#00ff88"

        return {
            "status": "ANALYZED", "threat_level": level, "threat_level_color": color,
            "threat_score": info.threat_score, "reputation_score": info.reputation_score
        }