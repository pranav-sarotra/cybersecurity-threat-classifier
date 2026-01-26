"""
URL Analysis Module.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, unquote

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False


@dataclass
class URLAnalysisResult:
    """URL analysis results container."""
    url: str
    is_valid: bool = False
    scheme: str = ""
    netloc: str = ""
    domain: str = ""
    subdomain: str = ""
    suffix: str = ""
    path: str = ""
    query: str = ""
    port: Optional[int] = None
    is_https: bool = False
    is_ip_url: bool = False
    is_shortened: bool = False
    is_suspicious: bool = False
    is_phishing: bool = False
    is_malware: bool = False
    has_suspicious_tld: bool = False
    has_suspicious_keywords: bool = False
    threat_score: float = 0.0
    phishing_score: float = 0.0
    malware_score: float = 0.0
    indicators: List[str] = field(default_factory=list)
    extracted_params: Dict = field(default_factory=dict)


class URLAnalyzer:
    """URL security analysis tool."""

    SHORTENERS = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'buff.ly', 'ow.ly', 'tiny.cc', 'rb.gy', 'cutt.ly', 'shorturl.at'}
    SUSPICIOUS_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link', 'loan', 'club', 'online', 'site', 'website'}
    PHISHING_TARGETS = {'paypal', 'amazon', 'ebay', 'apple', 'microsoft', 'google', 'facebook', 'netflix', 'chase', 'bankofamerica', 'wellsfargo'}
    SUSPICIOUS_KEYWORDS = ['login', 'signin', 'verify', 'confirm', 'account', 'update', 'secure', 'password', 'billing', 'suspend']
    MALWARE_KEYWORDS = ['download', 'install', 'setup', '.exe', '.dll', '.bat', '.vbs', '.ps1']

    def __init__(self, follow_redirects: bool = False, timeout: int = 10):
        self.follow_redirects = follow_redirects
        self.timeout = timeout

    def analyze(self, url: str) -> URLAnalysisResult:
        result = URLAnalysisResult(url=url)

        if not self._parse_url(url, result):
            return result

        self._check_scheme(result)
        self._check_ip_url(result)
        self._check_shortened(result)
        self._check_suspicious_tld(result)
        self._check_suspicious_keywords(result)
        self._check_encoding(result)
        self._check_subdomains(result)
        self._check_brand_impersonation(result)
        self._calculate_scores(result)

        return result

    def _parse_url(self, url: str, result: URLAnalysisResult) -> bool:
        try:
            if not url.startswith(('http://', 'https://', 'ftp://')):
                url = 'http://' + url

            parsed = urlparse(url)
            if not parsed.netloc:
                result.is_valid = False
                result.indicators.append("❌ Invalid URL format - no domain")
                return False

            result.is_valid = True
            result.scheme = parsed.scheme
            result.netloc = parsed.netloc
            result.path = parsed.path
            result.query = parsed.query

            if TLDEXTRACT_AVAILABLE:
                extracted = tldextract.extract(url)
                result.domain = extracted.domain
                result.subdomain = extracted.subdomain
                result.suffix = extracted.suffix
            else:
                parts = parsed.netloc.split(':')[0].split('.')
                if len(parts) >= 2:
                    result.suffix = parts[-1]
                    result.domain = parts[-2]
                    if len(parts) > 2:
                        result.subdomain = '.'.join(parts[:-2])

            return True
        except Exception as e:
            result.is_valid = False
            result.indicators.append(f"❌ URL parsing error: {str(e)}")
            return False

    def _check_scheme(self, result: URLAnalysisResult) -> None:
        result.is_https = result.scheme == 'https'
        if not result.is_https:
            result.phishing_score += 5
            result.indicators.append("⚠️ Non-HTTPS URL")

    def _check_ip_url(self, result: URLAnalysisResult) -> None:
        ip_pattern = r'^(?:\d{1,3}\.){3}\d{1,3}$'
        netloc = result.netloc.split(':')[0]
        if re.match(ip_pattern, netloc):
            result.is_ip_url = True
            result.is_suspicious = True
            result.phishing_score += 25
            result.indicators.append("🚨 URL uses IP address instead of domain")

    def _check_shortened(self, result: URLAnalysisResult) -> None:
        netloc_lower = result.netloc.lower()
        for shortener in self.SHORTENERS:
            if shortener in netloc_lower:
                result.is_shortened = True
                result.is_suspicious = True
                result.phishing_score += 15
                result.indicators.append(f"⚠️ URL shortener detected: {shortener}")
                break

    def _check_suspicious_tld(self, result: URLAnalysisResult) -> None:
        if result.suffix and result.suffix.lower() in self.SUSPICIOUS_TLDS:
            result.has_suspicious_tld = True
            result.is_suspicious = True
            result.phishing_score += 15
            result.indicators.append(f"⚠️ Suspicious TLD: .{result.suffix}")

    def _check_suspicious_keywords(self, result: URLAnalysisResult) -> None:
        url_lower = result.url.lower()
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in url_lower:
                result.has_suspicious_keywords = True
                result.phishing_score += 5
        for keyword in self.MALWARE_KEYWORDS:
            if keyword in url_lower:
                result.malware_score += 8

    def _check_encoding(self, result: URLAnalysisResult) -> None:
        encoded_chars = re.findall(r'%[0-9A-Fa-f]{2}', result.url)
        if len(encoded_chars) > 5:
            result.is_suspicious = True
            result.phishing_score += 10
            result.indicators.append(f"⚠️ Excessive URL encoding: {len(encoded_chars)} chars")

    def _check_subdomains(self, result: URLAnalysisResult) -> None:
        if result.subdomain:
            parts = result.subdomain.split('.')
            if len(parts) > 3:
                result.is_suspicious = True
                result.phishing_score += 10
                result.indicators.append(f"⚠️ Excessive subdomains: {len(parts)} levels")

    def _check_brand_impersonation(self, result: URLAnalysisResult) -> None:
        url_lower = result.url.lower()
        domain_lower = result.domain.lower() if result.domain else ""
        for brand in self.PHISHING_TARGETS:
            if brand in url_lower and brand != domain_lower:
                if result.subdomain and brand in result.subdomain.lower():
                    result.is_phishing = True
                    result.phishing_score += 25
                    result.indicators.append(f"🚨 Brand '{brand}' in subdomain - likely impersonation")
                    break

    def _calculate_scores(self, result: URLAnalysisResult) -> None:
        result.phishing_score = min(result.phishing_score, 100)
        result.malware_score = min(result.malware_score, 100)
        result.threat_score = max(result.phishing_score, result.malware_score)
        if result.phishing_score >= 50:
            result.is_phishing = True
        if result.malware_score >= 50:
            result.is_malware = True

    def extract_urls_from_text(self, text: str) -> List[str]:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return list(set(re.findall(url_pattern, text, re.IGNORECASE)))