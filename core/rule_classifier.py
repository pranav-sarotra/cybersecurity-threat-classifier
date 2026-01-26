"""
Rule-Based Threat Classifier.
"""

import re
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class RuleAnalysisResult:
    """Rule-based analysis results."""
    phishing_score: float = 0.0
    malware_score: float = 0.0
    safe_score: float = 0.0
    indicators: List[str] = field(default_factory=list)
    rule_matches: int = 0
    matched_patterns: List[Dict] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)


class RuleBasedClassifier:
    """Rule-based threat classifier using pattern matching."""

    def __init__(self):
        self._init_phishing_patterns()
        self._init_malware_patterns()
        self._init_safe_patterns()
        self._init_regex_patterns()
        self._init_contextual_rules()

    def _init_phishing_patterns(self) -> None:
        self.phishing_patterns = {
            'critical': {
                'patterns': [
                    'send your bank details', 'wire transfer immediately', 'social security number',
                    'ssn required', 'gift card payment', 'send money via', 'western union',
                    'claim your inheritance', 'nigerian prince', 'lottery winner',
                    'you have been selected', 'congratulations winner',
                ],
                'score': 30
            },
            'high': {
                'patterns': [
                    'verify your account', 'confirm your identity', 'account suspended',
                    'click here immediately', 'password expired', 'unusual activity',
                    'security alert', 'account locked', 'confirm within 24 hours',
                    'account will be terminated', 'unauthorized access', 'billing information',
                    'update payment', 'dear valued customer', 'verify your identity',
                ],
                'score': 20
            },
            'medium': {
                'patterns': [
                    'urgent', 'act now', 'limited time', 'expire', 'suspended',
                    'verify now', 'confirm now', 'dear customer', 'dear user',
                    'click here', 'reset password', 'final warning', 'last chance',
                ],
                'score': 10
            },
            'low': {
                'patterns': [
                    'account', 'verify', 'confirm', 'update', 'secure', 'bank',
                    'paypal', 'amazon', 'microsoft', 'apple', 'netflix', 'google',
                ],
                'score': 3
            }
        }

    def _init_malware_patterns(self) -> None:
        self.malware_patterns = {
            'critical': {
                'patterns': [
                    'powershell -encoded', 'powershell -enc', 'cmd /c', 'cmd.exe /c',
                    'wget http', 'curl -o', 'chmod +x', 'nc -e', 'reverse shell',
                    'keylogger', 'ransomware', 'mimikatz', 'reg add hklm',
                    'schtasks /create', 'disable defender', 'privilege escalation',
                    'invoke-mimikatz', 'invoke-expression', 'downloadstring',
                ],
                'score': 35
            },
            'high': {
                'patterns': [
                    '.exe download', 'enable macros', 'powershell', 'base64 decode',
                    'frombase64string', 'eval(', 'wscript', 'mshta', 'rundll32',
                    'certutil', 'payload', 'shellcode', 'exploit', 'bitsadmin',
                ],
                'score': 20
            },
            'medium': {
                'patterns': [
                    '.exe', '.dll', '.bat', '.vbs', '.ps1', '.scr',
                    'download now', 'macro', 'encoded', 'obfuscated',
                    'persistence', 'backdoor', 'dropper',
                ],
                'score': 8
            },
            'low': {
                'patterns': ['admin', 'root', 'system32', 'registry', 'process'],
                'score': 2
            }
        }

    def _init_safe_patterns(self) -> None:
        self.safe_patterns = {
            'high': {
                'patterns': [
                    'meeting reminder', 'project update', 'weekly newsletter',
                    'team sync', 'quarterly report', 'thank you for',
                    'best regards', 'kind regards', 'looking forward',
                ],
                'score': 15
            },
            'medium': {
                'patterns': [
                    'meeting', 'schedule', 'calendar', 'reminder', 'update',
                    'report', 'document', 'presentation', 'team', 'project',
                ],
                'score': 5
            }
        }

    def _init_regex_patterns(self) -> None:
        self.regex_patterns = {
            'base64_long': {'pattern': r'[A-Za-z0-9+/]{50,}={0,2}', 'malware_score': 15, 'phishing_score': 5, 'description': 'Long Base64 encoded string'},
            'ip_address': {'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'malware_score': 8, 'phishing_score': 5, 'description': 'IP address detected'},
            'url_shortener': {'pattern': r'(?:bit\.ly|tinyurl|goo\.gl|t\.co|is\.gd)', 'malware_score': 5, 'phishing_score': 15, 'description': 'URL shortener detected'},
            'suspicious_extension': {'pattern': r'\.(exe|dll|bat|cmd|vbs|ps1|scr)\b', 'malware_score': 15, 'phishing_score': 5, 'description': 'Suspicious file extension'},
            'encoded_command': {'pattern': r'(?:encodedcommand|frombase64string|invoke-expression)', 'malware_score': 25, 'phishing_score': 0, 'description': 'Encoded command execution'},
            'shell_command': {'pattern': r'(?:/bin/(?:ba)?sh|/dev/tcp)', 'malware_score': 30, 'phishing_score': 0, 'description': 'Shell command pattern'},
        }

    def _init_contextual_rules(self) -> None:
        self.urgency_words = ['immediately', 'urgent', 'asap', 'now', 'hurry', 'quickly', 'expires', 'deadline', 'last chance', 'final notice']
        self.threat_words = ['suspended', 'terminated', 'closed', 'locked', 'frozen', 'disabled', 'blocked', 'cancelled', 'deleted']
        self.action_words = ['click', 'download', 'install', 'verify', 'confirm', 'submit', 'send', 'transfer', 'pay']

    def analyze(self, text: str) -> RuleAnalysisResult:
        result = RuleAnalysisResult()
        text_lower = text.lower()

        self._check_patterns(text_lower, self.phishing_patterns, 'phishing', result)
        self._check_patterns(text_lower, self.malware_patterns, 'malware', result)
        self._check_patterns(text_lower, self.safe_patterns, 'safe', result)
        self._check_regex_patterns(text, result)
        self._contextual_analysis(text_lower, result)

        result.phishing_score = min(result.phishing_score, 100)
        result.malware_score = min(result.malware_score, 100)
        result.safe_score = min(result.safe_score, 100)

        return result

    def _check_patterns(self, text: str, patterns: Dict, pattern_type: str, result: RuleAnalysisResult) -> None:
        type_icons = {'phishing': '🎣', 'malware': '🦠', 'safe': '✅'}

        for severity, config in patterns.items():
            for keyword in config['patterns']:
                if keyword in text:
                    if pattern_type == 'phishing':
                        result.phishing_score += config['score']
                    elif pattern_type == 'malware':
                        result.malware_score += config['score']
                    else:
                        result.safe_score += config['score']

                    result.rule_matches += 1

                    if severity in ['critical', 'high']:
                        icon = type_icons.get(pattern_type, '⚠️')
                        result.indicators.append(f"{icon} {severity.title()}-risk {pattern_type}: '{keyword}'")

    def _check_regex_patterns(self, text: str, result: RuleAnalysisResult) -> None:
        for pattern_name, config in self.regex_patterns.items():
            matches = re.findall(config['pattern'], text, re.IGNORECASE)
            if matches:
                result.malware_score += config['malware_score']
                result.phishing_score += config['phishing_score']
                result.rule_matches += 1
                if config['malware_score'] > 10 or config['phishing_score'] > 10:
                    result.indicators.append(f"⚠️ {config['description']}: {len(matches)} match(es)")

    def _contextual_analysis(self, text: str, result: RuleAnalysisResult) -> None:
        urgency_count = sum(1 for word in self.urgency_words if word in text)
        if urgency_count >= 3:
            result.phishing_score *= 1.5
            result.indicators.append(f"⏰ High urgency language detected ({urgency_count} words)")
        elif urgency_count >= 2:
            result.phishing_score *= 1.2

        threat_count = sum(1 for word in self.threat_words if word in text)
        if threat_count >= 2:
            result.phishing_score *= 1.3
            result.indicators.append(f"⚠️ Threatening language detected ({threat_count} words)")

        caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
        if len(caps_words) >= 3:
            result.phishing_score += 10