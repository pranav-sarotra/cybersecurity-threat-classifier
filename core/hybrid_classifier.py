"""
Hybrid Classifier combining ML and Rule-Based approaches.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .ml_classifier import MLThreatClassifier
from .rule_classifier import RuleBasedClassifier
from .data_generator import ThreatDataGenerator
from .ip_checker import IPChecker
from .url_analyzer import URLAnalyzer


@dataclass
class ClassificationResult:
    """Classification results container."""
    classification: str
    raw_class: str
    confidence: float
    threat_score: float
    threat_level: str
    threat_level_color: str
    ml_scores: Dict[str, float] = field(default_factory=dict)
    rule_scores: Dict[str, float] = field(default_factory=dict)
    combined_scores: Dict[str, float] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    rule_matches: int = 0
    risk_factors: List[str] = field(default_factory=list)
    extracted_ips: List[str] = field(default_factory=list)
    extracted_urls: List[str] = field(default_factory=list)
    ip_analysis: List[Dict] = field(default_factory=list)
    url_analysis: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HybridThreatClassifier:
    """Hybrid threat classifier combining ML and rule-based analysis."""

    CLASS_LABELS = {
        'malware': '🦠 MALWARE THREAT',
        'phishing': '🎣 PHISHING ATTEMPT',
        'safe': '✅ SAFE'
    }

    def __init__(self, ml_weight: float = 0.6, rule_weight: float = 0.4,
                 analyze_ips: bool = True, analyze_urls: bool = True):
        self.ml_classifier = MLThreatClassifier()
        self.rule_classifier = RuleBasedClassifier()
        self.ip_checker = IPChecker()
        self.url_analyzer = URLAnalyzer()
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.analyze_ips = analyze_ips
        self.analyze_urls = analyze_urls
        self.is_trained = False

    def train(self, augment: bool = True) -> 'HybridThreatClassifier':
        texts, labels = ThreatDataGenerator.generate_training_data()
        if augment:
            texts, labels = ThreatDataGenerator.generate_augmented_data(texts, labels)
        self.ml_classifier.train(texts, labels)
        self.is_trained = True
        return self

    def classify(self, text: str) -> ClassificationResult:
        if not text.strip():
            return ClassificationResult(
                classification='No Input', raw_class='none', confidence=0,
                threat_score=0, threat_level='NONE', threat_level_color='#888888'
            )

        ml_class, ml_confidence, ml_probs = self.ml_classifier.predict(text)
        rule_results = self.rule_classifier.analyze(text)

        rule_total = rule_results.phishing_score + rule_results.malware_score + rule_results.safe_score + 50
        rule_probs = {
            'phishing': rule_results.phishing_score / rule_total,
            'malware': rule_results.malware_score / rule_total,
            'safe': (rule_results.safe_score + 50) / rule_total
        }

        combined_scores = {}
        for cls in ['phishing', 'malware', 'safe']:
            combined_scores[cls] = self.ml_weight * ml_probs[cls] + self.rule_weight * rule_probs[cls]

        total = sum(combined_scores.values())
        for cls in combined_scores:
            combined_scores[cls] /= total

        if rule_results.malware_score >= 50:
            combined_scores['malware'] += 0.2
        if rule_results.phishing_score >= 40:
            combined_scores['phishing'] += 0.15

        total = sum(combined_scores.values())
        for cls in combined_scores:
            combined_scores[cls] /= total

        final_class = max(combined_scores, key=combined_scores.get)
        final_confidence = combined_scores[final_class] * 100
        threat_score = max(combined_scores['phishing'] * 100, combined_scores['malware'] * 100)
        threat_level, threat_level_color = self._get_threat_level(final_class, final_confidence, threat_score)

        if final_class == 'safe' and final_confidence < 70:
            display_class = '⚠️ SUSPICIOUS'
        else:
            display_class = self.CLASS_LABELS[final_class]

        result = ClassificationResult(
            classification=display_class,
            raw_class=final_class,
            confidence=round(final_confidence, 1),
            threat_score=round(threat_score, 1),
            threat_level=threat_level,
            threat_level_color=threat_level_color,
            ml_scores={k: round(v * 100, 1) for k, v in ml_probs.items()},
            rule_scores={'phishing': round(rule_results.phishing_score, 1), 'malware': round(rule_results.malware_score, 1)},
            combined_scores={k: round(v * 100, 1) for k, v in combined_scores.items()},
            indicators=rule_results.indicators,
            rule_matches=rule_results.rule_matches,
            risk_factors=rule_results.risk_factors
        )

        if self.analyze_ips:
            self._analyze_extracted_ips(text, result)
        if self.analyze_urls:
            self._analyze_extracted_urls(text, result)

        result.recommendations = self._generate_recommendations(result)
        return result

    def _get_threat_level(self, cls: str, confidence: float, threat_score: float) -> Tuple[str, str]:
        if cls == 'safe':
            return ("NONE", "#00ff88") if confidence >= 70 else ("LOW", "#ffcc00")
        if threat_score > 75:
            return "CRITICAL", "#ff0000"
        elif threat_score > 50:
            return "HIGH", "#ff4444"
        elif threat_score > 30:
            return "MEDIUM", "#ff8800"
        return "LOW", "#ffaa00"

    def _analyze_extracted_ips(self, text: str, result: ClassificationResult) -> None:
        ips = self.ip_checker.extract_ips_from_text(text)
        result.extracted_ips = ips
        for ip in ips[:5]:
            ip_info = self.ip_checker.check_ip(ip, deep_scan=False)
            summary = self.ip_checker.get_threat_summary(ip_info)
            result.ip_analysis.append({'ip': ip, 'summary': summary})
            if ip_info.threat_score > 25:
                result.indicators.append(f"🌐 Suspicious IP: {ip} (Score: {ip_info.threat_score})")

    def _analyze_extracted_urls(self, text: str, result: ClassificationResult) -> None:
        urls = self.url_analyzer.extract_urls_from_text(text)
        result.extracted_urls = urls
        for url in urls[:5]:
            url_info = self.url_analyzer.analyze(url)
            result.url_analysis.append({
                'url': url, 'is_suspicious': url_info.is_suspicious,
                'threat_score': url_info.threat_score, 'domain': url_info.domain
            })
            if url_info.is_suspicious:
                result.indicators.append(f"🔗 Suspicious URL: {url[:50]}... (Score: {url_info.threat_score})")

    def _generate_recommendations(self, result: ClassificationResult) -> List[str]:
        if 'MALWARE' in result.classification:
            return [
                "🚨 ISOLATE the affected system immediately",
                "❌ DO NOT execute any attached files or scripts",
                "📧 Report to Cybersecurity Department",
                "🛡️ Update antivirus and run full scan"
            ]
        elif 'PHISHING' in result.classification:
            return [
                "⚠️ DO NOT click any links or download attachments",
                "🔒 DO NOT provide any personal information",
                "📧 Report to IT Security team",
                "🔑 If credentials were entered, reset passwords immediately"
            ]
        elif 'SUSPICIOUS' in result.classification:
            return [
                "✉️ Verify sender identity through official channels",
                "🔗 Do not click links - navigate to websites directly",
                "📤 Forward to IT for secondary review"
            ]
        return ["✅ No threats detected", "🛡️ Continue following security best practices"]

    def get_model_metrics(self) -> Optional[Dict]:
        if not self.is_trained or not self.ml_classifier.metrics:
            return None
        m = self.ml_classifier.metrics
        return {'accuracy': m.accuracy, 'precision': m.precision, 'recall': m.recall, 'f1_score': m.f1_score}