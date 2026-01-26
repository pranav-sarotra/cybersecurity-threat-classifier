"""
Machine Learning Threat Classifier.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    cross_val_scores: List[float]
    confusion_matrix: Optional[np.ndarray] = None


class MLThreatClassifier:
    """ML-based threat classifier using ensemble methods."""

    CLASS_NAMES = ['phishing', 'malware', 'safe']

    def __init__(self, model_config: Optional[Dict] = None):
        config = model_config or {}

        self.vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 5000),
            ngram_range=tuple(config.get('ngram_range', [1, 3])),
            min_df=config.get('min_df', 1),
            max_df=config.get('max_df', 0.95),
            sublinear_tf=True,
            stop_words='english'
        )

        self.classifiers = {
            'nb': MultinomialNB(alpha=0.1),
            'lr': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
            'rf': RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
            'svm': CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight='balanced', random_state=42), cv=3)
        }

        self.ensemble = VotingClassifier(
            estimators=[(name, clf) for name, clf in self.classifiers.items()],
            voting='soft',
            n_jobs=-1
        )

        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_names: Optional[List[str]] = None

    def train(self, texts: List[str], labels: List[int], evaluate: bool = True) -> 'MLThreatClassifier':
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.ensemble.fit(X, labels)
        self.is_trained = True

        if evaluate:
            self._compute_metrics(X, labels)

        return self

    def _compute_metrics(self, X, labels: List[int]) -> None:
        cv_scores = cross_val_score(self.ensemble, X, labels, cv=5, scoring='accuracy')
        predictions = self.ensemble.predict(X)
        report = classification_report(labels, predictions, target_names=self.CLASS_NAMES, output_dict=True)

        self.metrics = ModelMetrics(
            accuracy=float(np.mean(cv_scores)),
            precision={name: report[name]['precision'] for name in self.CLASS_NAMES},
            recall={name: report[name]['recall'] for name in self.CLASS_NAMES},
            f1_score={name: report[name]['f1-score'] for name in self.CLASS_NAMES},
            cross_val_scores=cv_scores.tolist(),
            confusion_matrix=confusion_matrix(labels, predictions)
        )

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_trained:
            return {'phishing': 0.33, 'malware': 0.33, 'safe': 0.34}

        X = self.vectorizer.transform([text])
        probs = self.ensemble.predict_proba(X)[0]

        return {
            'phishing': float(probs[0]),
            'malware': float(probs[1]),
            'safe': float(probs[2])
        }

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        probs = self.predict_proba(text)
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]
        return predicted_class, confidence, probs

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        if not self.is_trained or self.feature_names is None:
            return {}

        rf = self.classifiers['rf']
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [(self.feature_names[i], float(importances[i])) for i in indices]

        return {'overall': top_features}