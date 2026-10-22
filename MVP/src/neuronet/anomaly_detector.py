"""Isolation Forest baseline anomaly detector.

Train only on known-normal telemetry. For a research evaluation, keep train/test
scenarios separate and report precision/recall/F1 on labelled held-out faults.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .telemetry import METRIC_KEYS, validate_metrics

class AnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=150, contamination=contamination,
            random_state=random_state
        )
        self.is_fitted = False

    @staticmethod
    def _matrix(rows):
        return np.asarray([[validate_metrics(r)[k] for k in METRIC_KEYS] for r in rows], dtype=float)

    def fit(self, normal_rows):
        x = self._matrix(normal_rows)
        if len(x) < 20:
            raise ValueError("Provide at least 20 normal samples to fit the detector")
        self.scaler.fit(x)
        self.model.fit(self.scaler.transform(x))
        self.is_fitted = True
        return self

    def predict(self, metrics):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        x = self.scaler.transform(self._matrix([metrics]))
        # sklearn score_samples: lower is more anomalous.
        raw = float(self.model.score_samples(x)[0])
        threshold = float(self.model.offset_)
        anomaly = raw < threshold
        # Bounded ranking score for UI/logging only; not a calibrated probability.
        score = float(np.clip((threshold - raw) / max(abs(threshold), 1e-6), 0, 1))
        return {"is_anomaly": bool(anomaly), "anomaly_score": score, "raw_score": raw}
