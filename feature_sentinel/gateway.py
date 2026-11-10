from typing import Dict, List, Tuple
import pandas as pd
from feature_sentinel.drift import DualLayerDriftDetector, DriftTelemetry
from feature_sentinel.anomaly import AttributedAnomalySentinel
from feature_sentinel.circuit import IngestionCircuitBreaker, CircuitVerdict

class FeatureSentinelGateway:
    def __init__(self, features: List[str], anomaly_tolerance: float = 0.12):
        self.features = features
        self.drift_detector = DualLayerDriftDetector()
        self.anomaly_detector = AttributedAnomalySentinel()
        self.circuit_breaker = IngestionCircuitBreaker(max_anomaly_tolerance=anomaly_tolerance)

    def fit_baseline(self, baseline_df: pd.DataFrame):
        self.drift_detector.fit(baseline_df, self.features)
        self.anomaly_detector.fit(baseline_df, self.features)
        return self

    def inspect(self, batch_df: pd.DataFrame) -> Tuple[CircuitVerdict, Dict[str, DriftTelemetry], pd.DataFrame]:
        telemetry = self.drift_detector.inspect(batch_df)
        preds, scores, culprits = self.anomaly_detector.score(batch_df)
        
        enriched = batch_df.copy()
        enriched["is_quarantined"] = (preds == -1).astype(int)
        enriched["sentinel_score"] = scores
        enriched["root_cause_feature"] = culprits
        
        total = len(batch_df)
        anomalies = int((preds == -1).sum())
        rate = float(anomalies / total) if total > 0 else 0.0
        
        verdict = self.circuit_breaker.evaluate(telemetry, rate, total, anomalies)
        return verdict, telemetry, enriched
