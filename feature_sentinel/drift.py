from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

class DriftSeverity(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"

@dataclass
class DriftTelemetry:
    feature: str
    ks_statistic: float
    p_value: float
    wasserstein_distance: float
    psi: float
    severity: DriftSeverity

def calculate_psi(baseline: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    tgt_counts, _ = np.histogram(target, bins=bin_edges)
    base_pct = (base_counts + 1e-4) / (len(baseline) + 1e-4 * len(base_counts))
    tgt_pct = (tgt_counts + 1e-4) / (len(target) + 1e-4 * len(tgt_counts))
    return float(np.sum((tgt_pct - base_pct) * np.log(tgt_pct / base_pct)))

class DualLayerDriftDetector:
    def __init__(self, alpha: float = 0.01, psi_critical: float = 0.20):
        self.alpha = alpha
        self.psi_critical = psi_critical
        self.baseline_data: Optional[pd.DataFrame] = None
        self.features: List[str] = []

    def fit(self, baseline_df: pd.DataFrame, features: List[str]):
        self.features = features
        self.baseline_data = baseline_df[features].copy()
        return self

    def inspect(self, target_df: pd.DataFrame) -> Dict[str, DriftTelemetry]:
        effective_alpha = self.alpha / len(self.features)
        results = {}
        for col in self.features:
            b_vec = self.baseline_data[col].dropna().values
            t_vec = target_df[col].dropna().values
            ks = stats.ks_2samp(b_vec, t_vec)
            w_dist = stats.wasserstein_distance(b_vec, t_vec)
            psi = calculate_psi(b_vec, t_vec)
            
            if psi >= self.psi_critical or ks.statistic >= 0.35:
                severity = DriftSeverity.CRITICAL
            elif (ks.pvalue < effective_alpha) and (psi >= 0.10 or ks.statistic > 0.15):
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.NEGLIGIBLE
                
            results[col] = DriftTelemetry(
                feature=col,
                ks_statistic=float(ks.statistic),
                p_value=float(ks.pvalue),
                wasserstein_distance=float(w_dist),
                psi=float(psi),
                severity=severity
            )
        return results
