from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AttributedAnomalySentinel:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
            n_jobs=-1
        )
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.features: List[str] = []

    def fit(self, baseline_df: pd.DataFrame, features: List[str]):
        self.features = features
        self.model.fit(baseline_df[features])
        self.means = baseline_df[features].mean()
        self.stds = baseline_df[features].std().replace(0, 1e-5)
        return self

    def score(self, target_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        preds = self.model.predict(target_df[self.features])
        scores = self.model.decision_function(target_df[self.features])
        z_scores = np.abs((target_df[self.features] - self.means) / self.stds)
        culprits = z_scores.idxmax(axis=1).tolist()
        return preds, scores, culprits
