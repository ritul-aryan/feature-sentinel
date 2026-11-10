import unittest
import numpy as np
import pandas as pd
from feature_sentinel.gateway import FeatureSentinelGateway
from feature_sentinel.drift import calculate_psi, DriftSeverity
from feature_sentinel.circuit import CircuitState

class TestFeatureSentinelEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.n_samples = 2000
        self.features = ["feat_a", "feat_b"]
        self.baseline_df = pd.DataFrame({
            "feat_a": np.random.normal(50.0, 10.0, self.n_samples),
            "feat_b": np.random.exponential(5.0, self.n_samples),
        })
        self.gateway = FeatureSentinelGateway(self.features, anomaly_tolerance=0.10)
        self.gateway.fit_baseline(self.baseline_df)

    def test_null_hypothesis_stationary_data(self):
        stationary_prod = pd.DataFrame({
            "feat_a": np.random.normal(50.0, 10.0, 1000),
            "feat_b": np.random.exponential(5.0, 1000),
        })
        verdict, telemetry, _ = self.gateway.inspect(stationary_prod)
        self.assertEqual(verdict.state, CircuitState.CLOSED)
        self.assertEqual(telemetry["feat_a"].severity, DriftSeverity.NEGLIGIBLE)

    def test_psi_zero_on_identical_distributions(self):
        vec = np.random.normal(0, 1, 3000)
        psi_val = calculate_psi(vec, vec)
        self.assertAlmostEqual(psi_val, 0.0, places=3)

    def test_circuit_trips_on_extreme_covariate_shift(self):
        corrupted_prod = pd.DataFrame({
            "feat_a": np.random.normal(120.0, 10.0, 1000),
            "feat_b": np.random.exponential(5.0, 1000),
        })
        verdict, telemetry, _ = self.gateway.inspect(corrupted_prod)
        self.assertEqual(verdict.state, CircuitState.TRIPPED)
        self.assertIn("feat_a", verdict.quarantined_features)

if __name__ == "__main__":
    unittest.main()