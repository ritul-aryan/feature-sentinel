from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from feature_sentinel.drift import DriftTelemetry, DriftSeverity

class CircuitState(str, Enum):
    CLOSED = "CLOSED"              # Healthy Ingestion
    HALF_OPEN = "HALF_OPEN"        # Elevated Contamination Warning
    TRIPPED = "TRIPPED_QUARANTINE" # Critical Drift SLA Breach

@dataclass
class CircuitVerdict:
    state: CircuitState
    quarantined_features: List[str]
    trip_reason: Optional[str]
    total_records: int
    anomalous_records: int
    anomaly_rate: float

class IngestionCircuitBreaker:
    def __init__(self, max_anomaly_tolerance: float = 0.12):
        self.tolerance = max_anomaly_tolerance

    def evaluate(
        self,
        telemetry: Dict[str, DriftTelemetry],
        anomaly_rate: float,
        total: int,
        anomalies: int
    ) -> CircuitVerdict:
        crit = [k for k, v in telemetry.items() if v.severity == DriftSeverity.CRITICAL]
        mod = [k for k, v in telemetry.items() if v.severity == DriftSeverity.MODERATE]
        
        if crit:
            return CircuitVerdict(
                state=CircuitState.TRIPPED,
                quarantined_features=crit,
                trip_reason=f"Critical distribution drift on: {', '.join(crit)}",
                total_records=total,
                anomalous_records=anomalies,
                anomaly_rate=anomaly_rate
            )
        if anomaly_rate > self.tolerance or mod:
            return CircuitVerdict(
                state=CircuitState.HALF_OPEN,
                quarantined_features=mod,
                trip_reason=f"Contamination rate ({anomaly_rate:.1%}) breached SLA threshold" if anomaly_rate > self.tolerance else f"Moderate drift: {', '.join(mod)}",
                total_records=total,
                anomalous_records=anomalies,
                anomaly_rate=anomaly_rate
            )
        return CircuitVerdict(
            state=CircuitState.CLOSED,
            quarantined_features=[],
            trip_reason=None,
            total_records=total,
            anomalous_records=anomalies,
            anomaly_rate=anomaly_rate
        )
