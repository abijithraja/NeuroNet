"""Telemetry validation and standardized state vector generation."""
import math
from typing import Dict, Any, Optional
import numpy as np

METRIC_KEYS = (
    "latency_ms",
    "throughput_mbps",
    "packet_loss_pct",
    "congestion",
    "link_health",
    "cpu_util_pct",
)

STATE_DIM = 14

def validate_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Validate and sanitize telemetry fields."""
    metrics_dict = dict(metrics)
    # Default cpu_util_pct if not provided
    if "cpu_util_pct" not in metrics_dict:
        metrics_dict["cpu_util_pct"] = metrics_dict.get("congestion", 0.15) * 60.0 + 20.0

    missing = [k for k in METRIC_KEYS if k not in metrics_dict]
    if missing:
        raise ValueError(f"Missing required telemetry fields: {missing}")

    values = {k: float(metrics_dict[k]) for k in METRIC_KEYS}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Telemetry contains non-finite values (NaN or Inf)")

    if values["latency_ms"] < 0 or values["throughput_mbps"] < 0:
        raise ValueError("Latency and throughput must be non-negative")

    values["packet_loss_pct"] = float(np.clip(values["packet_loss_pct"], 0.0, 100.0))
    values["congestion"] = float(np.clip(values["congestion"], 0.0, 1.0))
    values["link_health"] = float(np.clip(values["link_health"], 0.0, 1.0))
    values["cpu_util_pct"] = float(np.clip(values["cpu_util_pct"], 0.0, 100.0))

    return values


def make_state(
    metrics: Dict[str, Any],
    anomaly_info: Optional[Dict[str, Any]] = None,
    recent_actions: Optional[list] = None,
    hour_of_day: float = 14.0,
) -> np.ndarray:
    """Constructs the standardized 14-dimensional state vector for the RL policy."""
    m = validate_metrics(metrics)
    a = anomaly_info or {
        "anomaly_score": 0.0,
        "lstm_error": 0.0,
        "iso_score": 0.0,
        "features": {
            "latency_anomalous": False,
            "loss_anomalous": False,
            "congestion_high": False,
        },
    }
    feats = a.get("features", {})
    history = recent_actions or []
    last_action_val = (history[-1] / 7.0) if history else 0.0

    # Cyclical hour encoding (sinusoidal)
    sin_hour = (math.sin(2 * math.pi * (hour_of_day % 24) / 24.0) + 1.0) / 2.0

    state_vec = np.asarray([
        np.clip(m["latency_ms"] / 200.0, 0.0, 1.0),            # 0: Latency (0-200ms -> 0-1)
        np.clip(m["throughput_mbps"] / 100.0, 0.0, 1.0),        # 1: Throughput (0-100Mbps -> 0-1)
        np.clip(m["packet_loss_pct"] / 10.0, 0.0, 1.0),         # 2: Packet loss (0-10% -> 0-1)
        m["congestion"],                                        # 3: Buffer congestion [0, 1]
        m["link_health"],                                       # 4: Link health [0, 1]
        np.clip(m["cpu_util_pct"] / 100.0, 0.0, 1.0),          # 5: CPU load [0, 1]
        np.clip(a.get("anomaly_score", 0.0), 0.0, 1.0),         # 6: Combined anomaly score [0, 1]
        np.clip(a.get("lstm_error", 0.0), 0.0, 1.0),            # 7: LSTM AE reconstruction error
        np.clip(a.get("iso_score", 0.0), 0.0, 1.0),             # 8: Isolation forest outlier score
        1.0 if feats.get("latency_anomalous", False) else 0.0,  # 9: Latency anomaly flag
        1.0 if feats.get("loss_anomalous", False) else 0.0,     # 10: Loss anomaly flag
        1.0 if feats.get("congestion_high", False) else 0.0,    # 11: High congestion flag
        float(sin_hour),                                        # 12: Diurnal time encoding
        float(last_action_val),                                 # 13: Recent action representation
    ], dtype=np.float32)

    return state_vec
