"""Telemetry validation and feature extraction."""
import numpy as np

METRIC_KEYS = ("latency_ms", "throughput_mbps", "packet_loss_pct", "congestion", "link_health")

def validate_metrics(metrics):
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:
        raise ValueError(f"Missing telemetry fields: {missing}")
    values = {k: float(metrics[k]) for k in METRIC_KEYS}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Telemetry contains non-finite values")
    if values["latency_ms"] < 0 or values["throughput_mbps"] < 0:
        raise ValueError("Latency and throughput must be non-negative")
    if not 0 <= values["packet_loss_pct"] <= 100:
        raise ValueError("packet_loss_pct must be in [0, 100]")
    if not 0 <= values["congestion"] <= 1 or not 0 <= values["link_health"] <= 1:
        raise ValueError("congestion and link_health must be in [0, 1]")
    return values

def make_state(metrics, fault="none", recent_actions=0):
    m = validate_metrics(metrics)
    fault = str(fault).lower()
    return np.asarray([
        np.clip(m["latency_ms"] / 200.0, 0, 1),
        np.clip(m["throughput_mbps"] / 100.0, 0, 1),
        np.clip(m["packet_loss_pct"] / 10.0, 0, 1),
        m["congestion"], m["link_health"],
        float(fault not in ("none", "normal", "")),
        float(fault in ("link_failure", "link failure", "failure")),
        min(max(int(recent_actions), 0), 3) / 3.0,
    ], dtype=np.float32)
