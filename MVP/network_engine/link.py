"""Link model for NeuroNet 20-node network engine.

Represents an interconnecting physical link between two nodes, with bandwidth,
propagation latency, loss rate, and status.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class NetworkLink:
    source: int
    target: int
    bandwidth_mbps: float = 100.0
    propagation_delay_ms: float = 2.0
    current_load_mbps: float = 25.0
    loss_pct: float = 0.0
    status: str = "normal"  # 'normal', 'congested', 'cut', 'degraded'
    is_primary: bool = True
    is_backup: bool = False

    @property
    def utilization_pct(self) -> float:
        return min(100.0, (self.current_load_mbps / max(1.0, self.bandwidth_mbps)) * 100.0)

    @property
    def latency_ms(self) -> float:
        if self.status == "cut":
            return 999.0
        # Add latency as link nears 100% capacity
        util = self.utilization_pct / 100.0
        congestion_penalty = (util ** 3) * 15.0 if util > 0.8 else 0.0
        degradation_penalty = 35.0 if self.status == "degraded" else 0.0
        return self.propagation_delay_ms + congestion_penalty + degradation_penalty

    def reset(self) -> None:
        self.bandwidth_mbps = 1000.0 if (self.source < 4 and self.target < 4) else 100.0
        self.current_load_mbps = 25.0
        self.loss_pct = 0.0
        self.status = "normal"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "load": round(self.utilization_pct, 1),
            "status": self.status,
            "bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": round(self.latency_ms, 2),
            "loss_pct": round(self.loss_pct, 2),
        }
