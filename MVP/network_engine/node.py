"""Node model for NeuroNet 20-node network engine.

Represents a network node (Core, Aggregation, or Edge) with real queue dynamics,
packet processing capacity, CPU load, and health state.
"""

from dataclasses import dataclass, field
import random
from typing import Dict, Any


@dataclass
class NetworkNode:
    node_id: int
    name: str
    tier: str  # 'core', 'agg', 'edge'
    x: int
    y: int

    # Queue & Traffic Parameters
    queue_capacity: int = 1000          # Maximum packets buffer
    current_queue: float = 20.0          # Current queued packets
    processing_rate_mbps: float = 100.0  # Packet processing throughput capacity
    incoming_rate_mbps: float = 0.0     # Live ingress traffic rate
    outgoing_rate_mbps: float = 0.0     # Live egress forwarded rate

    # Performance Counters
    dropped_packets: int = 0
    dropped_this_step: int = 0
    queuing_delay_ms: float = 1.0

    # System Health
    cpu_util_pct: float = 22.0
    status: str = "nominal"             # 'nominal', 'warning', 'congested', 'critical'
    qos_enabled: bool = False
    packet_size_bytes: int = 1500       # Standard Ethernet MTU

    def update_queue(self, incoming_mbps: float, dt: float = 1.0) -> Dict[str, Any]:
        """Calculates fluid queue dynamics: Q_{t+1} = min(Q_max, max(0, Q_t + A_t - S_t))."""
        self.incoming_rate_mbps = incoming_mbps

        # Effective processing rate (QoS reprioritizes and gives extra headroom if enabled)
        effective_processing = self.processing_rate_mbps * (1.2 if self.qos_enabled else 1.0)

        # Convert Mbps to packets per dt interval: (rate * 10^6 * dt) / (packet_size * 8)
        bytes_in = (incoming_mbps * 1e6 * dt) / 8.0
        bytes_processed = (effective_processing * 1e6 * dt) / 8.0

        packets_in = bytes_in / self.packet_size_bytes
        packets_processed = bytes_processed / self.packet_size_bytes

        # Fluid queue evolution
        net_change = packets_in - packets_processed
        potential_queue = self.current_queue + net_change

        if potential_queue > self.queue_capacity:
            self.dropped_this_step = int(potential_queue - self.queue_capacity)
            self.dropped_packets += self.dropped_this_step
            self.current_queue = float(self.queue_capacity)
        else:
            self.dropped_this_step = 0
            self.current_queue = max(0.0, potential_queue)

        # Outgoing forwarded rate
        self.outgoing_rate_mbps = min(incoming_mbps, effective_processing)

        # Queuing delay (Little's Law approximation: queue_packets / (packets_per_sec))
        packets_per_sec = (effective_processing * 1e6) / (self.packet_size_bytes * 8.0)
        self.queuing_delay_ms = max(0.5, (self.current_queue / (packets_per_sec + 1e-5)) * 1000.0)

        # CPU calculation based on processing load
        load_ratio = incoming_mbps / (self.processing_rate_mbps + 1e-5)
        base_cpu = 15.0 if self.tier == "edge" else (22.0 if self.tier == "core" else 26.0)
        self.cpu_util_pct = min(99.0, max(base_cpu, base_cpu + load_ratio * 55.0 + random.uniform(-1.0, 1.0)))

        # Update health status
        util_pct = self.buffer_utilization_pct
        if util_pct >= 85.0 or self.dropped_this_step > 50:
            self.status = "critical"
        elif util_pct >= 60.0 or self.dropped_this_step > 0:
            self.status = "congested"
        elif util_pct >= 40.0:
            self.status = "warning"
        else:
            self.status = "nominal"

        return {
            "current_queue": round(self.current_queue, 1),
            "buffer_util_pct": round(self.buffer_utilization_pct, 1),
            "dropped_this_step": self.dropped_this_step,
            "queuing_delay_ms": round(self.queuing_delay_ms, 2),
            "cpu_util_pct": round(self.cpu_util_pct, 1),
            "status": self.status,
        }

    @property
    def buffer_utilization_pct(self) -> float:
        return min(100.0, (self.current_queue / max(1.0, self.queue_capacity)) * 100.0)

    def drain(self, dt: float = 1.0) -> None:
        """Drains the queue when incoming traffic is rerouted away."""
        packets_processed = (self.processing_rate_mbps * 1e6 * dt) / (self.packet_size_bytes * 8.0)
        self.current_queue = max(5.0, self.current_queue - packets_processed * 0.8)
        self.incoming_rate_mbps = max(5.0, self.incoming_rate_mbps * 0.3)
        self.outgoing_rate_mbps = self.incoming_rate_mbps
        self.dropped_this_step = 0
        self.queuing_delay_ms = max(0.5, self.queuing_delay_ms * 0.35)
        self.cpu_util_pct = max(15.0, self.cpu_util_pct * 0.5)
        self.status = "nominal"

    def scale_capacity(self, factor: float = 2.0) -> None:
        """SCALE_RESOURCE: increases queue buffer capacity and processing rate."""
        self.queue_capacity = int(self.queue_capacity * factor)
        self.processing_rate_mbps *= factor

    def reset(self) -> None:
        """Restores baseline nominal parameters."""
        self.queue_capacity = 1000
        self.current_queue = random.uniform(10.0, 25.0)
        self.processing_rate_mbps = 200.0 if self.tier == "core" else (80.0 if self.tier == "agg" else 150.0)
        self.incoming_rate_mbps = 20.0
        self.outgoing_rate_mbps = 20.0
        self.dropped_packets = 0
        self.dropped_this_step = 0
        self.queuing_delay_ms = random.uniform(0.8, 1.8)
        self.cpu_util_pct = random.uniform(18.0, 26.0)
        self.status = "nominal"
        self.qos_enabled = False

    def as_dict(self) -> Dict[str, Any]:
        """Matches the frontend dashboard JSON schema."""
        return {
            "id": self.node_id,
            "name": self.name,
            "tier": self.tier,
            "x": self.x,
            "y": self.y,
            "cpu": round(self.cpu_util_pct, 1),
            "buffer": round(self.buffer_utilization_pct, 1),
            "status": self.status,
            "queue_depth": round(self.current_queue, 1),
            "queue_capacity": self.queue_capacity,
            "incoming_mbps": round(self.incoming_rate_mbps, 1),
            "processing_mbps": round(self.processing_rate_mbps, 1),
            "dropped_packets": self.dropped_packets,
            "delay_ms": round(self.queuing_delay_ms, 2),
        }
