"""Realistic Network Telemetry & Topology Simulation Environment.

Simulates dynamic 20-node network behavior with M/M/1 queuing delays, buffer overflows,
realistic failure scenarios (Congestion, Link Failure, Routing Flaps, DDoS, Degradation),
and measurable physical impacts for the 8 remediation actions.
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .actions import Action, ACTION_NAMES
from .telemetry import validate_metrics, make_state, STATE_DIM

@dataclass
class NetworkMetrics:
    latency_ms: float
    throughput_mbps: float
    packet_loss_pct: float
    congestion: float
    link_health: float
    cpu_util_pct: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "latency_ms": float(self.latency_ms),
            "throughput_mbps": float(self.throughput_mbps),
            "packet_loss_pct": float(self.packet_loss_pct),
            "congestion": float(self.congestion),
            "link_health": float(self.link_health),
            "cpu_util_pct": float(self.cpu_util_pct),
        }


class RealisticNetworkEnv:
    SCENARIOS = [
        "normal",
        "congestion",
        "link_failure",
        "routing_flap",
        "attack_ddos",
        "degradation",
    ]

    def __init__(self, seed: int = 42, num_nodes: int = 20, max_steps: int = 20):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.steps = 0
        self.metrics: Optional[NetworkMetrics] = None
        self.scenario: str = "normal"
        self.active_node: int = 0
        self.recent_actions: list = []
        self.last_action: Optional[Action] = None

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def _generate_baseline_metrics(self) -> NetworkMetrics:
        """Nominal operational baseline."""
        lat = max(5.0, float(self.rng.normal(20.0, 2.5)))
        tput = float(np.clip(self.rng.normal(92.0, 3.5), 75.0, 100.0))
        loss = float(max(0.01, self.rng.normal(0.12, 0.04)))
        cong = float(np.clip(self.rng.normal(0.14, 0.03), 0.05, 0.30))
        health = float(np.clip(self.rng.normal(0.98, 0.01), 0.90, 1.0))
        cpu = float(np.clip(self.rng.normal(24.0, 4.0), 10.0, 45.0))
        return NetworkMetrics(lat, tput, loss, cong, health, cpu)

    def reset(self, scenario: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.steps = 0
        self.recent_actions = []
        self.last_action = None
        self.active_node = int(self.rng.integers(0, self.num_nodes))

        if scenario is None or scenario == "all":
            self.scenario = str(self.rng.choice(self.SCENARIOS))
        else:
            self.scenario = scenario.lower()

        m = self._generate_baseline_metrics()

        # Inject realistic physical failure dynamics
        if self.scenario == "congestion":
            # Link buffer saturation, queuing delay escalates
            m.latency_ms += float(self.rng.normal(95.0, 10.0))
            m.throughput_mbps = float(np.clip(m.throughput_mbps - 42.0, 25.0, 70.0))
            m.packet_loss_pct += float(self.rng.normal(4.8, 0.8))
            m.congestion = float(np.clip(self.rng.normal(0.92, 0.03), 0.82, 1.0))
            m.cpu_util_pct = float(np.clip(self.rng.normal(78.0, 5.0), 65.0, 95.0))

        elif self.scenario == "link_failure":
            # Physical link down, sudden packet blackhole
            m.latency_ms += float(self.rng.normal(85.0, 12.0))
            m.throughput_mbps = float(self.rng.normal(4.5, 1.5))  # near collapse
            m.packet_loss_pct += float(self.rng.normal(9.2, 1.2))
            m.link_health = float(np.clip(self.rng.normal(0.12, 0.04), 0.0, 0.25))
            m.congestion = float(np.clip(self.rng.normal(0.85, 0.05), 0.70, 0.95))

        elif self.scenario == "routing_flap":
            # BGP/OSPF route flapping, path oscillation
            m.latency_ms += float(self.rng.normal(68.0, 14.0))
            m.throughput_mbps -= float(self.rng.normal(30.0, 6.0))
            m.packet_loss_pct += float(self.rng.normal(3.2, 0.6))
            m.congestion = float(np.clip(self.rng.normal(0.65, 0.08), 0.45, 0.85))

        elif self.scenario == "attack_ddos":
            # Volumetric UDP/SYN flood saturating router CPU and buffers
            m.latency_ms += float(self.rng.normal(120.0, 15.0))
            m.throughput_mbps = float(np.clip(self.rng.normal(98.0, 2.0), 90.0, 100.0))  # line rate saturated
            m.packet_loss_pct += float(self.rng.normal(8.5, 1.5))
            m.congestion = float(np.clip(self.rng.normal(0.96, 0.02), 0.90, 1.0))
            m.cpu_util_pct = float(np.clip(self.rng.normal(96.0, 2.0), 90.0, 100.0))

        elif self.scenario == "degradation":
            # Progressive optical/hardware degradation
            m.latency_ms += float(self.rng.normal(42.0, 6.0))
            m.throughput_mbps -= float(self.rng.normal(25.0, 5.0))
            m.packet_loss_pct += float(self.rng.normal(2.6, 0.4))
            m.link_health = float(np.clip(self.rng.normal(0.55, 0.05), 0.40, 0.70))

        # Clamp physical bounds
        m.latency_ms = max(2.0, m.latency_ms)
        m.throughput_mbps = max(0.5, m.throughput_mbps)
        m.packet_loss_pct = float(np.clip(m.packet_loss_pct, 0.0, 100.0))

        self.metrics = m
        info = {
            "scenario": self.scenario,
            "fault": self.scenario,
            "node_id": self.active_node,
            "metrics": self.metrics.as_dict(),
        }
        state = make_state(self.metrics.as_dict(), recent_actions=self.recent_actions)
        return state, info

    def step(self, action: Action, anomaly_info: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = Action(int(action))
        before = self.metrics.as_dict()
        self.steps += 1
        self.last_action = action
        self.recent_actions.append(int(action))

        m = self.metrics

        # Remediation physics: applying corrective actions
        if action == Action.REROUTE:
            # Shift traffic off congested link
            m.latency_ms = max(18.0, m.latency_ms * 0.45 + float(self.rng.normal(0, 1.5)))
            m.throughput_mbps = min(95.0, m.throughput_mbps * 1.6 + 15.0)
            m.packet_loss_pct = max(0.1, m.packet_loss_pct * 0.35)
            m.congestion = max(0.15, m.congestion * 0.38)
            m.cpu_util_pct = max(25.0, m.cpu_util_pct * 0.70)
            if self.scenario == "link_failure":
                m.link_health = 0.90  # Bypass failed link completely

        elif action == Action.QOS_ADJUST:
            # Prioritize queue: drops latency & loss for critical packets
            m.latency_ms = max(20.0, m.latency_ms * 0.65)
            m.packet_loss_pct = max(0.15, m.packet_loss_pct * 0.50)
            m.congestion = max(0.20, m.congestion * 0.75)

        elif action == Action.TRAFFIC_SHAPE:
            # Token bucket smoothing: caps bursty spikes
            m.congestion = max(0.25, m.congestion * 0.60)
            m.packet_loss_pct = max(0.18, m.packet_loss_pct * 0.60)
            m.throughput_mbps = max(30.0, m.throughput_mbps * 0.92)  # slight throughput tradeoff

        elif action == Action.FAILOVER:
            # Interface switch to standby hardware
            m.link_health = float(np.clip(self.rng.normal(0.96, 0.02), 0.90, 1.0))
            m.latency_ms = max(20.0, m.latency_ms * 0.50)
            m.throughput_mbps = float(np.clip(self.rng.normal(88.0, 4.0), 75.0, 95.0))
            m.packet_loss_pct = max(0.1, m.packet_loss_pct * 0.25)
            m.congestion = max(0.18, m.congestion * 0.40)

        elif action == Action.SCALE_RESOURCE:
            # Allocate more bandwidth/slice capacity
            m.throughput_mbps = min(100.0, m.throughput_mbps + 35.0)
            m.congestion = max(0.15, m.congestion * 0.45)
            m.latency_ms = max(19.0, m.latency_ms * 0.60)
            m.cpu_util_pct = max(30.0, m.cpu_util_pct * 0.65)

        elif action == Action.THROTTLE:
            # Throttle background slice to relieve DDoS or buffer pressure
            m.congestion = max(0.20, m.congestion * 0.50)
            m.cpu_util_pct = max(28.0, m.cpu_util_pct * 0.55)
            m.packet_loss_pct = max(0.15, m.packet_loss_pct * 0.40)
            m.latency_ms = max(22.0, m.latency_ms * 0.70)

        elif action == Action.ESCALATE:
            # Operator escalation: keeps metrics mostly unchanged pending manual review
            pass

        elif action == Action.MONITOR:
            # Passive observation: metrics persist or slightly deteriorate if anomaly active
            if self.scenario != "normal":
                m.latency_ms += float(self.rng.normal(2.0, 1.0))
                m.packet_loss_pct = min(10.0, m.packet_loss_pct + 0.1)

        after = self.metrics.as_dict()

        # Compute multi-objective reward
        from .reward import compute_reward
        reward = compute_reward(before, after, action, recent_actions=self.recent_actions)

        done = bool(self.steps >= self.max_steps or self.steps >= 1)  # Episode per incident
        truncated = False
        next_state = make_state(after, anomaly_info=anomaly_info, recent_actions=self.recent_actions)

        step_info = {
            "metrics_before": before,
            "metrics_after": after,
            "latency_improvement": before["latency_ms"] - after["latency_ms"],
            "loss_reduction": before["packet_loss_pct"] - after["packet_loss_pct"],
            "action_name": ACTION_NAMES[action],
            "scenario": self.scenario,
        }

        return next_state, reward, done, truncated, step_info
