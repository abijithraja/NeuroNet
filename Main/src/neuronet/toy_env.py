"""Synthetic Python toy environment for lightweight baseline checks."""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .actions import Action, ACTION_NAMES
from .telemetry import make_state, STATE_DIM

@dataclass
class Metrics:
    latency_ms: float
    throughput_mbps: float
    packet_loss_pct: float
    congestion: float
    link_health: float
    cpu_util_pct: float = 25.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "latency_ms": float(self.latency_ms),
            "throughput_mbps": float(self.throughput_mbps),
            "packet_loss_pct": float(self.packet_loss_pct),
            "congestion": float(self.congestion),
            "link_health": float(self.link_health),
            "cpu_util_pct": float(self.cpu_util_pct),
        }


class ToyNetworkEnv:
    def __init__(self, seed: int = 42, num_nodes: int = 20, max_steps: int = 20):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.steps = 0
        self.metrics = None
        self.fault = "none"
        self.scenario = "none"
        self.recent_actions = []
        self.last_action = None

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def _state(self, anomaly_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return make_state(self.metrics.as_dict(), anomaly_info=anomaly_info, recent_actions=self.recent_actions)

    def reset(self, scenario: Optional[str] = None, fault: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.steps = 0
        self.recent_actions = []
        self.last_action = None
        choice = scenario or fault or str(self.rng.choice(["none", "congestion", "link_failure", "degradation"]))
        self.fault = choice
        self.scenario = choice

        self.metrics = Metrics(
            latency_ms=float(self.rng.normal(20, 2)),
            throughput_mbps=float(self.rng.normal(90, 3)),
            packet_loss_pct=float(max(0.01, self.rng.normal(0.2, 0.08))),
            congestion=float(np.clip(self.rng.normal(0.15, 0.04), 0, 1)),
            link_health=float(np.clip(self.rng.normal(0.98, 0.01), 0, 1)),
            cpu_util_pct=float(np.clip(self.rng.normal(25.0, 3.0), 10, 50)),
        )
        if self.fault == "congestion":
            self.metrics.latency_ms += 95
            self.metrics.throughput_mbps -= 48
            self.metrics.packet_loss_pct += 5.5
            self.metrics.congestion = 0.92
            self.metrics.cpu_util_pct = 80.0
        elif self.fault in ("link_failure", "link failure"):
            self.metrics.latency_ms += 65
            self.metrics.throughput_mbps -= 58
            self.metrics.packet_loss_pct += 7.0
            self.metrics.link_health = 0.15
        elif self.fault == "degradation":
            self.metrics.latency_ms += 45
            self.metrics.throughput_mbps -= 28
            self.metrics.packet_loss_pct += 2.8
            self.metrics.link_health = 0.62

        self.metrics.latency_ms = max(1.0, self.metrics.latency_ms)
        self.metrics.throughput_mbps = max(1.0, self.metrics.throughput_mbps)
        self.metrics.packet_loss_pct = min(10.0, self.metrics.packet_loss_pct)
        return self._state(), {
            "scenario": self.scenario,
            "fault": self.fault,
            "metrics": self.metrics.as_dict(),
        }

    def step(self, action: Action, anomaly_info: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = Action(int(action))
        before = self.metrics.as_dict()
        self.steps += 1
        self.recent_actions.append(int(action))
        self.last_action = action

        m = self.metrics
        if action == Action.REROUTE:
            m.latency_ms *= 0.58
            m.packet_loss_pct *= 0.55
            m.throughput_mbps *= 1.12
            m.congestion *= 0.58
        elif action == Action.QOS_ADJUST:
            m.latency_ms *= 0.82
            m.packet_loss_pct *= 0.70
        elif action == Action.TRAFFIC_SHAPE:
            m.congestion *= 0.75
            m.throughput_mbps *= 0.95
        elif action == Action.FAILOVER:
            m.link_health = 0.95
            m.latency_ms = max(20.0, m.latency_ms * 0.60)
            m.throughput_mbps = max(80.0, m.throughput_mbps)
        elif action == Action.SCALE_RESOURCE:
            m.throughput_mbps = min(100.0, m.throughput_mbps * 1.30)
            m.congestion *= 0.70
        elif action == Action.THROTTLE:
            m.congestion *= 0.65
            m.latency_ms *= 0.85

        after = self.metrics.as_dict()
        from .reward import compute_reward
        reward = compute_reward(before, after, action, recent_actions=self.recent_actions)
        done = bool(self.steps >= self.max_steps or self.steps >= 1)
        truncated = False
        info = {
            "scenario": self.scenario,
            "fault": self.fault,
            "action": ACTION_NAMES[action],
            "action_name": ACTION_NAMES[action],
            "metrics_before": before,
            "metrics_after": after,
            "latency_improvement": before["latency_ms"] - after["latency_ms"],
        }
        return self._state(anomaly_info=anomaly_info), reward, done, truncated, info
