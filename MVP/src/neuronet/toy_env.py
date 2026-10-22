"""A small Python toy environment for validating the RL code path.

This is NOT an ns-3 network simulator. Its numbers are synthetic and must not
be reported as real network measurements.
"""
from dataclasses import dataclass
import numpy as np
from .actions import Action, N_ACTIONS

@dataclass
class Metrics:
    latency_ms: float
    throughput_mbps: float
    packet_loss_pct: float
    congestion: float
    link_health: float

    def as_dict(self):
        return {
            "latency_ms": float(self.latency_ms),
            "throughput_mbps": float(self.throughput_mbps),
            "packet_loss_pct": float(self.packet_loss_pct),
            "congestion": float(self.congestion),
            "link_health": float(self.link_health),
        }

class ToyNetworkEnv:
    def __init__(self, seed=42, num_nodes=20, max_steps=20):
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.steps = 0
        self.metrics = None
        self.fault = "none"
        self.recent_actions = []
        self.last_action = None

    @property
    def state_dim(self):
        return 8

    def _state(self):
        m = self.metrics
        # Fixed scale, bounded to roughly [0, 1].
        return np.asarray([
            np.clip(m.latency_ms / 200.0, 0, 1),
            np.clip(m.throughput_mbps / 100.0, 0, 1),
            np.clip(m.packet_loss_pct / 10.0, 0, 1),
            np.clip(m.congestion, 0, 1),
            np.clip(m.link_health, 0, 1),
            float(self.fault != "none"),
            float(self.fault == "link_failure"),
            min(len(self.recent_actions), 3) / 3.0,
        ], dtype=np.float32)

    def reset(self, fault=None):
        self.steps = 0
        self.recent_actions = []
        self.last_action = None
        self.fault = fault or self.rng.choice(["none", "congestion", "link_failure", "degradation"])
        self.metrics = Metrics(
            latency_ms=float(self.rng.normal(20, 2)),
            throughput_mbps=float(self.rng.normal(90, 3)),
            packet_loss_pct=float(max(0, self.rng.normal(0.2, 0.08))),
            congestion=float(np.clip(self.rng.normal(0.15, 0.04), 0, 1)),
            link_health=float(np.clip(self.rng.normal(0.98, 0.01), 0, 1)),
        )
        if self.fault == "congestion":
            self.metrics.latency_ms += 95
            self.metrics.throughput_mbps -= 48
            self.metrics.packet_loss_pct += 5.5
            self.metrics.congestion = 0.92
        elif self.fault == "link_failure":
            self.metrics.latency_ms += 65
            self.metrics.throughput_mbps -= 58
            self.metrics.packet_loss_pct += 7.0
            self.metrics.link_health = 0.15
        elif self.fault == "degradation":
            self.metrics.latency_ms += 45
            self.metrics.throughput_mbps -= 28
            self.metrics.packet_loss_pct += 2.8
            self.metrics.link_health = 0.62
        self.metrics.latency_ms = max(1, self.metrics.latency_ms)
        self.metrics.throughput_mbps = max(1, self.metrics.throughput_mbps)
        self.metrics.packet_loss_pct = min(10, self.metrics.packet_loss_pct)
        return self._state(), {"fault": self.fault, "metrics": self.metrics.as_dict()}

    def step(self, action):
        action = Action(int(action))
        before = self.metrics.as_dict()
        self.steps += 1
        # Synthetic deterministic response model with small noise.
        m = self.metrics
        if action == Action.REROUTE:
            m.latency_ms *= 0.58
            m.packet_loss_pct *= 0.55
            m.throughput_mbps *= 1.12
            m.congestion *= 0.58
        elif action == Action.QOS_ADJUST:
            m.latency_ms *= 0.82
            m.packet_loss_pct *= 0.88
            m.throughput_mbps *= 1.03
            m.congestion *= 0.86
        elif action == Action.TRAFFIC_SHAPE:
            m.latency_ms *= 0.75
            m.packet_loss_pct *= 0.68
            m.throughput_mbps *= 0.88
            m.congestion *= 0.62
        elif action == Action.FAILOVER:
            m.latency_ms *= 0.62
            m.packet_loss_pct *= 0.45
            m.throughput_mbps *= 0.92
            m.link_health = max(m.link_health, 0.88)
            m.congestion *= 0.72
        elif action == Action.SCALE_RESOURCE:
            m.latency_ms *= 0.72
            m.packet_loss_pct *= 0.8
            m.throughput_mbps *= 1.22
            m.congestion *= 0.64
        elif action == Action.ROLLBACK:
            # In this toy model rollback safely restores a moderate state.
            m.latency_ms = max(25, m.latency_ms * 0.9)
            m.packet_loss_pct = min(2.0, m.packet_loss_pct * 0.7)
            m.throughput_mbps = max(60, m.throughput_mbps * 0.9)
            m.link_health = max(0.85, m.link_health)
            m.congestion *= 0.8
        elif action == Action.ESCALATE:
            # No automatic change; human review is represented as a safe no-op.
            pass
        # MONITOR is also a no-op.
        m.latency_ms = float(max(1, m.latency_ms + self.rng.normal(0, 1.0)))
        m.throughput_mbps = float(np.clip(m.throughput_mbps + self.rng.normal(0, 1.0), 1, 100))
        m.packet_loss_pct = float(np.clip(m.packet_loss_pct + self.rng.normal(0, 0.08), 0, 10))
        m.congestion = float(np.clip(m.congestion + self.rng.normal(0, 0.015), 0, 1))
        m.link_health = float(np.clip(m.link_health, 0, 1))
        # Reward favors lower latency/loss/congestion and higher throughput/health.
        reward = (
            (before["latency_ms"] - m.latency_ms) * 0.20
            + (m.throughput_mbps - before["throughput_mbps"]) * 0.12
            + (before["packet_loss_pct"] - m.packet_loss_pct) * 1.2
            + (before["congestion"] - m.congestion) * 8.0
            + (m.link_health - before["link_health"]) * 5.0
            - {Action.MONITOR: 0.1, Action.REROUTE: 1.0, Action.QOS_ADJUST: 0.5,
               Action.TRAFFIC_SHAPE: 0.8, Action.FAILOVER: 1.5,
               Action.SCALE_RESOURCE: 2.0, Action.ROLLBACK: 1.0,
               Action.ESCALATE: 0.2}[action]
        )
        if int(action) in self.recent_actions[-3:]:
            reward -= 2.0
        self.recent_actions.append(int(action))
        self.last_action = int(action)
        # End an episode after one intervention or max steps.
        done = self.steps >= 1 or self.steps >= self.max_steps
        return self._state(), float(reward), done, False, {
            "fault": self.fault, "action": action.name.lower(),
            "metrics_before": before, "metrics_after": m.as_dict(),
        }
