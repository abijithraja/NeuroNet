"""Upgraded 20-Node Physical Network Environment for NeuroNet RL.

Integrates with the modular network_engine, providing actual queue dynamics,
link bandwidths, and physical packet drops without synthetic hardcoded numbers.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .actions import Action, ACTION_NAMES, N_ACTIONS
from .reward import compute_reward
from network_engine.simulation import NetworkEngine


class ToyNetworkEnv:
    def __init__(self, seed: int = 42, max_steps: int = 20, num_nodes: int = 20):
        self.engine = NetworkEngine(seed=seed)
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.steps = 0
        self.recent_actions = []
        self.last_action = None

    @property
    def state_dim(self) -> int:
        return 8

    def _state(self) -> np.ndarray:
        m = self.engine.metrics
        scen = self.engine.scenario
        return np.asarray([
            np.clip(m["latency_ms"] / 200.0, 0, 1),
            np.clip(m["throughput_mbps"] / 100.0, 0, 1),
            np.clip(m["packet_loss_pct"] / 10.0, 0, 1),
            np.clip(m["congestion"], 0, 1),
            np.clip(m["link_health"], 0, 1),
            float(scen != "nominal"),
            float(scen == "link_failure"),
            min(len(self.recent_actions), 3) / 3.0,
        ], dtype=np.float32)

    def reset(self, fault: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.steps = 0
        self.recent_actions = []
        self.last_action = None
        self.engine.reset()

        f = fault or self.engine.rng.choice(["none", "congestion", "link_failure", "degradation"])
        if f and f not in ("none", "nominal"):
            self.engine.inject_fault(f)

        return self._state(), {"fault": self.engine.scenario, "metrics": self.engine.metrics.copy()}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_enum = Action(int(action))
        before = self.engine.metrics.copy()
        self.steps += 1

        # Execute on physical simulation
        details = self.engine.remediate(action_enum.name)
        after = self.engine.metrics.copy()

        # Compute multi-objective SLA reward from real physical changes
        reward = compute_reward(
            before=before,
            after=after,
            action=action_enum.name.lower(),
            recent_actions=self.recent_actions
        )

        self.recent_actions.append(int(action))
        self.last_action = int(action)

        done = self.steps >= 1 or self.steps >= self.max_steps
        return self._state(), float(reward), done, False, {
            "fault": self.engine.scenario,
            "action": action_enum.name.lower(),
            "details": details,
            "metrics_before": before,
            "metrics_after": after,
            "active_path": self.engine.routing.path_description(),
        }
