"""Multi-objective transparent reward function for network remediation."""
from typing import Dict, Any, Optional, List
import numpy as np
from .actions import Action, ACTION_COSTS, ACTION_NAMES

def compute_reward(
    before: Dict[str, float],
    after: Dict[str, float],
    action: Action,
    recent_actions: Optional[List[int]] = None,
    sla_penalty: float = 0.0,
) -> float:
    """Computes a multi-component, bounded scalar reward for network state transition."""
    # 1. Latency delta (positive when latency drops)
    lat_delta = before["latency_ms"] - after["latency_ms"]
    latency_reward = min(lat_delta * 0.40, 50.0)

    # 2. Throughput delta (positive when throughput increases)
    tput_delta = after["throughput_mbps"] - before["throughput_mbps"]
    throughput_reward = min(tput_delta * 0.20, 30.0)

    # 3. Packet loss delta (positive when loss drops)
    loss_delta = before["packet_loss_pct"] - after["packet_loss_pct"]
    loss_reward = min(loss_delta * 2.5, 20.0)

    # 4. Congestion delta (positive when congestion drops)
    cong_delta = before["congestion"] - after["congestion"]
    congestion_reward = min(cong_delta * 12.0, 15.0)

    # 5. Link health recovery
    health_delta = after["link_health"] - before["link_health"]
    health_reward = health_delta * 10.0

    # 6. Action operational cost
    act = Action(int(action))
    cost = ACTION_COSTS.get(act, -0.5)

    # 7. Oscillation penalty (discourages thrashing back-and-forth)
    history = recent_actions or []
    oscillation_penalty = 0.0
    if len(history) >= 2 and int(act) in history[-3:-1]:
        oscillation_penalty = -10.0

    # 8. SLA penalty (e.g., if latency exceeds 80ms or loss > 3% post-action)
    sla_cost = 0.0
    if after["latency_ms"] > 100.0 or after["packet_loss_pct"] > 5.0:
        sla_cost -= 25.0
    sla_cost -= 50.0 * float(sla_penalty)

    total_reward = (
        0.45 * latency_reward
        + 0.20 * throughput_reward
        + 0.15 * loss_reward
        + 0.10 * congestion_reward
        + 0.10 * health_reward
        + cost
        + oscillation_penalty
        + sla_cost
    )

    return float(np.clip(total_reward, -100.0, 100.0))
