"""Transparent multi-objective reward. Tune weights using experiments, not guesswork."""
def compute_reward(before, after, action, recent_actions=None, sla_penalty=0.0):
    latency = min((before["latency_ms"] - after["latency_ms"]) * 0.20, 50.0)
    throughput = min((after["throughput_mbps"] - before["throughput_mbps"]) * 0.12, 30.0)
    loss = min((before["packet_loss_pct"] - after["packet_loss_pct"]) * 1.2, 20.0)
    congestion = min((before["congestion"] - after["congestion"]) * 8.0, 15.0)
    health = (after["link_health"] - before["link_health"]) * 5.0
    costs = {
        "monitor": -0.1, "reroute": -1.0, "qos_adjust": -0.5,
        "traffic_shape": -0.8, "failover": -1.5,
        "scale_resource": -2.0, "rollback": -1.0,
        "escalate": -0.2, "escalate_to_human": -0.2
    }
    repeat_penalty = -2.0 if action in (recent_actions or [])[-3:] else 0.0
    return float(latency + throughput + loss + congestion + health
                 + costs.get(str(action).lower(), -1.0)
                 + repeat_penalty - 100.0 * float(sla_penalty))
