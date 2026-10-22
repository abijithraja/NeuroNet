import numpy as np


ACTION_COST = {
    0: -2,    # reroute
    1: -1,    # qos_adjust
    2: -5,    # scale_slice  (heavyweight — penalise overuse)
    3: -3,    # throttle
    4: -1,    # traffic_shaping
    5: -3,    # failover
    6: -10,   # escalate     (very expensive — train agent to avoid)
    7: -0.5,  # monitor
}


def compute_reward(metrics_before, action, metrics_after, sla_penalty=0):
    latency_delta   = metrics_before['latency_ms']      - metrics_after['latency_ms']
    throughput_delta= metrics_after['throughput_gbps']  - metrics_before['throughput_gbps']
    loss_delta      = metrics_before['packet_loss_pct'] - metrics_after['packet_loss_pct']

    latency_reward    = np.clip(latency_delta   * 2.0, -50,  50)
    throughput_reward = np.clip(throughput_delta * 5.0, -20,  30)
    loss_reward       = np.clip(loss_delta       * 5.0, -20,  20)
    action_reward     = ACTION_COST.get(action, -1)
    sla_reward        = -sla_penalty * 100

    R = (
        0.50 * latency_reward    +
        0.20 * throughput_reward +
        0.15 * loss_reward       +
        0.10 * action_reward     +
        1.00 * sla_reward
    )
    return float(np.clip(R, -100, 100))