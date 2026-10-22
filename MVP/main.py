import argparse
import sys
from pathlib import Path
import numpy as np

from config import CONFIG, LOG_DIR
from src.neuronet.actions import N_ACTIONS, ACTION_NAMES, Action
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.anomaly_detector import AnomalyDetector
from src.neuronet.safety import apply_safety_gate
from src.neuronet.audit import log_decision

def build_detector(seed=42):
    rng = np.random.default_rng(seed)
    normal = []
    for _ in range(500):
        normal.append({
            "latency_ms": max(1, rng.normal(20, 3)),
            "throughput_mbps": np.clip(rng.normal(90, 5), 60, 100),
            "packet_loss_pct": np.clip(rng.normal(0.2, 0.08), 0, 1),
            "congestion": np.clip(rng.normal(0.15, 0.05), 0, 0.4),
            "link_health": np.clip(rng.normal(0.98, 0.01), 0.9, 1),
        })
    return AnomalyDetector(random_state=seed).fit(normal)

def main():
    parser = argparse.ArgumentParser(description="NeuroNet toy-loop starter (not ns-3)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    agent = DQNAgent(CONFIG.state_dim, N_ACTIONS, hidden_dim=CONFIG.hidden_dim,
                     capacity=CONFIG.replay_capacity, batch_size=CONFIG.batch_size,
                     gamma=CONFIG.gamma, lr=CONFIG.learning_rate, device=CONFIG.device,
                     target_update_every=CONFIG.target_update_every, seed=CONFIG.seed)
    if args.model:
        agent.load(args.model)
    detector = build_detector()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "decisions.jsonl"
    env = ToyNetworkEnv(seed=CONFIG.seed, num_nodes=CONFIG.num_nodes)
    print("=== NeuroNet starter: synthetic Python environment ===")
    print(f"Device: {agent.device}; episodes: {args.episodes}")
    print("Note: synthetic demo only. Not ns-3 data or validated results.")
    for ep in range(args.episodes):
        state, info = env.reset()
        metrics = info["metrics"]
        anomaly = detector.predict(metrics)
        action_id = agent.act(state, epsilon=0.0)
        confidence = agent.heuristic_confidence(state)
        gate = apply_safety_gate(action_id, confidence, CONFIG.confidence_threshold)
        chosen = Action.ESCALATE if not gate.approved else Action(action_id)
        next_state, reward, done, _, step_info = env.step(chosen)
        agent.remember(state, int(chosen), reward, next_state, done)
        loss = agent.learn()
        record = {
            "episode": ep, "environment": "python_toy_not_ns3",
            "fault": info["fault"], "anomaly": anomaly,
            "proposed_action": ACTION_NAMES[Action(action_id)],
            "executed_simulated_action": ACTION_NAMES[chosen],
            "confidence_heuristic": confidence, "safety_gate": gate.__dict__,
            "reward": reward, "metrics_before": step_info["metrics_before"],
            "metrics_after": step_info["metrics_after"], "loss": loss,
        }
        log_decision(log_path, record)
        print(f"[{ep+1:03d}] fault={info['fault']:<13} anomaly={anomaly['is_anomaly']} "
              f"proposed={ACTION_NAMES[Action(action_id)]:<14} "
              f"gate={'approved' if gate.approved else 'escalated':<9} reward={reward:7.2f}")
    print(f"Decision log: {log_path}")
    print("Interpret results cautiously; this run validates plumbing, not research claims.")

if __name__ == "__main__":
    main()
