"""Main entry point for NeuroNet end-to-end self-healing network control loop.

Integrates Telemetry Ingestion, LSTM Autoencoder + Isolation Forest Anomaly Detection,
Dueling DQN Decision Engine, Confidence-based Safety Gate, Remediation, and Explainability.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path so imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import List, Dict, Any
import numpy as np

from config import CONFIG, LOG_DIR, MODEL_DIR
from src.neuronet.actions import Action, ACTION_NAMES, N_ACTIONS
from src.neuronet.network_sim import RealisticNetworkEnv
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.ns3_csv import load_ns3_csv
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.anomaly_detector import AnomalyDetector
from src.neuronet.safety import apply_safety_gate
from src.neuronet.explainability import generate_decision_explanation
from src.neuronet.audit import log_decision, append_decision_log_json
from src.neuronet.telemetry import make_state, validate_metrics

def train_baseline_detector(seed: int = 42) -> AnomalyDetector:
    """Trains the LSTM Autoencoder + Isolation Forest ensemble on nominal network telemetry."""
    rng = np.random.default_rng(seed)
    normal_samples: List[Dict[str, float]] = []
    for _ in range(300):
        normal_samples.append({
            "latency_ms": max(5.0, float(rng.normal(20.0, 2.5))),
            "throughput_mbps": float(np.clip(rng.normal(92.0, 3.0), 80.0, 100.0)),
            "packet_loss_pct": float(max(0.01, rng.normal(0.12, 0.04))),
            "congestion": float(np.clip(rng.normal(0.14, 0.03), 0.05, 0.30)),
            "link_health": float(np.clip(rng.normal(0.98, 0.01), 0.90, 1.0)),
            "cpu_util_pct": float(np.clip(rng.normal(24.0, 4.0), 10.0, 45.0)),
        })
    detector = AnomalyDetector(random_state=seed, anomaly_threshold=0.70)
    detector.fit(normal_samples, epochs=6)
    return detector


def main():
    parser = argparse.ArgumentParser(description="NeuroNet Autonomous Self-Healing Control Plane")
    parser.add_argument("--episodes", type=int, default=20, help="Number of simulation episodes/incidents to execute")
    parser.add_argument("--env", type=str, choices=["realistic", "toy", "ns3"], default="realistic", help="Network environment backend")
    parser.add_argument("--scenario", type=str, default="all", help="Failure scenario: congestion, link_failure, routing_flap, attack_ddos, degradation, normal, or all")
    parser.add_argument("--model", type=str, default="", help="Path to custom model checkpoint")
    parser.add_argument("--threshold", type=float, default=CONFIG.confidence_threshold, help="Safety gate confidence threshold (default 0.75)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_log = LOG_DIR / "decisions.jsonl"
    json_log = LOG_DIR / "decision_log.json"
    root_json_log = Path("decision_log.json")

    print("\n" + "=" * 65)
    print("      NEURONET : AUTONOMOUS SELF-HEALING NETWORK CONTROL")
    print("=" * 65)
    print(f" Environment : {args.env.upper()}")
    print(f" Episodes    : {args.episodes}")
    print(f" Scenario    : {args.scenario}")
    print(f" Safety Gate : Confidence Threshold >= {args.threshold:.1%}")
    print("=" * 65 + "\n")

    # 1. Initialize Anomaly Detector
    print("[1/3] Calibrating Anomaly Detector (LSTM Autoencoder + Isolation Forest)...")
    detector = train_baseline_detector(CONFIG.seed)
    print("      Detector trained successfully on nominal telemetry.")

    # 2. Initialize DQN Agent
    print(f"[2/3] Initializing Dueling DQN Policy (Device: {CONFIG.device})...")
    agent = DQNAgent(
        state_dim=CONFIG.state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=CONFIG.hidden_dim,
        capacity=CONFIG.replay_capacity,
        batch_size=CONFIG.batch_size,
        gamma=CONFIG.gamma,
        lr=CONFIG.learning_rate,
        device=CONFIG.device,
        seed=CONFIG.seed,
    )

    model_path = args.model
    if not model_path:
        default_model = MODEL_DIR / "dueling_dqn.pt"
        baseline_model = MODEL_DIR / "pretrained_dqn.pt"
        if default_model.exists():
            model_path = str(default_model)
        elif baseline_model.exists():
            model_path = str(baseline_model)

    if model_path and Path(model_path).exists():
        print(f"      Loading model weights from: {model_path}")
        agent.load(model_path)
    else:
        print("      Operating with fresh policy weights (online exploration).")

    # 3. Initialize Environment
    print(f"[3/3] Initializing {args.env} environment backend...")
    if args.env == "toy":
        env = ToyNetworkEnv(seed=CONFIG.seed, num_nodes=CONFIG.num_nodes)
    elif args.env == "ns3":
        env = RealisticNetworkEnv(seed=CONFIG.seed, num_nodes=CONFIG.num_nodes)
    else:
        env = RealisticNetworkEnv(seed=CONFIG.seed, num_nodes=CONFIG.num_nodes)

    print("\n--- Beginning Autonomous Incident & Remediation Loop ---\n")
    print(f"{'Ep':<4} | {'Scenario':<13} | {'Anomaly':<7} | {'Action':<15} | {'Conf':<6} | {'Gate':<10} | {'Latency Fix':<12} | {'Reward':<7}")
    print("-" * 88)

    autonomous_count = 0
    escalated_count = 0
    total_rewards = []
    latency_improvements = []

    for ep in range(1, args.episodes + 1):
        # Scenario reset
        scen = None if args.scenario == "all" else args.scenario
        state, info = env.reset(scenario=scen)
        metrics_before = info["metrics"]

        # 1. Anomaly Detection Step
        anomaly = detector.predict(metrics_before)

        # 2. State Encoding (combining telemetry + anomaly features)
        state_vec = make_state(metrics_before, anomaly_info=anomaly, recent_actions=env.recent_actions)

        # 3. RL Decision & Softmax Confidence
        action_id, confidence, q_values = agent.act_with_confidence(state_vec, epsilon=0.05)
        proposed_action = Action(action_id)

        # 4. Safety Gate Verification
        gate = apply_safety_gate(
            proposed_action,
            confidence,
            threshold=args.threshold,
            metrics_before=metrics_before,
            recent_actions=env.recent_actions,
            allow_autonomy=True,
        )

        # Execute proposed action if approved, otherwise escalate
        executed_action = proposed_action if gate.approved else Action.ESCALATE

        # 5. Environment Remediation Execution
        next_state, reward, done, truncated, step_info = env.step(executed_action, anomaly_info=anomaly)
        metrics_after = step_info["metrics_after"]
        lat_fix = step_info.get("latency_improvement", metrics_before["latency_ms"] - metrics_after["latency_ms"])

        # 6. Experience Replay & Learning Step
        agent.remember(state_vec, int(executed_action), reward, next_state, done)
        loss = agent.learn()

        # 7. Explainability & Justification
        explanation = generate_decision_explanation(
            metrics_before=metrics_before,
            anomaly_info=anomaly,
            proposed_action=proposed_action,
            confidence=confidence,
            gate_result=gate.as_dict(),
            q_values=q_values,
        )

        if gate.approved:
            autonomous_count += 1
            gate_str = "APPROVED"
        else:
            escalated_count += 1
            gate_str = "ESCALATED"

        total_rewards.append(reward)
        latency_improvements.append(lat_fix)

        # 8. Record audit entry
        record = {
            "episode": ep,
            "scenario": info.get("scenario", info.get("fault", "unknown")),
            "anomaly_type": info.get("scenario", info.get("fault", "unknown")),
            "is_anomaly": anomaly["is_anomaly"],
            "anomaly_score": round(anomaly["anomaly_score"], 3),
            "action": ACTION_NAMES[executed_action],
            "proposed_action": ACTION_NAMES[proposed_action],
            "confidence": round(confidence, 3),
            "escalated": not gate.approved,
            "safety_gate": gate.as_dict(),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "latency_improvement_ms": round(lat_fix, 2),
            "reward": round(reward, 2),
            "explanation": explanation,
            "loss": round(loss, 4) if loss is not None else None,
        }

        log_decision(jsonl_log, record)
        append_decision_log_json(json_log, record)
        append_decision_log_json(root_json_log, record)

        act_label = ACTION_NAMES[executed_action]
        scen_label = str(info.get("scenario", info.get("fault", "normal")))
        print(f"{ep:03d}  | {scen_label:<13} | {str(anomaly['is_anomaly']):<7} | {act_label:<15} | {confidence:4.1%} | {gate_str:<10} | {lat_fix:+6.1f} ms    | {reward:+6.1f}")

    print("-" * 88)
    total_episodes = len(total_rewards)
    auto_rate = (autonomous_count / max(1, total_episodes)) * 100.0
    mean_reward = float(np.mean(total_rewards))
    mean_lat_fix = float(np.mean(latency_improvements))

    print("\n=== Executive Summary of Run ===")
    print(f" Total Incidents Handled : {total_episodes}")
    print(f" Autonomous Decisions    : {autonomous_count} ({auto_rate:.1f}%)")
    print(f" Escalations to Human NOC: {escalated_count} ({100.0 - auto_rate:.1f}%)")
    print(f" Mean Latency Recovered  : {mean_lat_fix:+.1f} ms")
    print(f" Mean Episode Reward     : {mean_reward:+.2f}")
    print(f" Audit Logs Saved To     : {json_log} & {jsonl_log}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
