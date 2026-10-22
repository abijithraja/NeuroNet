"""Comprehensive Evaluation & Benchmarking for NeuroNet.

Evaluates the integrated agent across 6 realistic scenarios: Normal, Congestion,
Link Failure, Routing Flap, DDoS Attack, and Degradation.
Compares NeuroNet against 3 baseline strategies:
1. Static Threshold Rule Engine (Industry Baseline)
2. Random Action Agent
3. Passive No-Op Monitor
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np

from config import CONFIG, MODEL_DIR, LOG_DIR
from src.neuronet.actions import Action, ACTION_NAMES, N_ACTIONS
from src.neuronet.network_sim import RealisticNetworkEnv
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.safety import apply_safety_gate
from src.neuronet.telemetry import make_state
from main import train_baseline_detector


def static_rule_engine(metrics: Dict[str, float]) -> Action:
    """Conventional threshold-based NOC rule engine."""
    if metrics.get("link_health", 1.0) < 0.30:
        return Action.FAILOVER
    if metrics.get("latency_ms", 0) > 65.0:
        return Action.REROUTE
    if metrics.get("packet_loss_pct", 0) > 3.0:
        return Action.QOS_ADJUST
    if metrics.get("congestion", 0) > 0.80:
        return Action.TRAFFIC_SHAPE
    return Action.MONITOR


def evaluate(
    model_path: str,
    episodes_per_scenario: int = 15,
    threshold: float = 0.75,
    seed: int = 123,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("      NEURONET : COMPREHENSIVE BENCHMARK & EVALUATION")
    print("=" * 70)
    print(f" Model Path    : {model_path}")
    print(f" Safety Gate   : Confidence >= {threshold:.1%}")
    print(f" Episodes/Scen : {episodes_per_scenario}")
    print("=" * 70 + "\n")

    detector = train_baseline_detector(seed)
    env = RealisticNetworkEnv(seed=seed, num_nodes=CONFIG.num_nodes)

    agent = DQNAgent(
        state_dim=CONFIG.state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=CONFIG.hidden_dim,
        device=CONFIG.device,
        seed=seed,
    )
    if Path(model_path).exists():
        agent.load(model_path)
    else:
        print(f"Warning: Model path {model_path} not found. Running with uninitialized weights.")

    scenarios = RealisticNetworkEnv.SCENARIOS

    strategies = ["NeuroNet RL Agent", "Static Rule Engine", "Random Action", "Passive Monitor"]
    results = {strat: defaultdict(list) for strat in strategies}

    for scen in scenarios:
        for ep in range(episodes_per_scenario):
            # Test each strategy on the exact same starting scenario conditions
            for strat in strategies:
                state, info = env.reset(scenario=scen)
                metrics_before = info["metrics"]
                anomaly = detector.predict(metrics_before)
                state_vec = make_state(metrics_before, anomaly_info=anomaly, recent_actions=env.recent_actions)

                if strat == "NeuroNet RL Agent":
                    action_id, conf, _ = agent.act_with_confidence(state_vec, epsilon=0.0)
                    gate = apply_safety_gate(
                        Action(action_id),
                        conf,
                        threshold=threshold,
                        metrics_before=metrics_before,
                        recent_actions=env.recent_actions,
                    )
                    chosen_action = Action(action_id) if gate.approved else Action.ESCALATE
                    approved = gate.approved

                elif strat == "Static Rule Engine":
                    chosen_action = static_rule_engine(metrics_before)
                    conf = 1.0
                    approved = True

                elif strat == "Random Action":
                    chosen_action = Action(np.random.randint(0, N_ACTIONS))
                    conf = 0.5
                    approved = True

                elif strat == "Passive Monitor":
                    chosen_action = Action.MONITOR
                    conf = 1.0
                    approved = True

                _, reward, _, _, step_info = env.step(chosen_action, anomaly_info=anomaly)

                lat_drop = metrics_before["latency_ms"] - step_info["metrics_after"]["latency_ms"]
                loss_drop = metrics_before["packet_loss_pct"] - step_info["metrics_after"]["packet_loss_pct"]

                results[strat][scen].append({
                    "reward": reward,
                    "latency_drop": lat_drop,
                    "loss_drop": loss_drop,
                    "approved": approved,
                    "action": ACTION_NAMES[chosen_action],
                })

    # Summary calculations
    print(f"{'Strategy':<20} | {'Mean Reward':<12} | {'Lat Fix (ms)':<14} | {'Loss Fix (%)':<14} | {'Autonomy Rate':<14}")
    print("-" * 80)

    summary_table = []
    for strat in strategies:
        all_rewards = [item["reward"] for s in scenarios for item in results[strat][s]]
        all_lats = [item["latency_drop"] for s in scenarios for item in results[strat][s]]
        all_losses = [item["loss_drop"] for s in scenarios for item in results[strat][s]]
        all_approved = [item["approved"] for s in scenarios for item in results[strat][s]]

        mean_rew = float(np.mean(all_rewards))
        mean_lat = float(np.mean(all_lats))
        mean_loss = float(np.mean(all_losses))
        auto_rate = float(np.mean(all_approved)) * 100.0

        summary_table.append({
            "strategy": strat,
            "mean_reward": mean_rew,
            "latency_improvement": mean_lat,
            "loss_reduction": mean_loss,
            "autonomy_rate": auto_rate,
        })
        print(f"{strat:<20} | {mean_rew:<+12.2f} | {mean_lat:<+14.2f} | {mean_loss:<+14.2f} | {auto_rate:13.1f}%")

    # Generate Markdown Report
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_file = LOG_DIR / "EVALUATION_REPORT.md"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("# NeuroNet — Empirical Evaluation & Benchmarking Report\n\n")
        f.write(f"**Evaluated Checkpoint:** `{model_path}`\n\n")
        f.write("## Overall Performance Comparison\n\n")
        f.write("| Strategy | Mean Reward | Latency Recovery (ms) | Packet Loss Reduction (%) | Autonomy / Approval Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for row in summary_table:
            f.write(f"| **{row['strategy']}** | {row['mean_reward']:+.2f} | {row['latency_improvement']:+.2f} ms | {row['loss_reduction']:+.2f}% | {row['autonomy_rate']:.1f}% |\n")
        
        f.write("\n## Scenario-by-Scenario Breakdown (NeuroNet RL Agent)\n\n")
        f.write("| Failure Scenario | Mean Reward | Latency Recovered | Loss Reduced | Autonomy Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for s in scenarios:
            items = results["NeuroNet RL Agent"][s]
            s_rew = np.mean([x["reward"] for x in items])
            s_lat = np.mean([x["latency_drop"] for x in items])
            s_loss = np.mean([x["loss_drop"] for x in items])
            s_auto = np.mean([x["approved"] for x in items]) * 100.0
            f.write(f"| `{s}` | {s_rew:+.2f} | {s_lat:+.2f} ms | {s_loss:+.2f}% | {s_auto:.1f}% |\n")

    print("\n" + "=" * 70)
    print(f" Detailed evaluation report written to: {report_file}")
    print("=" * 70 + "\n")
    return {"summary": summary_table, "report_path": str(report_file)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuroNet Agent against Baselines")
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per scenario (total = episodes * 6)")
    parser.add_argument("--model", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=CONFIG.confidence_threshold)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    m_path = args.model or str(MODEL_DIR / "dueling_dqn.pt")
    if not Path(m_path).exists():
        alt = MODEL_DIR / "pretrained_dqn.pt"
        if alt.exists():
            m_path = str(alt)

    evaluate(m_path, episodes_per_scenario=args.episodes, threshold=args.threshold, seed=args.seed)


if __name__ == "__main__":
    main()
