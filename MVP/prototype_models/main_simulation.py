import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from telemetry_emulator import NetworkTelemetry
from anomaly_detector   import AnomalyDetector
from dqn_agent          import DQNAgent
from reward_function    import compute_reward

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
BASE_DIR = Path(__file__).resolve().parent

ACTION_NAMES   = ['reroute', 'qos_adjust', 'scale_slice', 'throttle',
                  'traffic_shaping', 'failover', 'escalate', 'monitor']
ACTION_EFFECTS = {
    0: {'latency': -15, 'throughput': +2.0, 'loss': -1.5},
    1: {'latency':  -8, 'throughput':  0.0, 'loss': -0.5},
    2: {'latency': -20, 'throughput': +3.0, 'loss': -2.0},
    3: {'latency': -10, 'throughput': -1.0, 'loss': -1.0},
    4: {'latency':  -5, 'throughput':  0.0, 'loss': -0.3},
    5: {'latency': -30, 'throughput': +1.5, 'loss': -2.5},
    6: {'latency':   0, 'throughput':  0.0, 'loss':  0.0},
    7: {'latency':   0, 'throughput':  0.0, 'loss':  0.0},
}
CONFIDENCE_THRESHOLD = 0.75


class NeuroNetSimulation:
    def __init__(self):
        self.telemetry       = NetworkTelemetry()
        self.detector        = AnomalyDetector(window_size=5)
        self.agent           = DQNAgent(state_dim=8, action_dim=8)
        self.episode         = 0
        self.action_history  = []
        self.decision_log    = []

    def _encode_state(self, result):
        return np.array([
            result['anomaly_score'],
            result['lstm_error'],
            result['iso_score'],
            float(result['features']['latency_anomalous']),
            float(result['features']['loss_anomalous']),
            float(result['features']['congestion_high']),
            np.sin(2 * np.pi * datetime.now().hour / 24),
            (len(self.action_history) % 10) / 10,
        ], dtype=np.float32)

    def run_episode(self, anomaly_type='congestion'):
        m_before = self.telemetry.get_metrics('link_0', inject_anomaly=True,
                                               anomaly_type=anomaly_type)
        result   = self.detector.detect(m_before)

        if not result['is_anomaly']:
            # Force demo mode: still proceed
            result['is_anomaly'] = True

        state          = self._encode_state(result)
        action, conf   = self.agent.act(state)
        escalated      = conf < CONFIDENCE_THRESHOLD

        eff      = ACTION_EFFECTS[action]
        m_after  = {
            'latency_ms':      max(0.0, m_before['latency_ms']      + eff['latency']),
            'throughput_gbps': max(0.0, m_before['throughput_gbps'] + eff['throughput']),
            'packet_loss_pct': max(0.0, m_before['packet_loss_pct'] + eff['loss']),
            'congestion':      max(0.0, min(m_before['congestion']  + eff['throughput'] / 10, 1.0)),
        }

        reward     = compute_reward(m_before, action, m_after)
        next_res   = self.detector.detect(m_after)
        next_state = self._encode_state(next_res)
        done       = result['anomaly_score'] < 0.3

        self.agent.remember(state, action, reward, next_state, done)
        self.agent.train(batch_size=32)
        self.action_history.append(action)

        self.decision_log.append({
            'episode':      self.episode,
            'timestamp':    datetime.now().isoformat(),
            'anomaly_type': anomaly_type,
            'anomaly_score':result['anomaly_score'],
            'action':       ACTION_NAMES[action],
            'confidence':   round(conf, 4),
            'escalated':    escalated,
            'reward':       round(reward, 2),
            'metrics_before': {k: round(v, 2) for k, v in {
                'latency': m_before['latency_ms'],
                'loss':    m_before['packet_loss_pct'],
                'throughput': m_before['throughput_gbps']}.items()},
            'metrics_after':  {k: round(v, 2) for k, v in {
                'latency': m_after['latency_ms'],
                'loss':    m_after['packet_loss_pct'],
                'throughput': m_after['throughput_gbps']}.items()},
        })
        self.episode += 1
        return {
            'reward':              reward,
            'action':              ACTION_NAMES[action],
            'latency_improvement': m_before['latency_ms'] - m_after['latency_ms'],
            'confidence':          conf,
            'escalated':           escalated,
        }


def main():
    print("=== NeuroNet Modulator MVP ===\n")
    sim = NeuroNetSimulation()

    # --- Step 1: generate benign windows for anomaly detector training ---
    print("Step 1: Generating benign training data...")
    benign_data = []
    for _ in range(100):  # Fast demo mode
        window = []
        for _ in range(5):
            m = sim.telemetry.get_metrics('link_0', inject_anomaly=False)
            window.append([m['latency_ms'], m['throughput_gbps'],
                           m['packet_loss_pct'], m['congestion']])
        benign_data.append(np.array(window))

    # --- Step 2: train anomaly detector ---
    print("Step 2: Training anomaly detector (LSTM + IsolationForest)...")
    sim.detector.train(benign_data)
    print("  [OK] Anomaly detector ready\n")

    # --- Step 3: run simulation episodes ---
    print("Step 3: Running 30 simulation episodes...\n")
    console.print("[bold green]Starting AI Decision Engine...[/bold green]\n")
    anomaly_types = ['congestion', 'degradation', 'link_failure']
    results = {t: [] for t in anomaly_types}

    for ep in range(30):  # Fast demo mode
        atype  = anomaly_types[ep % 3]
        result = sim.run_episode(anomaly_type=atype)
        if result is not None:
            results[atype].append(result)

            # Create a visual alert based on whether it was escalated to a human or automated
            if result['escalated']:
                status_color = "[bold red]>> ESCALATED TO NOC[/bold red]"
            else:
                status_color = "[bold green]>> AUTONOMOUS ACTION[/bold green]"

            console.print(Panel(
                f"[cyan]Episode {ep:02d}[/cyan] | Anomaly: [bold yellow]{atype.upper()}[/bold yellow]\n"
                f"Action Selected: [bold magenta]{result['action'].upper()}[/bold magenta] (Confidence: {result['confidence']:.2f})\n"
                f"Result: Latency improved by [bold green]{result['latency_improvement']:+.1f}ms[/bold green]\n"
                f"Reward: [bold cyan]{result['reward']:+.2f}[/bold cyan]\n"
                f"Status: {status_color}",
                title="[bold white]NeuroNet AI Decision Engine[/bold white]",
                expand=False
            ))

    # --- Step 4: print summary ---
    print("\n=== RESULTS ===\n")
    for t in anomaly_types:
        if not results[t]:
            continue
        rewards  = [r['reward']              for r in results[t]]
        confs    = [r['confidence']          for r in results[t]]
        lats     = [r['latency_improvement'] for r in results[t]]
        escalated= sum(r['escalated']        for r in results[t])
        print(f"{t.upper()} ({len(results[t])} episodes)")
        print(f"  Mean reward:     {np.mean(rewards):+.2f}  (σ={np.std(rewards):.2f})")
        print(f"  Mean confidence: {np.mean(confs):.3f}")
        print(f"  Mean Δlatency:   {np.mean(lats):+.1f} ms")
        print(f"  Escalation rate: {escalated}/{len(results[t])} "
              f"({100*escalated/max(len(results[t]),1):.0f}%)\n")

    # --- Step 5: save outputs ---
    with (BASE_DIR / 'decision_log.json').open('w') as f:
        json.dump(sim.decision_log, f, indent=2)
    os.makedirs(BASE_DIR / 'models', exist_ok=True)
    sim.agent.save(str(BASE_DIR / 'models' / 'pretrained_dqn.pt'))
    print("[OK] decision_log.json saved")
    print("[OK] models/pretrained_dqn.pt saved")
    print("[OK] Simulation complete")


if __name__ == '__main__':
    main()