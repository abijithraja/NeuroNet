"""Unit tests for NeuroNet core modules."""
import unittest
import numpy as np
from src.neuronet.actions import Action, ACTION_NAMES, N_ACTIONS
from src.neuronet.telemetry import validate_metrics, make_state, STATE_DIM
from src.neuronet.anomaly_detector import AnomalyDetector
from src.neuronet.network_sim import RealisticNetworkEnv
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.safety import apply_safety_gate
from src.neuronet.reward import compute_reward
from src.neuronet.explainability import generate_decision_explanation
from src.neuronet.dqn_agent import DQNAgent

class TestNeuroNetCore(unittest.TestCase):
    def test_telemetry_validation(self):
        row = {
            "latency_ms": 22.0,
            "throughput_mbps": 88.0,
            "packet_loss_pct": 0.15,
            "congestion": 0.12,
            "link_health": 0.98,
            "cpu_util_pct": 25.0,
        }
        val = validate_metrics(row)
        self.assertEqual(val["latency_ms"], 22.0)
        state = make_state(row)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertTrue(all(np.isfinite(state)))

    def test_invalid_telemetry(self):
        with self.assertRaises(ValueError):
            validate_metrics({
                "latency_ms": -10.0,
                "throughput_mbps": 80.0,
                "packet_loss_pct": 0.2,
                "congestion": 0.1,
                "link_health": 0.9,
            })

    def test_anomaly_detector_ensemble(self):
        detector = AnomalyDetector(window_size=4, random_state=42)
        benign_data = [
            {
                "latency_ms": 20.0 + i * 0.1,
                "throughput_mbps": 90.0,
                "packet_loss_pct": 0.1,
                "congestion": 0.1,
                "link_health": 0.98,
                "cpu_util_pct": 22.0,
            }
            for i in range(25)
        ]
        detector.fit(benign_data, epochs=2)
        self.assertTrue(detector.is_trained)

        # Warm up buffer to window_size
        for i in range(detector.window_size):
            detector.predict(benign_data[i])

        # Predict anomaly spike
        spike = {
            "latency_ms": 160.0,
            "throughput_mbps": 25.0,
            "packet_loss_pct": 8.0,
            "congestion": 0.95,
            "link_health": 0.15,
            "cpu_util_pct": 95.0,
        }
        res_spike = detector.predict(spike)
        self.assertGreater(res_spike["anomaly_score"], 0.4)

    def test_realistic_network_env(self):
        env = RealisticNetworkEnv(seed=42)
        state, info = env.reset(scenario="congestion")
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(info["scenario"], "congestion")
        self.assertGreater(info["metrics"]["latency_ms"], 50.0)

        next_state, reward, done, truncated, step_info = env.step(Action.REROUTE)
        self.assertEqual(next_state.shape, (STATE_DIM,))
        self.assertTrue(np.isfinite(reward))
        self.assertIn("latency_improvement", step_info)
        self.assertGreater(step_info["latency_improvement"], 0.0)

    def test_toy_network_env(self):
        env = ToyNetworkEnv(seed=1)
        state, info = env.reset(fault="congestion")
        self.assertEqual(state.shape, (STATE_DIM,))
        ns, reward, done, truncated, step_info = env.step(Action.REROUTE)
        self.assertEqual(ns.shape, (STATE_DIM,))
        self.assertTrue(np.isfinite(reward))

    def test_safety_gate(self):
        # High confidence approval
        res_approved = apply_safety_gate(Action.REROUTE, 0.85, threshold=0.75)
        self.assertTrue(res_approved.approved)
        self.assertEqual(res_approved.action, "reroute")

        # Low confidence escalation
        res_low_conf = apply_safety_gate(Action.REROUTE, 0.60, threshold=0.75)
        self.assertFalse(res_low_conf.approved)
        self.assertEqual(res_low_conf.action, "escalate_to_human")

        # Explicit escalation
        res_esc = apply_safety_gate(Action.ESCALATE, 0.90)
        self.assertFalse(res_esc.approved)

        # Thrashing guard
        res_thrash = apply_safety_gate(Action.REROUTE, 0.95, recent_actions=[Action.REROUTE, Action.REROUTE])
        self.assertFalse(res_thrash.approved)
        self.assertIn("thrashing", res_thrash.reason)

    def test_reward_function(self):
        before = {"latency_ms": 120.0, "throughput_mbps": 40.0, "packet_loss_pct": 5.0, "congestion": 0.9, "link_health": 0.5}
        after = {"latency_ms": 30.0, "throughput_mbps": 85.0, "packet_loss_pct": 0.2, "congestion": 0.2, "link_health": 0.95}
        r = compute_reward(before, after, Action.REROUTE)
        self.assertGreater(r, 0.0)
        self.assertLessEqual(r, 100.0)
        self.assertGreaterEqual(r, -100.0)

    def test_dqn_agent_dueling(self):
        agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, seed=42)
        dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
        act, conf, q = agent.act_with_confidence(dummy_state)
        self.assertTrue(0 <= act < N_ACTIONS)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(q), N_ACTIONS)

    def test_explainability(self):
        metrics_before = {"latency_ms": 110.0, "congestion": 0.85}
        anomaly_info = {
            "is_anomaly": True,
            "anomaly_score": 0.85,
            "feature_attributions": {"latency": 0.6, "congestion": 0.4},
        }
        gate_result = {"approved": True, "reason": "Passed"}
        exp = generate_decision_explanation(metrics_before, anomaly_info, Action.REROUTE, 0.88, gate_result)
        self.assertIn("anomaly_summary", exp)
        self.assertIn("action_rationale", exp)
        self.assertEqual(exp["top_contributing_feature"], "latency")

if __name__ == "__main__":
    unittest.main()
