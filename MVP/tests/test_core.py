import unittest
import numpy as np
from src.neuronet.telemetry import validate_metrics, make_state
from src.neuronet.toy_env import ToyNetworkEnv
from src.neuronet.safety import apply_safety_gate
from src.neuronet.actions import Action

class TestCore(unittest.TestCase):
    def test_telemetry_validation(self):
        row = {"latency_ms": 20, "throughput_mbps": 80, "packet_loss_pct": 0.2,
               "congestion": 0.1, "link_health": 0.99}
        self.assertEqual(validate_metrics(row)["latency_ms"], 20)
        self.assertEqual(make_state(row).shape, (8,))
    def test_invalid_telemetry(self):
        with self.assertRaises(ValueError):
            validate_metrics({"latency_ms": -1, "throughput_mbps": 80,
                "packet_loss_pct": 0.2, "congestion": 0.1, "link_health": 0.9})
    def test_env(self):
        env = ToyNetworkEnv(seed=1)
        state, info = env.reset(fault="congestion")
        self.assertEqual(state.shape, (8,))
        self.assertGreater(info["metrics"]["latency_ms"], 50)
        ns, reward, done, _, info2 = env.step(Action.REROUTE)
        self.assertTrue(done)
        self.assertTrue(np.isfinite(reward))
    def test_safety_gate(self):
        self.assertTrue(apply_safety_gate(Action.REROUTE, 0.9).approved)
        self.assertFalse(apply_safety_gate(Action.REROUTE, 0.4).approved)
        self.assertFalse(apply_safety_gate(Action.REROUTE, 0.9, allow_autonomy=False).approved)

if __name__ == "__main__":
    unittest.main()
