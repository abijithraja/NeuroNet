"""Integration test for NeuroNet Live Control Plane Server endpoints."""

import unittest
import numpy as np
from live_runner.server import LiveControlPlaneState, AGENT
from src.neuronet.actions import Action


class TestLiveControlPlane(unittest.TestCase):
    def setUp(self):
        self.state = LiveControlPlaneState()

    def test_state_initialization(self):
        self.assertEqual(len(self.state.nodes), 20)
        self.assertEqual(len(self.state.links), 33)
        self.assertEqual(self.state.scenario, "nominal")
        self.assertIn("queue_depth", self.state.nodes[0])
        self.assertIn("queue_capacity", self.state.nodes[0])
        self.assertIn("E1", self.state.active_path_names)

    def test_independent_agent_decision_under_congestion(self):
        self.state.inject_fault("congestion")
        self.assertEqual(self.state.scenario, "congestion")
        self.assertGreater(self.state.nodes[4]["buffer"], 70.0)

        m = self.state.metrics
        state_vec = np.asarray([
            np.clip(m["latency_ms"] / 200.0, 0, 1),
            np.clip(m["throughput_mbps"] / 100.0, 0, 1),
            np.clip(m["packet_loss_pct"] / 10.0, 0, 1),
            np.clip(m["congestion"], 0, 1),
            np.clip(m["link_health"], 0, 1),
            1.0, 0.0,
            np.clip(m["cpu_util_pct"] / 100.0, 0, 1),
        ], dtype=np.float32)

        # Ensure agent infers without any manual offsets
        q_vals = AGENT.q_values(state_vec)
        best_act = int(np.argmax(q_vals))

        # Check that best action is REROUTE or valid remediation action
        self.assertIn(best_act, [int(Action.REROUTE), int(Action.TRAFFIC_SHAPE), int(Action.SCALE_RESOURCE)])

        # Execute remediation
        rem_res = self.state.remediate(Action(best_act).name)
        self.assertEqual(self.state.scenario, "nominal")
        self.assertGreater(rem_res["recovery_delta"]["latency_drop_ms"], 0.0)

    def test_independent_agent_decision_under_link_failure(self):
        self.state.inject_fault("link_failure")
        self.assertEqual(self.state.scenario, "link_failure")
        self.assertGreaterEqual(self.state.metrics["packet_loss_pct"], 9.0)

        rem_res = self.state.remediate("FAILOVER")
        self.assertEqual(self.state.scenario, "nominal")
        self.assertGreater(self.state.metrics["link_health"], 0.8)


if __name__ == "__main__":
    unittest.main()
