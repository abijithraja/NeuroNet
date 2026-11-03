"""Unit tests for the NeuroNet 20-node physical Network Engine, Fault Engine, and Remediation."""

import unittest
from network_engine.simulation import NetworkEngine
from network_engine.node import NetworkNode
from src.neuronet.actions import Action
from src.neuronet.remediation import RemediationEngine


class TestNetworkEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NetworkEngine(seed=42)

    def test_20_nodes_and_32_links_initialized(self):
        self.assertEqual(len(self.engine.nodes), 20)
        self.assertEqual(len(self.engine.links), 33)
        self.assertEqual(self.engine.scenario, "nominal")
        self.assertLess(self.engine.metrics["latency_ms"], 35.0)
        self.assertEqual(self.engine.metrics["packet_loss_pct"], 0.0)

    def test_node_fluid_queue_dynamics(self):
        node = NetworkNode(node_id=4, name="A1", tier="agg", x=280, y=170,
                           queue_capacity=500, processing_rate_mbps=50.0)
        # Normal arrival (30 Mbps < 50 Mbps processing) -> queue stays low
        res = node.update_queue(incoming_mbps=30.0, dt=1.0)
        self.assertEqual(res["dropped_this_step"], 0)
        self.assertLess(res["buffer_util_pct"], 20.0)

        # Overload arrival (150 Mbps > 50 Mbps processing) -> queue fills up
        for _ in range(5):
            res = node.update_queue(incoming_mbps=150.0, dt=1.0)
        self.assertGreater(node.current_queue, 200.0)
        self.assertGreater(res["queuing_delay_ms"], 5.0)

    def test_buffer_congestion_and_reroute(self):
        # Inject congestion
        self.engine.inject_fault("congestion")
        self.assertEqual(self.engine.scenario, "congestion")
        self.assertGreater(self.engine.metrics["latency_ms"], 50.0)
        self.assertGreater(self.engine.nodes[4].buffer_utilization_pct, 70.0)

        # REROUTE remediation
        res = RemediationEngine.execute(self.engine, Action.REROUTE)
        self.assertEqual(res["action"], "REROUTE")
        self.assertGreater(res["recovery_delta"]["latency_drop_ms"], 0.0)
        self.assertEqual(self.engine.scenario, "nominal")

    def test_link_failure_and_failover(self):
        # Cut trunk link
        self.engine.inject_fault("link_failure")
        self.assertEqual(self.engine.scenario, "link_failure")
        self.assertGreaterEqual(self.engine.metrics["packet_loss_pct"], 9.0)

        # FAILOVER remediation
        res = RemediationEngine.execute(self.engine, Action.FAILOVER)
        self.assertEqual(res["action"], "FAILOVER")
        self.assertEqual(self.engine.scenario, "nominal")
        self.assertGreater(self.engine.metrics["link_health"], 0.8)

    def test_traffic_shaping(self):
        self.engine.inject_fault("congestion")
        res = RemediationEngine.execute(self.engine, Action.TRAFFIC_SHAPE)
        self.assertEqual(res["action"], "TRAFFIC_SHAPE")
        self.assertEqual(self.engine.scenario, "nominal")

    def test_scale_resource(self):
        node = self.engine.nodes[4]
        orig_cap = node.queue_capacity
        res = RemediationEngine.execute(self.engine, Action.SCALE_RESOURCE)
        self.assertEqual(res["action"], "SCALE_RESOURCE")
        self.assertGreater(node.queue_capacity, orig_cap)

    def test_rollback(self):
        self.engine.inject_fault("congestion")
        res = RemediationEngine.execute(self.engine, Action.ROLLBACK)
        self.assertEqual(res["action"], "ROLLBACK")
        self.assertEqual(self.engine.scenario, "nominal")


if __name__ == "__main__":
    unittest.main()
