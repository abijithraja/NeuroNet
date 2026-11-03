"""Fault Injection Engine for NeuroNet.

Provides structured, measurable network perturbations based on real network
engineering failure models.
"""

from typing import Dict, Any, Tuple, Optional


class NetworkFault:
    name: str = "base"
    description: str = ""

    def apply(self, engine) -> Dict[str, Any]:
        raise NotImplementedError


class BufferCongestionFault(NetworkFault):
    name = "congestion"
    description = "Traffic arrival rate exceeds node processing capacity, causing queue buildup and packet drops."

    def __init__(self, target_node_id: int = 4, excess_rate_mbps: float = 145.0):
        self.target_node_id = target_node_id
        self.excess_rate_mbps = excess_rate_mbps

    def apply(self, engine) -> Dict[str, Any]:
        engine.scenario = "congestion"
        engine.affected_node_id = self.target_node_id
        engine.traffic_demand_mbps = self.excess_rate_mbps
        target_node = engine.nodes[self.target_node_id]
        target_node.current_queue = 920.0
        target_node.status = "congested"
        link = engine.find_link(0, self.target_node_id)
        if link:
            link.status = "congested"
        return engine.tick()


class CoreLinkFailureFault(NetworkFault):
    name = "link_failure"
    description = "Physical core-to-aggregation trunk link severed, dropping all in-flight packets."

    def __init__(self, link_nodes: Tuple[int, int] = (0, 4)):
        self.link_nodes = link_nodes

    def apply(self, engine) -> Dict[str, Any]:
        engine.scenario = "link_failure"
        engine.affected_link = self.link_nodes
        link = engine.find_link(*self.link_nodes)
        if link:
            link.status = "cut"
        engine.nodes[self.link_nodes[0]].status = "warning"
        engine.nodes[self.link_nodes[1]].status = "warning"
        return engine.tick()


class DDoSSYNFloodFault(NetworkFault):
    name = "attack_ddos"
    description = "High-volume SYN packet ingress flood overwhelming edge access switches."

    def __init__(self, edge_node_ids=(10, 11, 12, 13)):
        self.edge_node_ids = edge_node_ids

    def apply(self, engine) -> Dict[str, Any]:
        engine.scenario = "attack_ddos"
        for nid in self.edge_node_ids:
            engine.nodes[nid].incoming_rate_mbps = 180.0
            engine.nodes[nid].current_queue = 950.0
            engine.nodes[nid].cpu_util_pct = 98.0
            engine.nodes[nid].status = "critical"
        engine.traffic_demand_mbps = 120.0
        return engine.tick()


class RouteFlappingFault(NetworkFault):
    name = "routing_flap"
    description = "BGP/OSPF route flap oscillation causing intermittent packet loss and route ambiguity."

    def apply(self, engine) -> Dict[str, Any]:
        engine.scenario = "routing_flap"
        engine.flap_toggle = True
        return engine.tick()


class OpticalDegradationFault(NetworkFault):
    name = "degradation"
    description = "Optical fiber bending or transceiver degradation causing line latency and frame loss."

    def __init__(self, link_nodes: Tuple[int, int] = (0, 4)):
        self.link_nodes = link_nodes

    def apply(self, engine) -> Dict[str, Any]:
        engine.scenario = "degradation"
        link = engine.find_link(*self.link_nodes)
        if link:
            link.status = "degraded"
            link.loss_pct = 3.2
            link.propagation_delay_ms = 42.0
        return engine.tick()


FAULTS = {
    "congestion": BufferCongestionFault(),
    "link_failure": CoreLinkFailureFault(),
    "attack_ddos": DDoSSYNFloodFault(),
    "routing_flap": RouteFlappingFault(),
    "degradation": OpticalDegradationFault(),
}
