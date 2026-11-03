"""20-Node Topology definition for the NeuroNet network engine.

Defines the 4 Core, 6 Aggregation, and 10 Edge nodes (matching both the 60 FPS
HTML5 canvas and Mininet switch layout), along with the 32 interconnecting links.
"""

from typing import List, Tuple
import networkx as nx
from .node import NetworkNode
from .link import NetworkLink


def create_20_nodes() -> List[NetworkNode]:
    """Instantiates the 20 hierarchical network nodes."""
    nodes_def = [
        # Core Nodes (tier: core, id: 0..3) - High processing backbone
        (0, "C1", "core", 390, 240, 200.0),
        (1, "C2", "core", 510, 240, 200.0),
        (2, "C3", "core", 510, 320, 200.0),
        (3, "C4", "core", 390, 320, 200.0),

        # Aggregation Nodes (tier: agg, id: 4..9) - 80 Mbps distribution switches
        (4, "A1", "agg", 280, 170, 80.0),
        (5, "A2", "agg", 450, 140, 80.0),
        (6, "A3", "agg", 620, 170, 80.0),
        (7, "A4", "agg", 620, 390, 80.0),
        (8, "A5", "agg", 450, 420, 80.0),
        (9, "A6", "agg", 280, 390, 80.0),

        # Edge Nodes (tier: edge, id: 10..19) - 150 Mbps ingress switching
        (10, "E1",  "edge", 160, 110, 150.0),
        (11, "E2",  "edge", 310, 80,  150.0),
        (12, "E3",  "edge", 590, 80,  150.0),
        (13, "E4",  "edge", 740, 110, 150.0),
        (14, "E5",  "edge", 760, 280, 150.0),
        (15, "E6",  "edge", 740, 450, 150.0),
        (16, "E7",  "edge", 590, 480, 150.0),
        (17, "E8",  "edge", 310, 480, 150.0),
        (18, "E9",  "edge", 160, 450, 150.0),
        (19, "E10", "edge", 140, 280, 150.0),
    ]

    nodes = []
    for nid, name, tier, x, y, proc_rate in nodes_def:
        node = NetworkNode(
            node_id=nid,
            name=name,
            tier=tier,
            x=x,
            y=y,
            queue_capacity=1000,
            processing_rate_mbps=proc_rate,
        )
        node.reset()
        nodes.append(node)
    return nodes


def create_32_links() -> List[NetworkLink]:
    """Instantiates the 32 interconnecting links."""
    links_def = [
        # Core Full Mesh (High Bandwidth 1000 Mbps spine)
        (0, 1, 1000.0, 1.5),
        (1, 2, 1000.0, 1.5),
        (2, 3, 1000.0, 1.5),
        (3, 0, 1000.0, 1.5),
        (0, 2, 1000.0, 1.8),
        (1, 3, 1000.0, 1.8),

        # Core - Aggregation (Trunk links 100 Mbps)
        (0, 4, 100.0, 2.0),
        (0, 5, 100.0, 2.0),
        (1, 5, 100.0, 2.0),
        (1, 6, 100.0, 2.0),
        (2, 6, 100.0, 2.0),
        (2, 7, 100.0, 2.0),
        (2, 8, 100.0, 2.0),
        (3, 8, 100.0, 2.0),
        (3, 9, 100.0, 2.0),
        (0, 9, 100.0, 2.0),
        (3, 4, 100.0, 2.0),
        (1, 7, 100.0, 2.0),

        # Aggregation - Edge (Distribution links 100 Mbps)
        (4, 10, 100.0, 2.5),
        (4, 11, 100.0, 2.5),
        (5, 11, 100.0, 2.5),
        (5, 12, 100.0, 2.5),
        (6, 12, 100.0, 2.5),
        (6, 13, 100.0, 2.5),
        (6, 14, 100.0, 2.5),
        (7, 14, 100.0, 2.5),
        (7, 15, 100.0, 2.5),
        (8, 16, 100.0, 2.5),
        (8, 17, 100.0, 2.5),
        (9, 17, 100.0, 2.5),
        (9, 18, 100.0, 2.5),
        (9, 19, 100.0, 2.5),
        (4, 19, 100.0, 2.5),
    ]

    links = []
    for s, t, bw, delay in links_def:
        link = NetworkLink(source=s, target=t, bandwidth_mbps=bw, propagation_delay_ms=delay)
        links.append(link)
    return links


def build_network_graph(nodes: List[NetworkNode], links: List[NetworkLink]) -> nx.Graph:
    """Builds an undirected NetworkX graph with dynamic link and node attributes."""
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.node_id, name=n.name, tier=n.tier, obj=n)

    for l in links:
        weight = l.latency_ms + (100.0 if l.status == "congested" else 0.0)
        if l.status == "cut":
            weight = 1e9  # impassable
        G.add_edge(l.source, l.target, weight=weight, obj=l)

    return G
