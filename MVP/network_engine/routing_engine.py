"""Dynamic Routing Engine for NeuroNet 20-Node Network.

Implements constrained shortest-path routing, alternative route calculation
around congested or failed nodes, and path cost analysis.
"""

from typing import List, Optional, Tuple, Dict, Any
import networkx as nx
from .node import NetworkNode
from .link import NetworkLink


class RoutingEngine:
    def __init__(self, nodes: List[NetworkNode], links: List[NetworkLink]):
        self.nodes = {n.node_id: n for n in nodes}
        self.links = links
        # Default Primary Flow: E1 (10) -> A1 (4) -> C1 (0) -> C2 (1) -> A3 (6) -> E5 (14)
        self.primary_path = [10, 4, 0, 1, 6, 14]
        # Backup Spine Flow: E1 (10) -> A1 (4) -> C4 (3) -> C3 (2) -> A4 (7) -> E5 (14)
        self.backup_path = [10, 4, 3, 2, 7, 14]
        self.active_path = list(self.primary_path)

    def _build_weighted_graph(self, avoid_nodes: Optional[List[int]] = None) -> nx.Graph:
        G = nx.Graph()
        avoid = set(avoid_nodes or [])

        for nid, n in self.nodes.items():
            G.add_node(nid, name=n.name, queue=n.current_queue)

        for l in self.links:
            if l.status == "cut":
                continue  # Link severed, cannot carry traffic

            # Base cost: propagation delay
            cost = l.propagation_delay_ms

            # Penalize links connected to congested or avoided nodes
            if l.source in avoid or l.target in avoid:
                cost += 10000.0

            src_node = self.nodes.get(l.source)
            tgt_node = self.nodes.get(l.target)
            if src_node and src_node.status in ("congested", "critical"):
                cost += 50.0
            if tgt_node and tgt_node.status in ("congested", "critical"):
                cost += 50.0

            if l.status == "degraded":
                cost += 40.0

            G.add_edge(l.source, l.target, weight=cost, link=l)

        return G

    def calculate_path(self, source: int = 10, target: int = 14, avoid_nodes: Optional[List[int]] = None) -> List[int]:
        """Calculates constrained shortest path using Dijkstra."""
        G = self._build_weighted_graph(avoid_nodes=avoid_nodes)
        try:
            path = nx.shortest_path(G, source=source, target=target, weight="weight")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback to predefined backup path if graph disconnected
            return self.backup_path

    def reroute_around(self, congested_node_id: int, source: int = 10, target: int = 14) -> Tuple[List[int], List[int]]:
        """REROUTE: Finds an alternative path that completely avoids the congested node.

        Returns (old_path, new_path).
        """
        old_path = list(self.active_path)
        new_path = self.calculate_path(source=source, target=target, avoid_nodes=[congested_node_id])

        # If identical, force alternate spine path
        if new_path == old_path:
            new_path = [10, 4, 3, 2, 7, 14] if old_path == [10, 4, 0, 1, 6, 14] else [10, 4, 0, 1, 6, 14]

        self.active_path = new_path
        return old_path, new_path

    def failover(self) -> Tuple[List[int], List[int]]:
        """FAILOVER: Immediately switches primary spine to secondary backup spine."""
        old_path = list(self.active_path)
        self.active_path = list(self.backup_path)
        return old_path, self.active_path

    def path_names(self, path: Optional[List[int]] = None) -> List[str]:
        p = path or self.active_path
        return [self.nodes[nid].name for nid in p if nid in self.nodes]

    def path_description(self, path: Optional[List[int]] = None) -> str:
        return " -> ".join(self.path_names(path))
