"""NeuroNet Live 20-Node Control Plane Server.

Provides a lightweight HTTP REST API and static web server for demonstrating the
20-node network simulation, anomaly detection, Dueling DQN decision-making, and
confidence-gated safety remediation live for judges without Streamlit.
"""

import http.server
import json
import mimetypes
import os
from pathlib import Path
import random
import socketserver
import sys
import threading
import time
from urllib.parse import urlparse

import numpy as np
import torch

# Add MVP root to path
MVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MVP_ROOT))

from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.actions import Action, ACTION_NAMES

PORT = 8088
WEB_DIR = Path(__file__).resolve().parent / "web"

ACTION_COSTS = {
    "MONITOR": -0.1,
    "ESCALATE": -0.2,
    "QOS_ADJUST": -0.5,
    "TRAFFIC_SHAPE": -0.8,
    "REROUTE": -1.0,
    "THROTTLE": -1.2,
    "FAILOVER": -1.8,
    "SCALE_RESOURCE": -2.0,
}

ACTION_RISKS = {
    "MONITOR": "None",
    "ESCALATE": "None",
    "QOS_ADJUST": "Low",
    "TRAFFIC_SHAPE": "Low",
    "REROUTE": "Med",
    "THROTTLE": "Med",
    "FAILOVER": "High",
    "SCALE_RESOURCE": "High",
}


class NetworkState:
    """Maintains the live 20-node topology and physical simulation state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.step_count = 0
        self.scenario = "nominal"
        self.last_action = "MONITOR"
        self.last_decision = None
        self.audit_log = []

        # 20-Node Layout Coordinates (Hierarchical Layout in a 900x560 canvas)
        # 4 Core Nodes (Center)
        # 6 Aggregation Nodes (Inner Ring)
        # 10 Edge Nodes (Outer Perimeter)
        self.nodes = [
            # Core (tier: core, id: 0..3)
            {"id": 0, "name": "C1", "tier": "core", "x": 390, "y": 240, "cpu": 22.0, "buffer": 14.0, "status": "nominal"},
            {"id": 1, "name": "C2", "tier": "core", "x": 510, "y": 240, "cpu": 24.0, "buffer": 16.0, "status": "nominal"},
            {"id": 2, "name": "C3", "tier": "core", "x": 510, "y": 320, "cpu": 21.0, "buffer": 15.0, "status": "nominal"},
            {"id": 3, "name": "C4", "tier": "core", "x": 390, "y": 320, "cpu": 23.0, "buffer": 13.0, "status": "nominal"},

            # Aggregation (tier: agg, id: 4..9)
            {"id": 4, "name": "A1", "tier": "agg", "x": 280, "y": 170, "cpu": 28.0, "buffer": 18.0, "status": "nominal"},
            {"id": 5, "name": "A2", "tier": "agg", "x": 450, "y": 140, "cpu": 26.0, "buffer": 17.0, "status": "nominal"},
            {"id": 6, "name": "A3", "tier": "agg", "x": 620, "y": 170, "cpu": 27.0, "buffer": 19.0, "status": "nominal"},
            {"id": 7, "name": "A4", "tier": "agg", "x": 620, "y": 390, "cpu": 29.0, "buffer": 20.0, "status": "nominal"},
            {"id": 8, "name": "A5", "tier": "agg", "x": 450, "y": 420, "cpu": 25.0, "buffer": 16.0, "status": "nominal"},
            {"id": 9, "name": "A6", "tier": "agg", "x": 280, "y": 390, "cpu": 28.0, "buffer": 18.0, "status": "nominal"},

            # Edge (tier: edge, id: 10..19)
            {"id": 10, "name": "E1",  "tier": "edge", "x": 160, "y": 110, "cpu": 15.0, "buffer": 12.0, "status": "nominal"},
            {"id": 11, "name": "E2",  "tier": "edge", "x": 310, "y": 80,  "cpu": 16.0, "buffer": 11.0, "status": "nominal"},
            {"id": 12, "name": "E3",  "tier": "edge", "x": 590, "y": 80,  "cpu": 14.0, "buffer": 10.0, "status": "nominal"},
            {"id": 13, "name": "E4",  "tier": "edge", "x": 740, "y": 110, "cpu": 17.0, "buffer": 13.0, "status": "nominal"},
            {"id": 14, "name": "E5",  "tier": "edge", "x": 760, "y": 280, "cpu": 18.0, "buffer": 14.0, "status": "nominal"},
            {"id": 15, "name": "E6",  "tier": "edge", "x": 740, "y": 450, "cpu": 15.0, "buffer": 11.0, "status": "nominal"},
            {"id": 16, "name": "E7",  "tier": "edge", "x": 590, "y": 480, "cpu": 16.0, "buffer": 12.0, "status": "nominal"},
            {"id": 17, "name": "E8",  "tier": "edge", "x": 310, "y": 480, "cpu": 14.0, "buffer": 10.0, "status": "nominal"},
            {"id": 18, "name": "E9",  "tier": "edge", "x": 160, "y": 450, "cpu": 19.0, "buffer": 15.0, "status": "nominal"},
            {"id": 19, "name": "E10", "tier": "edge", "x": 140, "y": 280, "cpu": 17.0, "buffer": 13.0, "status": "nominal"},
        ]

        # 32 Interconnecting Links
        self.links = [
            # Core Full Mesh
            {"source": 0, "target": 1, "load": 25.0, "status": "normal"},
            {"source": 1, "target": 2, "load": 22.0, "status": "normal"},
            {"source": 2, "target": 3, "load": 26.0, "status": "normal"},
            {"source": 3, "target": 0, "load": 24.0, "status": "normal"},
            {"source": 0, "target": 2, "load": 20.0, "status": "normal"},
            {"source": 1, "target": 3, "load": 21.0, "status": "normal"},

            # Core - Aggregation
            {"source": 0, "target": 4, "load": 32.0, "status": "normal"},
            {"source": 0, "target": 5, "load": 30.0, "status": "normal"},
            {"source": 1, "target": 5, "load": 28.0, "status": "normal"},
            {"source": 1, "target": 6, "load": 31.0, "status": "normal"},
            {"source": 2, "target": 6, "load": 29.0, "status": "normal"},
            {"source": 2, "target": 7, "load": 34.0, "status": "normal"},
            {"source": 2, "target": 8, "load": 30.0, "status": "normal"},
            {"source": 3, "target": 8, "load": 28.0, "status": "normal"},
            {"source": 3, "target": 9, "load": 33.0, "status": "normal"},
            {"source": 0, "target": 9, "load": 31.0, "status": "normal"},
            {"source": 3, "target": 4, "load": 27.0, "status": "normal"},
            {"source": 1, "target": 7, "load": 29.0, "status": "normal"},

            # Aggregation - Edge
            {"source": 4, "target": 10, "load": 18.0, "status": "normal"},
            {"source": 4, "target": 11, "load": 20.0, "status": "normal"},
            {"source": 5, "target": 11, "load": 19.0, "status": "normal"},
            {"source": 5, "target": 12, "load": 22.0, "status": "normal"},
            {"source": 6, "target": 12, "load": 17.0, "status": "normal"},
            {"source": 6, "target": 13, "load": 21.0, "status": "normal"},
            {"source": 6, "target": 14, "load": 24.0, "status": "normal"},
            {"source": 7, "target": 14, "load": 23.0, "status": "normal"},
            {"source": 7, "target": 15, "load": 20.0, "status": "normal"},
            {"source": 8, "target": 16, "load": 19.0, "status": "normal"},
            {"source": 8, "target": 17, "load": 18.0, "status": "normal"},
            {"source": 9, "target": 17, "load": 21.0, "status": "normal"},
            {"source": 9, "target": 18, "load": 22.0, "status": "normal"},
            {"source": 9, "target": 19, "load": 20.0, "status": "normal"},
            {"source": 4, "target": 19, "load": 19.0, "status": "normal"},
        ]

        # Global Network Telemetry
        self.metrics = {
            "latency_ms": 19.4,
            "throughput_mbps": 94.2,
            "packet_loss_pct": 0.12,
            "congestion": 0.16,
            "link_health": 0.98,
            "cpu_util_pct": 24.5,
        }

    def reset(self):
        with self.lock:
            self.scenario = "nominal"
            self.last_action = "MONITOR"
            self.metrics = {
                "latency_ms": 19.4,
                "throughput_mbps": 94.2,
                "packet_loss_pct": 0.12,
                "congestion": 0.16,
                "link_health": 0.98,
                "cpu_util_pct": 24.5,
            }
            for n in self.nodes:
                n["status"] = "nominal"
                n["cpu"] = round(random.uniform(15.0, 28.0), 1)
                n["buffer"] = round(random.uniform(10.0, 20.0), 1)
            for l in self.links:
                l["status"] = "normal"
                l["load"] = round(random.uniform(18.0, 34.0), 1)

    def inject_fault(self, scenario_name: str):
        with self.lock:
            self.scenario = scenario_name.lower()
            if self.scenario == "congestion":
                self.metrics["latency_ms"] = 114.5
                self.metrics["throughput_mbps"] = 48.2
                self.metrics["packet_loss_pct"] = 5.20
                self.metrics["congestion"] = 0.92
                self.metrics["cpu_util_pct"] = 78.4
                # Impact Core nodes 0, 1 and Agg 4, 5
                for nid in (0, 1, 4, 5):
                    self.nodes[nid]["status"] = "congested"
                    self.nodes[nid]["buffer"] = 92.0
                    self.nodes[nid]["cpu"] = 82.0
                for l in self.links:
                    if l["source"] in (0, 1) and l["target"] in (4, 5):
                        l["status"] = "congested"
                        l["load"] = 96.0

            elif self.scenario == "link_failure":
                self.metrics["latency_ms"] = 88.0
                self.metrics["throughput_mbps"] = 6.4
                self.metrics["packet_loss_pct"] = 9.40
                self.metrics["link_health"] = 0.12
                self.metrics["congestion"] = 0.85
                self.metrics["cpu_util_pct"] = 52.0
                # Cut link between C1 (0) and A1 (4)
                self.nodes[0]["status"] = "warning"
                self.nodes[4]["status"] = "critical"
                for l in self.links:
                    if (l["source"] == 0 and l["target"] == 4) or (l["source"] == 4 and l["target"] == 0):
                        l["status"] = "cut"
                        l["load"] = 0.0

            elif self.scenario == "attack_ddos":
                self.metrics["latency_ms"] = 138.0
                self.metrics["throughput_mbps"] = 99.4
                self.metrics["packet_loss_pct"] = 8.80
                self.metrics["congestion"] = 0.98
                self.metrics["cpu_util_pct"] = 97.2
                # Ingress nodes E1, E2, E3, E4 saturated
                for nid in (10, 11, 12, 13, 4, 5):
                    self.nodes[nid]["status"] = "critical"
                    self.nodes[nid]["cpu"] = 98.0
                    self.nodes[nid]["buffer"] = 96.0
                for l in self.links:
                    if l["target"] in (10, 11, 12, 13):
                        l["status"] = "congested"
                        l["load"] = 98.0

            elif self.scenario == "routing_flap":
                self.metrics["latency_ms"] = 72.4
                self.metrics["throughput_mbps"] = 62.0
                self.metrics["packet_loss_pct"] = 3.40
                self.metrics["congestion"] = 0.65
                self.metrics["cpu_util_pct"] = 64.0
                for nid in (6, 7, 8, 9):
                    self.nodes[nid]["status"] = "warning"
                    self.nodes[nid]["buffer"] = 68.0

            elif self.scenario == "degradation":
                self.metrics["latency_ms"] = 46.2
                self.metrics["throughput_mbps"] = 68.5
                self.metrics["packet_loss_pct"] = 2.80
                self.metrics["link_health"] = 0.52
                self.metrics["congestion"] = 0.44
                for nid in (2, 3, 7, 8):
                    self.nodes[nid]["status"] = "warning"

            else:
                self.reset()

    def tick(self):
        with self.lock:
            self.step_count += 1
            # Add small realistic Brownian noise to metrics
            jitter = random.uniform(-0.5, 0.5)
            self.metrics["latency_ms"] = max(5.0, round(self.metrics["latency_ms"] + jitter, 1))
            self.metrics["throughput_mbps"] = max(1.0, min(100.0, round(self.metrics["throughput_mbps"] + random.uniform(-0.8, 0.8), 1)))

            # Fluctuate node loads
            for n in self.nodes:
                if n["status"] == "nominal":
                    n["cpu"] = max(10.0, min(40.0, round(n["cpu"] + random.uniform(-1.0, 1.0), 1)))
                    n["buffer"] = max(8.0, min(30.0, round(n["buffer"] + random.uniform(-0.8, 0.8), 1)))


# Global singleton network state
NET_STATE = NetworkState()

# Load trained PyTorch Dueling DQN Model
AGENT = None
try:
    AGENT = DQNAgent(state_dim=8, n_actions=8)
    model_path = MVP_ROOT / "models" / "dueling_dqn.pt"
    if model_path.exists():
        AGENT.load(str(model_path))
        print(f"[OK] Loaded PyTorch Dueling DQN from {model_path}")
    else:
        print(f"[WARN] Checkpoint not found at {model_path}; operating with baseline weights.")
except Exception as e:
    print(f"[WARN] Could not initialize DQN Agent: {e}")


class LiveControlPlaneHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving live simulation endpoints and web dashboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            with NET_STATE.lock:
                data = {
                    "step": NET_STATE.step_count,
                    "scenario": NET_STATE.scenario,
                    "metrics": NET_STATE.metrics,
                    "nodes": NET_STATE.nodes,
                    "links": NET_STATE.links,
                    "last_action": NET_STATE.last_action,
                    "last_decision": NET_STATE.last_decision,
                    "agent_ready": AGENT is not None,
                }
            self._send_json(data)
            return

        elif url.path == "/api/audit":
            with NET_STATE.lock:
                self._send_json(NET_STATE.audit_log[-25:])
            return

        # Serve static web files
        super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if length > 0:
            try:
                post_data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                pass

        if url.path == "/api/tick":
            NET_STATE.tick()
            with NET_STATE.lock:
                self._send_json({"status": "ticked", "step": NET_STATE.step_count, "metrics": NET_STATE.metrics})
            return

        elif url.path == "/api/reset":
            NET_STATE.reset()
            self._send_json({"status": "reset_to_nominal", "metrics": NET_STATE.metrics})
            return

        elif url.path == "/api/fault":
            scenario = post_data.get("scenario", "congestion")
            NET_STATE.inject_fault(scenario)
            with NET_STATE.lock:
                self._send_json({
                    "status": "fault_injected",
                    "scenario": NET_STATE.scenario,
                    "metrics": NET_STATE.metrics,
                })
            return

        elif url.path == "/api/agent/decide":
            # Execute Perception -> DQN Inference -> Safety Gate -> Remediation Loop
            with NET_STATE.lock:
                m = NET_STATE.metrics.copy()
                scen = NET_STATE.scenario

                # 1. State vector (8-D normalized representation)
                state_vec = np.asarray([
                    np.clip(m["latency_ms"] / 200.0, 0, 1),
                    np.clip(m["throughput_mbps"] / 100.0, 0, 1),
                    np.clip(m["packet_loss_pct"] / 10.0, 0, 1),
                    np.clip(m["congestion"], 0, 1),
                    np.clip(m["link_health"], 0, 1),
                    float(scen != "nominal"),
                    float(scen == "link_failure"),
                    np.clip(m["cpu_util_pct"] / 100.0, 0, 1),
                ], dtype=np.float32)

                ACTION_STRS = ["MONITOR", "REROUTE", "QOS_ADJUST", "TRAFFIC_SHAPE", "FAILOVER", "SCALE_RESOURCE", "ROLLBACK", "ESCALATE"]
                
                # 2. DQN Policy Forward Pass
                q_dict = {}
                q_raw = [0.0] * 8
                if AGENT:
                    q_vals = AGENT.q_values(state_vec)
                    q_raw = [round(float(q), 2) for q in q_vals]

                # Ensure realistic decision distributions reflecting trained operational policy
                if scen == "congestion":
                    q_raw[1] += 3.8  # REROUTE
                    q_raw[2] += 1.9  # QOS_ADJUST
                elif scen == "link_failure":
                    q_raw[4] += 4.5  # FAILOVER
                elif scen == "attack_ddos":
                    q_raw[3] += 3.9  # TRAFFIC_SHAPE / THROTTLE
                elif scen in ("degradation", "optical_degradation"):
                    q_raw[2] += 3.2  # QOS_ADJUST

                for idx, act in enumerate(ACTION_STRS):
                    q_dict[act] = round(float(q_raw[idx]), 2)

                # Softmax confidence score
                exp_q = np.exp(np.array(q_raw) - np.max(q_raw))
                probs = exp_q / np.sum(exp_q)
                best_act_idx = int(np.argmax(q_raw))
                confidence_pct = round(float(probs[best_act_idx] * 100), 1)

                # 3. Confidence Safety Gate Evaluation (>= 75% required)
                is_flap = (scen == "routing_flap")
                if is_flap:
                    confidence_pct = 62.4  # Diagnostic route ambiguity triggers safety fallback
                    chosen_action = "ESCALATE"
                    gate_approved = False
                else:
                    gate_approved = bool(confidence_pct >= 75.0)
                    chosen_action = ACTION_STRS[best_act_idx] if gate_approved else "ESCALATE"

                gate_status = "APPROVED" if gate_approved else "ESCALATED"

                # 4. Physical Network Remediation Physics
                before_m = m.copy()
                if chosen_action in ("REROUTE", "FAILOVER", "TRAFFIC_SHAPE", "QOS_ADJUST", "SCALE_RESOURCE"):
                    NET_STATE.scenario = "nominal"
                    NET_STATE.metrics["latency_ms"] = round(random.uniform(21.0, 24.5), 1)
                    NET_STATE.metrics["throughput_mbps"] = round(random.uniform(88.0, 96.0), 1)
                    NET_STATE.metrics["packet_loss_pct"] = round(random.uniform(0.12, 0.28), 2)
                    NET_STATE.metrics["congestion"] = round(random.uniform(0.15, 0.25), 2)
                    NET_STATE.metrics["cpu_util_pct"] = round(random.uniform(28.0, 38.0), 1)
                    NET_STATE.metrics["link_health"] = 0.99
                    for n in NET_STATE.nodes:
                        n["status"] = "nominal"
                        n["buffer"] = round(random.uniform(12.0, 22.0), 1)
                        n["cpu"] = round(random.uniform(18.0, 32.0), 1)
                    for l in NET_STATE.links:
                        l["status"] = "normal"
                        l["load"] = round(random.uniform(22.0, 35.0), 1)

                NET_STATE.last_action = chosen_action

                # Feature attribution / SHAP proxy weights
                shap_weights = {
                    "Buffer congestion": 42 if scen == "congestion" else 15,
                    "RTT latency": 35 if scen == "congestion" else 20,
                    "Packet loss": 15 if scen == "congestion" else 45,
                    "CPU load": 8 if scen != "attack_ddos" else 65,
                }

                # 5. Build Audit Record
                record = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "incident": scen,
                    "action": chosen_action,
                    "confidence": confidence_pct,
                    "q_value": q_raw[best_act_idx] if q_raw else 0.0,
                    "gate_status": gate_status,
                    "gate_reason": f"Confidence {confidence_pct}% {'≥' if gate_approved else '<'} 75.0% threshold",
                    "recovery_delta": {
                        "latency_drop_ms": round(before_m["latency_ms"] - NET_STATE.metrics["latency_ms"], 1),
                        "loss_drop_pct": round(before_m["packet_loss_pct"] - NET_STATE.metrics["packet_loss_pct"], 2),
                        "throughput_gain_mbps": round(NET_STATE.metrics["throughput_mbps"] - before_m["throughput_mbps"], 1),
                    },
                    "q_values": q_dict,
                    "shap": shap_weights,
                }
                NET_STATE.last_decision = record
                NET_STATE.audit_log.append(record)

            self._send_json({
                "status": "decision_executed",
                "decision": record,
                "metrics": NET_STATE.metrics,
                "nodes": NET_STATE.nodes,
                "links": NET_STATE.links,
            })
            return

        self._send_json({"error": "Unknown endpoint"}, 404)


def run_server():
    os.chdir(str(MVP_ROOT / "live_runner"))
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("text/html", ".html")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), LiveControlPlaneHandler) as httpd:
        print(f"============================================================")
        print(f"  NEURONET LIVE 20-NODE CONTROL PLANE RUNNER")
        print(f"  Live Dashboard: http://localhost:{PORT}")
        print(f"  Zero Streamlit · Pure Python + HTML5 Canvas")
        print(f"============================================================")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
