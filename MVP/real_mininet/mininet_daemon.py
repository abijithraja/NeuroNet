"""NeuroNet Persistent Real Mininet Daemon.

Runs continuously inside WSL2, keeping the real Linux Open vSwitch topology,
network namespaces (h1, h2), and iperf3 traffic alive. Serves a lightweight HTTP
API on port 8089 for instantaneous (<1s) fault injection and RL healing.
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

MVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MVP_ROOT))

from real_mininet.mininet_env import RealMininetEnv
from src.neuronet.actions import Action, ACTION_NAMES
from src.neuronet.dqn_agent import DQNAgent
from src.neuronet.anomaly_detector import AnomalyDetector
from src.neuronet.safety import apply_safety_gate
from src.neuronet.audit import log_decision
from config import CONFIG, LOG_DIR

DAEMON_PORT = 8089

ENV = None
AGENT = None
DETECTOR = None
LOCK = threading.Lock()


def build_detector(seed=42):
    rng = np.random.default_rng(seed)
    normal = []
    for _ in range(120):
        normal.append({
            "latency_ms": max(1, rng.normal(29, 2)),
            "throughput_mbps": np.clip(rng.normal(75, 4), 50, 100),
            "packet_loss_pct": np.clip(rng.normal(0.05, 0.04), 0, 1),
            "congestion": np.clip(rng.normal(0.50, 0.03), 0.4, 0.6),
            "link_health": np.clip(rng.normal(0.99, 0.01), 0.9, 1),
        })
    return AnomalyDetector(random_state=seed).fit(normal, epochs=10)


class MininetDaemonHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress excessive HTTP access logs

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            with LOCK:
                if ENV is None:
                    self._send_json({"status": "starting", "network": "initializing"}, 503)
                    return
                m = ENV.metrics or {"latency_ms": 25.0, "throughput_mbps": 75.0, "packet_loss_pct": 0.0}
                self._send_json({
                    "status": "ready",
                    "fault": ENV.fault,
                    "active_path": ENV.ovs.active_path,
                    "metrics": m,
                    "environment": "real_mininet_openvswitch"
                })
                return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if length > 0:
            try:
                post_data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                pass

        if url.path == "/api/fault":
            fault = post_data.get("fault", "link_failure")
            with LOCK:
                state, info = ENV.reset(fault=fault)
                self._send_json({
                    "status": "fault_injected",
                    "fault": ENV.fault,
                    "metrics": ENV.metrics,
                    "active_path": ENV.ovs.active_path
                })
            return

        elif url.path == "/api/heal":
            fault_req = post_data.get("fault", None)
            with LOCK:
                # If a specific fault is requested and different from current, inject it first
                if fault_req and fault_req not in ("nominal", "none") and ENV.fault != fault_req:
                    state, info = ENV.reset(fault=fault_req)
                else:
                    state = ENV._make_state(ENV.metrics)
                    info = {"fault": ENV.fault, "metrics": ENV.metrics}

                metrics = ENV.metrics
                anomaly = DETECTOR.predict(metrics) if DETECTOR else {"anomaly_score": 0.5, "is_anomaly": True}
                shap = DETECTOR.shap_attribution() if DETECTOR else {}

                # Independent DQN policy evaluation (zero artificial boosts)
                q_vals = AGENT.q_values(state).copy() if AGENT else np.zeros(8)

                action_id = int(np.argmax(q_vals))
                exp_q = np.exp(q_vals - np.max(q_vals))
                confidence = float(exp_q[action_id] / np.sum(exp_q))

                gate = apply_safety_gate(action_id, confidence, CONFIG.confidence_threshold)
                chosen = Action(action_id) if gate.approved else Action.ESCALATE

                # Execute action on real live switches
                next_state, reward, done, _, step_info = ENV.step(chosen)

                record = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "environment": "real_mininet_openvswitch",
                    "incident": f"{current_fault} [REAL MININET]",
                    "active_path": step_info["active_path"],
                    "action": ACTION_NAMES[chosen].upper(),
                    "confidence": round(confidence * 100, 1),
                    "q_value": round(float(q_vals[action_id]), 2),
                    "gate_status": "APPROVED" if gate.approved else "ESCALATED",
                    "gate_reason": "Real Mininet Kernel OVS Execution (WSL2 Live)",
                    "recovery_delta": {
                        "latency_drop_ms": round(step_info["metrics_before"].get("latency_ms", 30) - step_info["metrics_after"].get("latency_ms", 25), 1),
                        "loss_drop_pct": round(step_info["metrics_before"].get("packet_loss_pct", 0) - step_info["metrics_after"].get("packet_loss_pct", 0), 2),
                        "throughput_gain_mbps": round(step_info["metrics_after"].get("throughput_mbps", 75) - step_info["metrics_before"].get("throughput_mbps", 75), 1),
                    },
                    "q_values": {
                        "MONITOR": round(float(q_vals[0]), 2),
                        "REROUTE": round(float(q_vals[1]), 2),
                        "QOS_ADJUST": round(float(q_vals[2]), 2),
                        "TRAFFIC_SHAPE": round(float(q_vals[3]), 2),
                        "FAILOVER": round(float(q_vals[4]), 2),
                        "SCALE_RESOURCE": round(float(q_vals[5]), 2),
                        "ROLLBACK": round(float(q_vals[6]), 2),
                        "ESCALATE": round(float(q_vals[7]), 2),
                    },
                    "shap": shap,
                    "metrics_before": step_info["metrics_before"],
                    "metrics_after": step_info["metrics_after"],
                }

                # Log to disk
                log_path = MVP_ROOT / "logs" / "mininet_decisions.jsonl"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_decision(log_path, record)

                self._send_json({
                    "status": "mininet_executed",
                    "decision": record,
                    "metrics": step_info["metrics_after"]
                })
            return

        self._send_json({"error": "Unknown endpoint"}, 404)


def main():
    global ENV, AGENT, DETECTOR
    print("[Mininet Daemon] Initializing AI models...")
    AGENT = DQNAgent(CONFIG.state_dim, len(ACTION_NAMES), hidden_dim=CONFIG.hidden_dim,
                     gamma=CONFIG.gamma, lr=CONFIG.learning_rate, device="cpu")
    model_path = MVP_ROOT / "models" / "dueling_dqn.pt"
    if not model_path.exists():
        model_path = MVP_ROOT / "models" / "mithu_pretrained_dqn.pt"
    if model_path.exists():
        AGENT.load(str(model_path))
        print(f"[Mininet Daemon] Loaded {model_path.name}")
    DETECTOR = build_detector()
    print("[Mininet Daemon] Detector trained.")

    print("[Mininet Daemon] Starting Persistent Real Mininet Network...")
    ENV = RealMininetEnv(seed=CONFIG.seed)
    # Ensure nominal starting state
    ENV.reset(fault="none")
    print("[Mininet Daemon] Real Mininet Network is LIVE and ready!")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", DAEMON_PORT), MininetDaemonHandler) as httpd:
        print(f"[Mininet Daemon] HTTP Control Bridge listening on port {DAEMON_PORT}")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if ENV:
                ENV.close()


if __name__ == "__main__":
    main()

