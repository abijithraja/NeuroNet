"""Real Mininet RL Environment for NeuroNet.

Replaces ToyNetworkEnv with actual Linux Open vSwitch network namespaces,
real packet flows via iperf3, real ICMP RTT delays, and OpenFlow flow rules.
"""

import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Add MVP root to path
MVP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MVP_ROOT))

from src.neuronet.actions import Action, ACTION_NAMES
from src.neuronet.reward import compute_reward

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

from .topology import NeuroNet20NodeTopo
from .traffic import TrafficManager
from .ovs_controller import OVSController

def cleanup_stale_network():
    """Unlinks any lingering veth interfaces and deletes stale OVS bridges."""
    try:
        out = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True).stdout
        intfs = set(re.findall(r'(\b[\w\-]+-eth\d+\b)', out))
        for intf in intfs:
            subprocess.run(['ip', 'link', 'del', intf], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        br_out = subprocess.run(['ovs-vsctl', 'list-br'], capture_output=True, text=True).stdout
        for br in br_out.split():
            if br.strip():
                subprocess.run(['ovs-vsctl', '--if-exists', 'del-br', br.strip()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

class RealMininetEnv:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        setLogLevel('error')

        # Clean any stale veth interfaces or bridges before starting
        cleanup_stale_network()

        print("[Mininet] Building 20-Node Topology...")
        self.topo = NeuroNet20NodeTopo()
        self.net = Mininet(topo=self.topo, switch=OVSSwitch, link=TCLink, controller=None)
        
        print("[Mininet] Starting Open vSwitch Network...")
        self.net.start()

        print("[Mininet] Configuring static ARP cache across nodes...")
        self.net.staticArp()

        self.ovs = OVSController(self.net)
        self.traffic = TrafficManager(self.net)

        print("[Mininet] Installing baseline OpenFlow rules...")
        self.ovs.initialize_default_flows()

        print("[Mininet] Starting background traffic...")
        self.traffic.start_iperf_server()
        self.traffic.start_background_traffic(target_mbps=50)

        self.fault = "none"
        self.metrics = None
        self.recent_actions = []

    @property
    def state_dim(self):
        return 8

    def _make_state(self, m):
        return np.asarray([
            np.clip(m["latency_ms"] / 200.0, 0, 1),
            np.clip(m["throughput_mbps"] / 100.0, 0, 1),
            np.clip(m["packet_loss_pct"] / 10.0, 0, 1),
            np.clip(m["congestion"], 0, 1),
            np.clip(m["link_health"], 0, 1),
            float(self.fault != "none"),
            float(self.fault == "link_failure"),
            min(len(self.recent_actions), 3) / 3.0,
        ], dtype=np.float32)

    def reset(self, fault=None):
        """Restores physical network to nominal and injects fault."""
        self.recent_actions = []
        self.ovs.reset_nominal()
        time.sleep(0.8)

        self.fault = fault or self.rng.choice(["none", "congestion", "link_failure", "degradation"])

        if self.fault == "congestion":
            self.ovs.inject_congestion()
        elif self.fault == "link_failure":
            self.ovs.inject_link_failure()
        elif self.fault == "degradation":
            self.ovs.inject_degradation()

        # Let the physical impairment settle and measure REAL telemetry
        time.sleep(0.6)
        self.metrics = self.traffic.measure_telemetry()

        return self._make_state(self.metrics), {
            "fault": self.fault,
            "metrics": self.metrics,
            "active_path": self.ovs.active_path
        }

    def step(self, action):
        action = Action(int(action))
        before = self.metrics.copy()

        # Execute REAL network actions via OpenFlow and tc
        if action == Action.REROUTE:
            self.ovs.set_backup_path()
        elif action == Action.FAILOVER:
            self.ovs.net.configLinkStatus('a1', 'c1', 'down')
            self.ovs.set_backup_path()
        elif action == Action.QOS_ADJUST:
            self.ovs.apply_qos_priority()
        elif action == Action.TRAFFIC_SHAPE:
            self.ovs.apply_traffic_shaping(rate_mbps=40)
        elif action == Action.ROLLBACK:
            self.ovs.reset_nominal()
        elif action == Action.SCALE_RESOURCE:
            self.ovs.scale_resource(rate_mbps=100, limit_bytes=50000)
        elif action in (Action.MONITOR, Action.ESCALATE):
            pass

        # Wait for real packet flow convergence
        time.sleep(0.7)

        # Measure real physical post-action telemetry
        after = self.traffic.measure_telemetry()
        self.metrics = after

        # Compute multi-objective reward
        reward = compute_reward(
            before=before,
            after=after,
            action=ACTION_NAMES[action],
            recent_actions=self.recent_actions
        )

        self.recent_actions.append(int(action))
        done = True  # Single remediation step evaluation

        return self._make_state(after), float(reward), done, False, {
            "fault": self.fault,
            "action": ACTION_NAMES[action],
            "metrics_before": before,
            "metrics_after": after,
            "active_path": self.ovs.active_path
        }

    def close(self):
        """Clean shutdown of Mininet topology."""
        print("[Mininet] Stopping traffic and cleaning up OVS switches...")
        self.traffic.stop_traffic()
        self.ovs.clear_tc_rules()
        self.net.stop()

