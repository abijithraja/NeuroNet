"""Open vSwitch (OVS) Flow Controller & Linux Traffic Control (tc) Interface.

Dispatches REAL OpenFlow rules using ovs-ofctl and link perturbations via tc netem.
"""

import subprocess
import time

class OVSController:
    def __init__(self, net):
        self.net = net
        self.active_path = "path_a"

    def run_cmd(self, cmd):
        """Executes a bash command on the host/switch."""
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError as e:
            return e.output.decode('utf-8') if e.output else ""

    def initialize_default_flows(self):
        """Programs baseline OpenFlow flows on all switches (OpenFlow 1.3 normal forwarding + explicit pathing)."""
        # Set all switches to fail-standalone or install normal L2 learning
        for sw in self.net.switches:
            self.run_cmd(f'ovs-ofctl add-flow {sw.name} "priority=10,actions=NORMAL"')

        # Direct host delivery flows on ingress/egress edge switches
        self.run_cmd('ovs-ofctl add-flow e1 "priority=200,ip,nw_dst=10.0.0.1,actions=output:1"')
        self.run_cmd('ovs-ofctl add-flow e5 "priority=200,ip,nw_dst=10.0.0.2,actions=output:1"')

        self.set_primary_path()

    def set_primary_path(self):
        """Routes h1 <-> h2 traffic via Path A: e1 <-> a1 <-> c1 <-> c2 <-> a3 <-> e5."""
        self.active_path = "path_a"
        # Administrative port management (prevents L2 looping across dual spines)
        self.run_cmd('ovs-ofctl mod-port e1 e1-eth3 down 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e5 e5-eth3 down 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e1 e1-eth2 up 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e5 e5-eth2 up 2>/dev/null')

        # Forward traffic via Path A (port 2)
        self.run_cmd('ovs-ofctl del-flows e1 "ip,nw_dst=10.0.0.2" 2>/dev/null')
        self.run_cmd('ovs-ofctl add-flow e1 "priority=100,ip,nw_dst=10.0.0.2,actions=output:2" 2>/dev/null')
        self.run_cmd('ovs-ofctl del-flows e5 "ip,nw_dst=10.0.0.1" 2>/dev/null')
        self.run_cmd('ovs-ofctl add-flow e5 "priority=100,ip,nw_dst=10.0.0.1,actions=output:2" 2>/dev/null')

    def set_backup_path(self):
        """REROUTE: Changes OpenFlow flow on e1 and e5 to route via Path B: e1 <-> a6 <-> c4 <-> c3 <-> a4 <-> e5."""
        self.active_path = "path_b"
        # Administrative port management (prevents L2 looping across dual spines)
        self.run_cmd('ovs-ofctl mod-port e1 e1-eth2 down 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e5 e5-eth2 down 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e1 e1-eth3 up 2>/dev/null')
        self.run_cmd('ovs-ofctl mod-port e5 e5-eth3 up 2>/dev/null')

        # Forward traffic via Path B (port 3)
        self.run_cmd('ovs-ofctl del-flows e1 "ip,nw_dst=10.0.0.2" 2>/dev/null')
        self.run_cmd('ovs-ofctl add-flow e1 "priority=100,ip,nw_dst=10.0.0.2,actions=output:3" 2>/dev/null')
        self.run_cmd('ovs-ofctl del-flows e5 "ip,nw_dst=10.0.0.1" 2>/dev/null')
        self.run_cmd('ovs-ofctl add-flow e5 "priority=100,ip,nw_dst=10.0.0.1,actions=output:3" 2>/dev/null')

    def apply_traffic_shaping(self, iface='e1-eth2', rate_mbps=50):
        """Applies Linux tc token bucket filter (tbf) on the congested interface."""
        self.run_cmd(f'tc qdisc del dev {iface} root 2>/dev/null')
        self.run_cmd(f'tc qdisc add dev {iface} root tbf rate {rate_mbps}mbit burst 32kbit latency 400ms')

    def apply_qos_priority(self, iface='e1-eth2'):
        """Applies Linux prio qdisc to protect ICMP/TCP control traffic."""
        self.run_cmd(f'tc qdisc del dev {iface} root 2>/dev/null')
        self.run_cmd(f'tc qdisc add dev {iface} root handle 1: prio')

    def clear_tc_rules(self):
        """Clears all tc artificial impairments across key interfaces."""
        for iface in ['e1-eth2', 'e1-eth3', 'e5-eth2', 'e5-eth3', 'a1-eth1', 'a1-eth2', 'c1-eth1', 'c1-eth2']:
            self.run_cmd(f'tc qdisc del dev {iface} root 2>/dev/null')

    # =========================================================================
    # REAL FAULT INJECTIONS USING LINUX TC & OVS
    # =========================================================================
    def inject_congestion(self):
        """Injects artificial buffer congestion & latency on Path A interface."""
        self.run_cmd('tc qdisc del dev e1-eth2 root 2>/dev/null')
        # Add 90ms latency and 5% drop
        self.run_cmd('tc qdisc add dev e1-eth2 root netem delay 90ms 10ms loss 5.2%')

    def inject_link_failure(self):
        """Cuts the primary trunk link between a1 and c1."""
        self.net.configLinkStatus('a1', 'c1', 'down')

    def inject_degradation(self):
        """Simulates optical attenuation or line degradation with 45ms latency & 2.8% loss."""
        self.run_cmd('tc qdisc del dev e1-eth2 root 2>/dev/null')
        self.run_cmd('tc qdisc add dev e1-eth2 root netem delay 45ms 5ms loss 2.8%')

    def reset_nominal(self):
        """Restores all physical links and clears traffic control impairments."""
        self.net.configLinkStatus('a1', 'c1', 'up')
        self.clear_tc_rules()
        self.set_primary_path()

