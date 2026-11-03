import sys
import time
import subprocess
from pathlib import Path
MVP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MVP_ROOT))

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from real_mininet.topology import NeuroNet20NodeTopo
from real_mininet.ovs_controller import OVSController
from real_mininet.traffic import TrafficManager
from real_mininet.mininet_env import cleanup_stale_network

cleanup_stale_network()
setLogLevel('error')
topo = NeuroNet20NodeTopo()
net = Mininet(topo=topo, switch=OVSSwitch, link=TCLink, controller=None)
net.start()
net.staticArp()
ovs = OVSController(net)

# Test mod-port control
print("--- TESTING MOD-PORT LOGIC ---")
# Bring e1-eth3 and e5-eth3 down
ovs.run_cmd('ovs-ofctl mod-port e1 e1-eth3 down')
ovs.run_cmd('ovs-ofctl mod-port e5 e5-eth3 down')
ovs.run_cmd('ovs-ofctl mod-port e1 e1-eth2 up')
ovs.run_cmd('ovs-ofctl mod-port e5 e5-eth2 up')

# Wait for carrier stabilization
time.sleep(1.0)

traffic = TrafficManager(net)

print("--- PATH A NOMINAL ---")
m_a = traffic.measure_telemetry(count=2)
print("Path A Metrics:", m_a)

print("--- INJECTING CONGESTION ---")
ovs.inject_congestion()
time.sleep(0.5)
m_c = traffic.measure_telemetry(count=2)
print("Congested Metrics:", m_c)

print("--- REROUTE TO PATH B VIA MOD-PORT ---")
ovs.run_cmd('ovs-ofctl mod-port e1 e1-eth2 down')
ovs.run_cmd('ovs-ofctl mod-port e5 e5-eth2 down')
ovs.run_cmd('ovs-ofctl mod-port e1 e1-eth3 up')
ovs.run_cmd('ovs-ofctl mod-port e5 e5-eth3 up')
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=100,ip,nw_dst=10.0.0.2,actions=output:3"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=100,ip,nw_dst=10.0.0.1,actions=output:3"')
time.sleep(1.0)
m_b = traffic.measure_telemetry(count=2)
print("Path B Recovered Metrics:", m_b)

net.stop()
cleanup_stale_network()
