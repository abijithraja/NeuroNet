import sys
from pathlib import Path
MVP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MVP_ROOT))

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from real_mininet.topology import NeuroNet20NodeTopo
from real_mininet.ovs_controller import OVSController
from real_mininet.mininet_env import cleanup_stale_network

cleanup_stale_network()
setLogLevel('error')
topo = NeuroNet20NodeTopo()
net = Mininet(topo=topo, switch=OVSSwitch, link=TCLink, controller=None)
net.start()
net.staticArp()
ovs = OVSController(net)

# Drop backup path on e1 and e5 to prevent L2 loop
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=500,in_port=e1-eth3,actions=drop"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=500,in_port=e5-eth3,actions=drop"')

# Explicit delivery to hosts on ingress switches
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=200,ip,nw_dst=10.0.0.1,actions=output:e1-eth1"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=200,ip,nw_dst=10.0.0.2,actions=output:e5-eth1"')

ovs.initialize_default_flows()

h1 = net.get('h1')
h2 = net.get('h2')

print("Sending 2 pings from h1 on Path A...")
print(h1.cmd('ping -c 2 -W 1 10.0.0.2'))

print("\n--- SWITCHING TO PATH B (REROUTE) ---")
ovs.run_cmd('ovs-ofctl del-flows e1 "in_port=3"')
ovs.run_cmd('ovs-ofctl del-flows e5 "in_port=3"')
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=500,in_port=2,actions=drop"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=500,in_port=2,actions=drop"')
ovs.set_backup_path()

print("Sending 2 pings from h1 on Path B...")
print(h1.cmd('ping -c 2 -W 1 10.0.0.2'))

print("=== PATH B SWITCH FLOWS ===")
for sw in ['e1', 'a6', 'c3', 'c4', 'a4', 'e5']:
    print(f"[{sw}]")
    print(ovs.run_cmd(f'ovs-ofctl dump-flows {sw}'))

net.stop()
cleanup_stale_network()
