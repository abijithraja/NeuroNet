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

# Explicit delivery to hosts
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=200,ip,nw_dst=10.0.0.1,actions=output:1"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=200,ip,nw_dst=10.0.0.2,actions=output:1"')

# Isolate Path A: block port 2 on e1 and e5
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=500,in_port=2,actions=drop"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=500,in_port=2,actions=drop"')

# Set Path B forward & reverse
ovs.run_cmd('ovs-ofctl add-flow e1 "priority=100,ip,nw_dst=10.0.0.2,actions=output:3"')
ovs.run_cmd('ovs-ofctl add-flow e5 "priority=100,ip,nw_dst=10.0.0.1,actions=output:3"')

# Baseline normal on all
for sw in net.switches:
    ovs.run_cmd(f'ovs-ofctl add-flow {sw.name} "priority=10,actions=NORMAL"')

h1 = net.get('h1')
h2 = net.get('h2')

print("Ping 1 on Path B:")
print(h1.cmd('ping -c 1 -W 2 10.0.0.2'))

for sw in ['e1', 'a6', 'c3', 'c4', 'a4', 'e5']:
    print(f"=== {sw} PORT STATS ===")
    print(ovs.run_cmd(f'ovs-ofctl dump-ports {sw}'))

net.stop()
cleanup_stale_network()

