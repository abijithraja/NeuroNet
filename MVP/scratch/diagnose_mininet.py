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
setLogLevel('info')
topo = NeuroNet20NodeTopo()
net = Mininet(topo=topo, switch=OVSSwitch, link=TCLink, controller=None)
net.start()
net.staticArp()
ovs = OVSController(net)
ovs.initialize_default_flows()

h1 = net.get('h1')
h2 = net.get('h2')

print("--- H1 IFCONFIG & ARP ---")
print(h1.cmd('ifconfig; arp -n'))
print("--- H2 IFCONFIG & ARP ---")
print(h2.cmd('ifconfig; arp -n'))

print("--- E1 PORTS ---")
print(ovs.run_cmd('ovs-ofctl show e1'))
print("--- E5 PORTS ---")
print(ovs.run_cmd('ovs-ofctl show e5'))

print("--- E1 FLOWS ---")
print(ovs.run_cmd('ovs-ofctl dump-flows e1'))
print("--- E5 FLOWS ---")
print(ovs.run_cmd('ovs-ofctl dump-flows e5'))

print("--- PING TEST H1 -> H2 ---")
print(h1.cmd('ping -c 2 -W 1 10.0.0.2'))

net.stop()
cleanup_stale_network()

