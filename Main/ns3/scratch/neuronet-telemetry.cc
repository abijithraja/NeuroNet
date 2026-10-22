// NeuroNet starter ns-3 scenario: point-to-point UDP flow + FlowMonitor CSV.
// Copy into the scratch/ directory of an ns-3 source tree.
// Output path is relative to the ns-3 run working directory.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <iomanip>

using namespace ns3;

int main(int argc, char *argv[])
{
    double simulationTime = 10.0;
    std::string output = "neuronet_telemetry.csv";
    CommandLine cmd(__FILE__);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("output", "CSV output file", output);
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer devices = p2p.Install(nodes);

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    uint16_t port = 9000;
    UdpServerHelper server(port);
    ApplicationContainer serverApps = server.Install(nodes.Get(1));
    serverApps.Start(Seconds(0.5));
    serverApps.Stop(Seconds(simulationTime));

    UdpClientHelper client(interfaces.GetAddress(1), port);
    client.SetAttribute("MaxPackets", UintegerValue(1000000));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(5)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    ApplicationContainer clientApps = client.Install(nodes.Get(0));
    clientApps.Start(Seconds(1.0));
    clientApps.Stop(Seconds(simulationTime - 0.2));

    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    monitor->CheckForLostPackets();

    std::ofstream csv(output);
    csv << "time_s,flow_id,tx_packets,rx_packets,delay_ms,throughput_mbps\n";
    auto stats = monitor->GetFlowStats();
    for (const auto &entry : stats)
    {
        const FlowMonitor::FlowStats &st = entry.second;
        double delayMs = st.rxPackets > 0
            ? 1000.0 * st.delaySum.GetSeconds() / st.rxPackets : 0.0;
        double throughputMbps = simulationTime > 0
            ? (st.rxBytes * 8.0) / (simulationTime * 1e6) : 0.0;
        csv << std::fixed << std::setprecision(4)
            << simulationTime << "," << entry.first << ","
            << st.txPackets << "," << st.rxPackets << ","
            << delayMs << "," << throughputMbps << "\n";
    }
    csv.close();
    monitor->SerializeToXmlFile("neuronet-flowmon.xml", true, true);
    Simulator::Destroy();
    std::cout << "Wrote " << output << " and neuronet-flowmon.xml\n";
    return 0;
}
