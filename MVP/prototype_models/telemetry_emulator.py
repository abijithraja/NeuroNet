import numpy as np
from datetime import datetime

class NetworkTelemetry:
    def __init__(self, network=None):
        self.network = network
        self._links = [f'link_{i}' for i in range(20)]
        self.baseline_latency    = {l: 5.0 for l in self._links}
        self.baseline_throughput = {l: 8.0 for l in self._links}
        self.baseline_loss       = {l: 0.1 for l in self._links}

    def get_all_links(self):
        return self._links

    def get_metrics(self, link_id, inject_anomaly=False, anomaly_type='congestion'):
        latency    = self.baseline_latency[link_id]    + np.random.normal(0, 0.25)
        throughput = self.baseline_throughput[link_id] + np.random.normal(0, 0.40)
        loss       = self.baseline_loss[link_id]       + np.random.normal(0, 0.05)

        if inject_anomaly:
            if anomaly_type == 'congestion':
                throughput = 9.5
                latency    = self.baseline_latency[link_id] + 25
                loss       = 2.5
            elif anomaly_type == 'link_failure':
                latency    = 150.0
                throughput = 2.0
                loss       = 5.0
            elif anomaly_type == 'degradation':
                latency    = self.baseline_latency[link_id] + 35
                loss       = 1.2
                throughput = 7.5

        congestion = min(throughput / 10.0, 1.0)
        return {
            'timestamp':        datetime.now().isoformat(),
            'link_id':          link_id,
            'latency_ms':       max(0.0, latency),
            'throughput_gbps':  max(0.0, throughput),
            'packet_loss_pct':  max(0.0, min(loss, 100.0)),
            'congestion':       congestion,
            'is_anomaly':       inject_anomaly,
        }