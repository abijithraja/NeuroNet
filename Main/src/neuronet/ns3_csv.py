"""Load the included ns-3 FlowMonitor CSV into normalized telemetry rows."""
from pathlib import Path
import pandas as pd

REQUIRED = {"time_s", "flow_id", "tx_packets", "rx_packets", "delay_ms", "throughput_mbps"}

def load_ns3_csv(path):
    df = pd.read_csv(Path(path))
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df["packet_loss_pct"] = (
        (df["tx_packets"] - df["rx_packets"]).clip(lower=0)
        / df["tx_packets"].replace(0, 1) * 100.0
    ).clip(0, 100)
    df["latency_ms"] = df["delay_ms"].fillna(0).clip(lower=0)
    df["congestion"] = (df["packet_loss_pct"] / 10.0).clip(0, 1)
    df["link_health"] = (1.0 - df["packet_loss_pct"] / 100.0).clip(0, 1)
    return df
