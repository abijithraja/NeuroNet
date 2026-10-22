"""Ensemble Anomaly Detector combining LSTM Autoencoder and Isolation Forest.

Incorporates temporal sequence reconstruction (LSTM) and point-anomaly detection (Isolation Forest)
with feature-level attribution and explainability.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from .telemetry import validate_metrics, METRIC_KEYS

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)
        latent = torch.relu(self.encoder_fc(enc_out[:, -1, :]))
        dec_hidden = self.decoder_fc(latent).unsqueeze(1)
        decoded, _ = self.decoder(dec_hidden.expand(-1, x.size(1), -1))
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample mean squared error over sequence and features."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))

    def feature_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-feature mean squared error over the time window."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)  # shape: (batch, features)


class AnomalyDetector:
    def __init__(
        self,
        window_size: int = 5,
        contamination: float = 0.08,
        anomaly_threshold: float = 0.70,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.feature_names = ["latency_ms", "throughput_mbps", "packet_loss_pct", "congestion", "cpu_util_pct"]
        self.input_dim = len(self.feature_names)
        
        self.lstm_ae = LSTMAutoencoder(input_dim=self.input_dim, hidden_dim=32, latent_dim=8).to(self.device)
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.metrics_buffer: List[List[float]] = []
        self.is_trained: bool = False

    def _extract_feature_vector(self, metrics: Dict[str, Any]) -> List[float]:
        m = validate_metrics(metrics)
        return [
            m["latency_ms"],
            m["throughput_mbps"],
            m["packet_loss_pct"],
            m["congestion"],
            m["cpu_util_pct"],
        ]

    def fit(self, benign_data: List[Dict[str, Any]], epochs: int = 6):
        """Train both LSTM Autoencoder and Isolation Forest on benign normal traffic."""
        if len(benign_data) < self.window_size * 4:
            raise ValueError(f"Need at least {self.window_size * 4} benign samples to train detector")

        # Convert rows to 2D numpy array
        raw_rows = np.array([self._extract_feature_vector(row) for row in benign_data], dtype=np.float32)
        self.scaler.fit(raw_rows)
        scaled_rows = self.scaler.transform(raw_rows)

        # Build sliding windows
        windows = []
        for i in range(len(scaled_rows) - self.window_size + 1):
            windows.append(scaled_rows[i : i + self.window_size])
        windows_np = np.array(windows, dtype=np.float32)

        # 1. Train LSTM Autoencoder
        optimizer = torch.optim.Adam(self.lstm_ae.parameters(), lr=0.003)
        self.lstm_ae.train()
        batch_size = min(32, len(windows_np))
        num_batches = max(1, len(windows_np) // batch_size)

        for _ in range(epochs):
            indices = np.random.permutation(len(windows_np))
            for b in range(num_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                x_batch = torch.tensor(windows_np[batch_idx], dtype=torch.float32, device=self.device)
                loss = self.lstm_ae.reconstruction_error(x_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.lstm_ae.eval()

        # 2. Train Isolation Forest on flattened windows
        flattened_windows = windows_np.reshape(len(windows_np), -1)
        self.iso_forest.fit(flattened_windows)
        self.is_trained = True
        return self

    def predict(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies for incoming telemetry sample."""
        vec = self._extract_feature_vector(metrics_dict)
        self.metrics_buffer.append(vec)

        if not self.is_trained or len(self.metrics_buffer) < self.window_size:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "lstm_error": 0.0,
                "iso_score": 0.0,
                "features": {
                    "latency_anomalous": False,
                    "loss_anomalous": False,
                    "congestion_high": False,
                    "cpu_anomalous": False,
                },
                "feature_attributions": {
                    "latency": 0.25,
                    "packet_loss": 0.25,
                    "congestion": 0.25,
                    "cpu": 0.25,
                },
            }

        # Prepare window
        window = np.array(self.metrics_buffer[-self.window_size :], dtype=np.float32)
        scaled_window = self.scaler.transform(window)
        x = torch.tensor(scaled_window, dtype=torch.float32, device=self.device).unsqueeze(0)

        # LSTM Autoencoder scoring
        with torch.no_grad():
            lstm_error_tensor = self.lstm_ae.reconstruction_error(x)
            lstm_feat_errors = self.lstm_ae.feature_reconstruction_error(x)[0].cpu().numpy()
            lstm_error = float(lstm_error_tensor.cpu().numpy()[0])
            lstm_score = float(np.clip(lstm_error * 1.8, 0.0, 1.0))

        # Isolation Forest scoring
        flattened = scaled_window.reshape(1, -1)
        iso_raw = -float(self.iso_forest.decision_function(flattened)[0])
        iso_score = float(np.clip(iso_raw * 1.5, 0.0, 1.0))

        # Ensemble weighted score
        anomaly_score = float(np.clip(0.60 * lstm_score + 0.40 * iso_score, 0.0, 1.0))
        is_anomaly = bool(anomaly_score >= self.anomaly_threshold)

        # Baseline window comparison for rule heuristics
        lat_window = window[:, 0]
        loss_window = window[:, 2]
        cong_window = window[:, 3]
        cpu_window = window[:, 4]

        lat_flag = bool(lat_window[-1] > (np.mean(lat_window[:-1]) * 1.45) or lat_window[-1] > 65.0)
        loss_flag = bool(loss_window[-1] > (np.mean(loss_window[:-1]) * 1.8) or loss_window[-1] > 2.5)
        cong_flag = bool(cong_window[-1] > 0.75)
        cpu_flag = bool(cpu_window[-1] > 80.0)

        # Compute SHAP-inspired feature attribution breakdown
        feat_errs = np.maximum(lstm_feat_errors, 1e-6)
        total_err = np.sum(feat_errs)
        attributions = {
            "latency": float(np.round(feat_errs[0] / total_err, 3)),
            "throughput": float(np.round(feat_errs[1] / total_err, 3)),
            "packet_loss": float(np.round(feat_errs[2] / total_err, 3)),
            "congestion": float(np.round(feat_errs[3] / total_err, 3)),
            "cpu": float(np.round(feat_errs[4] / total_err, 3)),
        }

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "lstm_error": lstm_error,
            "iso_score": iso_score,
            "features": {
                "latency_anomalous": lat_flag,
                "loss_anomalous": loss_flag,
                "congestion_high": cong_flag,
                "cpu_anomalous": cpu_flag,
            },
            "feature_attributions": attributions,
        }
