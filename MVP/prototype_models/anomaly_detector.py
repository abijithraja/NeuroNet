import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder    = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder    = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        latent      = self.encoder_fc(enc_out[:, -1, :])
        dec_hidden  = self.decoder_fc(latent).unsqueeze(1)
        decoded, _  = self.decoder(dec_hidden.expand(-1, x.size(1), -1))
        return decoded

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))


class AnomalyDetector:
    def __init__(self, window_size=5):
        self.window_size    = window_size
        self.lstm_ae        = LSTMAutoencoder()
        self.iso_forest     = IsolationForest(contamination=0.1, random_state=42)
        self.metrics_buffer = []
        self.is_trained     = False

    def train(self, benign_data):
        """benign_data: list of numpy arrays, each shape (window_size, 4)"""
        optimizer = torch.optim.Adam(self.lstm_ae.parameters(), lr=0.001)
        for _ in range(5):  # Fast demo mode
            for window in benign_data:
                x    = torch.FloatTensor(window).unsqueeze(0)
                loss = self.lstm_ae.reconstruction_error(x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        flattened = np.array([w.flatten() for w in benign_data])
        self.iso_forest.fit(flattened)
        self.is_trained = True

    def detect(self, metrics_dict):
        if not self.is_trained:
            return {'anomaly_score': 0.0, 'is_anomaly': False,
                    'lstm_error': 0.0, 'iso_score': 0.0,
                    'features': {'latency_anomalous': False,
                                 'loss_anomalous': False,
                                 'congestion_high': False}}

        self.metrics_buffer.append([
            metrics_dict['latency_ms'],
            metrics_dict['throughput_gbps'],
            metrics_dict['packet_loss_pct'],
            metrics_dict['congestion'],
        ])
        if len(self.metrics_buffer) < self.window_size:
            return {'anomaly_score': 0.0, 'is_anomaly': False,
                    'lstm_error': 0.0, 'iso_score': 0.0,
                    'features': {'latency_anomalous': False,
                                 'loss_anomalous': False,
                                 'congestion_high': False}}

        window = np.array(self.metrics_buffer[-self.window_size:])
        x = torch.FloatTensor(window).unsqueeze(0)

        lstm_error = float(self.lstm_ae.reconstruction_error(x).detach().numpy()[0])
        lstm_score = min(lstm_error * 2.0, 1.0)

        iso_raw   = -self.iso_forest.decision_function([window.flatten()])[0]
        iso_score = float(max(0.0, min(iso_raw, 1.0)))

        anomaly_score = 0.6 * lstm_score + 0.4 * iso_score

        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly':    bool(anomaly_score > 0.7),
            'lstm_error':    float(lstm_error),
            'iso_score':     iso_score,
            'features': {
                'latency_anomalous': bool(window[-1, 0] > np.mean(window[:-1, 0]) * 1.5),
                'loss_anomalous':    bool(window[-1, 2] > np.mean(window[:-1, 2]) * 2.0),
                'congestion_high':   bool(window[-1, 3] > 0.8),
            },
        }