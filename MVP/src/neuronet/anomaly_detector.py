"""Hybrid LSTM-Autoencoder + Isolation Forest anomaly detector.

Integrates Mithu's calibrated hybrid detector (LSTM reconstruction error +
Isolation Forest margin) with the MVP telemetry pipeline.  Provides both the
original ``.fit()/.predict()`` API used by ``main.py`` and the newer
``.detect()/.shap_attribution()`` API.

Training uses z-score normalization, mini-batch SGD, and calibrates anomaly
thresholds on a held-out slice of the benign data so that normal traffic
scores well below 0.5.
"""
import numpy as np
from itertools import combinations
from math import factorial

from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn

from .telemetry import METRIC_KEYS, validate_metrics


# ---------------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder with a bottleneck latent vector."""

    def __init__(self, input_dim=5, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        latent = self.encoder_fc(encoded[:, -1, :])
        decoded_hidden = self.decoder_fc(latent).unsqueeze(1)
        decoded, _ = self.decoder(decoded_hidden.expand(-1, x.size(1), -1))
        return self.output(decoded)

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))


# ---------------------------------------------------------------------------
# Hybrid Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Hybrid LSTM-AE + Isolation Forest detector.

    Parameters
    ----------
    contamination : float
        Contamination parameter for the Isolation Forest.
    random_state : int
        Random seed for reproducibility.
    window_size : int
        Number of consecutive samples used per sliding window.
    """

    _N_FEATURES = len(METRIC_KEYS)  # 5

    def __init__(self, contamination=0.05, random_state=42, window_size=5):
        self.contamination = contamination
        self.random_state = random_state
        self.window_size = window_size

        self.lstm_ae = LSTMAutoencoder(input_dim=self._N_FEATURES)
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
        )
        self.metrics_buffer: list[list[float]] = []
        self.is_fitted = False

        # Set after training via calibration split.
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.e_q: float = 1.0
        self.d_med: float = 0.0
        self.d_low: float = -1.0

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _row_to_vec(row: dict) -> list[float]:
        """Convert a metrics dict to a list ordered by METRIC_KEYS."""
        v = validate_metrics(row)
        return [v[k] for k in METRIC_KEYS]

    @staticmethod
    def _matrix(rows):
        """List of dicts → (N, n_features) float array."""
        return np.asarray(
            [[validate_metrics(r)[k] for k in METRIC_KEYS] for r in rows],
            dtype=np.float32,
        )

    def _z(self, w):
        """Z-score normalise an array using training statistics."""
        return (np.asarray(w, dtype=np.float32) - self.mu) / self.sigma

    def _raw_scores(self, windows):
        """Return per-window (lstm_error, iso_decision) arrays."""
        z = self._z(windows)
        with torch.no_grad():
            e = self.lstm_ae.reconstruction_error(torch.from_numpy(z)).numpy()
        d = self.iso_forest.decision_function(z.reshape(len(z), -1))
        return e, d

    def _calibrated_scores(self, windows):
        """Return (combined, lstm_component, iso_component) arrays in [0, 1]."""
        e, d = self._raw_scores(windows)
        lstm = np.clip(0.5 * e / self.e_q, 0, 1)
        iso = np.clip(
            0.5 * (self.d_med - d) / (self.d_med - self.d_low + 1e-9), 0, 1
        )
        return 0.6 * lstm + 0.4 * iso, lstm, iso

    # -- training -----------------------------------------------------------

    def _train_hybrid(self, windows, epochs=50, batch_size=64):
        """Train LSTM-AE and Isolation Forest on 3-D benign windows."""
        data = np.asarray(windows, dtype=np.float32)
        if len(data) < 10:
            raise ValueError(
                "Provide enough normal samples to form at least 10 windows"
            )

        # Reserve 20% for calibration.
        n_cal = max(len(data) // 5, 1)
        cal, fit = data[:n_cal], data[n_cal:]

        # Compute z-score statistics on the training split.
        self.mu = fit.mean(axis=(0, 1))
        self.sigma = fit.std(axis=(0, 1)) + 1e-6

        # --- LSTM Autoencoder ---
        z = torch.from_numpy(self._z(fit))
        opt = torch.optim.Adam(self.lstm_ae.parameters(), lr=0.001)
        self.lstm_ae.train()
        for _ in range(epochs):
            for idx in torch.randperm(len(z)).split(batch_size):
                loss = self.lstm_ae.reconstruction_error(z[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.lstm_ae.eval()

        # --- Isolation Forest ---
        self.iso_forest.fit(self._z(fit).reshape(len(fit), -1))

        # --- Calibrate on held-out benign windows ---
        e, d = self._raw_scores(cal)
        self.e_q = float(np.quantile(e, 0.995))
        self.d_med = float(np.median(d))
        self.d_low = float(np.quantile(d, 0.005))
        self.is_fitted = True

    # -- public API: backward-compatible (main.py) --------------------------

    def fit(self, normal_rows, epochs=50, batch_size=64):
        """Train on a list of metric dicts (backward-compatible API).

        Internally creates sliding windows of size ``window_size`` from the
        flat list and delegates to the hybrid LSTM+IF training pipeline.

        Parameters
        ----------
        normal_rows : list[dict]
            Each dict must contain the keys in ``METRIC_KEYS``.
        """
        x = self._matrix(normal_rows)
        if len(x) < self.window_size + 10:
            raise ValueError(
                f"Provide at least {self.window_size + 10} normal samples "
                f"to fit the detector"
            )

        # Build sliding windows: (N - ws + 1, ws, n_features)
        windows = np.lib.stride_tricks.sliding_window_view(
            x, window_shape=self.window_size, axis=0
        )
        # sliding_window_view gives shape (N-ws+1, n_features, ws); transpose
        windows = windows.transpose(0, 2, 1).copy()

        self._train_hybrid(windows, epochs=epochs, batch_size=batch_size)
        return self

    def predict(self, metrics):
        """Score a single metrics dict (backward-compatible API).

        Calls ``.detect()`` internally and returns a superset of the original
        result format so that existing ``main.py`` code continues to work.

        Returns
        -------
        dict with keys: is_anomaly, anomaly_score, raw_score, lstm_error,
        iso_score.
        """
        result = self.detect(metrics)
        # ``raw_score`` mirrors the old Isolation Forest-only API.  Here we
        # expose the combined calibrated score as a stand-in; downstream code
        # only uses ``is_anomaly`` and ``anomaly_score``.
        result.setdefault("raw_score", -result["anomaly_score"])
        return result

    # -- public API: new (Mithu-style) --------------------------------------

    def detect(self, metrics_dict):
        """Score a single observation using the sliding buffer.

        Returns
        -------
        dict with anomaly_score, is_anomaly, lstm_error, iso_score.
        """
        if not self.is_fitted:
            return {"anomaly_score": 0.0, "is_anomaly": False,
                    "lstm_error": 0.0, "iso_score": 0.0}

        self.metrics_buffer.append(self._row_to_vec(metrics_dict))
        self.metrics_buffer = self.metrics_buffer[-self.window_size:]

        if len(self.metrics_buffer) < self.window_size:
            return {"anomaly_score": 0.0, "is_anomaly": False,
                    "lstm_error": 0.0, "iso_score": 0.0}

        window = np.array(self.metrics_buffer, dtype=np.float32)[None]
        s, l, i = self._calibrated_scores(window)
        return {
            "anomaly_score": float(s[0]),
            "is_anomaly": bool(s[0] > 0.7),
            "lstm_error": float(l[0]),
            "iso_score": float(i[0]),
        }

    def shap_attribution(self):
        """Exact Shapley values of the anomaly score over the 5 metrics.

        Computes marginal contributions by replacing absent features with
        their training-mean values and evaluating all 2^N coalitions.

        Returns
        -------
        dict mapping each metric name to a percentage weight (sums to ~100).
        """
        if not self.is_fitted or len(self.metrics_buffer) < self.window_size:
            return {k: 100.0 / self._N_FEATURES for k in METRIC_KEYS}

        w = np.array(self.metrics_buffer, dtype=np.float32)
        n = self._N_FEATURES

        # Build coalition windows: replace absent features with training mean.
        rows, keys = [], []
        for r in range(n + 1):
            for c in combinations(range(n), r):
                x = w.copy()
                x[-1] = self.mu
                x[-1, list(c)] = w[-1, list(c)]
                rows.append(x)
                keys.append(c)

        vals = dict(
            zip(keys, self._calibrated_scores(np.array(rows))[0].tolist())
        )

        phi = np.zeros(n)
        for i in range(n):
            for c, v in vals.items():
                if i in c:
                    continue
                wt = factorial(len(c)) * factorial(n - len(c) - 1) / factorial(n)
                phi[i] += wt * (vals[tuple(sorted(c + (i,)))] - v)

        total = np.abs(phi).sum() or 1.0
        pct = np.abs(phi) / total * 100.0
        return dict(zip(METRIC_KEYS, np.round(pct, 2).tolist()))
