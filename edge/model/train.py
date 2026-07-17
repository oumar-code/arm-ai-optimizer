"""
train.py
Trains two complementary models for predictive maintenance:
  1. Isolation Forest — unsupervised anomaly detection
  2. LSTM — Remaining Useful Life (RUL) regression

Generates synthetic training data that mirrors the telemetry simulator
output so that the pre-trained model bundled in the container is
immediately useful on demo day.

Usage:
    python train.py [--out-dir /path/to/models]
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("model-train")

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_HEALTHY = 5_000       # healthy samples for IF training
N_DEGRADED = 1_000      # degraded samples (still used for scaler fit)
WINDOW_LEN = 60         # time-steps per LSTM window
N_FEATURES = 21         # 3 sensors × 7 features each
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 30
LSTM_BATCH = 64
LSTM_LR = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    f"{sensor}__{feat}"
    for sensor in ["vibration_g", "temperature_c", "current_a"]
    for feat in ["mean", "std", "min", "max", "rms", "peak_to_peak", "kurtosis"]
]


def make_feature_vector(
    vib_mean: float, temp_mean: float, curr_mean: float, severity: float
) -> np.ndarray:
    """Build a feature vector matching the ingestor's extract_features output."""
    rng = np.random.default_rng()

    def sensor_features(mean: float, std_base: float, fault_mult: float) -> list:
        eff_mean = mean * (1 + (fault_mult - 1) * severity)
        eff_std = std_base * (1 + severity * 0.5)
        samples = rng.normal(eff_mean, eff_std, 60)
        return [
            float(np.mean(samples)),
            float(np.std(samples)),
            float(np.min(samples)),
            float(np.max(samples)),
            float(np.sqrt(np.mean(samples ** 2))),
            float(np.ptp(samples)),
            float(np.mean(((samples - np.mean(samples)) / (np.std(samples) + 1e-9)) ** 4)),
        ]

    feats = (
        sensor_features(vib_mean, 0.05, 3.2)
        + sensor_features(temp_mean, 1.5, 1.4)
        + sensor_features(curr_mean, 0.8, 1.6)
    )
    return np.array(feats, dtype=np.float32)


def generate_dataset(n_healthy: int, n_degraded: int):
    X_list, rul_list = [], []

    # Healthy samples (severity 0)
    for _ in range(n_healthy):
        x = make_feature_vector(0.5, 45.0, 12.0, 0.0)
        X_list.append(x)
        rul_list.append(48.0)

    # Degraded samples (severity 0.1 → 1.0)
    for _ in range(n_degraded):
        sev = np.random.uniform(0.1, 1.0)
        x = make_feature_vector(0.5, 45.0, 12.0, sev)
        X_list.append(x)
        rul_list.append(max(0.0, 48.0 * (1.0 - sev)))

    X = np.stack(X_list)
    rul = np.array(rul_list, dtype=np.float32)
    return X, rul


def build_lstm_sequences(X: np.ndarray, rul: np.ndarray, window: int = WINDOW_LEN):
    """Slide a window over the feature matrix to create LSTM input sequences."""
    n = len(X)
    if n < window:
        raise ValueError(f"Need at least {window} samples, got {n}")
    seqs, targets = [], []
    for i in range(n - window):
        seqs.append(X[i: i + window])
        targets.append(rul[i + window - 1])
    return np.stack(seqs), np.array(targets, dtype=np.float32)


class RULPredictor(nn.Module):
    def __init__(self, input_size: int, hidden: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_isolation_forest(X_healthy: np.ndarray) -> IsolationForest:
    log.info("Training Isolation Forest on %d healthy samples …", len(X_healthy))
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_healthy)
    log.info("Isolation Forest trained.")
    return clf


def train_lstm(X_seq: np.ndarray, y: np.ndarray) -> nn.Module:
    log.info("Training LSTM RUL predictor on %d sequences …", len(X_seq))
    X_t = torch.from_numpy(X_seq)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=LSTM_BATCH, shuffle=True)

    model = RULPredictor(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(LSTM_EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0:
            log.info("Epoch %3d/%d  loss=%.4f", epoch + 1, LSTM_EPOCHS, total_loss / len(dataset))

    log.info("LSTM training complete.")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/app/model", help="Directory to save models")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating training data …")
    X, rul = generate_dataset(N_HEALTHY, N_DEGRADED)

    # Fit and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    scaler_path = out_dir / "scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("Scaler saved to %s", scaler_path)

    # Isolation Forest (on healthy only)
    X_healthy = X_scaled[: N_HEALTHY]
    iforest = train_isolation_forest(X_healthy)
    if_path = out_dir / "isolation_forest.pkl"
    with open(if_path, "wb") as fh:
        pickle.dump(iforest, fh)
    log.info("Isolation Forest saved to %s", if_path)

    # LSTM RUL predictor
    try:
        X_seq, y_seq = build_lstm_sequences(X_scaled, rul)
        lstm = train_lstm(X_seq, y_seq)
        lstm_path = out_dir / "lstm_rul.pt"
        torch.save(lstm.state_dict(), lstm_path)
        log.info("LSTM state dict saved to %s", lstm_path)
    except ValueError as exc:
        log.warning("Skipping LSTM training: %s", exc)

    log.info("All models saved to %s", out_dir)


if __name__ == "__main__":
    main()
