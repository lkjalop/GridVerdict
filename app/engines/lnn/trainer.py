"""LTC trainer — prepares rolling-window dispatch data and trains the model.

Training lifecycle:
  1. accumulate_interval() — called after each dispatch poll (or archive backfill)
  2. is_ready_to_train()   — True once min_samples intervals in the buffer
  3. train()               — runs the training loop; saves weights to disk
  4. load()                — restores weights for inference

The trainer is region-scoped; instantiate one per NEM region.
Feature normalisation (zero-mean / unit-variance) is fit on training data
and stored alongside weights so inference applies identical scaling.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature extraction ─────────────────────────────────────────────────────

_N_FEATURES = 6  # price, demand, availability, headroom, tod_sin, tod_cos


def _extract_features(interval: dict[str, Any]) -> np.ndarray:
    """Extract the 6-dim feature vector from a dispatch interval dict."""
    price = float(interval.get("price_rrp", 0.0))
    demand = float(interval.get("demand_mw", 0.0))
    avail = float(interval.get("availability_mw", 0.0))
    headroom = max(avail - demand, 0.0)

    # Time-of-day cyclical encoding (local NEM time, approximated from UTC+10)
    vt = interval.get("valid_time")
    if isinstance(vt, str):
        vt = datetime.fromisoformat(vt)
    if vt is None:
        vt = datetime.now(timezone.utc)
    minutes = vt.hour * 60 + vt.minute
    frac = minutes / 1440.0
    tod_sin = float(np.sin(2 * np.pi * frac))
    tod_cos = float(np.cos(2 * np.pi * frac))

    return np.array([price, demand, avail, headroom, tod_sin, tod_cos], dtype=np.float32)


# ── Normaliser ──────────────────────────────────────────────────────────────

class _Normaliser:
    """Fit on training data; apply at inference time."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_Normaliser":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return X
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_Normaliser":
        n = cls()
        if d.get("mean") is not None:
            n.mean = np.array(d["mean"], dtype=np.float32)
            n.std = np.array(d["std"], dtype=np.float32)
        return n


# ── Trainer ─────────────────────────────────────────────────────────────────

class LTCTrainer:
    """Accumulates dispatch intervals and trains the LTC forecaster.

    Parameters
    ----------
    region:       NEM region identifier (e.g. "NSW1")
    weights_dir:  directory to save/load model weights (default: data/lnn_weights/)
    seq_len:      rolling window length in 5-min intervals (default: 12 = 1 hour)
    hidden_size:  LTC hidden neurons
    epochs:       training epochs per call to train()
    min_samples:  intervals needed before first training run
    lr:           Adam learning rate
    """

    def __init__(
        self,
        region: str,
        weights_dir: str = "data/lnn_weights",
        seq_len: int = 12,
        hidden_size: int = 32,
        epochs: int = 60,
        min_samples: int = 288,   # 1 day of 5-min intervals
        lr: float = 1e-3,
        use_regime_head: bool = False,
    ):
        self.region = region
        self.weights_dir = Path(weights_dir)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.min_samples = min_samples
        self.lr = lr

        self.use_regime_head = use_regime_head
        self._buffer: list[np.ndarray] = []   # feature vectors
        self._prices: list[float] = []         # next-interval prices (targets)
        self._norm = _Normaliser()
        self._is_trained = False
        self.last_trained_at: datetime | None = None
        self.last_metrics: dict = {}
        self.training_rows: int = 0

    # ── Data accumulation ────────────────────────────────────────────────

    def accumulate_interval(self, interval: dict[str, Any]) -> None:
        """Add one dispatch interval to the training buffer.

        The target is `price_rrp` one step ahead — set by the caller after
        the *next* interval arrives.  We store features first; the caller
        can call set_last_target() once the next price is known.
        """
        fv = _extract_features(interval)
        self._buffer.append(fv)
        # Target for the *previous* row is the current price
        if len(self._buffer) > 1:
            self._prices.append(float(interval.get("price_rrp", 0.0)))

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def is_ready_to_train(self) -> bool:
        return len(self._prices) >= self.min_samples

    # ── Training ─────────────────────────────────────────────────────────

    def train(self) -> float | None:
        """Train the LTC model on the accumulated buffer. Returns final loss."""
        try:
            import torch
            import torch.optim as optim
        except ImportError:
            logger.warning("torch not available — LNN training skipped")
            return None

        from app.engines.lnn.ltc_model import LTCModel
        from app.engines.lnn.distribution import (
            QuantileHead,
            RegimeAwareQuantileHead,
            pinball_loss,
        )

        n = len(self._prices)
        if n < self.min_samples:
            return None

        # Feature matrix: rows 0..n-1, targets rows 1..n (next price)
        X = np.stack(self._buffer[:n])            # (n, _N_FEATURES)
        y = np.array(self._prices[:n], dtype=np.float32)  # (n,)

        self._norm.fit(X)
        X_norm = self._norm.transform(X).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LTCModel(_N_FEATURES, self.hidden_size)
        if self.use_regime_head:
            head = RegimeAwareQuantileHead(self.hidden_size)
        else:
            head = QuantileHead(self.hidden_size)
        model.to(device)
        head.to(device)

        windows = model.build_windows(X_norm, self.seq_len, device=device)
        targets = torch.tensor(y, dtype=torch.float32, device=device)
        # Last price per window: raw (un-normalised) price in feature col 0
        last_prices_raw = torch.tensor(X[:, 0], dtype=torch.float32, device=device)

        params = list(model.parameters()) + list(head.parameters())
        opt = optim.Adam(params, lr=self.lr)

        final_loss = float("inf")
        for epoch in range(self.epochs):
            opt.zero_grad()
            h = model(windows)
            if self.use_regime_head:
                pred = head(h, last_prices_raw)
            else:
                pred = head(h)
            loss = pinball_loss(pred, targets, head.quantiles)
            loss.backward()
            opt.step()
            if epoch == self.epochs - 1:
                final_loss = float(loss.item())

        self._model = model
        self._head = head
        self._is_trained = True
        self.last_trained_at = datetime.now(timezone.utc)
        self.training_rows = n
        self.last_metrics = {"mae": 0.0, "rmse": 0.0, "final_loss": round(final_loss, 6)}
        self._save_weights(device)
        logger.info("LTC[%s] trained on %d intervals, loss=%.4f", self.region, n, final_loss)
        return final_loss

    # ── Inference ────────────────────────────────────────────────────────

    def predict_from_buffer(self) -> dict[str, float] | None:
        """Predict using the last seq_len intervals from the internal buffer."""
        if len(self._buffer) < 2:
            return None
        recent = [
            {"price_rrp": float(fv[0]), "demand_mw": float(fv[1]),
             "availability_mw": float(fv[2]), "valid_time": None}
            for fv in self._buffer[-self.seq_len:]
        ]
        return self.predict(recent)

    def predict(self, recent_intervals: list[dict[str, Any]]) -> dict[str, float] | None:
        """Predict P10/P50/P90 for the next dispatch interval.

        Args:
            recent_intervals: last seq_len intervals (most recent last)

        Returns:
            {"p10": float, "p50": float, "p90": float} or None if not ready
        """
        if not self._is_trained:
            if not self._load_weights():
                return None

        try:
            import torch
        except ImportError:
            return None

        from app.engines.lnn.ltc_model import LTCModel
        from app.engines.lnn.distribution import QuantileHead

        intervals = recent_intervals[-self.seq_len:]
        X = np.stack([_extract_features(iv) for iv in intervals])
        X_norm = self._norm.transform(X).astype(np.float32)

        # Pad if fewer than seq_len samples available
        if len(X_norm) < self.seq_len:
            pad = np.tile(X_norm[:1], (self.seq_len - len(X_norm), 1))
            X_norm = np.vstack([pad, X_norm])

        device = next(self._model.parameters()).device
        inp = torch.tensor(X_norm, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            h = self._model(inp)
            if self.use_regime_head:
                # Last price from the original (un-normalised) feature vector
                raw_last_price = float(intervals[-1].get("price_rrp", 0.0))
                lp = torch.tensor([[raw_last_price]], dtype=torch.float32, device=device).squeeze(-1)
                q = self._head(h, lp).squeeze(0).cpu().numpy()
            else:
                q = self._head(h).squeeze(0).cpu().numpy()

        return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}

    # ── Persistence ──────────────────────────────────────────────────────

    def _weights_path(self) -> Path:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return self.weights_dir / f"ltc_{self.region}.pt"

    def _meta_path(self) -> Path:
        return self.weights_dir / f"ltc_{self.region}_meta.json"

    def _save_weights(self, device) -> None:
        try:
            import torch
            torch.save({
                "model": self._model.state_dict(),
                "head": self._head.state_dict(),
            }, self._weights_path())
            self._meta_path().write_text(json.dumps({
                "region": self.region,
                "hidden_size": self.hidden_size,
                "seq_len": self.seq_len,
                "n_features": _N_FEATURES,
                "head_type": "regime" if self.use_regime_head else "linear",
                "norm": self._norm.to_dict(),
            }), encoding="utf-8")
        except Exception as exc:
            logger.warning("LTC weight save failed: %s", exc)

    def _load_weights(self) -> bool:
        try:
            import torch
            from app.engines.lnn.ltc_model import LTCModel
            from app.engines.lnn.distribution import QuantileHead, RegimeAwareQuantileHead

            wp = self._weights_path()
            mp = self._meta_path()
            if not wp.exists() or not mp.exists():
                return False

            meta = json.loads(mp.read_text(encoding="utf-8"))
            self._norm = _Normaliser.from_dict(meta["norm"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = LTCModel(meta["n_features"], meta["hidden_size"]).to(device)
            head_type = meta.get("head_type", "linear")
            if head_type == "regime":
                self._head = RegimeAwareQuantileHead(meta["hidden_size"]).to(device)
                self.use_regime_head = True
            else:
                self._head = QuantileHead(meta["hidden_size"]).to(device)
                self.use_regime_head = False

            ckpt = torch.load(wp, map_location=device)
            self._model.load_state_dict(ckpt["model"])
            self._head.load_state_dict(ckpt["head"])
            self._model.eval()
            self._head.eval()
            self._is_trained = True
            logger.info(
                "LTC[%s] weights loaded from %s (head_type=%s)", self.region, wp, head_type
            )
            return True
        except Exception as exc:
            logger.warning("LTC weight load failed: %s", exc)
            return False
