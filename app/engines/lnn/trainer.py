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
import shutil
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature extraction ─────────────────────────────────────────────────────

_N_FEATURES = 6    # v1: price, demand, avail, headroom, tod_sin, tod_cos (backward-compat)
_N_FEATURES_V2 = 11  # v2: v1 + dow, log1p_price, fcas_raise6s_norm, renewable_pct, constraint_norm

# Feature index constants for v2 (used by multistep rollout to advance TOD/DOW)
_F_PRICE      = 0
_F_DEMAND     = 1
_F_AVAIL      = 2
_F_HEADROOM   = 3
_F_TOD_SIN    = 4
_F_TOD_COS    = 5
_F_DOW        = 6   # day-of-week fraction 0-1
_F_LOG_PRICE  = 7   # log1p(|price|) * sign(price) — handles spikes without exploding gradients
_F_FCAS_R6S   = 8   # FCAS raise 6s price / 1000 — proxy for grid stress
_F_RENEW_PCT  = 9   # renewable % of demand (solar+wind / demand) — 0 if unknown
_F_CONSTR     = 10  # binding constraint count / 10 — 0 if unknown


def _extract_features(interval: dict[str, Any], n_features: int = _N_FEATURES_V2) -> np.ndarray:
    """Extract feature vector from a dispatch interval dict.

    n_features=6 → v1 (backward-compatible with old checkpoints).
    n_features=11 → v2 (enhanced: DOW, log-price, FCAS, renewable %, constraints).
    """
    price = float(interval.get("price_rrp", 0.0))
    demand = float(interval.get("demand_mw", 0.0))
    avail = float(interval.get("availability_mw", 0.0))
    headroom = max(avail - demand, 0.0)

    # Time-of-day cyclical encoding (local NEM time, UTC+10 approximation)
    vt = interval.get("valid_time")
    if isinstance(vt, str):
        try:
            vt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
        except ValueError:
            vt = None
    if vt is None:
        vt = datetime.now(timezone.utc)
    minutes = (vt.hour * 60 + vt.minute)
    frac = minutes / 1440.0
    tod_sin = float(np.sin(2 * np.pi * frac))
    tod_cos = float(np.cos(2 * np.pi * frac))

    if n_features == _N_FEATURES:
        return np.array([price, demand, avail, headroom, tod_sin, tod_cos], dtype=np.float32)

    # v2 extended features
    dow = float(vt.weekday()) / 6.0   # 0 (Mon) → 1 (Sun)
    # Log-scaled price: log1p(|p|)*sign(p) — compresses spikes, preserves negatives
    log_price = float(np.log1p(abs(price)) * np.sign(price)) if price != 0 else 0.0
    # FCAS raise 6s: market stress indicator, scaled to ~0-1 range ($0-$1000)
    fcas_r6s = float(interval.get("fcas_raise_6s", 0.0) or 0.0) / 1000.0
    # Renewable % of demand (solar+wind cleared MW / demand) — 0 if not provided
    renewable_pct = float(interval.get("renewable_pct", 0.0) or 0.0)
    # Binding constraint count, normalised — 0 if not provided
    constraint_norm = min(float(interval.get("constraint_count", 0) or 0), 20.0) / 20.0

    return np.array([
        price, demand, avail, headroom, tod_sin, tod_cos,
        dow, log_price, fcas_r6s, renewable_pct, constraint_norm,
    ], dtype=np.float32)


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
        feature_version: int = 2,  # 1=6 features (compat), 2=11 features (enriched)
    ):
        self.region = region
        self.weights_dir = Path(weights_dir)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.min_samples = min_samples
        self.lr = lr

        self.use_regime_head = use_regime_head
        self._feature_version = feature_version
        self.n_features: int = _N_FEATURES_V2 if feature_version == 2 else _N_FEATURES
        self._buffer: list[np.ndarray] = []   # feature vectors
        self._prices: list[float] = []         # next-interval prices (targets)
        self._norm = _Normaliser()
        self._is_trained = False
        self.last_trained_at: datetime | None = None
        self.last_metrics: dict = {}
        self.training_rows: int = 0
        # Model ops
        self.torch_version: str | None = None
        self.last_inference_latency_ms: float | None = None
        self._has_backup: bool = False         # True after first rollback-safe save

    # ── Data accumulation ────────────────────────────────────────────────

    def accumulate_interval(self, interval: dict[str, Any]) -> None:
        """Add one dispatch interval to the training buffer."""
        fv = _extract_features(interval, n_features=self.n_features)
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
            raise RuntimeError(
                "torch is not installed — LNN cannot train. "
                "Add torch to the ml extras and rebuild the image: "
                "`pip install torch --index-url https://download.pytorch.org/whl/cpu`"
            )
        self.torch_version = torch.__version__

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
        X = np.stack(self._buffer[:n])            # (n, n_features)
        y = np.array(self._prices[:n], dtype=np.float32)  # (n,)

        # Derive n_features from the actual buffer — guards against any race where
        # _load_weights set self.n_features to a different value between accumulation
        # and training. The buffer is authoritative.
        actual_n = X.shape[1]
        if actual_n != self.n_features:
            logger.info(
                "LTC[%s] buffer has %d features, self.n_features=%d — using buffer count",
                self.region, actual_n, self.n_features,
            )
            self.n_features = actual_n
            self._feature_version = 2 if actual_n == _N_FEATURES_V2 else 1

        self._norm.fit(X)
        X_norm = self._norm.transform(X).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LTCModel(self.n_features, self.hidden_size)
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

    def predict_multistep_from_buffer(self, steps: int) -> list[dict[str, float]] | None:
        """Autoregressive rollout from the internal buffer.

        Each step predicts P10/P50/P90, then feeds P50 back as the next
        interval's price. TOD features advance by one 5-minute interval per
        step. Returns None if the model is not trained or buffer is too short.
        """
        if len(self._buffer) < 2:
            return None

        _INTERVAL_RAD = 2.0 * np.pi * 5.0 / (24.0 * 60.0)

        seq: list[np.ndarray] = list(self._buffer[-self.seq_len:])
        results: list[dict[str, float]] = []

        for step in range(steps):
            recent = [
                {"price_rrp": float(fv[0]), "demand_mw": float(fv[1]),
                 "availability_mw": float(fv[2]), "valid_time": None}
                for fv in seq[-self.seq_len:]
            ]
            pred = self.predict(recent)
            if pred is None:
                return results if results else None

            results.append(pred)

            # Build next feature vector: use predicted P50 as price, advance TOD
            last_fv = seq[-1]
            p50 = float(pred["p50"])
            demand = float(last_fv[1])
            avail = float(last_fv[2])
            headroom = max(avail - demand, 0.0)
            tod_sin = float(last_fv[4])
            tod_cos = float(last_fv[5])
            angle = np.arctan2(tod_sin, tod_cos)
            new_angle = angle + _INTERVAL_RAD
            next_fv = np.array(
                [p50, demand, avail, headroom, float(np.sin(new_angle)), float(np.cos(new_angle))],
                dtype=np.float32,
            )
            seq = seq[1:] + [next_fv]  # slide the window forward

        return results

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
        X = np.stack([_extract_features(iv, n_features=self.n_features) for iv in intervals])
        X_norm = self._norm.transform(X).astype(np.float32)

        # Pad if fewer than seq_len samples available
        if len(X_norm) < self.seq_len:
            pad = np.tile(X_norm[:1], (self.seq_len - len(X_norm), 1))
            X_norm = np.vstack([pad, X_norm])

        device = next(self._model.parameters()).device
        inp = torch.tensor(X_norm, dtype=torch.float32, device=device).unsqueeze(0)

        _t0 = _time.monotonic()
        with torch.no_grad():
            h = self._model(inp)
            if self.use_regime_head:
                raw_last_price = float(intervals[-1].get("price_rrp", 0.0))
                lp = torch.tensor([[raw_last_price]], dtype=torch.float32, device=device).squeeze(-1)
                q = self._head(h, lp).squeeze(0).cpu().numpy()
            else:
                q = self._head(h).squeeze(0).cpu().numpy()
        self.last_inference_latency_ms = round((_time.monotonic() - _t0) * 1000, 2)

        return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}

    # ── Persistence ──────────────────────────────────────────────────────

    def _weights_path(self) -> Path:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return self.weights_dir / f"ltc_{self.region}.pt"

    def _meta_path(self) -> Path:
        return self.weights_dir / f"ltc_{self.region}_meta.json"

    def _backup_weights_path(self) -> Path:
        return self.weights_dir / f"ltc_{self.region}.backup.pt"

    def _backup_meta_path(self) -> Path:
        return self.weights_dir / f"ltc_{self.region}_meta.backup.json"

    def _save_weights(self, device) -> None:
        try:
            import torch
            wp = self._weights_path()
            mp = self._meta_path()
            # Copy current checkpoint to backup before overwriting (enables rollback)
            if wp.exists():
                shutil.copy2(wp, self._backup_weights_path())
            if mp.exists():
                shutil.copy2(mp, self._backup_meta_path())
            self._has_backup = self._backup_weights_path().exists()

            torch.save({
                "model": self._model.state_dict(),
                "head": self._head.state_dict(),
            }, wp)
            mp.write_text(json.dumps({
                "region": self.region,
                "hidden_size": self.hidden_size,
                "seq_len": self.seq_len,
                "n_features": self.n_features,
                "head_type": "regime" if self.use_regime_head else "linear",
                "norm": self._norm.to_dict(),
            }), encoding="utf-8")
        except Exception as exc:
            logger.warning("LTC weight save failed: %s", exc)

    def rollback(self) -> bool:
        """Restore the previous checkpoint (the one saved before the last train()).

        Returns True on success, False when no backup exists or load fails.
        Useful when the latest checkpoint degrades evaluation metrics.
        """
        bwp = self._backup_weights_path()
        bmp = self._backup_meta_path()
        if not bwp.exists() or not bmp.exists():
            logger.warning("LTC[%s] rollback requested but no backup checkpoint found", self.region)
            return False
        try:
            # Swap backup → active
            shutil.copy2(bwp, self._weights_path())
            shutil.copy2(bmp, self._meta_path())
            ok = self._load_weights()
            if ok:
                logger.info("LTC[%s] rolled back to previous checkpoint", self.region)
            return ok
        except Exception as exc:
            logger.warning("LTC[%s] rollback failed: %s", self.region, exc)
            return False

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
            saved_n = int(meta.get("n_features", _N_FEATURES))
            # Only sync n_features from the checkpoint when the buffer is still empty.
            # If the buffer already has data (e.g. from startup bootstrap), keep the
            # current n_features so accumulate_interval and train() stay consistent.
            # This prevents a race between forecast-warmup _load_weights and the
            # background training thread, which would otherwise cause a feature-count
            # mismatch in np.stack(self._buffer).
            if not self._buffer:
                self.n_features = saved_n
                self._feature_version = 2 if saved_n == _N_FEATURES_V2 else 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = LTCModel(saved_n, meta["hidden_size"]).to(device)
            head_type = meta.get("head_type", "linear")
            if head_type == "regime":
                self._head = RegimeAwareQuantileHead(meta["hidden_size"]).to(device)
                self.use_regime_head = True
            else:
                self._head = QuantileHead(meta["hidden_size"]).to(device)
                self.use_regime_head = False

            ckpt = torch.load(wp, map_location=device, weights_only=True)
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
