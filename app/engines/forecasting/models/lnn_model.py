"""Liquid Neural Network forecaster (CfC via the ncps library).

The continuous-time battery. Its genuine value is volatility handling and
robustness, NOT headline accuracy — it sits in the ensemble, not on a pedestal.
Trained with pinball loss so it emits the same quantiles as the other batteries.

Also exposes a spike-risk head (trained jointly with BCE loss) that outputs
event probabilities: P(price > $300), P(price > $1000), P(price < $0).

Requires: torch, ncps. Install with `pip install torch ncps`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import QuantileForecast, DEFAULT_QUANTILES
from .base import ForecastModel

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from ncps.torch import CfC
    _HAS_NCPS = True
except ImportError:
    _HAS_NCPS = False

# Spike thresholds — order determines column indices in the spike head output.
# Each entry: (output_key, direction, threshold_$/MWh)
_SPIKE_THRESHOLDS: list[tuple[str, str, float]] = [
    ("gt_300",   "gt", 300.0),
    ("gt_1000",  "gt", 1000.0),
    ("lt_0",     "lt", 0.0),
]
_N_SPIKE = len(_SPIKE_THRESHOLDS)


def _pinball_torch(pred, target, quantiles):
    """Vectorised pinball loss across quantiles (pred: (B, Q), target: (B, 1))."""
    qs = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    diff = target - pred
    return torch.maximum(qs * diff, (qs - 1.0) * diff).mean()


def _spike_labels(y: np.ndarray) -> np.ndarray:
    """Build (N, _N_SPIKE) binary labels from price targets y (shape (N,))."""
    cols = []
    for _, direction, thresh in _SPIKE_THRESHOLDS:
        if direction == "gt":
            cols.append((y > thresh).astype(np.float32))
        else:
            cols.append((y < thresh).astype(np.float32))
    return np.column_stack(cols)


class LNNQuantileModel(ForecastModel):
    """CfC with a quantile head (pinball) and a spike-risk head (BCE).

    The two heads share the same CfC backbone and are trained jointly.
    Spike probabilities are available via predict_spike_probs() after fit().
    """

    name = "lnn_cfc"

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        hidden: int = 32,
        epochs: int = 60,
        lr: float = 1e-3,
        seq_len: int = 12,
        spike_loss_weight: float = 0.5,
        device: str | None = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch/ncps not installed; `pip install torch ncps`")
        if not _HAS_NCPS:
            raise ImportError("ncps not installed; `pip install ncps`")
        self.quantiles = quantiles
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seq_len = seq_len
        self.spike_loss_weight = spike_loss_weight
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._cell: "CfC | None" = None
        self._head: "nn.Linear | None" = None
        self._spike_head: "nn.Linear | None" = None

    @property
    def _is_fitted(self) -> bool:
        return (
            self._cell is not None
            and self._head is not None
            and self._spike_head is not None
        )

    def _build(self, n_features: int) -> tuple["CfC", "nn.Linear", "nn.Linear"]:
        from ncps.torch import CfC
        cell = CfC(n_features, self.hidden).to(self.device)
        head = nn.Linear(self.hidden, len(self.quantiles)).to(self.device)
        spike_head = nn.Linear(self.hidden, _N_SPIKE).to(self.device)
        return cell, head, spike_head

    def _windows(self, X: np.ndarray) -> "torch.Tensor":
        """Build (N, seq_len, d) overlapping windows; left-pad the start."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        n, d = x.shape
        pad = x[:1].repeat(self.seq_len - 1, 1)
        xp = torch.cat([pad, x], dim=0)
        return torch.stack([xp[i:i + self.seq_len] for i in range(n)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LNNQuantileModel":
        self._cell, self._head, self._spike_head = self._build(X.shape[1])
        seqs = self._windows(X)
        target = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        spike_target = torch.tensor(
            _spike_labels(y), dtype=torch.float32, device=self.device
        )
        params = (
            list(self._cell.parameters())
            + list(self._head.parameters())
            + list(self._spike_head.parameters())
        )
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            out, _ = self._cell(seqs)
            hidden_last = out[:, -1, :]
            pred = self._head(hidden_last)
            logits = self._spike_head(hidden_last)
            pinball = _pinball_torch(pred, target, self.quantiles)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, spike_target)
            loss = pinball + self.spike_loss_weight * bce
            loss.backward()
            opt.step()
        return self

    def predict_quantiles(self, X: np.ndarray, target_times: Sequence) -> QuantileForecast:
        if not self._is_fitted:
            raise RuntimeError("LNNQuantileModel.fit() must be called before predict_quantiles()")
        with torch.no_grad():
            seqs = self._windows(X)
            out, _ = self._cell(seqs)
            pred = self._head(out[:, -1, :]).cpu().numpy()
        return self._as_forecast(target_times, pred)

    def predict_spike_probs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-row spike probabilities after fit().

        Returns a dict with keys ``gt_300``, ``gt_1000``, ``lt_0``, each an
        ndarray of shape (N,) containing probabilities in [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LNNQuantileModel.fit() must be called before predict_spike_probs()"
            )
        with torch.no_grad():
            seqs = self._windows(X)
            out, _ = self._cell(seqs)
            logits = self._spike_head(out[:, -1, :]).cpu().numpy()
        logits = np.clip(logits, -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {key: probs[:, i] for i, (key, _, _) in enumerate(_SPIKE_THRESHOLDS)}

    def save(self, path: str) -> None:
        """Persist weights — call after fit() if you want to reuse without retraining."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        torch.save({
            "cell": self._cell.state_dict(),
            "head": self._head.state_dict(),
            "spike_head": self._spike_head.state_dict(),
            "config": {
                "hidden": self.hidden,
                "quantiles": list(self.quantiles),
                "seq_len": self.seq_len,
                "spike_loss_weight": self.spike_loss_weight,
            }
        }, path)

    def load(self, path: str, n_features: int) -> "LNNQuantileModel":
        """Restore weights. n_features must match the training shape."""
        self._cell, self._head, self._spike_head = self._build(n_features)
        ckpt = torch.load(path, map_location=self.device)
        self._cell.load_state_dict(ckpt["cell"])
        self._head.load_state_dict(ckpt["head"])
        if "spike_head" in ckpt:
            self._spike_head.load_state_dict(ckpt["spike_head"])
        return self
