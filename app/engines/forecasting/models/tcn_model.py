"""Temporal Convolutional Network (TCN) forecaster.

TCNs replace recurrence with dilated causal convolutions, giving:
  - Parallelisable training (vs sequential RNN unrolls)
  - Explicit receptive field control via dilation
  - Better gradient flow on long sequences

Architecture: stacked residual dilated-conv blocks (WaveNet / Bai et al. 2018
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling", https://arxiv.org/abs/1803.01271).

Like the LNN, trained with pinball loss and exposes the same interface
(fit / predict_quantiles / predict_spike_probs / save / load).

Requires: torch. Install with `pip install torch`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import QuantileForecast, DEFAULT_QUANTILES
from .base import ForecastModel
from .lnn_model import _spike_labels, _SPIKE_THRESHOLDS, _N_SPIKE, _pinball_torch

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:
    torch = None

    class _MissingModule:
        pass

    class _MissingNN:
        Module = _MissingModule

    nn = _MissingNN()
    _HAS_TORCH = False


class _CausalConv1d(nn.Module):
    """Causal conv1d that zero-pads the left so output length == input length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )

    def forward(self, x):
        return self.conv(x)[:, :, : x.shape[2]]  # trim the right-side padding


class _ResidualBlock(nn.Module):
    """Dilated causal-conv residual block with weight-norm and dropout."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.causal1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.causal2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        # Apply weight_norm to the underlying nn.Conv1d (which has a 'weight' param)
        nn.utils.weight_norm(self.causal1.conv)
        nn.utils.weight_norm(self.causal2.conv)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.causal1(x))
        out = self.drop(out)
        out = self.relu(self.causal2(out))
        out = self.drop(out)
        return self.relu(out + x)  # residual


class _TCNBackbone(nn.Module):
    """TCN backbone: input projection → stacked residual blocks."""

    def __init__(
        self,
        n_features: int,
        channels: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            _ResidualBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x):
        # x: (B, T, F) → transpose to (B, F, T) for Conv1d
        h = self.input_proj(x.permute(0, 2, 1))
        for block in self.blocks:
            h = block(h)
        return h[:, :, -1]  # take the final time-step: (B, channels)


class TCNQuantileModel(ForecastModel):
    """Dilated TCN with a quantile head (pinball) and a spike-risk head (BCE).

    The two heads share the TCN backbone and are trained jointly.
    Spike probabilities are available via predict_spike_probs() after fit().
    """

    name = "tcn"

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        channels: int = 32,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 60,
        lr: float = 1e-3,
        seq_len: int = 12,
        spike_loss_weight: float = 0.5,
        device: str | None = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch not installed; `pip install torch`")
        self.quantiles = quantiles
        self.channels = channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.seq_len = seq_len
        self.spike_loss_weight = spike_loss_weight
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._backbone: "_TCNBackbone | None" = None
        self._head: "nn.Linear | None" = None
        self._spike_head: "nn.Linear | None" = None

    @property
    def _is_fitted(self) -> bool:
        return (
            self._backbone is not None
            and self._head is not None
            and self._spike_head is not None
        )

    def _build(self, n_features: int):
        backbone = _TCNBackbone(
            n_features, self.channels, self.n_layers, self.kernel_size, self.dropout
        ).to(self.device)
        head = nn.Linear(self.channels, len(self.quantiles)).to(self.device)
        spike_head = nn.Linear(self.channels, _N_SPIKE).to(self.device)
        return backbone, head, spike_head

    def _windows(self, X: np.ndarray) -> "torch.Tensor":
        """Build (N, seq_len, d) overlapping windows; left-pad the start."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        n, d = x.shape
        pad = x[:1].repeat(self.seq_len - 1, 1)
        xp = torch.cat([pad, x], dim=0)
        return torch.stack([xp[i: i + self.seq_len] for i in range(n)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNQuantileModel":
        self._backbone, self._head, self._spike_head = self._build(X.shape[1])
        seqs = self._windows(X)
        target = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        spike_target = torch.tensor(
            _spike_labels(y), dtype=torch.float32, device=self.device
        )
        params = (
            list(self._backbone.parameters())
            + list(self._head.parameters())
            + list(self._spike_head.parameters())
        )
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            hidden = self._backbone(seqs)
            pred = self._head(hidden)
            logits = self._spike_head(hidden)
            pinball = _pinball_torch(pred, target, self.quantiles)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, spike_target)
            loss = pinball + self.spike_loss_weight * bce
            loss.backward()
            opt.step()
        return self

    def predict_quantiles(self, X: np.ndarray, target_times: Sequence) -> QuantileForecast:
        if not self._is_fitted:
            raise RuntimeError("TCNQuantileModel.fit() must be called before predict_quantiles()")
        self._backbone.eval()
        self._head.eval()
        try:
            with torch.no_grad():
                seqs = self._windows(X)
                pred = self._head(self._backbone(seqs)).cpu().numpy()
        finally:
            self._backbone.train()
            self._head.train()
        return self._as_forecast(target_times, pred)

    def predict_spike_probs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-row spike probabilities after fit().

        Returns dict with keys ``gt_300``, ``gt_1000``, ``lt_0``, each (N,).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TCNQuantileModel.fit() must be called before predict_spike_probs()"
            )
        self._backbone.eval()
        self._spike_head.eval()
        try:
            with torch.no_grad():
                seqs = self._windows(X)
                logits = self._spike_head(self._backbone(seqs)).cpu().numpy()
        finally:
            self._backbone.train()
            self._spike_head.train()
        logits = np.clip(logits, -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return {key: probs[:, i] for i, (key, _, _) in enumerate(_SPIKE_THRESHOLDS)}

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        torch.save({
            "backbone": self._backbone.state_dict(),
            "head": self._head.state_dict(),
            "spike_head": self._spike_head.state_dict(),
            "config": {
                "channels": self.channels,
                "n_layers": self.n_layers,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "quantiles": list(self.quantiles),
                "seq_len": self.seq_len,
                "spike_loss_weight": self.spike_loss_weight,
            },
        }, path)

    def load(self, path: str, n_features: int) -> "TCNQuantileModel":
        self._backbone, self._head, self._spike_head = self._build(n_features)
        ckpt = torch.load(path, map_location=self.device)
        self._backbone.load_state_dict(ckpt["backbone"])
        self._head.load_state_dict(ckpt["head"])
        if "spike_head" in ckpt:
            self._spike_head.load_state_dict(ckpt["spike_head"])
        return self
