"""Quantile output heads for the LTC model.

Two head variants are provided:

  QuantileHead (original) —
    Single linear projection: hidden_size → 3 quantiles, with monotone sort.

  RegimeAwareQuantileHead (Sprint H) —
    Two-layer MLP with regime conditioning. The hidden state is concatenated
    with 3 regime context features derived from the last observed price:
    [log1p(|last_price|), is_elevated (≥$300), is_spike (≥$1000)].
    P10/P50/P90 are constructed via softplus deltas: P10 = P50 − softplus(lo),
    P90 = P50 + softplus(hi), guaranteeing non-crossing without sorting.

Pinball (quantile regression) loss is defined here so the trainer
imports from one place.
"""
from __future__ import annotations

from typing import Sequence

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# Regime thresholds — match domain.nem.adapter.classify_regime defaults
_ELEVATED_THRESHOLD = 300.0
_SPIKE_THRESHOLD = 1000.0


class QuantileHead(nn.Module):
    """Linear projection: hidden_size → len(quantiles), monotone-sorted.

    Parameters
    ----------
    hidden_size: must match LTCModel.hidden_size
    quantiles:   ordered quantile levels (default: P10, P50, P90)
    """

    def __init__(
        self,
        hidden_size: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch required — pip install torch")
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.head = nn.Linear(hidden_size, len(quantiles))

    def forward(self, h: "torch.Tensor") -> "torch.Tensor":
        """Return (batch, n_quantiles) — monotone non-crossing."""
        raw = self.head(h)
        sorted_q, _ = torch.sort(raw, dim=-1)
        return sorted_q


class RegimeAwareQuantileHead(nn.Module):
    """Two-layer probabilistic head with regime-conditioning (Sprint H).

    Architecture:
        concat([h, regime_ctx])  → fc1 → ReLU → {fc_p50, fc_lo, fc_hi}
        P50 = fc_p50(z)
        P10 = P50 − softplus(fc_lo(z))   # guaranteed P10 ≤ P50
        P90 = P50 + softplus(fc_hi(z))   # guaranteed P50 ≤ P90

    regime_ctx = [log1p(|last_price|), is_elevated, is_spike] — 3 scalars.
    """

    def __init__(
        self,
        hidden_size: int,
        mid_size: int | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch required — pip install torch")
        super().__init__()
        self.quantiles = tuple(quantiles)
        _mid = mid_size if mid_size is not None else max(16, hidden_size // 2)
        _in = hidden_size + 3   # h dims + 3 regime context dims
        self.fc1 = nn.Linear(_in, _mid)
        self.fc_p50 = nn.Linear(_mid, 1)
        self.fc_lo = nn.Linear(_mid, 1)   # softplus → delta below P50 = P50 - P10
        self.fc_hi = nn.Linear(_mid, 1)   # softplus → delta above P50 = P90 - P50

    def _regime_context(self, last_price: "torch.Tensor") -> "torch.Tensor":
        """Derive 3 regime features from the last observed price in the window."""
        log_p = torch.log1p(last_price.abs())
        is_elev = (last_price >= _ELEVATED_THRESHOLD).float()
        is_spike = (last_price >= _SPIKE_THRESHOLD).float()
        return torch.stack([log_p, is_elev, is_spike], dim=-1)

    def forward(
        self,
        h: "torch.Tensor",
        last_price: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return (batch, 3) — [P10, P50, P90], non-crossing by construction.

        Args:
            h:          (batch, hidden_size) — LTC final hidden state
            last_price: (batch,)             — most recent price in the window
        """
        ctx = self._regime_context(last_price)        # (batch, 3)
        x = torch.cat([h, ctx], dim=-1)               # (batch, hidden_size + 3)
        z = torch.relu(self.fc1(x))                   # (batch, mid_size)
        p50 = self.fc_p50(z).squeeze(-1)              # (batch,)
        delta_lo = nn.functional.softplus(self.fc_lo(z)).squeeze(-1)  # (batch,) ≥ 0
        delta_hi = nn.functional.softplus(self.fc_hi(z)).squeeze(-1)  # (batch,) ≥ 0
        p10 = p50 - delta_lo
        p90 = p50 + delta_hi
        return torch.stack([p10, p50, p90], dim=-1)   # (batch, 3)


def pinball_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    quantiles: Sequence[float],
) -> "torch.Tensor":
    """Mean pinball loss across quantiles and batch.

    Args:
        pred:      (batch, n_quantiles)
        target:    (batch,) or (batch, 1) — actual price
        quantiles: matching quantile levels

    Returns:
        scalar loss tensor
    """
    qs = torch.tensor(
        list(quantiles), dtype=pred.dtype, device=pred.device
    ).view(1, -1)
    t = target.view(-1, 1)
    diff = t - pred
    return torch.maximum(qs * diff, (qs - 1.0) * diff).mean()
