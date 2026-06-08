"""LTC sequence model — stacks LTCCell layers over a rolling input window.

Processes (batch, seq_len, n_features) → final hidden state (batch, hidden_size),
which the QuantileHead in distribution.py projects to P10/P50/P90.

The model optionally accepts per-step Δt tensors so irregular dispatch
intervals (maintenance windows, clock changes) are handled correctly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from app.engines.lnn.ltc_cell import LTCCell


class LTCModel(nn.Module):
    """Multi-layer LTC recurrent model.

    Parameters
    ----------
    input_size:  number of input features (matches FEATURE_COLUMNS length)
    hidden_size: neurons per layer
    n_layers:    depth of the LTC stack (1 is sufficient for NEM; 2 for richer dynamics)
    dropout:     applied between layers (not after final layer)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        n_layers: int = 1,
        tau_min: float = 0.1,
        dropout: float = 0.0,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch required — pip install torch")
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        cells = []
        for i in range(n_layers):
            in_sz = input_size if i == 0 else hidden_size
            cells.append(LTCCell(in_sz, hidden_size, tau_min=tau_min))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        seq: "torch.Tensor",
        dts: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Run the full sequence and return the final hidden state.

        Args:
            seq:  (batch, seq_len, input_size)
            dts:  (batch, seq_len) optional per-step Δt; defaults to 1.0 per step

        Returns:
            h_final: (batch, hidden_size)
        """
        batch, seq_len, _ = seq.shape
        device = seq.device

        h_stack = [cell.init_hidden(batch, device) for cell in self.cells]

        for t in range(seq_len):
            x = seq[:, t, :]
            dt = dts[:, t].unsqueeze(-1) if dts is not None else 1.0

            for i, cell in enumerate(self.cells):
                h_stack[i] = cell(x, h_stack[i], dt)
                x = h_stack[i]
                if self.dropout is not None and i < self.n_layers - 1:
                    x = self.dropout(x)

        return h_stack[-1]

    def build_windows(
        self,
        X: np.ndarray,
        seq_len: int,
        device=None,
    ) -> "torch.Tensor":
        """Construct overlapping windows from a (n, d) feature array.

        Left-pads with the first row so every sample has seq_len steps,
        preserving the full training set size.
        """
        import torch
        x = torch.tensor(X, dtype=torch.float32, device=device)
        n, d = x.shape
        pad = x[:1].expand(seq_len - 1, d)
        xp = torch.cat([pad, x], dim=0)
        windows = torch.stack([xp[i: i + seq_len] for i in range(n)])
        return windows  # (n, seq_len, d)

    def integrated_gradients(
        self,
        seq: "torch.Tensor",
        head_module: "nn.Module",
        target_quantile_idx: int = 1,
        baseline: "torch.Tensor | None" = None,
        steps: int = 50,
    ) -> "torch.Tensor":
        """Integrated Gradients attribution for the LTC sequence model.

        Standard SHAP/TreeSHAP does not work for ODE-based recurrent networks.
        IG works for any differentiable PyTorch model: linearly interpolate from
        baseline to input in `steps` steps, accumulate ∂output/∂input at each
        step via autograd, then scale by (input - baseline).

        Args:
            seq:                (1, seq_len, input_size) input sequence tensor
            head_module:        nn.Module that maps h_final → quantile outputs
            target_quantile_idx: index into head output to differentiate (1 = P50)
            baseline:           reference input (zeros if None)
            steps:              Riemann approximation steps (50 is sufficient)

        Returns:
            attribution: (seq_len, input_size) tensor — which feature at which
                         timestep drove the prediction. Positive = pushed price up.

        NLP usage: "LNN forecast $167 driven by: notice_lor_active t-4 (+$23),
                    demand_ramp t-1 (+$18), headroom t-2 (+$12), renew_frac (+$8)"
        """
        import torch
        if baseline is None:
            baseline = torch.zeros_like(seq)

        self.eval()
        head_module.eval()

        alphas = torch.linspace(0.0, 1.0, steps, device=seq.device)
        grad_accum = torch.zeros_like(seq[0])  # (seq_len, input_size)

        for alpha in alphas:
            inp = (baseline + alpha * (seq - baseline)).detach().requires_grad_(True)
            h = self.forward(inp)
            out = head_module(h)
            target = out[0, target_quantile_idx]
            target.backward()
            if inp.grad is not None:
                grad_accum = grad_accum + inp.grad[0].detach()

        avg_grad = grad_accum / steps
        ig = (seq[0] - baseline[0]) * avg_grad  # (seq_len, input_size)
        return ig
