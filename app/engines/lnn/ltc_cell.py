"""LTC cell — explicit ODE implementation.

The Liquid Time-Constant ODE (Hasani et al., 2021):

    τᵢ(x, h) * dh/dt = -h(t) + f_gate(x, h) * A

where:
  h      = hidden state
  x      = current input
  τ      = state-dependent time constant (strictly > τ_min)
  f_gate = synaptic gating (sigmoid)
  A      = learnable per-neuron attractor bias

Semi-implicit Euler discretisation (stable for stiff ODEs):

    h(t+Δt) = [h(t) + Δt * f_gate * A / τ] / [1 + Δt / τ]

This is the exact solution for the linear ODE (treating f and τ as constant
over [t, t+Δt]), avoiding the instability of plain Euler for large Δt.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class LTCCell(nn.Module):
    """Single Liquid Time-Constant recurrent cell.

    Parameters
    ----------
    input_size:  dimensionality of the input x
    hidden_size: number of LTC neurons
    tau_min:     floor for time constants (prevents division by near-zero)
    """

    def __init__(self, input_size: int, hidden_size: int, tau_min: float = 0.1):
        if not _HAS_TORCH:
            raise ImportError("torch required — pip install torch")
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_min = tau_min

        combined = input_size + hidden_size

        # Synaptic gating: f(x, h) → [0, 1]^hidden
        self.W_gate = nn.Linear(combined, hidden_size)

        # Time constant head: τ(x, h) > τ_min
        self.W_tau = nn.Linear(combined, hidden_size)

        # Per-neuron learnable attractor bias A
        self.A = nn.Parameter(torch.ones(hidden_size))

        # Bias τ toward larger initial values so training starts stable
        nn.init.constant_(self.W_tau.bias, 1.0)
        nn.init.xavier_uniform_(self.W_gate.weight, gain=0.5)

    def forward(
        self,
        x: "torch.Tensor",
        h: "torch.Tensor",
        dt: "float | torch.Tensor" = 1.0,
    ) -> "torch.Tensor":
        """Advance hidden state by one time step.

        Args:
            x:  (batch, input_size) — current input
            h:  (batch, hidden_size) — previous hidden state
            dt: scalar or (batch,) — elapsed time in units (default 1 = one 5-min interval)

        Returns:
            h_new: (batch, hidden_size)
        """
        xh = torch.cat([x, h], dim=-1)                      # (B, in+hid)

        gate = torch.sigmoid(self.W_gate(xh))                # (B, hid)
        tau  = F.softplus(self.W_tau(xh)) + self.tau_min     # (B, hid) > tau_min

        # Semi-implicit Euler for ẋ = (-x + gate * A) / τ
        h_new = (h + dt * gate * self.A / tau) / (1.0 + dt / tau)
        return h_new

    def init_hidden(self, batch_size: int, device=None) -> "torch.Tensor":
        return torch.zeros(batch_size, self.hidden_size, device=device)
