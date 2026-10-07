"""
stackem.skn_model
===================

The Segment Kernel Network (SKN): a neural surrogate of the
three single-segment kernels of the Korhonen equation as functions of the
similarity-scaled inputs

    u = [xi, log10 tau, theta, r, rho_J, lam]  ->  [s_G, s_M/m_amp, a] / sqrt(tau).

Architecture: per-feature standardisation -> random Fourier features on
(xi, log10 tau) (the two coordinates along which the kernels have sharp
early-time boundary layers) concatenated with the standardised descriptors ->
MLP (SiLU) -> 3 outputs -> per-target de-standardisation.

Training loss = data MSE (per-target weighted) + w_pde * PDE residual, where
the residual of every kernel in its own dimensionless form

    d s/d tau - [ kappa_hat' (s_xi - S) + kappa_hat s_xi_xi ] = 0

is evaluated by automatic differentiation at the training points (kappa_hat,
kappa_hat', S are stored with the dataset).  The networks of the final experiments use
w_pde = 0, i.e. they are trained on reference-solver data only (the physics enters through the
similarity-scaled inputs/targets and the closed-form tails, not through the loss); w_pde = 0.05
is kept as an ablation (output folder ablation_pde).

The file also defines the *exact* numpy re-implementation of the forward pass
(``forward_numpy``) so that trained weights can be evaluated without torch
(used by the closure on CPU machines and by the tests that compare the two
implementations).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import numpy as np

try:  # torch is optional at import time (the numpy path needs none of it)
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None; nn = None
    _HAS_TORCH = False


@dataclass
class SKNConfig:
    n_in: int = 6
    n_out: int = 3
    n_fourier: int = 24          # frequencies for (xi, log10 tau)
    fourier_scale: float = 2.0
    hidden: List[int] = field(default_factory=lambda: [256, 256, 256, 256, 256, 256])
    activation: str = "silu"
    target_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    seed: int = 0


def _act_np(x, name):
    if name == "silu":
        return x * (0.5 * (1.0 + np.tanh(0.5 * x)))          # x * sigmoid(x), overflow-safe
    if name == "tanh":
        return np.tanh(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    raise ValueError(name)


def make_fourier_matrix(n_fourier: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 1.0, (2, n_fourier)) * scale).astype(np.float32)   # (2, F)


@dataclass
class SKNWeights:
    """Framework-independent container of trained weights (saved as .npz)."""
    cfg: SKNConfig
    x_mu: np.ndarray; x_sd: np.ndarray
    y_mu: np.ndarray; y_sd: np.ndarray
    fourier_B: np.ndarray                  # (2, F)
    W: List[np.ndarray]; b: List[np.ndarray]

    def save(self, path: str):
        np.savez(path, cfg=np.array([json.dumps(asdict(self.cfg))]), x_mu=self.x_mu, x_sd=self.x_sd,
                 y_mu=self.y_mu, y_sd=self.y_sd, fourier_B=self.fourier_B,
                 **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": v for i, v in enumerate(self.b)})

    @staticmethod
    def random(cfg: SKNConfig, seed: int = 0) -> "SKNWeights":
        """Untrained weights of the right shapes (tests, smoke runs)."""
        rng = np.random.default_rng(seed)
        dims = [cfg.n_in + 2 * cfg.n_fourier] + list(cfg.hidden) + [cfg.n_out]
        W = [rng.normal(0.0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32) for i in range(len(dims) - 1)]
        b = [np.zeros(dims[i + 1], np.float32) for i in range(len(dims) - 1)]
        return SKNWeights(cfg, np.zeros(cfg.n_in, np.float32), np.ones(cfg.n_in, np.float32), np.zeros(cfg.n_out, np.float32),
                          np.ones(cfg.n_out, np.float32), make_fourier_matrix(cfg.n_fourier, cfg.fourier_scale, seed), W, b)

    @staticmethod
    def load(path: str) -> "SKNWeights":
        d = np.load(path, allow_pickle=False)
        cfg = SKNConfig(**json.loads(str(d["cfg"][0])))
        n = len(cfg.hidden) + 1
        return SKNWeights(cfg, d["x_mu"], d["x_sd"], d["y_mu"], d["y_sd"], d["fourier_B"],
                          [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)])


def forward_numpy(w: SKNWeights, X: np.ndarray) -> np.ndarray:
    """Exact numpy forward pass. X (N, 6) raw features -> (N, 3) raw targets."""
    X = np.asarray(X, dtype=np.float32)
    u = (X - w.x_mu) / w.x_sd
    proj = u[:, :2] @ w.fourier_B                       # (N, F)
    h = np.concatenate([u, np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], 1)
    for i in range(len(w.W) - 1):
        h = _act_np(h @ w.W[i] + w.b[i], w.cfg.activation)
    y = h @ w.W[-1] + w.b[-1]
    return y * w.y_sd + w.y_mu


if _HAS_TORCH:

    class SKN(nn.Module):
        """PyTorch module with the identical forward pass."""

        def __init__(self, cfg: SKNConfig, x_mu, x_sd, y_mu, y_sd):
            super().__init__()
            self.cfg = cfg
            self.register_buffer("x_mu", torch.as_tensor(np.asarray(x_mu, np.float32)))
            self.register_buffer("x_sd", torch.as_tensor(np.asarray(x_sd, np.float32)))
            self.register_buffer("y_mu", torch.as_tensor(np.asarray(y_mu, np.float32)))
            self.register_buffer("y_sd", torch.as_tensor(np.asarray(y_sd, np.float32)))
            self.register_buffer("B", torch.as_tensor(make_fourier_matrix(cfg.n_fourier, cfg.fourier_scale, cfg.seed)))
            dims = [cfg.n_in + 2 * cfg.n_fourier] + list(cfg.hidden) + [cfg.n_out]
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            self.act = {"silu": nn.SiLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[cfg.activation]
            for m in self.layers:
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        def forward(self, X):
            u = (X - self.x_mu) / self.x_sd
            proj = u[:, :2] @ self.B
            h = torch.cat([u, torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], 1)
            for lyr in self.layers[:-1]:
                h = self.act(lyr(h))
            y = self.layers[-1](h)
            return y * self.y_sd + self.y_mu

        def export(self) -> SKNWeights:
            W = [l.weight.detach().cpu().numpy().T.astype(np.float32) for l in self.layers]
            b = [l.bias.detach().cpu().numpy().astype(np.float32) for l in self.layers]
            return SKNWeights(self.cfg, self.x_mu.cpu().numpy(), self.x_sd.cpu().numpy(), self.y_mu.cpu().numpy(),
                              self.y_sd.cpu().numpy(), self.B.cpu().numpy(), W, b)

        @staticmethod
        def from_weights(w: SKNWeights) -> "SKN":
            m = SKN(w.cfg, w.x_mu, w.x_sd, w.y_mu, w.y_sd)
            with torch.no_grad():
                m.B.copy_(torch.as_tensor(w.fourier_B))
                for l, W, b in zip(m.layers, w.W, w.b):
                    l.weight.copy_(torch.as_tensor(W.T)); l.bias.copy_(torch.as_tensor(b))
            return m

    def pde_residual(model: "SKN", X: "torch.Tensor", AUX: "torch.Tensor") -> "torch.Tensor":
        """Dimensionless Korhonen residual of the three kernels at points X.

        The network predicts y = s / sqrt(tau).  With s = sqrt(tau) y:
            s_tau = y/(2 sqrt(tau)) + sqrt(tau) y_tau,  s_xi = sqrt(tau) y_xi,  s_xixi = sqrt(tau) y_xixi
        and the residual (divided by sqrt(tau)) reads
            y/(2 tau) + y_tau - kappa_hat' (y_xi - S/sqrt(tau)) - kappa_hat y_xixi = 0
        with S = 1 (s_G), M_hat/m_amp (s_M), 0 (a).  X columns: xi, log10 tau, ...;
        AUX columns: kappa_hat, dkappa_hat/dxi, M_hat/m_amp.  y_tau = y_logtau / (tau ln 10).
        """
        X = X.clone().requires_grad_(True)
        Y = model(X)
        res = []
        tau = 10.0 ** X[:, 1:2]
        st = torch.sqrt(tau)
        ln10 = math.log(10.0)
        kh, dkh, Mh = AUX[:, 0:1], AUX[:, 1:2], AUX[:, 2:3]
        for k, S in enumerate([torch.ones_like(kh), Mh, torch.zeros_like(kh)]):
            y = Y[:, k:k + 1]
            g = torch.autograd.grad(y, X, torch.ones_like(y), create_graph=True)[0]
            y_xi, y_lt = g[:, 0:1], g[:, 1:2]
            y_tau = y_lt / (tau * ln10)
            g2 = torch.autograd.grad(y_xi, X, torch.ones_like(y_xi), create_graph=True)[0]
            y_xixi = g2[:, 0:1]
            r = y / (2 * tau) + y_tau - dkh * (y_xi - S / st) - kh * y_xixi
            # weight by tau: every term becomes O(1) over the whole tau range (y/(2 tau) * tau = y/2,
            # y_tau * tau = y_logtau / ln 10); with sqrt(tau) the residual was dominated by tau ~ 1e-6 (|r| ~ 500)
            res.append(r * tau)
        return torch.cat(res, 1)


def numpy_kernel_provider_from_weights(w: SKNWeights):
    """Factory used by the closure module (avoids a circular import)."""
    from .skn_numpy import SKNNumpyProvider
    return SKNNumpyProvider(w)
