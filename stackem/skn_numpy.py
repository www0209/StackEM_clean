"""
stackem.skn_numpy
===================

Numpy evaluation of a trained Segment Kernel Network as a ``KernelProvider``
for the junction closure, plus an sklearn (torch-free) fallback trainer that
produces weights in the very same container so that the whole pipeline can be
exercised on a CPU-only machine.

Beyond the trained tau range the kernels are in their exact steady state:
    s_G(xi, tau > tau_max) = s_G(xi, tau_max)
    s_M(xi, tau > tau_max) = s_M(xi, tau_max)
    a  (xi, tau > tau_max) = a(xi, tau_max) - (tau - tau_max)      (unit outflow drift)
and below the trained range they vanish like sqrt(tau) (early-time boundary layer).
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from .closure import KernelProvider
from .skn_model import SKNWeights, SKNConfig, forward_numpy, make_fourier_matrix
from .kernel_dataset import m_amp_of


class SKNNumpyProvider(KernelProvider):
    def __init__(self, weights: SKNWeights, log10_tau_range=(-6.0, 1.0), chunk: int = 250_000):
        self.w = weights
        self.lt_min, self.lt_max = log10_tau_range
        self.chunk = int(chunk)          # points per forward pass (256 hidden x 4 B x 250k = 0.25 GB per activation)

    def raw(self, X: np.ndarray) -> np.ndarray:
        return forward_numpy(self.w, X)

    def evaluate(self, desc, xi, tau) -> Dict[str, np.ndarray]:
        """Chunked: a stack of long rails asks for 10^7-10^8 kernel points at once (B x S x M^2/2); evaluating them
        in one numpy forward pass would need tens of GB."""
        n = len(tau)
        if n <= self.chunk:
            return self._evaluate(desc, xi, tau)
        parts = [self._evaluate(desc[i:i + self.chunk], xi[i:i + self.chunk], tau[i:i + self.chunk]) for i in range(0, n, self.chunk)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _evaluate(self, desc, xi, tau) -> Dict[str, np.ndarray]:
        desc = np.asarray(desc, np.float64); xi = np.asarray(xi, np.float64); tau = np.asarray(tau, np.float64)
        lt = np.log10(np.clip(tau, 1e-300, None))
        ltc = np.clip(lt, self.lt_min, self.lt_max)
        from .kernel_dataset import LAM_CLAMP
        X = np.stack([xi, ltc, desc[:, 0], desc[:, 1], desc[:, 2], np.clip(desc[:, 3], *LAM_CLAMP)], 1)
        Y = self.raw(X).astype(np.float64)
        tau_c = 10.0 ** ltc
        st = np.sqrt(tau_c)
        sG = Y[:, 0] * st; sM = Y[:, 1] * st * m_amp_of(desc[:, 0], desc[:, 1], desc[:, 2], desc[:, 3]); a = Y[:, 2] * st
        small = lt < self.lt_min                      # below range: sqrt(tau) scaling of the boundary layer
        if np.any(small):
            f = np.sqrt(tau[small] / 10.0 ** self.lt_min)
            sG[small] *= f; sM[small] *= f; a[small] *= f
        large = lt > self.lt_max                      # above range: steady state + unit outflow drift
        if np.any(large):
            a[large] -= (tau[large] - tau_c[large])
        return dict(a=a, s_G=sG, s_M=sM)


# ----------------------------------------------------------------------------
# torch-free fallback trainer (sklearn MLPRegressor with the same Fourier features)
# ----------------------------------------------------------------------------
def train_sklearn_fallback(X: np.ndarray, Y: np.ndarray, cfg: SKNConfig | None = None, max_iter: int = 200,
                           seed: int = 0, verbose: bool = True) -> SKNWeights:
    """Train a tanh MLP with scikit-learn on the augmented (Fourier) features and export
    it as SKNWeights.  Accuracy is below the torch trainer (no cosine schedule, no
    physics term, tanh instead of SiLU) but it exercises the identical forward pass."""
    from sklearn.neural_network import MLPRegressor
    cfg = cfg or SKNConfig(hidden=[128, 128, 128, 128], activation="tanh", n_fourier=16)
    cfg.activation = "tanh"
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
    B = make_fourier_matrix(cfg.n_fourier, cfg.fourier_scale, cfg.seed)
    u = (X - x_mu) / x_sd
    proj = u[:, :2] @ B
    H = np.concatenate([u, np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], 1)
    T = (Y - y_mu) / y_sd
    mlp = MLPRegressor(hidden_layer_sizes=tuple(cfg.hidden), activation="tanh", solver="adam", batch_size=4096,
                       learning_rate_init=1e-3, max_iter=max_iter, random_state=seed, verbose=False,
                       early_stopping=True, n_iter_no_change=20, tol=1e-6)
    mlp.fit(H, T)
    W = [c.astype(np.float32) for c in mlp.coefs_]; b = [c.astype(np.float32) for c in mlp.intercepts_]
    w = SKNWeights(cfg, x_mu, x_sd, y_mu, y_sd, B, W, b)
    if verbose:
        pred = forward_numpy(w, X)
        rel = np.linalg.norm(pred - Y, axis=0) / np.linalg.norm(Y - Y.mean(0), axis=0)
        print(f"sklearn fallback: iterations {mlp.n_iter_}, rel-L2 per target {np.round(rel, 4)}")
    return w
