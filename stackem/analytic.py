"""
stackem.analytic
==================

Closed-form reference solutions (uniform kappa) used to validate the
finite-volume solver and the learned kernels.

All functions are dimensionless:  xi = x/L,  tau = kappa t / L^2.
"""
from __future__ import annotations
import numpy as np


def s_G_blocked_series(xi, tau, n_terms: int = 400):
    """EM response of a blocked line, zero initial stress, unit source (S=1):
        s(xi,tau) = (xi - 1/2) + sum_{n odd} 4/(n^2 pi^2) cos(n pi xi) exp(-n^2 pi^2 tau)
    (Korhonen 1993).  Steady state s = xi - 1/2, tensile at xi = 1 for G > 0."""
    xi = np.asarray(xi, dtype=float)[None, :]
    tau = np.asarray(tau, dtype=float)[:, None]
    s = (xi - 0.5) * np.ones_like(tau)
    for n in range(1, 2 * n_terms, 2):
        s = s + 4.0 / (n * n * np.pi * np.pi) * np.cos(n * np.pi * xi) * np.exp(-n * n * np.pi * np.pi * tau)
    return s


def a_unit_flux_series(xi, tau, n_terms: int = 600):
    """Unit-flux Green's function of a line with q(0) = 1, q(1) = 0, zero IC:
        a(xi,tau) = -tau + (xi - xi^2/2 - 1/3) + sum_{n>=1} 2/(n^2 pi^2) cos(n pi xi) exp(-n^2 pi^2 tau)
    The mean stress decreases linearly (atoms leave through x = 0)."""
    xi = np.asarray(xi, dtype=float)[None, :]
    tau = np.asarray(tau, dtype=float)[:, None]
    a = -tau + (xi - 0.5 * xi ** 2 - 1.0 / 3.0)
    for n in range(1, n_terms + 1):
        a = a + 2.0 / (n * n * np.pi * np.pi) * np.cos(n * np.pi * xi) * np.exp(-n * n * np.pi * np.pi * tau)
    return a


def s_G_semi_infinite_end(tau):
    """Early-time end stress of the EM problem: s(1,tau) = 2 sqrt(tau/pi)  (valid for tau << 0.1)."""
    return 2.0 * np.sqrt(np.asarray(tau) / np.pi)


def a_semi_infinite_end(tau):
    """Early-time boundary value of the unit-flux Green's function: a(0,tau) = -2 sqrt(tau/pi)."""
    return -2.0 * np.sqrt(np.asarray(tau) / np.pi)
