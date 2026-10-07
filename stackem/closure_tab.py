"""
stackem.closure_tab
=====================

Tabulated kernel look-up for the junction closure (the look-up of the final engine).

Why
---
The closure asks the kernel provider for the end-point kernels of every segment at all
M(M+1)/2 lags of the output time grid: B x S x M^2/2 network evaluations per batch of rails,
i.e. the surrogate cost grows quadratically with the number of output times although the
kernels of one segment are smooth functions of a single variable, log tau.

What
----
``TabulatedProvider`` wraps any ``KernelProvider``.  In every ``evaluate`` call it detects the
runs of identical (descriptor, xi) rows - the closure expands each segment to P or M copies -
and, per run, evaluates the wrapped provider on only ``n_tab`` log-spaced tau points spanning
the run's tau range.  The sqrt(tau)-scaled kernels are then linearly interpolated in log tau
(they are smooth there; the network itself learns them as functions of log tau) and the two
closed-form tails of the providers are re-applied outside the trained range:

    tau < tau_min :  s = s(tau_min) sqrt(tau / tau_min)         (early-time boundary layer)
    tau > tau_max :  a = a(tau_max) - (tau - tau_max),  s_G, s_M frozen  (steady state + unit outflow)

so the wrapper reproduces the wrapped provider up to the interpolation error (~1e-4 relative at
n_tab = 192 over 7 decades, see tests).  Cost per segment: 4 x n_tab evaluations, independent
of M.  ``TabulatedTorchProvider`` does the same in torch and stays differentiable (the table
values depend on the descriptors through the network, the interpolation weights on tau).

Together with a distilled small network (``train_distill.py``) and a coarser output grid
(M = 32, closure error 3e-3 rel-L2, 0.1 % in t_nuc, E0) this is the engine of the final
experiments (config closure.mode = "tabulated"); the teacher-engine comparison keeps the direct
evaluation (mode "direct", configs/base3_teacher.json).
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from .closure import KernelProvider


class TabulatedProvider(KernelProvider):
    def __init__(self, base: KernelProvider, n_tab: int = 192):
        self.base = base
        self.n_tab = int(n_tab)
        self.lt_min = float(getattr(base, "lt_min", -6.0)); self.lt_max = float(getattr(base, "lt_max", 1.0))
        self.n_base_evals = 0          # bookkeeping for the timing experiment
        self.n_points = 0

    def evaluate(self, desc: np.ndarray, xi: np.ndarray, tau: np.ndarray) -> Dict[str, np.ndarray]:
        desc = np.asarray(desc, np.float64); xi = np.asarray(xi, np.float64); tau = np.asarray(tau, np.float64)
        N = len(tau); K = self.n_tab
        if N == 0:
            return {k: np.zeros(0) for k in ("a", "s_G", "s_M")}
        key = np.concatenate([desc, xi[:, None]], 1)                                   # (N,5)
        change = np.any(key[1:] != key[:-1], axis=1)
        starts = np.flatnonzero(np.concatenate([[True], change])); G = len(starts)
        gid = np.cumsum(np.concatenate([[0], change.astype(np.int64)]))                # (N,)
        lt = np.log10(np.clip(tau, 1e-300, None)); ltc = np.clip(lt, self.lt_min, self.lt_max)
        lo = np.minimum.reduceat(ltc, starts); hi = np.maximum.reduceat(ltc, starts); hi = np.maximum(hi, lo + 1e-3)
        u = np.linspace(0.0, 1.0, K)
        LT = lo[:, None] + (hi - lo)[:, None] * u[None, :]                              # (G,K)
        kd = key[starts]
        dg = np.repeat(kd[:, :4], K, axis=0); xg = np.repeat(kd[:, 4], K); tg = (10.0 ** LT).reshape(-1)
        base = self.base.evaluate(dg, xg, tg); self.n_base_evals += G * K; self.n_points += N
        stg = np.sqrt(tg)
        tab = {k: (v / stg).reshape(G, K) for k, v in base.items()}                    # sqrt(tau)-scaled tables
        pos = (ltc - lo[gid]) / (hi - lo)[gid] * (K - 1)
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, K - 2); w = np.clip(pos - i0, 0.0, 1.0)
        tau_c = 10.0 ** ltc; st = np.sqrt(tau_c)
        f_small = np.where(lt < self.lt_min, np.sqrt(tau / 10.0 ** self.lt_min), 1.0)
        out = {}
        for k, T in tab.items():
            y = T[gid, i0] * (1.0 - w) + T[gid, i0 + 1] * w
            v = y * st * f_small
            if k == "a":
                v = v - np.where(lt > self.lt_max, tau - tau_c, 0.0)
            out[k] = v
        return out


class TabulatedTorchProvider:
    """Torch twin of ``TabulatedProvider`` (wraps ``closure_torch.SKNTorchProvider``); differentiable."""

    def __init__(self, base, n_tab: int = 192):
        import torch  # noqa: F401
        self.base = base; self.n_tab = int(n_tab); self.device = base.device
        self.lt_min = float(getattr(base, "lt_min", -6.0)); self.lt_max = float(getattr(base, "lt_max", 1.0))
        self.n_base_evals = 0; self.n_points = 0

    def evaluate(self, desc, xi, tau):
        import torch
        N = tau.shape[0]; K = self.n_tab; dev = tau.device
        if N == 0:
            return {k: torch.zeros(0, device=dev, dtype=tau.dtype) for k in ("a", "s_G", "s_M")}
        key = torch.cat([desc, xi[:, None]], 1)
        change = (key[1:] != key[:-1]).any(dim=1)
        first = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), change])
        starts = torch.nonzero(first).flatten(); G = starts.numel()
        gid = torch.cumsum(first.to(torch.int64), 0) - 1
        lt = torch.log10(torch.clamp(tau, min=1e-300)); ltc = torch.clamp(lt, self.lt_min, self.lt_max)
        lo = torch.full((G,), float("inf"), device=dev, dtype=tau.dtype).scatter_reduce(0, gid, ltc.detach(), reduce="amin")
        hi = torch.full((G,), float("-inf"), device=dev, dtype=tau.dtype).scatter_reduce(0, gid, ltc.detach(), reduce="amax")
        hi = torch.maximum(hi, lo + 1e-3)
        u = torch.linspace(0.0, 1.0, K, device=dev, dtype=tau.dtype)
        LT = lo[:, None] + (hi - lo)[:, None] * u[None, :]
        kd = key[starts]
        dg = kd[:, :4].repeat_interleave(K, dim=0); xg = kd[:, 4].repeat_interleave(K); tg = (10.0 ** LT).reshape(-1)
        base = self.base.evaluate(dg, xg, tg); self.n_base_evals += G * K; self.n_points += N
        stg = torch.sqrt(tg)
        tab = {k: (v / stg).reshape(G, K) for k, v in base.items()}
        pos = (ltc - lo[gid]) / (hi - lo)[gid] * (K - 1)
        i0 = torch.clamp(torch.floor(pos.detach()).to(torch.int64), 0, K - 2); w = torch.clamp(pos - i0.to(pos.dtype), 0.0, 1.0)
        tau_c = 10.0 ** ltc; st = torch.sqrt(tau_c)
        f_small = torch.where(lt < self.lt_min, torch.sqrt(tau / 10.0 ** self.lt_min), torch.ones_like(tau))
        out = {}
        flat_i0 = gid * K + i0
        for k, T in tab.items():
            Tf = T.reshape(-1)
            y = Tf[flat_i0] * (1.0 - w) + Tf[flat_i0 + 1] * w
            v = y * st * f_small
            if k == "a":
                v = v - torch.where(lt > self.lt_max, tau - tau_c, torch.zeros_like(tau))
            out[k] = v
        return out


def wrap_provider(provider, mode: str = "direct", n_tab: int = 192):
    """Return the provider as configured: mode 'direct' (unchanged) or 'tabulated' (wrapped)."""
    if mode == "direct" or provider is None:
        return provider
    if mode == "tabulated":
        if hasattr(provider, "device") and not isinstance(provider, KernelProvider):
            return TabulatedTorchProvider(provider, n_tab)
        return TabulatedProvider(provider, n_tab)
    raise ValueError(mode)


# ----------------------------------------------------------------------------------------------------------
# direct table build for the closure (no run detection on the expanded (B,S,P) inputs)
# ----------------------------------------------------------------------------------------------------------
# ``close_rails`` / ``close_rails_torch`` call these when their provider is a Tabulated*Provider.  The result is the
# same as feeding the expanded arrays through ``evaluate`` (one table per segment and end, same lo/hi grid), but the
# (B,S,P) descriptor expansion, the row comparison and the scatter reductions of the generic wrapper are skipped:
# the tables are built from the (B,S) descriptors and queried by gather.  This is what makes the tabulated closure
# cheaper than the direct one at every M, not only asymptotically.

def _np_grid(ltc):
    lo = ltc.min(axis=-1); hi = np.maximum(ltc.max(axis=-1), lo + 1e-3)
    return lo, hi


def _np_tables(base, desc_flat, xi_val, lo, hi, K):
    """sqrt(tau)-scaled kernel tables (Gn,K) of the wrapped provider on the per-row grid lo..hi (log10 tau)."""
    Gn = desc_flat.shape[0]
    u = np.linspace(0.0, 1.0, K)
    LT = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    tg = (10.0 ** LT).reshape(-1)
    out = base.evaluate(np.repeat(desc_flat, K, axis=0), np.full(Gn * K, float(xi_val)), tg)
    stg = np.sqrt(tg)
    return {k: (v / stg).reshape(Gn, K) for k, v in out.items()}


def _np_interp(T, tau, lt, ltc, lo, hi, K, lt_min, lt_max, is_a):
    pos = (ltc - lo[:, None]) / (hi - lo)[:, None] * (K - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, K - 2); w = np.clip(pos - i0, 0.0, 1.0)
    y = np.take_along_axis(T, i0, 1) * (1.0 - w) + np.take_along_axis(T, i0 + 1, 1) * w
    tau_c = 10.0 ** ltc
    v = y * np.sqrt(tau_c) * np.where(lt < lt_min, np.sqrt(tau / 10.0 ** lt_min), 1.0)
    if is_a:
        v = v - np.where(lt > lt_max, tau - tau_c, 0.0)
    return v


def closure_kernels_tabulated_np(prov: TabulatedProvider, desc, tau_lag, tau_out):
    """desc (B,S,4), tau_lag (B,S,P), tau_out (B,S,M) -> a-kernels kL0,kL1,kR0,kR1 (B,S,P) and the end
    particular solutions p0, p1 (dicts of (B,S,M)); identical in meaning to the six evaluate() calls of close_rails."""
    base, K, lt_min, lt_max = prov.base, prov.n_tab, prov.lt_min, prov.lt_max
    B, S, P = tau_lag.shape; M = tau_out.shape[2]; n = B * S
    d = desc.reshape(n, 4); dm = d.copy(); dm[:, 1] *= -1.0
    tl = tau_lag.reshape(n, P); ltl = np.log10(np.clip(tl, 1e-300, None)); ltlc = np.clip(ltl, lt_min, lt_max)
    lo, hi = _np_grid(ltlc)
    aL0 = _np_tables(base, d, 0.0, lo, hi, K)["a"]; aL1 = _np_tables(base, d, 1.0, lo, hi, K)["a"]
    aR0 = _np_tables(base, dm, 1.0, lo, hi, K)["a"]; aR1 = _np_tables(base, dm, 0.0, lo, hi, K)["a"]
    ks = [_np_interp(T, tl, ltl, ltlc, lo, hi, K, lt_min, lt_max, True).reshape(B, S, P) for T in (aL0, aL1, aR0, aR1)]
    to = tau_out.reshape(n, M); lto = np.log10(np.clip(to, 1e-300, None)); ltoc = np.clip(lto, lt_min, lt_max)
    lo2, hi2 = _np_grid(ltoc)
    p = []
    for xi_val in (0.0, 1.0):
        T = _np_tables(base, d, xi_val, lo2, hi2, K)
        p.append({k: _np_interp(T[k], to, lto, ltoc, lo2, hi2, K, lt_min, lt_max, k == "a").reshape(B, S, M) for k in T})
    prov.n_base_evals += 6 * n * K; prov.n_points += 4 * n * P + 2 * n * M
    return ks[0], ks[1], ks[2], ks[3], p[0], p[1]


def _t_grid(ltc):
    import torch
    lo = ltc.min(dim=-1).values.detach(); hi = torch.maximum(ltc.max(dim=-1).values.detach(), lo + 1e-3)
    return lo, hi


def _t_tables(base, desc_flat, xi_val, lo, hi, K):
    import torch
    Gn = desc_flat.shape[0]
    u = torch.linspace(0.0, 1.0, K, device=desc_flat.device, dtype=lo.dtype)
    LT = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    tg = (10.0 ** LT).reshape(-1)
    out = base.evaluate(desc_flat.repeat_interleave(K, dim=0), torch.full((Gn * K,), float(xi_val), device=tg.device, dtype=tg.dtype), tg)
    stg = torch.sqrt(tg)
    return {k: (v / stg).reshape(Gn, K) for k, v in out.items()}


def _t_interp(T, tau, lt, ltc, lo, hi, K, lt_min, lt_max, is_a):
    import torch
    pos = (ltc - lo[:, None]) / (hi - lo)[:, None] * (K - 1)
    i0 = torch.clamp(torch.floor(pos.detach()).to(torch.int64), 0, K - 2); w = torch.clamp(pos - i0.to(pos.dtype), 0.0, 1.0)
    y = torch.gather(T, 1, i0) * (1.0 - w) + torch.gather(T, 1, i0 + 1) * w
    tau_c = 10.0 ** ltc
    v = y * torch.sqrt(tau_c) * torch.where(lt < lt_min, torch.sqrt(tau / 10.0 ** lt_min), torch.ones_like(tau))
    if is_a:
        v = v - torch.where(lt > lt_max, tau - tau_c, torch.zeros_like(tau))
    return v


def closure_kernels_tabulated_torch(prov: TabulatedTorchProvider, desc, tau_lag, tau_out):
    """Torch twin of closure_kernels_tabulated_np (differentiable w.r.t. desc and tau)."""
    import torch
    base, K, lt_min, lt_max = prov.base, prov.n_tab, prov.lt_min, prov.lt_max
    B, S, P = tau_lag.shape; M = tau_out.shape[2]; n = B * S
    d = desc.reshape(n, 4); dm = torch.cat([d[:, :1], -d[:, 1:2], d[:, 2:]], 1)
    tl = tau_lag.reshape(n, P); ltl = torch.log10(torch.clamp(tl, min=1e-300)); ltlc = torch.clamp(ltl, lt_min, lt_max)
    lo, hi = _t_grid(ltlc)
    aL0 = _t_tables(base, d, 0.0, lo, hi, K)["a"]; aL1 = _t_tables(base, d, 1.0, lo, hi, K)["a"]
    aR0 = _t_tables(base, dm, 1.0, lo, hi, K)["a"]; aR1 = _t_tables(base, dm, 0.0, lo, hi, K)["a"]
    ks = [_t_interp(T, tl, ltl, ltlc, lo, hi, K, lt_min, lt_max, True).reshape(B, S, P) for T in (aL0, aL1, aR0, aR1)]
    to = tau_out.reshape(n, M); lto = torch.log10(torch.clamp(to, min=1e-300)); ltoc = torch.clamp(lto, lt_min, lt_max)
    lo2, hi2 = _t_grid(ltoc)
    p = []
    for xi_val in (0.0, 1.0):
        T = _t_tables(base, d, xi_val, lo2, hi2, K)
        p.append({k: _t_interp(T[k], to, lto, ltoc, lo2, hi2, K, lt_min, lt_max, k == "a").reshape(B, S, M) for k in T})
    prov.n_base_evals += 6 * n * K; prov.n_points += 4 * n * P + 2 * n * M
    return ks[0], ks[1], ks[2], ks[3], p[0], p[1]
