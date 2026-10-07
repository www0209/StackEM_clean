"""
stackem.closure_torch
=======================

PyTorch implementation of the junction closure (same algorithm as
``closure.close_rails``) evaluated with the SKN as a torch module, so that

* a whole stack (10^3-10^6 rails) is closed in a few batched GPU kernel
  evaluations plus M batched tridiagonal solves, and
* the nucleation times are *differentiable* with respect to every segment's
  temperature, Joule rise, current density and initial stress - the basis of
  the design-sensitivity experiment (E5), where those inputs are themselves
  linear functions of the block powers (HotSpot and the PG solver are linear).

The code is written functionally (no in-place writes into tensors that carry
gradients) so that autograd works through the Thomas recursion.

Only this module and ``train_skn.py`` import torch; everything else runs with
numpy.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple
import torch

from .constants import EMParams, K_B, K_B_EV, E_CHARGE
from .skn_model import SKN, SKNWeights
from .kernel_dataset import m_amp_of


# ----------------------------------------------------------------------------
# physics in torch
# ----------------------------------------------------------------------------
def t_log_kappa(T: torch.Tensor, em: EMParams) -> torch.Tensor:
    """ln kappa(T) with kappa = D0 exp(-Ea/kT) B Omega/(k_B T)."""
    return math.log(em.D0) - em.Ea_eV / (K_B_EV * T) + math.log(em.B * em.Omega / K_B) - torch.log(T)


def t_descriptors(L, T_L, T_R, T_m, Gamma, em: EMParams):
    """(theta, r, rhoJ, lam, kappa_bar) for batched segments (any broadcastable shapes)."""
    Tb = 0.5 * (T_L + T_R)
    lk_b = t_log_kappa(Tb, em)
    r = t_log_kappa(T_R, em) - t_log_kappa(T_L, em)
    rhoJ = t_log_kappa(Tb + T_m, em) - lk_b
    theta = K_B_EV * Tb / em.Ea_eV
    lam = L / Gamma
    return theta, r, rhoJ, lam, torch.exp(lk_b)


def t_G_of_j(j, em: EMParams):
    return E_CHARGE * em.Z_eff * em.rho * j / em.Omega


def t_sigma_T(Tbar, em: EMParams):
    if em.sigma_T_model == "constant":
        return torch.full_like(Tbar, em.sigma_T_const)
    return torch.clamp(em.B * em.dalpha * (em.T_zero_stress - Tbar), min=0.0)


class SKNTorchProvider:
    """Evaluate the SKN kernels in torch (differentiable)."""

    def __init__(self, weights: SKNWeights, device: str | torch.device = "cpu", log10_tau_range=(-6.0, 1.0),
                 chunk: int = 500_000):
        self.model = SKN.from_weights(weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.lt_min, self.lt_max = log10_tau_range
        self.chunk = int(chunk)          # points per forward pass: 6 x 256 hidden units x 4 B x 500k = 0.5 GB per activation

    def evaluate(self, desc: torch.Tensor, xi: torch.Tensor, tau: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Chunked so that a whole stack (10^7 points) never materialises a multi-GB activation on the GPU
        (on Windows/WSL an oversubscribed GPU silently spills into system RAM and becomes 1000x slower)."""
        n = tau.shape[0]
        if n <= self.chunk:
            return self._evaluate(desc, xi, tau)
        parts = [self._evaluate(desc[i:i + self.chunk], xi[i:i + self.chunk], tau[i:i + self.chunk]) for i in range(0, n, self.chunk)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def _evaluate(self, desc: torch.Tensor, xi: torch.Tensor, tau: torch.Tensor) -> Dict[str, torch.Tensor]:
        lt = torch.log10(torch.clamp(tau, min=1e-300))
        ltc = torch.clamp(lt, self.lt_min, self.lt_max)
        from .kernel_dataset import LAM_CLAMP
        X = torch.stack([xi, ltc, desc[:, 0], desc[:, 1], desc[:, 2], torch.clamp(desc[:, 3], *LAM_CLAMP)], 1).to(torch.float32)
        Y = self.model(X).to(tau.dtype)
        tau_c = 10.0 ** ltc
        st = torch.sqrt(tau_c)
        ma = m_amp_torch(desc[:, 0], desc[:, 1], desc[:, 2], desc[:, 3])
        sG = Y[:, 0] * st; sM = Y[:, 1] * st * ma; a = Y[:, 2] * st
        f_small = torch.where(lt < self.lt_min, torch.sqrt(tau / 10.0 ** self.lt_min), torch.ones_like(tau))
        a = a * f_small; sG = sG * f_small; sM = sM * f_small
        a = a - torch.where(lt > self.lt_max, tau - tau_c, torch.zeros_like(tau))
        return dict(a=a, s_G=sG, s_M=sM)


def m_amp_torch(theta, r, rhoJ, lam):
    return theta * (torch.abs(r) + torch.abs(rhoJ) * lam * torch.tanh(0.5 * lam)) + 1e-3


# ----------------------------------------------------------------------------
# batched Thomas algorithm (functional)
# ----------------------------------------------------------------------------
def tridiag_solve_batch(a, b, c, d):
    """Solve the batched tridiagonal systems (sub a, diag b, sup c) x = d.  On the GPU one batched dense LU
    (torch.linalg.solve on (B,J,J), J = S-1 <= ~100) replaces the 2J-step Python Thomas recursion, i.e. ~6 kernel
    launches instead of ~8J per time step - launch latency, not arithmetic, dominates the march on WSL2/Windows.
    The Thomas recursion is kept for the CPU and for J <= 2 (identical result, functional, differentiable)."""
    J = b.shape[1]
    if J > 2 and b.is_cuda and hasattr(torch.linalg, "solve"):
        if J <= 96:                                   # small systems: one batched dense LU on the GPU
            A = torch.diag_embed(b) + torch.diag_embed(a[:, 1:], offset=-1) + torch.diag_embed(c[:, :-1], offset=1)
            return torch.linalg.solve(A, d[:, :, None])[:, :, 0]
        if not any(t.requires_grad for t in (a, b, c, d)):
            # long rails (IBM-PG: 100-250 junctions): cuSOLVER's batched LU degenerates to a per-matrix loop at this
            # size (1 % GPU utilisation, minutes per step); the Thomas recursion vectorised over the batch on the
            # CPU costs ~1 ms per time step and the transfer is a few MB
            import numpy as np
            an, bn, cn, dn = (t.detach().cpu().numpy() for t in (a, b, c, d))
            return torch.as_tensor(thomas_numpy(an, bn, cn, dn), device=b.device, dtype=b.dtype)
    return thomas_batch(a, b, c, d)


def thomas_numpy(a, b, c, d):
    """Batched Thomas recursion in numpy, arrays (B, J); loop over J, vectorised over B."""
    import numpy as np
    n = b.shape[1]
    cp = np.empty_like(b); dp = np.empty_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]; dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        m = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / m
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / m
    x = np.empty_like(b); x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def thomas_batch(a, b, c, d):
    n = b.shape[1]
    cp = [c[:, 0] / b[:, 0]]; dp = [d[:, 0] / b[:, 0]]
    for i in range(1, n):
        m = b[:, i] - a[:, i] * cp[-1]
        cp.append(c[:, i] / m)
        dp.append((d[:, i] - a[:, i] * dp[-1]) / m)
    x = [dp[-1]]
    for i in range(n - 2, -1, -1):
        x.append(dp[i] - cp[i] * x[-1])
    x.reverse()
    return torch.stack(x, 1)


# ----------------------------------------------------------------------------
# closure
# ----------------------------------------------------------------------------
def close_rails_torch(L, j, T_L, T_R, T_m, Gamma, sigma_T, em: EMParams, provider: SKNTorchProvider,
                      times: torch.Tensor, with_tm: bool = True, jump_at: str = "mid") -> Dict[str, torch.Tensor]:
    """All rail arrays are (B, S) tensors on the provider's device (Gamma (B,) or scalar).
    Returns dict(sigma_nodes (B,M,S+1), F (B,M,S+1), sigma_max (B,M), t_nuc (B,))."""
    dev = provider.device
    B, S = L.shape; M = len(times)
    _t0 = _tick(dev)
    theta, r, rhoJ, lam, kb = t_descriptors(L, T_L, T_R, T_m, Gamma if torch.is_tensor(Gamma) else torch.tensor(Gamma, device=dev), em)
    desc = torch.stack([theta, r, rhoJ, lam], -1)                     # (B,S,4)
    G = t_G_of_j(j, em)
    scaleA = L / kb
    QO = em.Q_J / em.Omega
    t_prev = torch.cat([torch.zeros(1, device=dev, dtype=times.dtype), times[:-1]])
    t_jump = t_prev if jump_at == "start" else 0.5 * (t_prev + times)
    lag = times[:, None] - t_jump[None, :]
    nn_, mm_ = torch.tril_indices(M, M, device=dev)
    lag_v = lag[nn_, mm_]; P = lag_v.shape[0]
    tau_lag = kb[:, :, None] * lag_v[None, None, :] / L[:, :, None] ** 2          # (B,S,P)
    tau_out = kb[:, :, None] * times[None, None, :] / L[:, :, None] ** 2          # (B,S,M)
    from .closure_tab import TabulatedTorchProvider, closure_kernels_tabulated_torch
    if isinstance(provider, TabulatedTorchProvider):          # tables built per segment directly (no (B,S,P) expansion)
        kL0, kL1, kR0, kR1, p0, p1 = closure_kernels_tabulated_torch(provider, desc, tau_lag, tau_out)
    else:
        d_rep = desc[:, :, None, :].expand(B, S, P, 4).reshape(-1, 4)
        d_mir = torch.cat([d_rep[:, :1], -d_rep[:, 1:2], d_rep[:, 2:]], 1)
        t_flat = tau_lag.reshape(-1)
        z = torch.zeros_like(t_flat); o = torch.ones_like(t_flat)
        kL0 = provider.evaluate(d_rep, z, t_flat)["a"].reshape(B, S, P)
        kL1 = provider.evaluate(d_rep, o, t_flat)["a"].reshape(B, S, P)
        kR0 = provider.evaluate(d_mir, o, t_flat)["a"].reshape(B, S, P)
        kR1 = provider.evaluate(d_mir, z, t_flat)["a"].reshape(B, S, P)
        d2 = desc[:, :, None, :].expand(B, S, M, 4).reshape(-1, 4); t2 = tau_out.reshape(-1)
        p0 = {k: v.reshape(B, S, M) for k, v in provider.evaluate(d2, torch.zeros_like(t2), t2).items()}
        p1 = {k: v.reshape(B, S, M) for k, v in provider.evaluate(d2, torch.ones_like(t2), t2).items()}
    AL0 = scaleA[:, :, None] * kL0; AL1 = scaleA[:, :, None] * kL1
    AR0 = -scaleA[:, :, None] * kR0; AR1 = -scaleA[:, :, None] * kR1
    GL = (G * L)[:, :, None]
    P0 = sigma_T[:, :, None] + GL * p0["s_G"]
    P1 = sigma_T[:, :, None] + GL * p1["s_G"]
    if with_tm:
        P0 = P0 + QO * p0["s_M"]; P1 = P1 + QO * p1["s_M"]
    _t1 = _tick(dev); _prof_add("kernels", _t1 - _t0)
    pos = torch.full((M, M), -1, dtype=torch.long, device=dev); pos[nn_, mm_] = torch.arange(P, device=dev)
    zero_col = torch.zeros(B, 1, device=dev, dtype=L.dtype)
    dF_list = []; sig_list = []; F_prev = torch.zeros(B, S + 1, device=dev, dtype=L.dtype); F_list = []
    for n in range(M):
        if n > 0:                                            # history of the earlier flux increments, one einsum per end instead of n ops
            pn = pos[n, :n]
            D = torch.stack(dF_list, 1)                      # (B,n,S+1)
            DL, DR = D[:, :, :-1], D[:, :, 1:]
            hist0 = torch.einsum("bms,bsm->bs", DL, AL0[:, :, pn]) + torch.einsum("bms,bsm->bs", DR, AR0[:, :, pn])
            hist1 = torch.einsum("bms,bsm->bs", DL, AL1[:, :, pn]) + torch.einsum("bms,bsm->bs", DR, AR1[:, :, pn])
        else:
            hist0 = torch.zeros(B, S, device=dev, dtype=L.dtype); hist1 = torch.zeros(B, S, device=dev, dtype=L.dtype)
        pd = pos[n, n]
        aL0, aL1, aR0, aR1 = AL0[:, :, pd], AL1[:, :, pd], AR0[:, :, pd], AR1[:, :, pd]
        if S > 1:
            rhs = (P0[:, 1:, n] + hist0[:, 1:]) - (P1[:, :-1, n] + hist1[:, :-1])
            sub = torch.cat([zero_col, aL1[:, 1:-1]], 1) if S > 2 else zero_col
            diag = aR1[:, :-1] - aL0[:, 1:]
            sup = torch.cat([-aR0[:, 1:-1], zero_col], 1) if S > 2 else zero_col
            x = tridiag_solve_batch(sub, diag, sup, rhs)
            dFn = torch.cat([zero_col, x, zero_col], 1)
        else:
            dFn = torch.zeros(B, 2, device=dev, dtype=L.dtype)
        dF_list.append(dFn)
        F_prev = F_prev + dFn; F_list.append(F_prev)
        s0 = P0[:, :, n] + hist0 + dFn[:, :-1] * aL0 + dFn[:, 1:] * aR0
        s1 = P1[:, :, n] + hist1 + dFn[:, :-1] * aL1 + dFn[:, 1:] * aR1
        sig_list.append(torch.cat([s0, s1[:, -1:]], 1))
    sigma_nodes = torch.stack(sig_list, 1)            # (B,M,S+1)
    F = torch.stack(F_list, 1)
    smax = sigma_nodes.max(dim=2).values               # (B,M)
    t_nuc = nucleation_time_torch(times, smax, em.sigma_crit)
    _prof_add("march", _tick(dev) - _t1)
    return dict(sigma_nodes=sigma_nodes, F=F, sigma_max=smax, t_nuc=t_nuc)


PROFILE: Optional[Dict[str, float]] = None      # set to {} (e.g. E4 --profile) to accumulate seconds per phase


def _tick(dev):
    if PROFILE is None:
        return 0.0
    import time
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    return time.time()


def _prof_add(key, dt):
    if PROFILE is not None:
        PROFILE[key] = PROFILE.get(key, 0.0) + dt


def nucleation_time_torch(times: torch.Tensor, smax: torch.Tensor, sigma_crit: float, horizon_factor: float = 1e3) -> torch.Tensor:
    """Differentiable interpolation (linear in sqrt t) of the first crossing of sigma_crit.
    Rails that never cross get horizon_factor * times[-1] (a finite sentinel)."""
    B, M = smax.shape
    crossed = smax >= sigma_crit
    any_cross = crossed.any(dim=1)
    first = torch.where(any_cross, crossed.float().argmax(dim=1), torch.full((B,), M - 1, device=smax.device))
    k = torch.clamp(first, min=1)
    idx = torch.arange(B, device=smax.device)
    s0 = smax[idx, k - 1]; s1 = smax[idx, k]
    r0 = torch.sqrt(times[k - 1]); r1 = torch.sqrt(times[k])                 # linear in sqrt(t), see korhonen_fdm.crossing_time
    frac = (sigma_crit - s0) / (s1 - s0 + 1e-30)
    t_int = (r0 + frac * (r1 - r0)) ** 2
    t_first = torch.where(first == 0, times[0].expand(B), t_int)
    return torch.where(any_cross, t_first, torch.full((B,), float(times[-1]) * horizon_factor, device=smax.device, dtype=smax.dtype))


def rails_to_tensors(rails, device="cpu", dtype=torch.float64):
    """Stack a list of Rail objects (same S) into the (B,S) tensors used above."""
    import numpy as np
    f = lambda name: torch.as_tensor(np.stack([getattr(r, name) for r in rails]), dtype=dtype, device=device)
    Gamma = torch.as_tensor(np.array([r.Gamma for r in rails]), dtype=dtype, device=device)[:, None]
    return dict(L=f("L"), j=f("j"), T_L=f("T_L"), T_R=f("T_R"), T_m=f("T_m"), Gamma=Gamma, sigma_T=f("sigma_T"))
