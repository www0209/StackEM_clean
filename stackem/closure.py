"""
stackem.closure
=================

Junction Closure by Flux-history Superposition (JCX): the exact, training-free
assembly of a multi-segment rail from single-segment kernels.

Idea
----
The Korhonen equation is linear.  On a segment k (length L_k, diffusivity
profile kappa_k(x), EM source G_k, TM source M_k, initial stress sigma_T,k)
with junction fluxes F at its two ends, the stress is the superposition

    sigma_k(x,t) = sigma_T,k
                 + G_k L_k  s_G,k(xi, tau)                    EM particular solution
                 + (Q*/Omega) s_M,k(xi, tau)                  TM particular solution
                 + sum_m  dF^L_{k,m}  A^L_k(xi, t - t_{m-1})  Duhamel sum over the
                 + sum_m  dF^R_{k,m}  A^R_k(xi, t - t_{m-1})  piecewise-constant flux history

with  A^L_k(xi,t) = (L_k/kappa_k) a_k(xi, kappa_k t / L_k^2)   (unit-flux Green's function)
and   A^R_k(xi,t) = -(L_k/kappa_k) a_k^mirror(1-xi, ...)       (mirror: r -> -r).

At every junction the stress must be continuous and the atomic flux conserved.
For a straight rail (equal cross-sections) that is one unknown flux per junction
and one continuity equation per junction; with the flux history known up to
t_{n-1}, the increments dF_{.,n} follow from a *tridiagonal* linear system.
Nothing is learned here; the only approximation is the piecewise-constant flux
history on the (logarithmic) output time grid, which converges with grid
refinement (verified against the full-tree finite-volume solver in tests).

Because of the parabolic maximum principle, the stress extrema of a segment
with (near-)uniform kappa sit at its ends, so the time march only needs kernel
values at xi = 0 and xi = 1; interior profiles are reconstructed on demand for
plotting.

Kernel providers
----------------
``KernelProvider.evaluate(desc, xi, tau)`` returns the three dimensionless
kernels point-wise.  ``FDMKernelProvider`` computes them with the reference
solver (slow, exact; used for validation).  ``skn_numpy.SKNNumpyProvider``
evaluates the trained Segment Kernel Network (fast, differentiable in the
torch variant).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .constants import EMParams, SEC_PER_YEAR
from .korhonen_fdm import crossing_time, solve_segment_hat, default_tau_grid
from .thermal_profile import SegmentProfile, descriptors_to_profile
from .power_grid import Rail


# ----------------------------------------------------------------------------
# kernel providers
# ----------------------------------------------------------------------------
class KernelProvider:
    """Interface: evaluate(desc (N,4) = [theta, r, rhoJ, lam], xi (N,), tau (N,)) -> dict of (N,) arrays
    with keys 'a' (unit-flux Green's function, flux at xi=0), 's_G', 's_M'."""

    def evaluate(self, desc: np.ndarray, xi: np.ndarray, tau: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class FDMKernelProvider(KernelProvider):
    """Exact kernels from the finite-volume solver, cached per unique descriptor and
    interpolated (linearly in xi, log-linearly in tau) from a dense tau grid."""

    def __init__(self, em: EMParams, tau_grid: np.ndarray | None = None, n_cells: int = 200, L_ref: float = 100e-6):
        self.em = em
        self.tau_grid = default_tau_grid(1e-7, 10.0, 400) if tau_grid is None else tau_grid
        self.n_cells = n_cells
        self.L_ref = L_ref
        self._cache: Dict[Tuple[float, float, float, float], Dict[str, np.ndarray]] = {}
        self.xi_grid = None

    def _kernels_for(self, key):
        if key in self._cache:
            return self._cache[key]
        theta, r, rhoJ, lam = key
        prof = descriptors_to_profile(theta, r, rhoJ, lam, self.L_ref, self.em)
        kh = lambda xi: prof.kappa_hat(xi, self.em)
        sG = solve_segment_hat(kh, 1.0, 0.0, 0.0, self.tau_grid, 0.0, self.n_cells)
        a = solve_segment_hat(kh, 0.0, 1.0, 0.0, self.tau_grid, 0.0, self.n_cells)
        sM = solve_segment_hat(kh, lambda xi: prof.M_hat(xi, self.em), 0.0, 0.0, self.tau_grid, 0.0, self.n_cells)
        self.xi_grid = sG.xi
        self._cache[key] = dict(a=a.s, s_G=sG.s, s_M=sM.s)
        return self._cache[key]

    def evaluate(self, desc, xi, tau):
        desc = np.asarray(desc, float); xi = np.asarray(xi, float); tau = np.asarray(tau, float)
        out = {k: np.zeros(len(xi)) for k in ("a", "s_G", "s_M")}
        keys = [tuple(np.round(d, 10)) for d in desc]
        uniq = {}
        for i, k in enumerate(keys):
            uniq.setdefault(k, []).append(i)
        lt_grid = np.log(self.tau_grid)
        for k, idx in uniq.items():
            ker = self._kernels_for(k)
            idx = np.asarray(idx)
            lt = np.log(np.clip(tau[idx], self.tau_grid[0], self.tau_grid[-1]))
            it = np.clip(np.searchsorted(lt_grid, lt) - 1, 0, len(lt_grid) - 2)
            wt = (lt - lt_grid[it]) / (lt_grid[it + 1] - lt_grid[it])
            xg = self.xi_grid
            ix = np.clip(np.searchsorted(xg, xi[idx]) - 1, 0, len(xg) - 2)
            wx = (xi[idx] - xg[ix]) / (xg[ix + 1] - xg[ix])
            for name in out:
                K = ker[name]
                v = ((1 - wt) * ((1 - wx) * K[it, ix] + wx * K[it, ix + 1])
                     + wt * ((1 - wx) * K[it + 1, ix] + wx * K[it + 1, ix + 1]))
                # tau below the grid: kernels vanish like sqrt(tau); scale
                small = tau[idx] < self.tau_grid[0]
                if np.any(small):
                    v[small] *= np.sqrt(tau[idx][small] / self.tau_grid[0])
                out[name][idx] = v
        return out


# ----------------------------------------------------------------------------
# rail closure
# ----------------------------------------------------------------------------
@dataclass
class RailResult:
    times: np.ndarray            # (M,) s
    sigma_nodes: np.ndarray      # (B, M, S+1) Pa   stress at the S+1 nodes (junction/terminal) of each rail
    F: np.ndarray                # (B, M, S+1) Pa m/s   junction flux history (0 at terminals)
    t_nuc: np.ndarray            # (B,) s  (inf if no nucleation within the horizon)
    nuc_node: np.ndarray         # (B,) int node index of first nucleation
    sigma_max: np.ndarray        # (B, M)

    def t_nuc_years(self):
        return self.t_nuc / SEC_PER_YEAR


def rail_descriptors(rail: Rail, em: EMParams):
    """Per-segment (theta, r, rhoJ, lam, kappa_bar) for a rail."""
    S = rail.n_seg
    desc = np.zeros((S, 4)); kb = np.zeros(S)
    for k in range(S):
        p = SegmentProfile(rail.L[k], rail.T_L[k], rail.T_R[k], rail.T_m[k], rail.Gamma)
        th, r, rJ, lam, kbar = p.descriptors(em)
        desc[k] = (th, r, rJ, lam); kb[k] = kbar
    return desc, kb


def jump_times(times: np.ndarray, jump_at: str = "mid") -> np.ndarray:
    t_prev = np.concatenate([[0.0], times[:-1]])
    if jump_at == "start":
        return t_prev
    if jump_at == "mid":
        return 0.5 * (t_prev + times)
    raise ValueError(jump_at)


def _thomas_batch(a, b, c, d):
    """Solve batched tridiagonal systems: a sub-diag (B,n) [a[:,0] unused], b diag (B,n),
    c super-diag (B,n) [c[:,-1] unused], d rhs (B,n)."""
    n = b.shape[1]
    cp = np.zeros_like(b); dp = np.zeros_like(b); x = np.zeros_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]; dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        m = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / m
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / m
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def close_rails(rails: Sequence[Rail], em: EMParams, provider: KernelProvider, times: np.ndarray,
                sigma_crit: float | None = None, with_tm: bool = True, jump_at: str = "mid") -> RailResult:
    """Run the junction closure for a batch of rails with the same number of segments.

    Complexity: one kernel evaluation batch of size B*S*n_lag*2 (ends) and M tridiagonal
    solves of size S-1 per rail.
    """
    times = np.asarray(times, float); M = len(times)
    B = len(rails); S = rails[0].n_seg
    assert all(r.n_seg == S for r in rails), "batch rails must share the segment count"
    sigma_crit = em.sigma_crit if sigma_crit is None else sigma_crit
    # --- gather per-segment scalars -------------------------------------------------
    L = np.stack([r.L for r in rails])                       # (B,S)
    j = np.stack([r.j for r in rails])
    sT = np.stack([r.sigma_T for r in rails])
    desc = np.zeros((B, S, 4)); kb = np.zeros((B, S))
    for b, r in enumerate(rails):
        desc[b], kb[b] = rail_descriptors(r, em)
    G = em.G_of_j(j)                                          # (B,S) signed
    scaleA = L / kb                                           # (B,S) Pa per unit flux (Pa m/s)
    QO = em.Q_J / em.Omega
    # --- lags: t_n - t_jump(m), m = 1..n -------------------------------------------
    # The flux is piecewise constant on (t_{m-1}, t_m]; its jump is placed either at the
    # interval start (first-order Duhamel) or at the interval mid-point (second-order,
    # default), see tests for the convergence study.
    t_jump = jump_times(times, jump_at)
    lag = times[:, None] - t_jump[None, :]                     # (M,M), valid for m<=n
    nn, mm = np.tril_indices(M)
    lag_v = lag[nn, mm]                                        # (P,) with P = M(M+1)/2
    P = len(lag_v)
    # --- kernel evaluation at both ends for all lags ----------------------------------
    # arrays: (B,S,P) tau_lag = kb*lag/L^2
    tau_lag = kb[:, :, None] * lag_v[None, None, :] / L[:, :, None] ** 2
    # particular solutions at the ends at the output times: tau_n = kb t_n / L^2
    tau_out = kb[:, :, None] * times[None, None, :] / L[:, :, None] ** 2     # (B,S,M)
    from .closure_tab import TabulatedProvider, closure_kernels_tabulated_np
    if isinstance(provider, TabulatedProvider):                # tables built per segment directly (no (B,S,P) expansion)
        a_L0, a_L1, a_R0, a_R1, p0, p1 = closure_kernels_tabulated_np(provider, desc, tau_lag, tau_out)
    else:
        desc_rep = np.repeat(desc[:, :, None, :], P, axis=2)      # (B,S,P,4)
        mirror = desc_rep.copy(); mirror[..., 1] *= -1.0          # r -> -r
        N = B * S * P
        d_flat = desc_rep.reshape(N, 4); dm_flat = mirror.reshape(N, 4); t_flat = tau_lag.reshape(N)
        xi0 = np.zeros(N); xi1 = np.ones(N)
        a_L0 = provider.evaluate(d_flat, xi0, t_flat)["a"].reshape(B, S, P)      # left-flux kernel at xi=0
        a_L1 = provider.evaluate(d_flat, xi1, t_flat)["a"].reshape(B, S, P)      # at xi=1
        a_R0 = provider.evaluate(dm_flat, xi1, t_flat)["a"].reshape(B, S, P)     # mirrored profile at 1-xi=1 -> xi=0
        a_R1 = provider.evaluate(dm_flat, xi0, t_flat)["a"].reshape(B, S, P)     # mirrored at 1-xi=0 -> xi=1
        dflat2 = np.repeat(desc[:, :, None, :], M, axis=2).reshape(B * S * M, 4)
        tflat2 = tau_out.reshape(-1)
        p0 = {k: v.reshape(B, S, M) for k, v in provider.evaluate(dflat2, np.zeros(B * S * M), tflat2).items()}
        p1 = {k: v.reshape(B, S, M) for k, v in provider.evaluate(dflat2, np.ones(B * S * M), tflat2).items()}
    AL0 = scaleA[:, :, None] * a_L0; AL1 = scaleA[:, :, None] * a_L1
    AR0 = -scaleA[:, :, None] * a_R0; AR1 = -scaleA[:, :, None] * a_R1
    P0 = sT[:, :, None] + (G * L)[:, :, None] * p0["s_G"]
    P1 = sT[:, :, None] + (G * L)[:, :, None] * p1["s_G"]
    if with_tm:
        P0 = P0 + QO * p0["s_M"]
        P1 = P1 + QO * p1["s_M"]
    # index helper: lag (n,m) -> position in lag_v
    pos = np.full((M, M), -1, int); pos[nn, mm] = np.arange(P)
    # --- time march ----------------------------------------------------------------
    dF = np.zeros((B, M, S + 1))          # flux increments at nodes (terminals stay 0)
    F = np.zeros((B, M, S + 1))
    sig_nodes = np.zeros((B, M, S + 1))
    J = S - 1                              # number of junctions
    for n in range(M):
        # history contributions to sigma at the ends of every segment (m < n), vectorised over m
        if n > 0:
            pn = pos[n, :n]                                            # (n,) lag indices of the earlier increments
            DL = dF[:, :n, :-1]; DR = dF[:, :n, 1:]                    # (B,n,S)
            hist0 = np.einsum("bms,bsm->bs", DL, AL0[:, :, pn]) + np.einsum("bms,bsm->bs", DR, AR0[:, :, pn])
            hist1 = np.einsum("bms,bsm->bs", DL, AL1[:, :, pn]) + np.einsum("bms,bsm->bs", DR, AR1[:, :, pn])
        else:
            hist0 = np.zeros((B, S)); hist1 = np.zeros((B, S))
        pd = pos[n, n]
        aL0, aL1, aR0, aR1 = AL0[:, :, pd], AL1[:, :, pd], AR0[:, :, pd], AR1[:, :, pd]
        if J > 0:
            # unknown x_k = dF at node k (k=1..S-1). Equation at junction k:
            #   sigma_{k-1}(1) = sigma_k(0)
            # left seg k-1: contributions x_{k-1} aL1[k-1] + x_k aR1[k-1]
            # right seg k : contributions x_k aL0[k] + x_{k+1} aR0[k]
            rhs = (P0[:, 1:, n] + hist0[:, 1:]) - (P1[:, :-1, n] + hist1[:, :-1])     # (B,J)
            sub = aL1[:, :-1].copy()                 # coefficient of x_{k-1}  (index k-1 = seg k-1)
            diag = aR1[:, :-1] - aL0[:, 1:]
            sup = -aR0[:, 1:]
            sub[:, 0] = 0.0; sup[:, -1] = 0.0        # terminals x_0 = x_S = 0
            x = _thomas_batch(sub, diag, sup, rhs)
            dF[:, n, 1:-1] = x
        F[:, n] = (F[:, n - 1] if n > 0 else 0.0) + dF[:, n]
        # node stresses: left end of segment k (node k) and right end of last segment
        s0 = P0[:, :, n] + hist0 + dF[:, n, :-1] * aL0 + dF[:, n, 1:] * aR0
        s1 = P1[:, :, n] + hist1 + dF[:, n, :-1] * aL1 + dF[:, n, 1:] * aR1
        sig_nodes[:, n, :-1] = s0
        sig_nodes[:, n, -1] = s1[:, -1]
        # (continuity makes s1[:, :-1] == s0[:, 1:] up to solver tolerance)
    smax = sig_nodes.max(axis=2)
    t_nuc = np.full(B, np.inf); nuc_node = np.zeros(B, int)
    for b in range(B):
        idx = np.where(smax[b] >= sigma_crit)[0]
        if len(idx):
            k = idx[0]; nuc_node[b] = int(np.argmax(sig_nodes[b, k]))
            if k == 0:
                t_nuc[b] = times[0]
            else:
                t0, t1, s0_, s1_ = times[k - 1], times[k], smax[b, k - 1], smax[b, k]
                t_nuc[b] = crossing_time(t0, t1, s0_, s1_, sigma_crit)
        else:
            nuc_node[b] = int(np.argmax(sig_nodes[b, -1]))
    return RailResult(times, sig_nodes, F, t_nuc, nuc_node, smax)


def reconstruct_profile(rail: Rail, res: RailResult, b: int, em: EMParams, provider: KernelProvider,
                        n_xi: int = 21, with_tm: bool = True, jump_at: str = "mid") -> Tuple[np.ndarray, np.ndarray]:
    """Full sigma(x,t) of rail b from the closure result: returns (x (S*n_xi,), sigma (M, S*n_xi))."""
    times = res.times; M = len(times); S = rail.n_seg
    desc, kb = rail_descriptors(rail, em)
    G = em.G_of_j(rail.j); L = rail.L; scaleA = L / kb; QO = em.Q_J / em.Omega
    xi = np.linspace(0, 1, n_xi)
    t_prev = jump_times(times, jump_at)
    xs = []; sig = np.zeros((M, S * n_xi))
    dF = np.diff(np.concatenate([np.zeros((1, S + 1)), res.F[b]], axis=0), axis=0)   # (M,S+1)
    for k in range(S):
        xs.append(rail.L[:k].sum() + xi * L[k])
        d = np.repeat(desc[k][None, :], n_xi, 0); dm = d.copy(); dm[:, 1] *= -1
        for n in range(M):
            tau = kb[k] * times[n] / L[k] ** 2
            p = provider.evaluate(d, xi, np.full(n_xi, tau))
            s = rail.sigma_T[k] + G[k] * L[k] * p["s_G"] + (QO * p["s_M"] if with_tm else 0.0)
            for m in range(n + 1):
                lagt = kb[k] * (times[n] - t_prev[m]) / L[k] ** 2
                aL = provider.evaluate(d, xi, np.full(n_xi, lagt))["a"] * scaleA[k]
                aR = -provider.evaluate(dm, 1 - xi, np.full(n_xi, lagt))["a"] * scaleA[k]
                s = s + dF[m, k] * aL + dF[m, k + 1] * aR
            sig[n, k * n_xi:(k + 1) * n_xi] = s
    return np.concatenate(xs), sig


# ----------------------------------------------------------------------------
# general trees (branching junctions): per-tree dense closure
# ----------------------------------------------------------------------------
@dataclass
class TreeSpec:
    """A general interconnect tree for the closure: segments (n0, n1, L, A, j, profile params, sigma_T)."""
    n_nodes: int
    n0: np.ndarray; n1: np.ndarray            # (S,) node ids
    L: np.ndarray; A: np.ndarray; j: np.ndarray
    T_L: np.ndarray; T_R: np.ndarray; T_m: np.ndarray; Gamma: float
    sigma_T: np.ndarray


def close_tree_general(tree: TreeSpec, em: EMParams, provider: KernelProvider, times: np.ndarray,
                       sigma_crit: float | None = None, with_tm: bool = True, jump_at: str = "mid"):
    """Junction closure for a tree with arbitrary branching.  Unknowns: the flux at every
    segment end (2S); equations: q = 0 at terminal ends, sum(+-A q) = 0 at every junction,
    and stress continuity between all ends meeting at a junction.  One dense solve
    (size 2S) per time step.  Returns (sigma_ends (M,S,2), t_nuc, node_of_nucleation)."""
    times = np.asarray(times, float); M = len(times); S = len(tree.L)
    sigma_crit = em.sigma_crit if sigma_crit is None else sigma_crit
    desc = np.zeros((S, 4)); kb = np.zeros(S)
    for k in range(S):
        p = SegmentProfile(tree.L[k], tree.T_L[k], tree.T_R[k], tree.T_m[k], tree.Gamma)
        th, r, rJ, lam, kbar = p.descriptors(em); desc[k] = (th, r, rJ, lam); kb[k] = kbar
    G = em.G_of_j(tree.j); scaleA = tree.L / kb; QO = em.Q_J / em.Omega
    t_jump = jump_times(times, jump_at)
    lag = times[:, None] - t_jump[None, :]; nn, mm = np.tril_indices(M); lag_v = lag[nn, mm]; P = len(lag_v)
    tau_lag = kb[:, None] * lag_v[None, :] / tree.L[:, None] ** 2
    d_rep = np.repeat(desc[:, None, :], P, 1).reshape(-1, 4); d_mir = d_rep.copy(); d_mir[:, 1] *= -1; t_flat = tau_lag.reshape(-1)
    z, o = np.zeros(S * P), np.ones(S * P)
    AL0 = scaleA[:, None] * provider.evaluate(d_rep, z, t_flat)["a"].reshape(S, P)
    AL1 = scaleA[:, None] * provider.evaluate(d_rep, o, t_flat)["a"].reshape(S, P)
    AR0 = -scaleA[:, None] * provider.evaluate(d_mir, o, t_flat)["a"].reshape(S, P)
    AR1 = -scaleA[:, None] * provider.evaluate(d_mir, z, t_flat)["a"].reshape(S, P)
    tau_out = kb[:, None] * times[None, :] / tree.L[:, None] ** 2
    d2 = np.repeat(desc[:, None, :], M, 1).reshape(-1, 4); t2 = tau_out.reshape(-1)
    p0 = provider.evaluate(d2, np.zeros(S * M), t2); p1 = provider.evaluate(d2, np.ones(S * M), t2)
    P0 = tree.sigma_T[:, None] + (G * tree.L)[:, None] * p0["s_G"].reshape(S, M)
    P1 = tree.sigma_T[:, None] + (G * tree.L)[:, None] * p1["s_G"].reshape(S, M)
    if with_tm:
        P0 = P0 + QO * p0["s_M"].reshape(S, M); P1 = P1 + QO * p1["s_M"].reshape(S, M)
    pos = np.full((M, M), -1, int); pos[nn, mm] = np.arange(P)
    # topology: ends e = 2k (left of seg k, node n0) and 2k+1 (right, node n1)
    ends_at = {}
    for k in range(S):
        ends_at.setdefault(int(tree.n0[k]), []).append(2 * k); ends_at.setdefault(int(tree.n1[k]), []).append(2 * k + 1)
    sign = np.array([+1.0, -1.0] * S)                # + for left ends (flux leaves node into segment), - for right ends
    Aend = np.repeat(tree.A, 2)
    dq = np.zeros((M, 2 * S)); sig_ends = np.zeros((M, S, 2))
    for n in range(M):
        h0 = np.zeros(S); h1 = np.zeros(S)
        for m in range(n):
            p = pos[n, m]
            h0 += dq[m, 0::2] * AL0[:, p] + dq[m, 1::2] * AR0[:, p]
            h1 += dq[m, 0::2] * AL1[:, p] + dq[m, 1::2] * AR1[:, p]
        pd = pos[n, n]
        # sigma at end e as affine function of the unknown increments x (2S):
        #   sigma_left(k)  = P0 + h0 + x[2k] AL0 + x[2k+1] AR0
        #   sigma_right(k) = P1 + h1 + x[2k] AL1 + x[2k+1] AR1
        C = np.zeros((2 * S, 2 * S)); c0 = np.zeros(2 * S)
        for k in range(S):
            C[2 * k, 2 * k] = AL0[k, pd]; C[2 * k, 2 * k + 1] = AR0[k, pd]; c0[2 * k] = P0[k, n] + h0[k]
            C[2 * k + 1, 2 * k] = AL1[k, pd]; C[2 * k + 1, 2 * k + 1] = AR1[k, pd]; c0[2 * k + 1] = P1[k, n] + h1[k]
        Amat = np.zeros((2 * S, 2 * S)); rhs = np.zeros(2 * S); row = 0
        for node, es in ends_at.items():
            if len(es) == 1:                                   # terminal: total flux stays zero
                Amat[row, es[0]] = 1.0; row += 1
            else:
                for e in es:                                   # conservation of the increments
                    Amat[row, e] = sign[e] * Aend[e]
                row += 1
                for e in es[1:]:                               # continuity: sigma_e = sigma_es[0]
                    Amat[row] = C[e] - C[es[0]]; rhs[row] = c0[es[0]] - c0[e]; row += 1
        assert row == 2 * S
        rs = np.abs(Amat).max(axis=1); rs[rs == 0] = 1.0          # row equilibration (areas ~1e-12 vs kernels ~1e10)
        x = np.linalg.solve(Amat / rs[:, None], rhs / rs)
        dq[n] = x
        s = C @ x + c0
        sig_ends[n, :, 0] = s[0::2]; sig_ends[n, :, 1] = s[1::2]
    smax = sig_ends.reshape(M, -1).max(axis=1)
    idx = np.where(smax >= sigma_crit)[0]
    if len(idx):
        k = idx[0]; e = int(np.argmax(sig_ends[k].reshape(-1)))
        node = int(tree.n0[e // 2] if e % 2 == 0 else tree.n1[e // 2])
        if k == 0:
            t_nuc = times[0]
        else:
            t0, t1, s0_, s1_ = times[k - 1], times[k], smax[k - 1], smax[k]
            t_nuc = crossing_time(t0, t1, s0_, s1_, sigma_crit)
    else:
        t_nuc = np.inf; node = -1
    return sig_ends, t_nuc, node
