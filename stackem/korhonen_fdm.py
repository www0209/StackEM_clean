"""
stackem.korhonen_fdm
======================

Reference finite-volume solver for the Korhonen EM/TM stress equation on a
single segment (dimensionless kernel problems) and on multi-segment
interconnect trees (physical truth).  This solver is the *ground truth* of
StackEM: it generates the SKN training kernels, validates the junction
closure, and provides the full-tree stress histories used in every experiment.

Governing equation (physical, per segment, x in [0, L])
------------------------------------------------------
    d sigma/dt = d/dx [ kappa(x) ( d sigma/dx - G - M(x) ) ]

    kappa(x) = D_a(T(x)) B Omega / (k_B T(x))          stress diffusivity
    G        = e Z* rho j / Omega                       EM driving gradient
    M(x)     = Q*/(Omega T(x)) dT/dx                    TM driving gradient

Sign convention: for j > 0 atoms are pushed towards +x, so tensile stress
builds at x = L and compressive at x = 0 (steady state sigma' = G).  For a
positive temperature gradient (hot at x = L) atoms move to the cold end x = 0,
so the hot end becomes tensile: steady state sigma' = M > 0.  Both facts are
checked by the test-suite.

Relation to the notation of the paper: the paper writes the bracket as (d sigma/dx + G + M) with a
negative effective charge Z* = -10; this code writes (d sigma/dx - G - M) with ``Z_eff = +10``.  The two
forms are the same physics.  When the equation is typed into another solver (see comsol/), use the form
and the signs of THIS file together with the positive Z_eff exported in constants.json.

Stress flux q(x,t) := kappa (sigma' - G - M)   [Pa m/s].  The atomic flux is
J = (C/B) q, so at a junction shared by segments with cross-sections A_k the
conservation law is  sum_k (+-) A_k q_k = 0  (+ for a left end, - for a right
end).  A blocked end (diffusion barrier) has q = 0.

Discretisation
--------------
Vertex-centred finite volumes on a Chebyshev-graded mesh (dense near the
ends where the early-time boundary layer sqrt(kappa t) lives), theta time
stepping with Rannacher start-up (4 implicit-Euler steps, then
Crank-Nicolson), geometrically growing time steps that land exactly on the
requested output times.  Chains (straight multi-segment rails) are solved with
a banded O(N) solver; general trees with sparse LU.

Validated against (see analytic.py / tests):
  * the blocked-line EM series solution (uniform kappa),
  * the unit-flux Green's function series solution,
  * the early-time semi-infinite solution,
  * mass conservation and grid/time refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import EMParams, default_em
from .thermal_profile import SegmentProfile


# ----------------------------------------------------------------------------
# meshes and time grids
# ----------------------------------------------------------------------------
def chebyshev_mesh(n_cells: int) -> np.ndarray:
    """Node coordinates xi_0..xi_n in [0,1], dense at both ends."""
    i = np.arange(n_cells + 1)
    return 0.5 * (1.0 - np.cos(np.pi * i / n_cells))


def uniform_mesh(n_cells: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_cells + 1)


def default_tau_grid(tau_min: float = 1e-6, tau_max: float = 3.0, n: int = 64) -> np.ndarray:
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n)


def internal_time_steps(t_out: np.ndarray, first_dt: float | None = None, growth: float = 1.10,
                        n_rannacher: int = 4, max_rel_dt: float = 0.04) -> List[Tuple[float, float]]:
    """Build the internal step list [(dt, theta), ...] that lands exactly on every
    output time.  Steps grow geometrically from first_dt (default t_out[0]/32).
    The relative step size is capped at max_rel_dt (dt <= max_rel_dt * t), which
    bounds the Crank-Nicolson error uniformly on a logarithmic output grid.
    Returns the list of (dt, theta)."""
    t_out = np.asarray(t_out, dtype=float)
    assert np.all(np.diff(t_out) > 0) and t_out[0] > 0
    dt = t_out[0] / 32.0 if first_dt is None else first_dt
    steps: List[Tuple[float, float]] = []
    t = 0.0
    k = 0
    while k < len(t_out):
        target = t_out[k]
        while t < target - 1e-12 * target:
            d = min(dt, target - t)
            if target - t - d < 0.25 * dt:      # avoid a tiny last step
                d = target - t
            theta = 1.0 if len(steps) < n_rannacher else 0.5
            steps.append((d, theta))
            t += d
            dt = min(dt * growth, max_rel_dt * t) if t > 0 else dt * growth
        k += 1
    return steps


# ----------------------------------------------------------------------------
# dimensionless single-segment kernel solver
# ----------------------------------------------------------------------------
@dataclass
class KernelSolution:
    xi: np.ndarray            # nodes (n+1,)
    tau: np.ndarray           # output times (m,)
    s: np.ndarray             # (m, n+1) dimensionless stress


def solve_segment_hat(kappa_hat: Callable[[np.ndarray], np.ndarray],
                      source_hat: Callable[[np.ndarray], np.ndarray] | float,
                      q_left: float, q_right: float,
                      tau_out: np.ndarray,
                      ic: float | np.ndarray = 0.0,
                      n_cells: int = 200, mesh: str = "chebyshev",
                      growth: float = 1.10, first_dt: float | None = None, max_rel_dt: float = 0.04) -> KernelSolution:
    """Solve  ds/dtau = d/dxi[ kappa_hat(xi) ( ds/dxi - S(xi) ) ]  on xi in [0,1]
    with prescribed constant boundary fluxes q_hat(0) = q_left, q_hat(1) = q_right
    (q_hat = kappa_hat (s' - S)) and initial condition ic.

    The three SKN kernels are:
        s_G : S = 1,        q = 0, 0     (EM response, scale G L)
        s_M : S = M_hat(xi),q = 0, 0     (TM response, scale Q*/Omega)
        a   : S = 0,        q = 1, 0     (unit-flux Green's function, scale L/kappa_bar)
    """
    tau_out = np.asarray(tau_out, dtype=float)
    xi = chebyshev_mesh(n_cells) if mesh == "chebyshev" else uniform_mesh(n_cells)
    n = n_cells
    xf = 0.5 * (xi[1:] + xi[:-1])                      # faces
    dx = np.diff(xi)
    kf = np.asarray(kappa_hat(xf), dtype=float)
    Sf = (np.full(n, float(source_hat)) if np.isscalar(source_hat)
          else np.asarray(source_hat(xf), dtype=float))
    c = kf / dx                                        # face conductances
    V = np.zeros(n + 1)
    V[0] = xf[0] - xi[0]
    V[1:-1] = xf[1:] - xf[:-1]
    V[-1] = xi[-1] - xf[-1]
    # tridiagonal A (banded storage: upper, diag, lower)
    diag = np.zeros(n + 1); upper = np.zeros(n + 1); lower = np.zeros(n + 1)
    diag[:-1] -= c; diag[1:] -= c
    upper[1:] = c            # A[i, i+1] stored at upper[i+1]
    lower[:-1] = c           # A[i+1, i] stored at lower[i]
    b = np.zeros(n + 1)
    src = kf * Sf            # kappa_hat * S at faces  (flux contribution -src)
    b[:-1] -= src; b[1:] += src
    b[0] -= q_left
    b[-1] += q_right
    s = np.full(n + 1, float(ic)) if np.isscalar(ic) else np.asarray(ic, dtype=float).copy()
    out = np.zeros((len(tau_out), n + 1))
    steps = internal_time_steps(tau_out, first_dt=first_dt, growth=growth, max_rel_dt=max_rel_dt)
    t = 0.0; k_out = 0; last = None
    for dt, theta in steps:
        if last != (dt, theta):
            ab = np.zeros((3, n + 1))
            ab[0] = -theta * upper
            ab[1] = V / dt - theta * diag
            ab[2] = -theta * lower
            last = (dt, theta)
        As = diag * s
        As[:-1] += upper[1:] * s[1:]
        As[1:] += lower[:-1] * s[:-1]
        rhs = V * s / dt + (1.0 - theta) * As + b
        s = sla.solve_banded((1, 1), ab, rhs)
        t += dt
        if k_out < len(tau_out) and abs(t - tau_out[k_out]) <= 1e-9 * tau_out[k_out]:
            out[k_out] = s; k_out += 1
    assert k_out == len(tau_out), "time marching did not hit all output times"
    return KernelSolution(xi=xi, tau=tau_out, s=out)


def segment_kernels(profile: SegmentProfile, em: EMParams, tau_out: np.ndarray,
                    n_cells: int = 200, with_tm: bool = True) -> Dict[str, KernelSolution]:
    """Compute the three dimensionless kernels for a segment profile."""
    kh = lambda xi: profile.kappa_hat(xi, em)
    ker = {
        "s_G": solve_segment_hat(kh, 1.0, 0.0, 0.0, tau_out, 0.0, n_cells),
        "a": solve_segment_hat(kh, 0.0, 1.0, 0.0, tau_out, 0.0, n_cells),
    }
    if with_tm:
        ker["s_M"] = solve_segment_hat(kh, lambda xi: profile.M_hat(xi, em), 0.0, 0.0, tau_out, 0.0, n_cells)
    return ker


# ----------------------------------------------------------------------------
# physical multi-segment tree solver
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    """One Cu segment between two tree nodes."""
    n0: int
    n1: int
    L: float
    A: float                       # cross-section W*H (m^2)
    j: float                       # A/m^2 (sign = direction along n0->n1)
    profile: SegmentProfile
    sigma_T: float = 0.0           # initial (thermal residual) stress (Pa)
    qL: float = 0.0                # prescribed flux at x=0 if n0 is a terminal (normally 0)
    qR: float = 0.0


@dataclass
class TreeSolution:
    times: np.ndarray                          # (m,)
    sigma: np.ndarray                          # (m, N_nodes_global)
    seg_index: List[np.ndarray]                # per segment: global node indices
    seg_x: List[np.ndarray]                    # per segment: local x coordinates (m)
    node_of_tree: int
    t_fine: np.ndarray | None = None           # every internal time step (dt <= 4 % of t) ...
    smax_fine: np.ndarray | None = None        # ... and the stress maximum over the tree there: t_nuc from this
                                               # history does not depend on the (coarse, log-spaced) output grid

    def segment_profile(self, k: int, it: int | None = None):
        if it is None:
            return self.seg_x[k], self.sigma[:, self.seg_index[k]]
        return self.seg_x[k], self.sigma[it, self.seg_index[k]]

    def max_stress_history(self):
        return self.sigma.max(axis=1)

    def nucleation(self, sigma_crit: float):
        """(t_nuc, global node index, sigma_max at crossing); t_nuc = inf if never."""
        return nucleation_from_history(self.times, self.sigma, sigma_crit)


def nucleation_from_fine_history(t_fine: np.ndarray, smax_fine: np.ndarray, sigma_crit: float) -> float:
    """First crossing of sigma_crit on the internal-step history (steps <= 4 % apart in time, so the linear
    interpolation error is ~1e-4 in t_nuc, independent of the output grid).  inf when never crossed."""
    idx = np.where(smax_fine >= sigma_crit)[0]
    if len(idx) == 0:
        return np.inf
    k = int(idx[0])
    if k == 0:
        return float(t_fine[0])
    t0, t1, s0, s1 = t_fine[k - 1], t_fine[k], smax_fine[k - 1], smax_fine[k]
    return float(t0 + (sigma_crit - s0) / (s1 - s0) * (t1 - t0)) if s1 > s0 else float(t1)


def crossing_time(t0: float, t1: float, s0: float, s1: float, sigma_crit: float) -> float:
    """First crossing of sigma_crit between two output times, interpolating linearly in sqrt(t): the EM stress
    maximum grows like sqrt(t) over most of its history, so this rule is nearly exact even on a coarse
    log-spaced grid (M = 32: 0.01 % error vs 1 % for log-linear interpolation, measured against the fine-step
    reference), whereas log-linear interpolation put a ~1 % floor under every t_nuc comparison at M = 32.
    (These two percentages were measured once during development; no result file of that measurement is shipped.)"""
    r0, r1 = np.sqrt(t0), np.sqrt(t1)
    return float((r0 + (sigma_crit - s0) / (s1 - s0) * (r1 - r0)) ** 2) if s1 > s0 else float(t1)


def nucleation_from_history(times: np.ndarray, sigma: np.ndarray, sigma_crit: float):
    smax = sigma.max(axis=1)
    idx = np.where(smax >= sigma_crit)[0]
    if len(idx) == 0:
        return np.inf, int(np.argmax(sigma[-1])), float(smax[-1])
    k = int(idx[0])
    node = int(np.argmax(sigma[k]))
    if k == 0:
        return float(times[0]), node, float(smax[0])
    return crossing_time(times[k - 1], times[k], smax[k - 1], smax[k], sigma_crit), node, float(sigma_crit)


class TreeFDM:
    """Finite-volume Korhonen solver on an interconnect tree.

    Usage
    -----
        tr = TreeFDM(em)
        a = tr.add_node(); b = tr.add_node(); c = tr.add_node()
        tr.add_segment(a, b, L, A, j, profile, sigma_T)
        tr.add_segment(b, c, ...)
        sol = tr.solve(times)
    """

    def __init__(self, em: EMParams | None = None, n_cells: int = 60, mesh: str = "chebyshev"):
        self.em = em or default_em()
        self.n_cells = n_cells
        self.mesh = mesh
        self.n_nodes = 0
        self.segments: List[Segment] = []

    def add_node(self) -> int:
        self.n_nodes += 1
        return self.n_nodes - 1

    def add_segment(self, n0: int, n1: int, L: float, A: float, j: float,
                    profile: SegmentProfile, sigma_T: float = 0.0, qL: float = 0.0, qR: float = 0.0) -> int:
        self.segments.append(Segment(n0, n1, L, A, j, profile, sigma_T, qL, qR))
        return len(self.segments) - 1

    # ---- topology helpers -------------------------------------------------
    def degree(self) -> np.ndarray:
        d = np.zeros(self.n_nodes, int)
        for s in self.segments:
            d[s.n0] += 1; d[s.n1] += 1
        return d

    def is_chain(self) -> bool:
        """True if the segments form a simple path n0 -> n1 -> n2 ... in order."""
        for k in range(len(self.segments) - 1):
            if self.segments[k].n1 != self.segments[k + 1].n0:
                return False
        d = self.degree()
        return d.max() <= 2

    # ---- assembly -------------------------------------------------------------
    def _assemble(self):
        em = self.em
        n = self.n_cells
        xi = chebyshev_mesh(n) if self.mesh == "chebyshev" else uniform_mesh(n)
        chain = self.is_chain()
        S = len(self.segments)
        # global numbering
        if chain:
            # sequential: node0, interior seg0, node1, interior seg1, node2 ...
            N = 1 + S * n
            gidx = [np.arange(k * n, k * n + n + 1) for k in range(S)]
        else:
            N = self.n_nodes + S * (n - 1)
            gidx = []
            off = self.n_nodes
            for s in self.segments:
                g = np.concatenate(([s.n0], np.arange(off, off + n - 1), [s.n1]))
                off += n - 1
                gidx.append(g)
        rows, cols, vals = [], [], []
        V = np.zeros(N); b = np.zeros(N); sig0 = np.zeros(N); w0 = np.zeros(N)
        seg_x = []
        for k, s in enumerate(self.segments):
            g = gidx[k]
            x = xi * s.L; seg_x.append(x)
            xf = 0.5 * (x[1:] + x[:-1]); dx = np.diff(x)
            Tf = s.profile.T_of_xi(xf / s.L)
            kf = em.kappa(Tf)
            G = em.G_of_j(s.j)
            M = em.M_of(Tf, s.profile.dTdx_of_xi(xf / s.L))
            c = kf / dx * s.A
            src = kf * (G + M) * s.A
            a_, b_ = g[:-1], g[1:]
            rows += list(a_) + list(a_) + list(b_) + list(b_)
            cols += list(a_) + list(b_) + list(b_) + list(a_)
            vals += list(-c) + list(c) + list(-c) + list(c)
            np.add.at(b, a_, -src); np.add.at(b, b_, src)
            vol = np.zeros(n + 1)
            vol[0] = xf[0] - x[0]; vol[1:-1] = xf[1:] - xf[:-1]; vol[-1] = x[-1] - xf[-1]
            np.add.at(V, g, vol * s.A)
            b[g[0]] -= s.qL * s.A; b[g[-1]] += s.qR * s.A
            # initial stress: volume-weighted average at shared nodes
            np.add.at(sig0, g, s.sigma_T * vol * s.A); np.add.at(w0, g, vol * s.A)
        Amat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        sig0 = sig0 / w0
        self._N, self._A, self._V, self._b, self._sig0 = N, Amat, V, b, sig0
        self._gidx, self._seg_x, self._chain = gidx, seg_x, chain
        return self

    # ---- time integration -----------------------------------------------------
    def solve(self, times: Sequence[float], growth: float = 1.10, first_dt: float | None = None,
              max_rel_dt: float = 0.04) -> TreeSolution:
        times = np.asarray(times, dtype=float)
        self._assemble()
        N, A, V, b, sig = self._N, self._A, self._V, self._b, self._sig0.copy()
        steps = internal_time_steps(times, first_dt=first_dt, growth=growth, max_rel_dt=max_rel_dt)
        out = np.zeros((len(times), N)); t = 0.0; k_out = 0; last = None
        node_idx = np.unique(np.array([g[0] for g in self._gidx] + [g[-1] for g in self._gidx], int))   # junctions and ends
        t_fine = [0.0]; smax_fine = [float(sig[node_idx].max())]                                        # (same nodes the closure sees)
        if self._chain:
            # banded storage of A
            diag = A.diagonal(); upper = np.zeros(N); lower = np.zeros(N)
            coo = A.tocoo()
            for r, cc, v in zip(coo.row, coo.col, coo.data):
                if cc == r + 1: upper[cc] = v
                elif cc == r - 1: lower[cc] = v
            for dt, theta in steps:
                if last != (dt, theta):
                    ab = np.zeros((3, N)); ab[0] = -theta * upper; ab[1] = V / dt - theta * diag; ab[2] = -theta * lower
                    last = (dt, theta)
                As = A @ sig
                rhs = V * sig / dt + (1 - theta) * As + b
                sig = sla.solve_banded((1, 1), ab, rhs)
                t += dt
                t_fine.append(t); smax_fine.append(float(sig[node_idx].max()))
                if k_out < len(times) and abs(t - times[k_out]) <= 1e-9 * times[k_out]:
                    out[k_out] = sig; k_out += 1
        else:
            I = sp.diags(V)
            for dt, theta in steps:
                if last != (dt, theta):
                    lu = spla.splu((I / dt - theta * A).tocsc()); last = (dt, theta)
                rhs = V * sig / dt + (1 - theta) * (A @ sig) + b
                sig = lu.solve(rhs)
                t += dt
                t_fine.append(t); smax_fine.append(float(sig[node_idx].max()))
                if k_out < len(times) and abs(t - times[k_out]) <= 1e-9 * times[k_out]:
                    out[k_out] = sig; k_out += 1
        assert k_out == len(times)
        return TreeSolution(times=times, sigma=out, seg_index=self._gidx, seg_x=self._seg_x, node_of_tree=N,
                            t_fine=np.array(t_fine), smax_fine=np.array(smax_fine))

    def total_stress_integral(self, sol: TreeSolution) -> np.ndarray:
        """int sigma dV over the tree at each time (conserved when all ends are blocked)."""
        return sol.sigma @ self._V


def build_chain_tree(em: EMParams, segs: Sequence[dict], n_cells: int = 60) -> TreeFDM:
    """Convenience: segs = [dict(L, A, j, profile, sigma_T), ...] forming a straight rail."""
    tr = TreeFDM(em, n_cells=n_cells)
    nodes = [tr.add_node() for _ in range(len(segs) + 1)]
    for k, s in enumerate(segs):
        tr.add_segment(nodes[k], nodes[k + 1], s["L"], s["A"], s["j"], s["profile"], s.get("sigma_T", 0.0))
    return tr


def single_segment_physical(em: EMParams, L: float, A: float, j: float, profile: SegmentProfile,
                            times: Sequence[float], sigma_T: float = 0.0, qL: float = 0.0, qR: float = 0.0,
                            n_cells: int = 200) -> TreeSolution:
    tr = TreeFDM(em, n_cells=n_cells)
    a, b_ = tr.add_node(), tr.add_node()
    tr.add_segment(a, b_, L, A, j, profile, sigma_T, qL, qR)
    return tr.solve(times)
