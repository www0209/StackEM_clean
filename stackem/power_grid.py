"""
stackem.power_grid
====================

Power-Grid Solve (PGS): per-die DC power-delivery network, current densities,
and extraction of the straight multi-segment rails ("trees") analysed by the
EM solver.  Also the Thermal Field Embedding part 2: attaching the via
temperatures, the Joule bump and the thermal residual stress to every segment.

Model
-----
* One horizontal + one vertical global metal layer per die (the pair that
  carries ``current_fraction`` of the die current; the remaining fraction flows
  in parallel metal pairs that are not analysed).  Vias at every crossing,
  ideal.  Node lattice n x n with pitch ``pitch`` over a square die.
* Current sinks: each floorplan block's power P_b / Vdd is spread uniformly
  over the lattice nodes inside the block.
* Vertical feeds (TSV clusters for a die fed from below, hybrid-bond pad
  clusters for a die fed from the die beneath) sit on a coarser lattice with
  pitch ``feed_pitch`` and are modelled as Vdd sources through R_feed.
* The DC solution is a sparse linear system (MNA); current densities follow
  from branch voltages.  Because the network is linear, the exact sensitivity
  d j / d P_block is obtained from unit-power solves (used by E5).

Rails / trees
-------------
Every horizontal (vertical) stripe is one continuous Cu line with a diffusion
barrier only at its two ends; the crossings are junctions where the current
changes.  A stripe is therefore a straight multi-segment tree with n-1
segments of length ``pitch``.  The barrier at via bottoms blocks atomic flux
between metal layers, so stripes are independent Korhonen problems coupled
only through j and T (assumption A1 of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import EMParams, ThermalParams
from .hotspot_stack import DieSpec, grid_T_at


@dataclass
class PGParams:
    pitch: float = 100e-6       # m  lattice pitch of the analysed metal pair
    W: float = 2.0e-6           # m  rail width
    H: float = 1.0e-6           # m  rail thickness
    rho: float = 3.0e-8         # Ohm m
    Vdd: float = 0.8            # V
    current_fraction: float = 0.5   # share of the die current carried by this metal pair
    R_feed: float = 0.0         # Ohm, series resistance of a feed site (0 = ideal)

    @property
    def A(self) -> float:
        return self.W * self.H

    @property
    def R_seg(self) -> float:
        return self.rho * self.pitch / self.A


@dataclass
class DieGridSolution:
    die: str
    n: int
    pitch: float
    V: np.ndarray            # (n, n) node voltages [r (y index), c (x index)]
    I_h: np.ndarray          # (n, n-1) current (+x) on horizontal branches, A
    I_v: np.ndarray          # (n-1, n) current (+y) on vertical branches, A
    feed_mask: np.ndarray    # (n, n) bool
    sink_A: np.ndarray       # (n, n) node current sinks, A
    Vdd: float

    @property
    def ir_drop(self) -> np.ndarray:
        return self.Vdd - self.V

    def j_h(self, A: float) -> np.ndarray:
        return self.I_h / A

    def j_v(self, A: float) -> np.ndarray:
        return self.I_v / A


class DieGrid:
    """DC solver for one die's analysed metal pair."""

    def __init__(self, die: DieSpec, pg: PGParams):
        from dataclasses import replace as _replace
        self.die = die
        self.pg = _replace(pg, W=float(die.rail_W)) if die.rail_W else pg      # per-die strap width (sign-off sizing)
        self.n = int(round(die.size / pg.pitch)) + 1
        self.xs = np.arange(self.n) * pg.pitch
        self.ys = np.arange(self.n) * pg.pitch
        self._G = self._laplacian()

    # ---- lattice helpers ----
    def nid(self, r: int, c: int) -> int:
        return r * self.n + c

    def _laplacian(self) -> sp.csr_matrix:
        n = self.n; g = 1.0 / self.pg.R_seg
        r = np.repeat(np.arange(n), n - 1); c = np.tile(np.arange(n - 1), n)
        a = r * n + c; b = a + 1                                  # horizontal
        r2 = np.tile(np.arange(n), n - 1); c2 = np.repeat(np.arange(n - 1), n)
        a2 = c2 * n + r2; b2 = a2 + n                              # vertical (row c2 -> c2+1)
        A = np.concatenate([a, b, a, b, a2, b2, a2, b2])
        Bc = np.concatenate([a, b, b, a, a2, b2, b2, a2])
        V = np.concatenate([np.full(len(a), g), np.full(len(a), g), np.full(len(a), -g), np.full(len(a), -g),
                            np.full(len(a2), g), np.full(len(a2), g), np.full(len(a2), -g), np.full(len(a2), -g)])
        return sp.csr_matrix((V, (A, Bc)), shape=(n * n, n * n))

    def feed_mask(self, feed_pitch: float | None = None) -> np.ndarray:
        fp = self.die.feed_pitch if feed_pitch is None else feed_pitch
        step = max(1, int(round(fp / self.pg.pitch)))
        m = np.zeros((self.n, self.n), bool)
        m[step // 2::step, step // 2::step] = True
        return m

    def sinks_from_blocks(self, power: np.ndarray | None = None) -> np.ndarray:
        """Node current sinks (A) from the block power map (nblk x nblk, index [i(x), j(y)])."""
        pm = self.die.power if power is None else power
        nblk = self.die.nblk; bw = self.die.size / nblk
        n = self.n
        bi = np.minimum((self.xs / bw).astype(int), nblk - 1)      # block x-index of each column
        bj = np.minimum((self.ys / bw).astype(int), nblk - 1)      # block y-index of each row
        sinks = np.zeros((n, n))
        for i in range(nblk):
            for j in range(nblk):
                rows = np.where(bj == j)[0]; cols = np.where(bi == i)[0]
                sinks[np.ix_(rows, cols)] += pm[i, j] * self.pg.current_fraction / self.pg.Vdd / (len(rows) * len(cols))
        return sinks

    def solve(self, sinks: np.ndarray, feed_mask: np.ndarray) -> DieGridSolution:
        n = self.n; N = n * n; Vdd = self.pg.Vdd
        G = self._G.tolil(copy=True)
        I = -sinks.ravel().astype(float)
        feeds = np.where(feed_mask.ravel())[0]
        if self.pg.R_feed > 0:
            gf = 1.0 / self.pg.R_feed
            for k in feeds:
                G[k, k] += gf; I[k] += gf * Vdd
        else:
            for k in feeds:
                G.rows[k] = [int(k)]; G.data[k] = [1.0]; I[k] = Vdd
        V = spla.spsolve(G.tocsc(), I).reshape(n, n)
        R = self.pg.R_seg
        I_h = (V[:, :-1] - V[:, 1:]) / R
        I_v = (V[:-1, :] - V[1:, :]) / R
        return DieGridSolution(self.die.name, n, self.pg.pitch, V, I_h, I_v, feed_mask, sinks, Vdd)

    def solve_from_blocks(self, power: np.ndarray | None = None, feed_pitch: float | None = None) -> DieGridSolution:
        return self.solve(self.sinks_from_blocks(power), self.feed_mask(feed_pitch))

    def unit_power_current_responses(self, feed_pitch: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Exact linear sensitivities: returns (S_h, S_v) with shapes
        (nblk*nblk, n, n-1) and (nblk*nblk, n-1, n): I_h = sum_b P_b S_h[b]  (block order i-major)."""
        nblk = self.die.nblk
        fm = self.feed_mask(feed_pitch)
        Sh = np.zeros((nblk * nblk, self.n, self.n - 1)); Sv = np.zeros((nblk * nblk, self.n - 1, self.n))
        for b in range(nblk * nblk):
            pm = np.zeros((nblk, nblk)); pm[b // nblk, b % nblk] = 1.0
            sol = self.solve(self.sinks_from_blocks(pm), fm)
            Sh[b] = sol.I_h; Sv[b] = sol.I_v
        return Sh, Sv


# ----------------------------------------------------------------------------
# rails (straight multi-segment trees) with thermal embedding
# ----------------------------------------------------------------------------
@dataclass
class Rail:
    """A straight multi-segment Cu rail: one Korhonen tree."""
    die: str
    kind: str                 # 'H' or 'V'
    index: int                # stripe index (row for H, column for V)
    x0: float; y0: float      # start coordinate (m)
    L: np.ndarray             # (S,) segment lengths (m)
    W: float; H: float
    j: np.ndarray             # (S,) A/m^2 signed along +x / +y
    T_L: np.ndarray           # (S,) via temperature at segment start (K)
    T_R: np.ndarray           # (S,) via temperature at segment end (K)
    T_m: np.ndarray           # (S,) Joule bump amplitude (K)
    Gamma: float              # m
    sigma_T: np.ndarray       # (S,) initial thermal residual stress (Pa)
    feed_at_node: np.ndarray  # (S+1,) bool, node is a vertical feed site

    @property
    def n_seg(self) -> int:
        return len(self.L)

    @property
    def A(self) -> float:
        return self.W * self.H

    def node_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        s = np.concatenate([[0.0], np.cumsum(self.L)])
        if self.kind == "H":
            return self.x0 + s, np.full_like(s, self.y0)
        return np.full_like(s, self.x0), self.y0 + s

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Rail":
        d = dict(d)
        for k in ("L", "j", "T_L", "T_R", "T_m", "sigma_T", "feed_at_node"):
            d[k] = np.asarray(d[k])
        d["feed_at_node"] = d["feed_at_node"].astype(bool)
        return Rail(**d)


def embed_thermal(grid: DieGrid, sol: DieGridSolution, Tgrid: np.ndarray, em: EMParams, th: ThermalParams,
                  two_sided: bool | None = None, joule: bool = True, sigma_T_mode: str = "linear",
                  uniform_T: float | None = None) -> List[Rail]:
    """Attach temperatures / Joule bump / sigma_T to every stripe and return the rails.

    Tgrid : (rows, cols) die temperature field (K) from HotSpot.
    uniform_T : if given, every node gets this temperature (the "uniform-temperature" baseline).
    joule : include wire self-heating (T_m from j) or not (E3d baseline).
    sigma_T_mode : "linear" (temperature dependent) | "constant" (E3c baseline).
    """
    n = grid.n; A = grid.pg.A; p = grid.pg.pitch
    X, Y = np.meshgrid(grid.xs, grid.ys)            # [r, c] -> (x=xs[c], y=ys[r])
    if uniform_T is None:
        Tn = grid_T_at(Tgrid, grid.die.size, X, Y)
    else:
        Tn = np.full((n, n), float(uniform_T))
    Gam = th.Gamma(two_sided)
    fm = sol.feed_mask
    rails: List[Rail] = []
    em_sT = em
    if sigma_T_mode == "constant" and em.sigma_T_model != "constant":
        from dataclasses import replace
        em_sT = replace(em, sigma_T_model="constant")
    for r in range(n):                                # horizontal stripes (y = ys[r])
        j = sol.I_h[r, :] / A
        TL, TR = Tn[r, :-1], Tn[r, 1:]
        Tm = th.T_m(np.abs(j), two_sided) if joule else np.zeros_like(j)
        Tbar = 0.5 * (TL + TR) + (Tm * (1.0 - np.tanh(0.5 * p / Gam) / (0.5 * p / Gam)) if joule else 0.0)
        rails.append(Rail(grid.die.name, "H", r, 0.0, float(grid.ys[r]), np.full(n - 1, p), grid.pg.W, grid.pg.H,
                          j, TL, TR, Tm, Gam, np.asarray(em_sT.sigma_T(Tbar)), fm[r, :].copy()))
    for c in range(n):                                # vertical stripes (x = xs[c])
        j = sol.I_v[:, c] / A
        TL, TR = Tn[:-1, c], Tn[1:, c]
        Tm = th.T_m(np.abs(j), two_sided) if joule else np.zeros_like(j)
        Tbar = 0.5 * (TL + TR) + (Tm * (1.0 - np.tanh(0.5 * p / Gam) / (0.5 * p / Gam)) if joule else 0.0)
        rails.append(Rail(grid.die.name, "V", c, float(grid.xs[c]), 0.0, np.full(n - 1, p), grid.pg.W, grid.pg.H,
                          j, TL, TR, Tm, Gam, np.asarray(em_sT.sigma_T(Tbar)), fm[:, c].copy()))
    return rails


def save_rails(path: str, rails: Sequence[Rail], meta: dict | None = None):
    with open(path, "w") as f:
        json.dump(dict(meta=meta or {}, rails=[r.to_dict() for r in rails]), f)


def load_rails(path: str) -> Tuple[List[Rail], dict]:
    d = json.load(open(path))
    return [Rail.from_dict(r) for r in d["rails"]], d.get("meta", {})


def pg_summary(sol: DieGridSolution, pg: PGParams) -> Dict[str, float]:
    jh = np.abs(sol.j_h(pg.A)); jv = np.abs(sol.j_v(pg.A))
    jall = np.concatenate([jh.ravel(), jv.ravel()])
    return dict(ir_drop_max_mV=float(sol.ir_drop.max() * 1e3), ir_drop_mean_mV=float(sol.ir_drop.mean() * 1e3),
                j_max_A_per_cm2=float(jall.max() / 1e4), j_p50_A_per_cm2=float(np.median(jall) / 1e4),
                j_p90_A_per_cm2=float(np.percentile(jall, 90) / 1e4),
                total_sink_A=float(sol.sink_A.sum()), n_feeds=int(sol.feed_mask.sum()))


def check_current_regime(summary: Dict[str, float], j_lo: float = 1e5, j_hi: float = 3e6) -> List[str]:
    """Sanity warnings: EM analysis is meaningful for 1e5 <= j_max <= 3e6 A/cm^2 and IR drop < 10 % Vdd."""
    msgs = []
    if not (j_lo <= summary["j_max_A_per_cm2"] <= j_hi):
        msgs.append(f"j_max={summary['j_max_A_per_cm2']:.3g} A/cm^2 outside [{j_lo:g},{j_hi:g}] - adjust power, current_fraction or W/H")
    return msgs
