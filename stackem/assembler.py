"""
stackem.assembler
===================

Differentiable Stack Assembler (DSA): the stack-level driver that turns a
stack description into rail-level stress histories, nucleation statistics,
critical-rail rankings and the physics-decomposition variants used by the
experiments.

Pipeline for one stack case
---------------------------
    StackSpec ── HotSpot (stack + every die alone) ──► thermal fields
             ── per die: PG solve ──► j on every rail
             ── thermal embedding (variant) ──► rails with T_L, T_R, T_m, sigma_T
             ── BIS steady-state screen ──► immortal rails skipped
             ── junction closure (SKN kernels) or FDM truth ──► sigma_nodes(t), t_nuc
             ── statistics ──► per rail / per die / per stack metrics

Physics variants (E3 decomposition)
-----------------------------------
    full        3-D temperature field, sigma_T(T), Joule bump, TM           (StackEM)
    alone       die simulated as a standalone 2-D chip                       (E3a baseline)
    uniform     die-average temperature everywhere                           (E3b baseline)
    sigmaT_const initial stress frozen at the standalone-die temperature       (E3c baseline)
    no_joule    no wire self-heating                                          (E3d baseline)
    no_tm       Q* = 0                                                        (E3e baseline)
    uniform_signoff  uniform worst-case junction temperature (105 C) and constant sigma_T:
                     what an industrial rule-based sign-off assumes           (industry baseline)

Evaluation quantities (the "metric ladder" of the paper)
--------------------------------------------------------
    level 1  fields      T (K) per die, IR drop (mV), j (A/cm^2)
    level 2  segment     kappa, G L (MPa), Joule rise (K), sigma_T (MPa), Blech ratio
    level 3  rail        sigma_max(t) (MPa), t_nuc (yr), nucleation node (x,y), steady max
    level 4  die         mortal rails within 1/10/100 yr, earliest t_nuc, t_nuc CDF, top-k rails
    level 5  stack       earliest t_nuc, critical die, cross-die alignment score,
                         effect ratios of the variants
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .constants import EMParams, ThermalParams, SEC_PER_YEAR
from .hotspot_stack import StackSpec, DieSpec, HotSpotRunner, ThermalField, run_stack, grid_T_at
from .power_grid import PGParams, DieGrid, DieGridSolution, Rail, embed_thermal, pg_summary, check_current_regime
from .blech_screen import screen_rails, vertical_screen, screen_table
from .closure import KernelProvider, close_rails, RailResult
from .korhonen_fdm import nucleation_from_fine_history, TreeFDM, nucleation_from_history
from .thermal_profile import SegmentProfile

VARIANTS = ["full", "alone", "uniform_signoff", "uniform", "sigmaT_const", "no_joule", "no_tm"]
T_SIGNOFF_DEFAULT = 378.15     # K, uniform worst-case junction temperature of an industrial EM rule (105 C)


@dataclass
class RailMetrics:
    die: str
    kind: str
    index: int
    t_nuc_s: float
    nuc_node: int
    nuc_x: float
    nuc_y: float
    sigma_ss_max: float
    immortal: bool
    j_max: float
    T_max: float
    sigma_T_mean: float
    margin_h: float = float("nan")      # 1 - max sigma within the sign-off horizon / sigma_crit  (design margin)


@dataclass
class DieMetrics:
    die: str
    n_rails: int
    n_immortal: int
    mortal_within: Dict[str, int]
    earliest_t_nuc_s: float
    earliest_rail: str
    T_min: float; T_max: float; T_mean: float
    ir_drop_max_mV: float
    j_max_A_per_cm2: float
    min_margin_h: float = float("nan")


@dataclass
class VariantResult:
    variant: str
    rails: List[Rail]
    metrics: List[RailMetrics]
    die_metrics: Dict[str, DieMetrics]
    sigma_nodes: Dict[str, np.ndarray]          # die -> (n_rails, M, S+1)
    t_nuc: Dict[str, np.ndarray]                # die -> (n_rails,)
    immortal: Dict[str, np.ndarray]
    seconds: float


def rails_for_variant(variant: str, grid: DieGrid, sol: DieGridSolution, tf: ThermalField, em: EMParams,
                      th: ThermalParams, T_signoff: float = T_SIGNOFF_DEFAULT) -> Tuple[List[Rail], EMParams]:
    name = grid.die.name
    if variant == "full":
        return embed_thermal(grid, sol, tf.T_die[name], em, th), em
    if variant == "uniform_signoff":
        # industrial rule: one worst-case junction temperature for the whole die, residual stress a constant
        em_c = replace(em, sigma_T_model="constant", sigma_T_const=float(em.sigma_T(T_signoff)))
        return embed_thermal(grid, sol, tf.T_die[name], em_c, th, uniform_T=T_signoff), em_c
    if variant == "alone":
        return embed_thermal(grid, sol, tf.T_alone[name], em, th, two_sided=False), em
    if variant == "uniform":
        return embed_thermal(grid, sol, tf.T_die[name], em, th, uniform_T=tf.T_uniform(name)), em
    if variant == "sigmaT_const":
        # what a die-level sign-off would use: the residual stress at the *standalone* die
        # temperature, kept fixed while the 3-D field only accelerates diffusion
        T_ref = float(tf.T_alone[name].mean()) if name in tf.T_alone else 353.0
        em_c = replace(em, sigma_T_model="constant", sigma_T_const=float(em.sigma_T(T_ref)))
        return embed_thermal(grid, sol, tf.T_die[name], em_c, th), em_c
    if variant == "no_joule":
        return embed_thermal(grid, sol, tf.T_die[name], em, th, joule=False), em
    if variant == "no_tm":
        return embed_thermal(grid, sol, tf.T_die[name], em, th), replace(em, Q_eV=0.0)
    raise ValueError(variant)


def _sigma_max_within(times: np.ndarray, sig: np.ndarray, t_h: float) -> np.ndarray:
    """max over nodes and over t <= t_h of sigma (B,M,S+1), with sigma at t_h itself interpolated in log t between
    the bracketing grid points.  NOTE: t_nuc itself is interpolated linearly in sqrt(t) (korhonen_fdm.crossing_time), not in
    log t, so the two rules are not identical: a rail whose t_nuc lies within one output interval of t_h can show a margin
    of the opposite sign by the interpolation difference."""
    M = len(times)
    if t_h <= times[0]:
        return np.nanmax(sig[:, 0, :], axis=1)
    k = int(np.searchsorted(times, t_h, side="right") - 1)          # times[k] <= t_h
    m = np.nanmax(sig[:, :k + 1, :], axis=(1, 2))
    if k + 1 < M:
        w = (np.log(t_h) - np.log(times[k])) / (np.log(times[k + 1]) - np.log(times[k]))
        s_h = (1 - w) * sig[:, k, :] + w * sig[:, k + 1, :]
        m = np.maximum(m, np.nanmax(s_h, axis=1))
    return m


def cells_for_length(L_max: float, base: int = 60) -> int:
    """Finite-volume cells per segment so that the early-time boundary layer stays resolved on long segments.
    (The percentages below were measured once during development; no result file of that measurement is shipped.)
    The Chebyshev mesh with `base` cells is converged (<0.5 % in t_nuc) up to ~300 um; measured on a 5-segment rail
    at j = 1 MA/cm^2 the 60-cell error is +5 % at 1 mm, +15 % at 3 mm and +32 % at 10 mm, while 240 cells give
    +0.3 % / +3 % and 960 cells are converged at 10 mm.  Benchmarks with millimetre segments (IBM-PG) need this."""
    if L_max <= 300e-6:
        return base
    if L_max <= 3e-3:
        return 4 * base
    return 16 * base


def solve_rails_truth(rails: Sequence[Rail], em: EMParams, times: np.ndarray, n_cells: int = 60,
                      with_tm: bool = True, adaptive_cells: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference: full finite-volume tree solve of every rail. Returns
    (sigma_nodes (B,M,S+1), t_nuc (B,), nuc_node (B,)).  adaptive_cells: raise n_cells per rail with its longest
    segment (cells_for_length), for benchmarks whose segments are much longer than the 100-um straps of the stacks."""
    B = len(rails); M = len(times); S = rails[0].n_seg
    em_ = em if with_tm else replace(em, Q_eV=0.0)
    sig = np.zeros((B, M, S + 1)); tn = np.zeros(B); nn_ = np.zeros(B, int)
    for b, r in enumerate(rails):
        nc = max(n_cells, cells_for_length(float(np.max(r.L)), n_cells)) if adaptive_cells else n_cells
        tr = TreeFDM(em_, n_cells=nc)
        nodes = [tr.add_node() for _ in range(S + 1)]
        for k in range(S):
            prof = SegmentProfile(r.L[k], r.T_L[k], r.T_R[k], r.T_m[k], r.Gamma)
            tr.add_segment(nodes[k], nodes[k + 1], r.L[k], r.A, r.j[k], prof, r.sigma_T[k])
        sol = tr.solve(times)
        idx = [tr._gidx[k][0] for k in range(S)] + [tr._gidx[-1][-1]]
        sig[b] = sol.sigma[:, idx]
        t, node, _ = nucleation_from_history(times, sig[b], em.sigma_crit)
        # t_nuc from the fine internal history: the reference must not inherit the interpolation error of the coarse
        # output grid (with M = 32 log-spaced points that error is ~1 %, and it would cancel against the same error in
        # the surrogate when both interpolate on the same grid)
        if sol.t_fine is not None:
            t = nucleation_from_fine_history(sol.t_fine, sol.smax_fine, em.sigma_crit)
        tn[b] = t; nn_[b] = node
    return sig, tn, nn_


class StackAssembler:
    """Runs the full StackEM flow for one stack case."""

    def __init__(self, spec: StackSpec, pg: PGParams, em: EMParams, th: ThermalParams, runner: HotSpotRunner,
                 times: np.ndarray, workdir: str, provider: Optional[KernelProvider] = None,
                 jump_at: str = "mid", with_tm: bool = True, truth_cells: int = 60, screen: bool = True,
                 horizon_years: float = 10.0, T_signoff: float = T_SIGNOFF_DEFAULT):
        self.spec, self.pg, self.em, self.th, self.runner = spec, pg, em, th, runner
        self.horizon_years, self.T_signoff = horizon_years, T_signoff
        self.times = np.asarray(times, float); self.workdir = workdir; os.makedirs(workdir, exist_ok=True)
        self.provider = provider; self.jump_at = jump_at; self.with_tm = with_tm
        self.truth_cells = truth_cells; self.screen = screen
        self.closure_batch = 32
        self.tf: Optional[ThermalField] = None
        self.grids: Dict[str, DieGrid] = {}
        self.pg_sol: Dict[str, DieGridSolution] = {}
        self.pg_stats: Dict[str, Dict] = {}

    # ---- stage 1: thermal + electrical fields ----------------------------------
    def run_fields(self, with_alone: bool = True) -> "StackAssembler":
        self.tf = run_stack(self.spec, os.path.join(self.workdir, "hotspot"), self.runner, "stack", with_alone)
        for d in self.spec.dies:
            g = DieGrid(d, self.pg); sol = g.solve_from_blocks()
            self.grids[d.name] = g; self.pg_sol[d.name] = sol
            s = pg_summary(sol, g.pg); s["warnings"] = check_current_regime(s)      # g.pg carries the die's own strap width
            self.pg_stats[d.name] = s
        return self

    # ---- stage 2: rails for a variant ------------------------------------------------
    def rails(self, variant: str = "full") -> Tuple[Dict[str, List[Rail]], EMParams]:
        assert self.tf is not None, "run_fields() first"
        out = {}; em_v = self.em
        for d in self.spec.dies:
            rails, em_v = rails_for_variant(variant, self.grids[d.name], self.pg_sol[d.name], self.tf, self.em, self.th, self.T_signoff)
            out[d.name] = rails
        return out, em_v

    # ---- stage 3: solve (closure or truth) ----------------------------------------------
    def solve_variant(self, variant: str = "full", method: str = "closure", n_max_per_die: int | None = None) -> VariantResult:
        t0 = time.time()
        rails_by_die, em_v = self.rails(variant)
        with_tm = self.with_tm and (variant != "no_tm")
        metrics: List[RailMetrics] = []; die_metrics = {}; sig_all = {}; tn_all = {}; imm_all = {}
        M = len(self.times)
        for d in self.spec.dies:
            rails = rails_by_die[d.name]
            if n_max_per_die:
                rails = rails[:n_max_per_die]
            S = rails[0].n_seg; B = len(rails)
            immortal, ss_max = screen_rails(rails, em_v, with_tm) if self.screen else (np.zeros(B, bool), np.zeros(B))
            sig = np.full((B, M, S + 1), np.nan); tn = np.full(B, np.inf); nn_ = np.zeros(B, int)
            todo = np.where(~immortal)[0]
            if len(todo):
                sub = [rails[i] for i in todo]
                if method == "closure":
                    assert self.provider is not None, "a kernel provider is required for method='closure'"
                    for i0 in range(0, len(sub), self.closure_batch):        # bounded memory: (batch x S x M^2/2) x 4 kernel arrays
                        res = close_rails(sub[i0:i0 + self.closure_batch], em_v, self.provider, self.times, em_v.sigma_crit, with_tm, self.jump_at)
                        t_ = todo[i0:i0 + self.closure_batch]
                        sig[t_] = res.sigma_nodes; tn[t_] = res.t_nuc; nn_[t_] = res.nuc_node
                elif method == "truth":
                    s_, t_, n_ = solve_rails_truth(sub, em_v, self.times, self.truth_cells, with_tm)
                    sig[todo] = s_; tn[todo] = t_; nn_[todo] = n_
                else:
                    raise ValueError(method)
            # immortal rails: report their exact steady state at the nodes (constant in time) for completeness
            for i in np.where(immortal)[0]:
                from .blech_screen import rail_steady_state
                sig[i] = rail_steady_state(rails[i], em_v, with_tm)[0][None, :]
            marg = 1.0 - _sigma_max_within(self.times, sig, self.horizon_years * SEC_PER_YEAR) / em_v.sigma_crit
            for b, r in enumerate(rails):
                xs, ys = r.node_xy()
                metrics.append(RailMetrics(d.name, r.kind, r.index, float(tn[b]), int(nn_[b]), float(xs[nn_[b]]), float(ys[nn_[b]]),
                                           float(ss_max[b]), bool(immortal[b]), float(np.abs(r.j).max()),
                                           float(max(r.T_L.max(), r.T_R.max()) + r.T_m.max()), float(r.sigma_T.mean()), float(marg[b])))
            hz = {f"{h:g}yr": int(np.sum(tn <= h * SEC_PER_YEAR)) for h in (1.0, 10.0, 100.0)}
            k_first = int(np.argmin(tn))
            Tg = self.tf.T_alone[d.name] if variant == "alone" else self.tf.T_die[d.name]
            die_metrics[d.name] = DieMetrics(d.name, B, int(immortal.sum()), hz, float(tn.min()),
                                             f"{rails[k_first].kind}{rails[k_first].index}", float(Tg.min()), float(Tg.max()),
                                             float(Tg.mean()), self.pg_stats[d.name]["ir_drop_max_mV"], self.pg_stats[d.name]["j_max_A_per_cm2"],
                                             float(np.nanmin(marg)))
            sig_all[d.name] = sig; tn_all[d.name] = tn; imm_all[d.name] = immortal
        all_rails = [r for d in self.spec.dies for r in (rails_by_die[d.name][:n_max_per_die] if n_max_per_die else rails_by_die[d.name])]
        return VariantResult(variant, all_rails, metrics, die_metrics, sig_all, tn_all, imm_all, time.time() - t0)

    # ---- stack-level statistics ------------------------------------------------------------
    def stack_summary(self, res: VariantResult) -> Dict:
        tn_all = np.concatenate([res.t_nuc[d.name] for d in self.spec.dies])
        die_of = np.concatenate([[d.name] * len(res.t_nuc[d.name]) for d in self.spec.dies])
        k = int(np.argmin(tn_all))
        return dict(variant=res.variant, earliest_t_nuc_years=float(tn_all[k] / SEC_PER_YEAR), critical_die=str(die_of[k]),
                    n_rails=int(len(tn_all)), n_mortal_10yr=int(np.sum(tn_all <= 10 * SEC_PER_YEAR)),
                    n_mortal_100yr=int(np.sum(tn_all <= 100 * SEC_PER_YEAR)),
                    dies={n: asdict(m) for n, m in res.die_metrics.items()}, seconds=res.seconds)

    def vertical_screen_table(self) -> List[Dict]:
        die_T = {d.name: self.tf.T_uniform(d.name) for d in self.spec.dies}
        return screen_table(vertical_screen(self.spec, self.pg, self.em, die_T))

    def cross_die_alignment(self, res: VariantResult, bottom: str, top: str, horizon_years: float | None = None) -> Dict:
        """Which rails of the bottom die fail, relative to the top die's power map?

        Nucleation itself happens at rail ends (parabolic maximum principle, blocked ends), so the
        alignment shows up in *which rails* fail, not where along a rail.  A bottom-die rail "crosses"
        the top hot spot when one of its segment mid-points lies under a top-die block with power
        > 1.5 x the block mean.
        * crossing_enrichment_signoff / _100yr : share of the mortal rails (within 10 yr / 100 yr) that cross
                                                 the hot footprint, divided by the share of all rails that cross
                                                 (1 = no alignment, > 1 = the hot spot selects the failing rails);
        * tnuc_ratio_cross_over_noncross       : median t_nuc of crossing / non-crossing mortal rails;
        * spearman_rho                         : Spearman correlation between a rail's mean temperature and its
                                                 log t_nuc over the mortal rails (negative = hotter rails fail earlier)."""
        from scipy.stats import spearmanr
        dt = next(d for d in self.spec.dies if d.name == top)
        nblk = dt.nblk; bw = dt.size / nblk
        hot = dt.power > 1.5 * dt.power.mean()
        rails = [r for r in res.rails if r.die == bottom]; tn = np.asarray(res.t_nuc[bottom], float)
        cross = np.zeros(len(rails), bool); Tmean = np.zeros(len(rails))
        for k, r in enumerate(rails):
            xs, ys = r.node_xy(); xm = 0.5 * (xs[:-1] + xs[1:]); ym = 0.5 * (ys[:-1] + ys[1:])
            bi = np.minimum((xm / bw).astype(int), nblk - 1); bj = np.minimum((ym / bw).astype(int), nblk - 1)
            cross[k] = bool(hot[bi, bj].any()) if hot.any() else False
            Tmean[k] = float(0.5 * (r.T_L + r.T_R).mean())
        share_all = float(cross.mean()) if len(rails) else float("nan")
        out = dict(top=top, bottom=bottom, n_rails=len(rails), share_of_rails_crossing_hot_block=share_all,
                   hot_block_area_share=float(hot.sum()) / hot.size if hot.any() else float("nan"))
        for label, h in (("signoff", (horizon_years or self.horizon_years)), ("100yr", 100.0)):
            m = np.isfinite(tn) & (tn <= h * SEC_PER_YEAR)
            n_m = int(m.sum()); n_c = int((m & cross).sum())
            out[f"n_mortal_{label}"] = n_m; out[f"n_mortal_crossing_{label}"] = n_c
            out[f"crossing_enrichment_{label}"] = float((n_c / n_m) / share_all) if (n_m and share_all > 0) else float("nan")
        m = np.isfinite(tn)
        if (m & cross).any() and (m & ~cross).any():
            out["tnuc_ratio_cross_over_noncross"] = float(np.median(tn[m & cross]) / np.median(tn[m & ~cross]))
        else:
            out["tnuc_ratio_cross_over_noncross"] = float("nan")
        if m.sum() > 2 and Tmean[m].std() > 0:
            rho, pval = spearmanr(Tmean[m], np.log(tn[m]))
        else:
            rho, pval = float("nan"), float("nan")
        out.update(spearman_rho=float(rho), p_value=float(pval), earliest_rail_crosses_hot_block=bool(cross[int(np.argmin(tn))]) if len(rails) else None)
        return out


# ----------------------------------------------------------------------------
# ranking utilities (E7)
# ----------------------------------------------------------------------------
def topk_hit_rate(t_ref: np.ndarray, t_test: np.ndarray, k: int) -> float:
    a = set(np.argsort(t_ref)[:k].tolist()); b = set(np.argsort(t_test)[:k].tolist())
    return len(a & b) / k


def kendall_tau(t_ref: np.ndarray, t_test: np.ndarray, top_n: int | None = None) -> float:
    from scipy.stats import kendalltau
    if top_n:
        idx = np.argsort(t_ref)[:top_n]; t_ref, t_test = t_ref[idx], t_test[idx]
    return float(kendalltau(np.log(np.clip(t_ref, 1, 1e13)), np.log(np.clip(t_test, 1, 1e13)))[0])


def save_variant(path: str, res: VariantResult, summary: Dict, times: np.ndarray | None = None):
    """Raw data of one variant: node-stress histories, t_nuc, immortality per die (npz), the output time grid,
    the rail metrics (json) and the rails themselves (geometry, j, T_L/T_R, T_m, sigma_T; <stem>_rails.json)."""
    from .power_grid import save_rails
    np.savez_compressed(path, times=np.asarray(times if times is not None else [0.0], float),
                        **{f"sigma_{d}": v for d, v in res.sigma_nodes.items()},
                        **{f"tnuc_{d}": v for d, v in res.t_nuc.items()},
                        **{f"immortal_{d}": v for d, v in res.immortal.items()})
    json.dump(dict(summary=summary, rails=[asdict(m) for m in res.metrics]), open(path.replace(".npz", ".json"), "w"), indent=1)
    save_rails(path.replace(".npz", "_rails.json"), res.rails, dict(variant=res.variant))
