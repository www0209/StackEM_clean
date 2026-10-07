"""
stackem.sensitivity
=====================

Differentiable stack-level EM (experiment E5): exact gradients of every
rail's nucleation time with respect to the block powers of every die,

    d t_nuc / d P_b  =  sum_nodes dt/dT_node * dT_node/dP_b  +  sum_segs dt/dj_seg * dj_seg/dP_b,

obtained by automatic differentiation through the SKN + junction closure
(torch), with the two linear maps  T_node(P)  (HotSpot grid model, exact by
unit-power runs) and  j_seg(P)  (DC power grid, exact by unit-power solves).

Provided
--------
* ``LinearMaps``       : the two response matrices per die.
* ``StackSensitivity`` : forward pass P -> t_nuc (torch), gradients for
                         selected rails, cross-die sensitivity maps.
* ``finite_difference_check`` : numpy re-evaluation of the whole flow with a
                         perturbed block power, to validate the autograd values.
* ``optimize_power_budget``   : projected-gradient reallocation of block powers
                         (per-die totals fixed) that maximises the earliest
                         nucleation time of the stack - the reliability-aware
                         power-budgeting case study.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .constants import EMParams, ThermalParams, SEC_PER_YEAR
from .hotspot_stack import StackSpec, HotSpotRunner, unit_power_responses, grid_T_at, block_names
from .power_grid import PGParams, DieGrid, Rail
from .assembler import StackAssembler, solve_rails_truth


@dataclass
class LinearMaps:
    block_labels: List[str]                 # global block order (die_name/block)
    die_block_slices: Dict[str, slice]      # global indices of each die's blocks
    RT: Dict[str, np.ndarray]               # die -> (n_rails, S+1, nb_total)  node temperature response (K/W)
    SJ: Dict[str, np.ndarray]               # die -> (n_rails, S, nb_die)      current-density response (A/m^2/W)
    rails: Dict[str, List[Rail]]            # die -> rails in the same order (full variant)
    T_amb: float


def build_linear_maps(asm: StackAssembler, workdir: str) -> LinearMaps:
    spec, pg = asm.spec, asm.pg
    R = unit_power_responses(spec, os.path.join(workdir, "unit_hotspot"), asm.runner)
    labels = []; slices = {}; off = 0
    for d in spec.dies:
        nb = d.nblk ** 2
        labels += [f"{d.name}/{n}" for n in block_names(d)]
        slices[d.name] = slice(off, off + nb); off += nb
    nb_total = off
    rails_by_die, _ = asm.rails("full")
    RT = {}; SJ = {}
    for d in spec.dies:
        g = asm.grids[d.name]; rails = rails_by_die[d.name]; pg = g.pg
        Sh, Sv = g.unit_power_current_responses()
        n_r = len(rails); S = rails[0].n_seg
        rt = np.zeros((n_r, S + 1, nb_total)); sj = np.zeros((n_r, S, d.nblk ** 2))
        for i, r in enumerate(rails):
            xs, ys = r.node_xy()
            for b in range(nb_total):
                rt[i, :, b] = grid_T_at(R[d.name][b], d.size, xs, ys)
            if r.kind == "H":
                sj[i] = (Sh[:, r.index, :] / pg.A).T
            else:
                sj[i] = (Sv[:, :, r.index] / pg.A).T
        RT[d.name] = rt; SJ[d.name] = sj
    return LinearMaps(labels, slices, RT, SJ, rails_by_die, spec.ambient)


def block_power_vector(spec: StackSpec) -> np.ndarray:
    return np.concatenate([[d.power[i, j] for i in range(d.nblk) for j in range(d.nblk)] for d in spec.dies])


class StackSensitivity:
    """Torch forward pass P -> t_nuc for every rail of every die, and its gradients."""

    def __init__(self, maps: LinearMaps, em: EMParams, th: ThermalParams, weights_path: str, times: np.ndarray,
                 device: str = "cpu", with_tm: bool = True, jump_at: str = "mid", two_sided: bool = True,
                 closure_mode: str = "direct", n_tab: int = 192):
        import torch
        from .closure_torch import SKNTorchProvider
        from .closure_tab import wrap_provider
        from .skn_model import SKNWeights
        self.torch = torch
        self.maps, self.em, self.th = maps, em, th
        self.provider = wrap_provider(SKNTorchProvider(SKNWeights.load(weights_path), device), closure_mode, n_tab)
        self.dev = torch.device(device)
        self.times = torch.as_tensor(np.asarray(times, float), dtype=torch.float64, device=self.dev)
        self.with_tm, self.jump_at, self.two_sided = with_tm, jump_at, two_sided
        self.Gamma = th.Gamma(two_sided)
        self.RT = {d: torch.as_tensor(v, dtype=torch.float64, device=self.dev) for d, v in maps.RT.items()}
        self.SJ = {d: torch.as_tensor(v, dtype=torch.float64, device=self.dev) for d, v in maps.SJ.items()}
        self.L = {d: torch.as_tensor(np.stack([r.L for r in rs]), dtype=torch.float64, device=self.dev) for d, rs in maps.rails.items()}

    def forward(self, P, dies: Sequence[str] | None = None, rail_idx: Dict[str, Sequence[int]] | None = None):
        """P: torch tensor (nb_total,) of block powers (W). Returns {die: t_nuc tensor (n_rails_sel,)}."""
        torch = self.torch
        from .closure_torch import close_rails_torch, t_sigma_T
        out = {}
        for d, sl in self.maps.die_block_slices.items():
            if dies and d not in dies:
                continue
            RT, SJ, L = self.RT[d], self.SJ[d], self.L[d]
            if rail_idx and d in rail_idx:
                idx = torch.as_tensor(np.asarray(rail_idx[d]), device=self.dev)
                RT, SJ, L = RT[idx], SJ[idx], L[idx]
            T_nodes = self.maps.T_amb + RT @ P                          # (B,S+1)
            j = SJ @ P[sl]                                              # (B,S)
            T_L, T_R = T_nodes[:, :-1], T_nodes[:, 1:]
            T_m = (j ** 2) * self.th.rho_ref * self.Gamma ** 2 / self.th.k_cu
            lam = L / self.Gamma
            Tbar = 0.5 * (T_L + T_R) + T_m * (1.0 - torch.tanh(0.5 * lam) / (0.5 * lam))
            sigma_T = t_sigma_T(Tbar, self.em)
            Gam = torch.full((L.shape[0], 1), self.Gamma, dtype=torch.float64, device=self.dev)
            res = close_rails_torch(L, j, T_L, T_R, T_m, Gam, sigma_T, self.em, self.provider, self.times,
                                    self.with_tm, self.jump_at)
            out[d] = res["t_nuc"]
        return out

    def gradients(self, P0: np.ndarray, targets: Dict[str, Sequence[int]]) -> Dict[str, np.ndarray]:
        """d log t_nuc / d P_b for the selected rails: returns {die: (n_sel, nb_total)} in 1/W."""
        torch = self.torch
        grads = {}
        for d, idx in targets.items():
            g = np.zeros((len(idx), P0.size))
            for i, ri in enumerate(idx):
                P = torch.as_tensor(P0, dtype=torch.float64, device=self.dev).requires_grad_(True)
                t = self.forward(P, dies=[d], rail_idx={d: [ri]})[d][0]
                torch.log(t).backward()
                g[i] = P.grad.detach().cpu().numpy()
            grads[d] = g
        return grads

    def earliest(self, P: np.ndarray, dies: Sequence[str] | None = None) -> Dict[str, float]:
        torch = self.torch
        with torch.no_grad():
            r = self.forward(torch.as_tensor(P, dtype=torch.float64, device=self.dev), dies)
        return {d: float(v.min()) for d, v in r.items()}


def finite_difference_check(asm: StackAssembler, P0: np.ndarray, block: int, delta_W: float, die: str, rail_index: int,
                            maps: LinearMaps, provider) -> Tuple[float, float]:
    """Numpy re-evaluation of one rail's t_nuc at P0 and P0 + delta e_block using the linear
    maps (no HotSpot re-run needed: the response is exactly linear). Returns (t0, t1) in s."""
    from .closure import close_rails
    from .power_grid import Rail
    from dataclasses import replace as rep
    def rail_at(P):
        r = maps.rails[die][rail_index]
        RT = maps.RT[die][rail_index]; SJ = maps.SJ[die][rail_index]
        Tn = maps.T_amb + RT @ P
        j = SJ @ P[maps.die_block_slices[die]]
        Tm = asm.th.T_m(np.abs(j)) if r.T_m.max() > 0 or True else np.zeros_like(j)
        lam = r.L / r.Gamma
        Tbar = 0.5 * (Tn[:-1] + Tn[1:]) + Tm * (1 - np.tanh(0.5 * lam) / (0.5 * lam))
        return rep(r, j=j, T_L=Tn[:-1], T_R=Tn[1:], T_m=Tm, sigma_T=np.asarray(asm.em.sigma_T(Tbar)))
    P1 = P0.copy(); P1[block] += delta_W
    t0 = close_rails([rail_at(P0)], asm.em, provider, asm.times).t_nuc[0]
    t1 = close_rails([rail_at(P1)], asm.em, provider, asm.times).t_nuc[0]
    return float(t0), float(t1)


def optimize_power_budget(sens: StackSensitivity, P0: np.ndarray, die_slices: Dict[str, slice], n_iter: int = 30,
                          step: float = 0.05, p_min_frac: float = 0.3, p_max_frac: float = 3.0, softmin_T: float = 0.15,
                          chunk: int = 4, verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """Projected gradient ascent on the soft-minimum of log t_nuc over all rails of all dies,
    keeping each die's total power fixed and each block within [p_min_frac, p_max_frac] x mean.

    The objective is  J = -T logsumexp(-lt_i / T)  with lt_i = log t_nuc,i (years).  Its gradient is
    sum_i w_i d lt_i / dP  with softmax weights w_i = exp(-lt_i/T) / sum exp(-lt_j/T), so it is
    accumulated rail-chunk by rail-chunk (autograd through 4 rails at a time keeps the retained
    activations at a few GB; a whole stack at once would need >100 GB)."""
    torch = sens.torch
    P = P0.copy(); hist = []
    for it in range(n_iter):
        # 1. forward without grad: all rails, soft-min weights
        with torch.no_grad():
            r = sens.forward(torch.as_tensor(P, dtype=torch.float64, device=sens.dev))
        lt_all = {d: torch.log(v / SEC_PER_YEAR) for d, v in r.items()}
        lt_cat = torch.cat(list(lt_all.values()))
        w_cat = torch.softmax(-lt_cat / softmin_T, 0)
        obj = float(-softmin_T * torch.logsumexp(-lt_cat / softmin_T, 0)); hard_min = float(lt_cat.min().exp())
        # 2. gradient accumulated over rail chunks (weights treated as constants: exact gradient of J)
        g = np.zeros_like(P); off = 0
        for d, v in lt_all.items():
            n_r = v.shape[0]; w_d = w_cat[off:off + n_r]; off += n_r
            for i0 in range(0, n_r, chunk):
                idx = list(range(i0, min(n_r, i0 + chunk)))
                if float(w_d[idx].sum()) < 1e-9:          # rails with negligible weight do not move the soft-min
                    continue
                Pt = torch.as_tensor(P, dtype=torch.float64, device=sens.dev).requires_grad_(True)
                lt = torch.log(sens.forward(Pt, dies=[d], rail_idx={d: idx})[d] / SEC_PER_YEAR)
                (w_d[idx] * lt).sum().backward()
                g += Pt.grad.detach().cpu().numpy()
        hist.append(dict(iter=it, softmin_log10_tnuc_yr=obj / np.log(10), min_tnuc_yr=hard_min, P=P.copy()))
        if verbose:
            print(f"  iter {it:3d}  earliest t_nuc {hard_min:.3f} yr  soft {obj/np.log(10):.3f}", flush=True)
        # 3. gradient step, then project onto per-die sum constraint and box
        for d, sl in die_slices.items():
            gd = g[sl]; Pd = P[sl]; mean = Pd.mean()
            gd = gd - gd.mean()                                       # keep the die total fixed
            Pd = Pd + step * mean * gd / (np.abs(gd).max() + 1e-30)
            Pd = np.clip(Pd, p_min_frac * mean, p_max_frac * mean)
            Pd *= Pd.size * mean / Pd.sum()
            P[sl] = Pd
    return P, hist


# ----------------------------------------------------------------------------
# stack-aware repair: targeted vs blanket strap widening (E5, numpy closure only)
# ----------------------------------------------------------------------------
def _widen(rail: Rail, f: float, th: ThermalParams, em: EMParams, two_sided: bool = True) -> Rail:
    """The same rail with straps f times wider: j/f, Joule bump recomputed, sigma_T re-evaluated."""
    from dataclasses import replace as rep
    j = rail.j / f
    Tm = th.T_m(np.abs(j), two_sided)
    lam = rail.L / rail.Gamma
    Tbar = 0.5 * (rail.T_L + rail.T_R) + Tm * (1 - np.tanh(0.5 * lam) / (0.5 * lam))
    return rep(rail, W=rail.W * f, j=j, T_m=Tm, sigma_T=np.asarray(em.sigma_T(Tbar)))


def widening_factor(rail: Rail, em: EMParams, th: ThermalParams, provider, times: np.ndarray, target_s: float,
                    f_max: float = 20.0, rel_tol: float = 0.02) -> float:
    """Smallest widening factor f >= 1 such that t_nuc(rail widened by f) >= target_s (closure; bisection in log f)."""
    from .closure import close_rails
    from .blech_screen import screen_rails

    def t_of(f):
        r = _widen(rail, f, th, em)
        if screen_rails([r], em, True)[0][0]:
            return float("inf")
        return float(close_rails([r], em, provider, times, em.sigma_crit, True, "mid").t_nuc[0])

    if t_of(1.0) >= target_s:
        return 1.0
    lo, hi = 0.0, np.log(f_max)
    if t_of(f_max) < target_s:
        return f_max
    while hi - lo > np.log(1 + rel_tol):
        mid = 0.5 * (lo + hi)
        if t_of(np.exp(mid)) >= target_s:
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def targeted_vs_blanket_widening(asm: StackAssembler, provider, horizon_years: float, verbose: bool = True) -> Dict:
    """For every die: the rails that violate the horizon in the stack (full variant, closure), the per-rail widening
    factors that restore it, and the metal area of (a) widening only those rails by their own factors and
    (b) widening every rail of the die by the largest factor (what a die-level rule without rail resolution
    would do).  Returns per-die dicts and the area ratio; both repairs are verified with the reference solver."""
    from .closure import close_rails
    from .blech_screen import screen_rails
    target = horizon_years * SEC_PER_YEAR
    rails_by_die, em = asm.rails("full")
    out = {}
    for d in asm.spec.dies:
        rails = rails_by_die[d.name]
        imm, _ = screen_rails(rails, em, True)
        tn = np.full(len(rails), np.inf); todo = np.where(~imm)[0]
        if len(todo):
            tn[todo] = close_rails([rails[i] for i in todo], em, provider, asm.times, em.sigma_crit, True, asm.jump_at).t_nuc
        viol = np.where(tn < target)[0]
        f = np.ones(len(rails))
        for i in viol:
            f[i] = widening_factor(rails[i], em, asm.th, provider, asm.times, target)
        W = rails[0].W; Lsum = float(rails[0].L.sum())
        area_targeted = float(np.sum((f - 1.0) * W * Lsum))
        f_blanket = float(f.max())
        area_blanket = float(len(rails) * (f_blanket - 1.0) * W * Lsum)
        area_grid = float(len(rails) * W * Lsum)
        # verification with the reference solver
        t_targeted = float("inf"); t_blanket = float("inf")
        if len(viol):
            rt = [_widen(rails[i], f[i], asm.th, em) for i in range(len(rails))]
            rb = [_widen(rails[i], f_blanket, asm.th, em) for i in range(len(rails))]
            for label, rs in (("targeted", rt), ("blanket", rb)):
                im, _ = screen_rails(rs, em, True); td = np.where(~im)[0]
                t = float("inf")
                if len(td):
                    _, tt, _ = solve_rails_truth([rs[i] for i in td], em, asm.times, asm.truth_cells, True); t = float(tt.min())
                if label == "targeted": t_targeted = t
                else: t_blanket = t
        out[d.name] = dict(n_rails=len(rails), n_violating=int(len(viol)), violating=[dict(rail=f"{rails[i].kind}{rails[i].index}", t_nuc_yr=float(tn[i] / SEC_PER_YEAR), factor=float(f[i])) for i in viol],
                           W_um=W * 1e6, f_blanket=f_blanket, area_targeted_m2=area_targeted, area_blanket_m2=area_blanket, area_grid_m2=area_grid,
                           area_ratio_blanket_over_targeted=(area_blanket / area_targeted if area_targeted > 0 else None),
                           verify_truth_earliest_targeted_yr=t_targeted / SEC_PER_YEAR, verify_truth_earliest_blanket_yr=t_blanket / SEC_PER_YEAR)
        if verbose:
            print(f"  {d.name}: {len(viol)} / {len(rails)} rails violate {horizon_years:g} yr; targeted +{100*area_targeted/area_grid:.1f}% metal, "
                  f"blanket +{100*area_blanket/area_grid:.1f}% (x{f_blanket:.2f}); truth after repair: targeted {t_targeted/SEC_PER_YEAR:.1f} yr, blanket {t_blanket/SEC_PER_YEAR:.1f} yr", flush=True)
    return out
