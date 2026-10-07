"""
stackem.signoff
=================

Sign-off normalisation of a stack case (the "rule-sized" benchmark definition).

Every die of a benchmark stack is sized the way a die-level EM sign-off would size
it: the strap width W of its analysed power-rail pair is the smallest width for
which the die, simulated *standalone* (its own package, its own HotSpot field,
single-sided heat loss), meets the lifetime target with the prescribed margin,

    earliest t_nuc(die alone; W)  =  horizon * margin        (default 10 yr x 2).

The rule is monotone in W (j = I/(W H), so wider straps -> smaller j -> later
nucleation), which makes the sizing a bracketed root-find in log W with the
reference solver on every rail of the die.  Nothing else of the case changes
(powers, geometry, feed pitch, materials), so the stack-level results of the
paper are the answer to the question "what happens to three dies that each pass
their own sign-off when they are bonded together?".

The same routine sizes the dies of HotSpot's ev6 example (E8) from their
"other dies off" temperature field.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Callable, Dict, List, Sequence
import numpy as np

from .constants import EMParams, ThermalParams, SEC_PER_YEAR
from .hotspot_stack import DieSpec, StackSpec, ThermalField
from .power_grid import PGParams, DieGrid, Rail, embed_thermal
from .blech_screen import screen_rails
from .assembler import solve_rails_truth


def earliest_t_nuc(rails: Sequence[Rail], em: EMParams, times: np.ndarray, n_cells: int = 60, with_tm: bool = True) -> float:
    """Earliest nucleation time (s) over a set of rails with the reference solver (immortal rails screened out)."""
    imm, _ = screen_rails(rails, em, with_tm)
    todo = np.where(~imm)[0]
    if len(todo) == 0:
        return float("inf")
    _, tn, _ = solve_rails_truth([rails[i] for i in todo], em, times, n_cells, with_tm)
    return float(tn.min())


def size_strap_width(rails_of_W: Callable[[float], List[Rail]], em: EMParams, times: np.ndarray, target_s: float,
                     W_min: float = 0.5e-6, W_max: float = 40e-6, n_cells: int = 60, rel_tol: float = 0.02,
                     verbose: bool = False) -> Dict:
    """Smallest W in [W_min, W_max] with earliest_t_nuc(rails_of_W(W)) >= target_s (bisection in log W).

    Returns dict(W, t_earliest_s, iterations, saturated) - saturated=True when even W_max fails
    or W_min already passes (then W is that bound)."""
    lo, hi = np.log(W_min), np.log(W_max)
    hist = []                                                   # every (W, earliest t_nuc) evaluated: the sizing curve
    t_lo = earliest_t_nuc(rails_of_W(W_min), em, times, n_cells); hist.append((float(W_min), float(t_lo)))
    if t_lo >= target_s:
        return dict(W=W_min, t_earliest_s=t_lo, iterations=1, saturated=True, history=hist)
    t_hi = earliest_t_nuc(rails_of_W(W_max), em, times, n_cells); hist.append((float(W_max), float(t_hi)))
    if t_hi < target_s:
        return dict(W=W_max, t_earliest_s=t_hi, iterations=2, saturated=True, history=hist)
    it = 2
    while hi - lo > np.log(1.0 + rel_tol):
        mid = 0.5 * (lo + hi); t_mid = earliest_t_nuc(rails_of_W(np.exp(mid)), em, times, n_cells); it += 1
        hist.append((float(np.exp(mid)), float(t_mid)))
        if verbose:
            print(f"    W = {np.exp(mid)*1e6:6.2f} um -> earliest {t_mid/SEC_PER_YEAR:8.2f} yr", flush=True)
        if t_mid >= target_s:
            hi, t_hi = mid, t_mid
        else:
            lo = mid
    return dict(W=float(np.exp(hi)), t_earliest_s=float(t_hi), iterations=it, saturated=False, history=sorted(hist))


def size_stack(spec: StackSpec, pg: PGParams, em: EMParams, th: ThermalParams, tf: ThermalField, times: np.ndarray,
               horizon_years: float, margin: float, W_min: float, W_max: float, n_cells: int = 60, verbose: bool = True) -> Dict[str, Dict]:
    """Sign-off sizing of every die of a stack from its standalone temperature field tf.T_alone[die]."""
    target = horizon_years * margin * SEC_PER_YEAR
    out = {}
    for d in spec.dies:
        T_alone = tf.T_alone[d.name]

        def rails_of_W(W, d=d, T_alone=T_alone):
            g = DieGrid(replace(d, rail_W=float(W)), pg)
            return embed_thermal(g, g.solve_from_blocks(), T_alone, em, th, two_sided=False)

        if verbose:
            print(f"  sizing {d.name} (standalone T {T_alone.min():.1f}-{T_alone.max():.1f} K, target {horizon_years*margin:.0f} yr)", flush=True)
        r = size_strap_width(rails_of_W, em, times, target, W_min, W_max, n_cells, verbose=verbose)
        g = DieGrid(replace(d, rail_W=r["W"]), pg); sol = g.solve_from_blocks()
        jmax = float(max(np.abs(sol.I_h).max(), np.abs(sol.I_v).max()) / g.pg.A)
        r.update(die=d.name, W_um=r["W"] * 1e6, t_earliest_years=r["t_earliest_s"] / SEC_PER_YEAR, j_max_A_per_cm2=jmax * 1e-4,
                 T_alone_mean=float(T_alone.mean()), T_alone_max=float(T_alone.max()))
        out[d.name] = r
        if verbose:
            print(f"  -> {d.name}: W = {r['W_um']:.2f} um, earliest {r['t_earliest_years']:.1f} yr, j_max {jmax*1e-4:.3g} A/cm^2{' (saturated)' if r['saturated'] else ''}", flush=True)
    return out


def sizing_path(root_dir: str) -> str:
    return os.path.join(root_dir, "signoff", "signoff_sizing.json")


def apply_sizing(spec: StackSpec, sizing: Dict[str, Dict]) -> StackSpec:
    return replace(spec, dies=[replace(d, rail_W=float(sizing[d.name]["W"])) if d.name in sizing else d for d in spec.dies])


def load_sizing(path: str) -> Dict[str, Dict] | None:
    if not os.path.exists(path):
        return None
    return json.load(open(path))["dies"]
