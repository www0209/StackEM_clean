"""
stackem.blech_screen
======================

Blech Immortality Screen (BIS): the vertical-element and steady-state screen
of the stack.

1.  Vertical elements (TSVs, hybrid-bond pad pairs) are Cu bodies surrounded
    by diffusion barriers; each is an independent, very short Korhonen line.
    They are immortal when their Blech product j*L is below the critical
    product  (jL)_c = 2 Omega (sigma_crit - sigma_T) / (e Z* rho).  We report
    the ratio jL / (jL)_c per element type, which is 1-2 orders of magnitude
    below 1 for realistic TSV / pad currents (this is why the transient
    solver only needs the BEOL rails).

2.  Rails: the exact steady state of a blocked multi-segment rail is
        sigma_ss(x) = sigma_T + Phi(x) - <Phi>,   Phi(x) = int_0^x (G + M) dx'
    (mass conservation fixes the mean).  A rail whose max sigma_ss stays below
    sigma_crit can never nucleate and is skipped by the transient solver
    (the same idea as EMSpice-3's steady-state screen, here in closed form).
    For rails with different sigma_T per segment the volume-weighted mean is
    used, which is exact for equal cross-sections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np

from .constants import EMParams, E_CHARGE
from .hotspot_stack import DieSpec, StackSpec
from .power_grid import Rail, PGParams
from .thermal_profile import SegmentProfile

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


@dataclass
class VerticalElement:
    kind: str            # "TSV" | "pad"
    die: str
    diameter: float      # m
    height: float        # m
    current: float       # A per element
    T: float             # K
    j: float             # A/m^2
    jL: float            # A/m
    jL_crit: float       # A/m
    ratio: float
    immortal: bool


def vertical_screen(spec: StackSpec, pg: PGParams, em: EMParams, die_T: Dict[str, float],
                    pad_diameter: float = 3e-6, pad_height: float = 3e-6, pad_max_current: float = 5e-3,
                    current_fraction_vertical: float = 1.0) -> List[VerticalElement]:
    """Screen the TSVs and bond pads of every die.

    The die current I = P/Vdd is shared by the feed sites; a TSV cluster at a feed
    site holds (feed_pitch/tsv_pitch)^2 TSVs, so I_TSV = I / (n_feeds * n_tsv_per_site).
    Bond pads: bounded by pad_max_current per pad pair (AMD/TSMC hybrid bonding scale)."""
    out = []
    for k, d in enumerate(spec.dies):
        I_die = d.total_power() / pg.Vdd * current_fraction_vertical
        n_feeds = max(1, int(round(d.size / d.feed_pitch)) ** 2)
        T = die_T[d.name]
        jl_c = em.blech_critical_product(T)
        if d.feed_kind == "tsv":
            n_per_site = max(1, int(round(d.feed_pitch / d.tsv_pitch)) ** 2)
            I_tsv = I_die / (n_feeds * n_per_site)
            h = d.tsv_height or d.thickness
            j = I_tsv / (np.pi * (d.tsv_diameter / 2) ** 2)
            out.append(VerticalElement("TSV", d.name, d.tsv_diameter, h, I_tsv, T, j, j * h, jl_c, j * h / jl_c, j * h < jl_c))
        # pad pairs exist between this die and the one above (or below for pad-fed dies)
        I_pad = min(pad_max_current, I_die / max(1, n_feeds * 4))
        j = I_pad / (np.pi * (pad_diameter / 2) ** 2)
        out.append(VerticalElement("pad", d.name, pad_diameter, pad_height, I_pad, T, j, j * pad_height, jl_c,
                                   j * pad_height / jl_c, j * pad_height < jl_c))
    return out


def rail_steady_state(rail: Rail, em: EMParams, with_tm: bool = True):
    """Exact steady-state stress at the nodes of a blocked rail and its maximum."""
    S = rail.n_seg
    G = em.G_of_j(rail.j)
    dPhi = G * rail.L
    if with_tm:
        for k in range(S):
            p = SegmentProfile(rail.L[k], rail.T_L[k], rail.T_R[k], rail.T_m[k], rail.Gamma)
            xi = np.linspace(0, 1, 33)
            M = em.M_of(p.T_of_xi(xi), p.dTdx_of_xi(xi))
            dPhi[k] += _trapz(M, xi) * rail.L[k]
    Phi = np.concatenate([[0.0], np.cumsum(dPhi)])
    # mean of Phi over the rail (piecewise linear -> trapezoid with segment weights)
    seg_mean = 0.5 * (Phi[:-1] + Phi[1:])
    mean_Phi = float(np.sum(seg_mean * rail.L) / np.sum(rail.L))
    sT_mean = float(np.sum(rail.sigma_T * rail.L) / np.sum(rail.L))
    sigma_ss = sT_mean + Phi - mean_Phi
    return sigma_ss, float(sigma_ss.max())


def screen_rails(rails: Sequence[Rail], em: EMParams, with_tm: bool = True, margin: float = 1.0):
    """Returns (immortal mask, steady max stress array). margin < 1 keeps a safety band."""
    smax = np.array([rail_steady_state(r, em, with_tm)[1] for r in rails])
    return smax < margin * em.sigma_crit, smax


def screen_table(elements: Sequence[VerticalElement]) -> List[Dict]:
    return [dict(kind=e.kind, die=e.die, diameter_um=e.diameter * 1e6, height_um=e.height * 1e6, current_mA=e.current * 1e3,
                 T_K=e.T, j_A_per_cm2=e.j / 1e4, jL_A_per_cm=e.jL / 100, jL_crit_A_per_cm=e.jL_crit / 100,
                 ratio=e.ratio, immortal=bool(e.immortal)) for e in elements]
