"""
Fig. 2 of the paper: the framework as a block diagram, drawn programmatically in the paper style
(no hand-made drawing to keep in sync with the code).

    python tools/framework_figure.py --out ~/stackem_work/outputs/base3/figs/framework

Row 1 (offline, once):   FDM kernel data  ->  SKN training  ->  frozen kernels s_G, s_M, a(theta, r, rho_J, lambda; tau)
Row 2 (per stack case):  stack + floorplan -> TFE (HotSpot 3-D + Joule fin) -> PGS (DC grid, rails) -> BIS (Blech screen)
                         -> segment descriptors -> JCX (exact junction closure, batched, differentiable) -> t_nuc per rail
Row 3 (uses):            sign-off (margins, mortal set, ranking) | targeted repair | sensitivities & power budgeting
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from stackem.viz import style as st                   # noqa: E402


def box(ax, x, y, w, h, title, body="", fc="#FFFFFF", ec="#333333", lw=0.8, tfs=6.8, bfs=5.4, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.10, title, ha="center", va="top", fontsize=tfs, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + 0.09, body, ha="center", va="bottom", fontsize=bfs, color="#333333", linespacing=1.3)


def arrow(ax, p, q, color="#333333", lw=0.9, style="-|>", ls="-", text=None, tfs=5.4, toff=(0, 0.08), ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha=ha, va="center", fontsize=tfs, color=color)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_2COL, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0.55, 5.0); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    C_OFF, C_ON, C_USE = "#EEF3F8", "#FFF4E6", "#EDF7EF"
    # ---- offline row -------------------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.15, 3.75), 9.7, 1.2, boxstyle="round,pad=0.02", fc=C_OFF, ec="none"))
    ax.text(0.3, 4.88, "offline, once per technology (no stack, no netlist involved)", fontsize=6.0, color="#456", va="top", style="italic")
    box(ax, 0.4, 3.85, 2.3, 0.8, "kernel data (reference FDM)", "random segment profiles $(\\theta, r, \\rho_J, \\lambda)$\n96 $\\tau$ points per profile")
    box(ax, 3.4, 3.85, 2.3, 0.8, "SKN training", "4$\\times$128 SiLU + Fourier features\n$\\sqrt{\\tau}$-scaled targets, teacher-densified labels")
    box(ax, 6.4, 3.85, 3.1, 0.8, "frozen segment kernels", "$s_G,\\ s_M,\\ a\\,(\\xi;\\ \\theta, r, \\rho_J, \\lambda;\\ \\tau)$\nreused by every stack case")
    arrow(ax, (2.7, 4.25), (3.4, 4.25)); arrow(ax, (5.7, 4.25), (6.4, 4.25))
    # ---- per-case row --------------------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.15, 2.15), 9.7, 1.45, boxstyle="round,pad=0.02", fc=C_ON, ec="none"))
    ax.text(0.3, 3.53, "per stack case (seconds): stack-level EM sign-off", fontsize=6.0, color="#764", va="top", style="italic")
    y = 2.26; h = 0.98
    box(ax, 0.4, y, 1.45, h, "stack case", "dies, block powers,\nfeeds, package,\nrule-sized straps")
    box(ax, 2.05, y, 1.75, h, "3-D thermal field", "HotSpot 7 3-D grid\n+ Joule fin ($\\Gamma$, $T_m$)\nstack and standalone")
    box(ax, 4.0, y, 1.55, h, "grid currents", "DC power-grid solve\n$j$ on every rail\nsegments between feeds")
    box(ax, 5.75, y, 1.35, h, "Blech screen", "Blech immortality\n$jL < (jL)_c$\nfeeds and rails")
    box(ax, 7.3, y, 2.2, h, "JCX  exact closure", "Duhamel superposition of junction\nfluxes; tridiagonal per time step;\ntabulated kernels; batched; autograd")
    for x0, x1 in [(1.85, 2.05), (3.8, 4.0), (5.55, 5.75), (7.1, 7.3)]:
        arrow(ax, (x0, y + h / 2), (x1, y + h / 2))
    arrow(ax, (8.4, 3.85), (8.4, y + h), text="kernel look-up at the\nsegment end points", toff=(-0.12, 0.0), ha="right")
    # ---- outputs / uses row --------------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.15, 0.6), 9.7, 1.4, boxstyle="round,pad=0.02", fc=C_USE, ec="none"))
    ax.text(0.3, 1.93, "outputs: $\\sigma(t)$ at every node, $t_{nuc}$ of every rail  $\\rightarrow$  the sign-off metric ladder", fontsize=6.0, color="#465", va="top", style="italic")
    yb = 0.7; hb = 0.82
    box(ax, 0.4, yb, 2.9, hb, "stack-level sign-off", "mortal set, earliest $t_{nuc}$, stress margin,\nTop-$k$ ranking, cross-die alignment")
    box(ax, 3.55, yb, 2.9, hb, "targeted repair", "widen only the violating rails\n(7.9-22$\\times$ less metal than blanket)")
    box(ax, 6.7, yb, 2.8, hb, "sensitivities & budgeting", "$\\partial \\log t_{nuc}/\\partial P_{block}$ by autograd,\nprojected-gradient power re-allocation")
    arrow(ax, (8.4, y), (8.4, 1.68), style="-")
    ax.plot([1.85, 8.4], [1.68, 1.68], color="#333333", lw=0.9)
    for xc in (1.85, 5.0, 8.1):
        arrow(ax, (xc, 1.68), (xc, yb + hb + 0.02))
    ax.text(9.75, 4.88, "physics results: reference FDM on every rail (truth);  SKN + JCX: the fast, differentiable surrogate", ha="right", va="top", fontsize=5.2, color="#555555", style="italic")
    os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
    st.save(fig, os.path.expanduser(a.out)); print("wrote", a.out)


if __name__ == "__main__":
    main()
