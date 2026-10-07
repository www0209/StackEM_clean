"""
E0 - solver and closure validation.

    python -m stackem.experiments.e0_validate [--config cfg.json] [--root DIR] [--hotspot DIR]

Produces  <root>/<name>/e0/e0_validation.json  and the figures
    e0_kernels_vs_series.(pdf|png)   FDM kernels vs closed-form series (uniform kappa)
    e0_closure_convergence.(pdf|png)  junction closure error vs number of time steps
    e0_tabulation.(pdf|png)           error of the tabulated kernel look-up vs table size

Checks.  "asserted" = a failure stops the pipeline; "reported" = the number is written to the JSON without a threshold.
  1. s_G and a kernels vs analytic series:      max |err| < 1e-4 (n_cells = 200)                       asserted
  2. early-time semi-infinite limit at tau = 1e-6 (relative deviation of the end values)             reported
  3. mass conservation of a blocked tree:        relative drift < 1e-9                                 asserted
  4. chain vs general-tree assembly identical:   < 1e-9 relative                                       asserted
  5. closure (exact kernels) vs full-tree FDM:   rel-L2 < 5e-3 at M = 64 (mid-jump), converging with M  asserted (M = 64 only)
  6. parabolic maximum principle: gap between the max over all nodes and over the junction nodes     reported
  7. tabulated look-up vs direct look-up for several table sizes                                     reported
  8. HotSpot linearity (if HotSpot is available): superposition error < 0.3 K                          asserted (not in --smoke)
"""
from __future__ import annotations

import os
import time
import numpy as np

from ._common import base_parser, setup, record, out_dir
from ..constants import default_em, verify_constants, SEC_PER_YEAR
from ..korhonen_fdm import solve_segment_hat, default_tau_grid, TreeFDM
from ..analytic import s_G_blocked_series, a_unit_flux_series, s_G_semi_infinite_end, a_semi_infinite_end
from ..thermal_profile import SegmentProfile
from ..closure import FDMKernelProvider, close_rails
from ..power_grid import Rail
from ..viz import style as st


def main(argv=None):
    ap = base_parser("E0: validate the FDM solver and the junction closure")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; em = ctx["em"]
    d = out_dir(cfg, "e0")
    rep = dict(constants=verify_constants(em, ctx["th"]))
    st.use_paper_style()
    import matplotlib.pyplot as plt

    # 1-2: kernels vs series -------------------------------------------------------------
    tau = default_tau_grid(1e-6, 3.0, 64)
    k = solve_segment_hat(lambda x: np.ones_like(x), 1.0, 0.0, 0.0, tau, 0.0, 200)
    a = solve_segment_hat(lambda x: np.ones_like(x), 0.0, 1.0, 0.0, tau, 0.0, 200)
    eG = np.abs(k.s - s_G_blocked_series(k.xi, tau)); eA = np.abs(a.s - a_unit_flux_series(a.xi, tau))
    rep["kernel_vs_series"] = dict(sG_max_abs_err=float(eG.max()), a_max_abs_err=float(eA.max()),
                                   sG_end_semi_inf_rel=float(abs(k.s[0, -1] - s_G_semi_infinite_end(tau[0])) / s_G_semi_infinite_end(tau[0])),
                                   a_end_semi_inf_rel=float(abs(a.s[0, 0] - a_semi_infinite_end(tau[0])) / abs(a_semi_infinite_end(tau[0]))))
    assert eG.max() < 1e-4 and eA.max() < 1e-4, rep["kernel_vs_series"]
    fig, axes = plt.subplots(1, 2, figsize=(st.IEEE_2COL, 2.3), constrained_layout=True)
    for it in [0, 16, 32, 48, 63]:
        axes[0].plot(k.xi, s_G_blocked_series(k.xi, tau[it:it + 1])[0], color="k", lw=0.8)
        axes[0].plot(k.xi[::8], k.s[it, ::8], "o", ms=2.2, color=st.OKABE_ITO["vermilion"])
        axes[1].plot(a.xi, a_unit_flux_series(a.xi, tau[it:it + 1])[0], color="k", lw=0.8)
        axes[1].plot(a.xi[::8], a.s[it, ::8], "o", ms=2.2, color=st.OKABE_ITO["blue"])
    axes[0].set_title(r"$\hat s_G(\xi,\tau)$: series (line) vs FDM (dots)"); axes[1].set_title(r"$\hat a(\xi,\tau)$")
    for ax in axes: ax.set_xlabel(r"$\xi$")
    st.save(fig, os.path.join(d, "e0_kernels_vs_series"))

    # 3-4: tree assembly ----------------------------------------------------------------------
    L = np.array([100e-6, 150e-6, 80e-6, 120e-6, 100e-6, 90e-6]); A = 2e-12
    j = np.array([1.2e10, 0.6e10, -0.3e10, -0.9e10, 0.4e10, 1.0e10])
    TL = np.array([340, 343, 347, 352, 356, 358.]); TR = np.array([343, 347, 352, 356, 358, 357.]); Tm = np.array([2.0, 0.5, 0.1, 1.0, 0.2, 1.5])
    Gam = 2e-5; sT = np.asarray(em.sigma_T(0.5 * (TL + TR)))
    times = np.logspace(4, 9, 64)
    tr = TreeFDM(em, n_cells=120); ns = [tr.add_node() for _ in range(7)]
    for kk in range(6):
        tr.add_segment(ns[kk], ns[kk + 1], L[kk], A, j[kk], SegmentProfile(L[kk], TL[kk], TR[kk], Tm[kk], Gam), sT[kk])
    sol = tr.solve(times)
    mass = tr.total_stress_integral(sol); rep["mass_drift_rel"] = float((mass[-1] - mass[0]) / abs(mass[0]))
    assert abs(rep["mass_drift_rel"]) < 1e-9
    tr2 = TreeFDM(em, n_cells=120); ns2 = [tr2.add_node() for _ in range(7)]
    for kk in [3, 0, 5, 1, 4, 2]:
        tr2.add_segment(ns2[kk], ns2[kk + 1], L[kk], A, j[kk], SegmentProfile(L[kk], TL[kk], TR[kk], Tm[kk], Gam), sT[kk])
    sol2 = tr2.solve(times)
    x0, p0 = sol.segment_profile(0); x1, p1 = sol2.segment_profile(1)
    rep["chain_vs_general_rel"] = float(np.abs(p0 - p1).max() / np.abs(p0).max()); assert rep["chain_vs_general_rel"] < 1e-9
    node_idx = [tr._gidx[kk][0] for kk in range(6)] + [tr._gidx[-1][-1]]
    # 6: maximum principle (reported, not asserted)
    rep["max_principle_gap_MPa"] = float((sol.sigma.max(axis=1) - sol.sigma[:, node_idx].max(axis=1)).max() / 1e6)

    # 5: closure convergence -------------------------------------------------------------------
    rail = Rail("X", "H", 0, 0.0, 0.0, L, 2e-6, 1e-6, j, TL, TR, Tm, Gam, sT, np.zeros(7, bool))
    prov = FDMKernelProvider(em)
    conv = {}
    Ms = [16, 32, 64, 128] if not args.smoke else [32, 64]
    for mode in ["start", "mid"]:
        conv[mode] = []
        for M in Ms:
            tt = np.logspace(4, 9, M); sM = tr.solve(tt); truth = sM.sigma[:, node_idx]
            t0 = time.time(); res = close_rails([rail], em, prov, tt, jump_at=mode); dt = time.time() - t0
            err = res.sigma_nodes[0] - truth
            tn_truth = sM.nucleation(em.sigma_crit)[0]
            conv[mode].append(dict(M=M, rel_L2=float(np.linalg.norm(err) / np.linalg.norm(truth)),
                                   max_rel=float(np.abs(err).max() / np.abs(truth).max()),
                                   t_nuc_rel_err=float(abs(res.t_nuc[0] - tn_truth) / tn_truth), seconds=dt))
    rep["closure_convergence"] = conv
    mid64 = next(c for c in conv["mid"] if c["M"] == 64)
    assert mid64["rel_L2"] < 5e-3, mid64
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.2))
    for mode, c in [("start", st.OKABE_ITO["grey"]), ("mid", st.OKABE_ITO["blue"])]:
        ax.plot([r["M"] for r in conv[mode]], [r["rel_L2"] for r in conv[mode]], marker="o", ms=3, color=c, label=f"flux jump at interval {mode}")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("time steps M"); ax.set_ylabel("rel-L2 error of node stresses"); ax.legend()
    st.save(fig, os.path.join(d, "e0_closure_convergence"))

    # 7: tabulated kernel look-up: interpolation error vs table size, same rail, exact kernels ------
    from ..closure_tab import TabulatedProvider
    M_tab = int(cfg["times"]["n"]); tt = np.logspace(4, 9, M_tab)
    ref = close_rails([rail], em, prov, tt, jump_at="mid")
    tab_err = []
    for K in ([48, 96, 192, 384] if not args.smoke else [48, 192]):
        tp = TabulatedProvider(prov, K); t0 = time.time(); rt = close_rails([rail], em, tp, tt, jump_at="mid"); dt = time.time() - t0
        e = rt.sigma_nodes[0] - ref.sigma_nodes[0]
        tab_err.append(dict(n_tab=K, rel_L2=float(np.linalg.norm(e) / np.linalg.norm(ref.sigma_nodes[0])), max_rel=float(np.abs(e).max() / np.abs(ref.sigma_nodes[0]).max()),
                            t_nuc_rel_err=float(abs(rt.t_nuc[0] - ref.t_nuc[0]) / ref.t_nuc[0]) if np.isfinite(ref.t_nuc[0]) else None,
                            kernel_evals=int(tp.n_base_evals), kernel_points=int(tp.n_points), seconds=dt))
    rep["tabulation_error"] = dict(M=M_tab, rows=tab_err)
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.2))
    ax.plot([r["n_tab"] for r in tab_err], [r["rel_L2"] for r in tab_err], marker="D", ms=3, color=st.OKABE_ITO["green"], label="node stresses, rel-L2")
    ax.plot([r["n_tab"] for r in tab_err], [r["max_rel"] for r in tab_err], marker="o", ms=3, color=st.OKABE_ITO["orange"], ls="--", label="node stresses, max rel")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("table points per segment and end"); ax.set_ylabel("error of the tabulated look-up")
    ax.set_title(f"vs direct look-up, M = {M_tab} output times", fontsize=7); ax.legend(fontsize=6)
    st.save(fig, os.path.join(d, "e0_tabulation"))

    # 8: HotSpot linearity (optional) ---------------------------------------------------------------
    try:
        from ..hotspot_stack import HotSpotRunner, run_stack, unit_power_responses, block_values
        from ..config import hotspot_dir
        runner = HotSpotRunner(hotspot_dir(cfg)); spec = ctx["spec"]
        wd = os.path.join(d, "hotspot_lin")
        tf = run_stack(spec, wd, runner, "stack", with_alone=False)
        if not args.smoke:
            R = unit_power_responses(spec, wd, runner)
            P = np.concatenate([block_values(dd) for dd in spec.dies])
            errs = {dd.name: float(np.abs(spec.ambient + np.tensordot(P, R[dd.name], axes=1) - tf.T_die[dd.name]).max()) for dd in spec.dies}
            rep["hotspot_linearity_max_K"] = errs
            assert max(errs.values()) < 0.3
        rep["hotspot_summary"] = tf.summary()
    except FileNotFoundError as e:
        rep["hotspot_linearity_max_K"] = f"skipped: {e}"
    record(cfg, "e0", "e0_validation.json", rep)
    print("E0 PASSED")


if __name__ == "__main__":
    main()
