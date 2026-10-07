"""
E3 - the three-dimensional physics decomposition (main scientific result).

    python -m stackem.experiments.e3_decomposition --root DIR [--weights ...] [--smoke]

For the base stack every physics variant is solved with the reference FDM on
every rail (truth), and the full variant additionally with StackEM (overlay):

    alone / uniform_signoff / uniform / sigmaT_const / no_joule / no_tm / full      (seven variants)

Outputs
    e3_summary.json      per-die mortal counts within 1/10/100 yr, earliest t_nuc, effect ratios, stress margins,
                         cross-die alignment of every die below the top die (crossing enrichment of the mortal
                         rails with the top hot footprint), hotspot-position sweep
    e3_cdf.(pdf|png)     per-die cumulative nucleation curves (main figure)
    e3_effect_ratios     dot plot of earliest-t_nuc ratios (simplified / full)
    e3_diemaps_<v>       per-die maps for full and alone
    e3_stack3d_full      exploded 3-D view of the stack with rails coloured by t_nuc
    e3_stress_surface    sigma(x,t) of the most critical rail of the bottom die
    e3_thermal_slice     vertical cut through the stack under the top hot spot (in-stack vs standalone T)
    e3_margin_<v>        per-die stress-margin maps at the sign-off horizon (full / alone / uniform_signoff)
    e3_hotspot_alignment same top-die power, different hot-block position: maps of the die under the top die,
                         earliest t_nuc and crossing enrichment of every lower die (data in e3_summary.json)
"""
from __future__ import annotations

import copy
import os
import time
import numpy as np

from ._common import base_parser, setup, record, out_dir, make_assembler, kernel_provider
from ..assembler import VARIANTS, save_variant
from ..constants import SEC_PER_YEAR
from ..config import stack_from_config
from ..viz.plots import plot_cdf_variants, plot_effect_ratios
from ..viz.stack3d import plot_stack_exploded, plot_die_maps, plot_stress_surface, plot_thermal_slice, plot_hotspot_alignment, plot_margin_maps
from ..viz import style as st
from ..korhonen_fdm import TreeFDM
from ..thermal_profile import SegmentProfile


def main(argv=None):
    ap = base_parser("E3: 3-D physics decomposition")
    ap.add_argument("--no-overlay", action="store_true", help="skip the StackEM overlay (truth only)")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; spec = ctx["spec"]; em = ctx["em"]; times = ctx["times"]
    prov, kind = (None, "none") if args.no_overlay else kernel_provider(ctx)
    asm = make_assembler(ctx, "e3", prov).run_fields()
    d = out_dir(cfg, "e3"); dies = [dd.name for dd in spec.dies]
    nmax = 10 if args.smoke else None
    variants = ["alone", "uniform_signoff", "uniform", "sigmaT_const", "no_joule", "no_tm", "full"] if not args.smoke else ["alone", "uniform_signoff", "full"]
    results = {}; tn = {}
    for v in variants:
        t0 = time.time(); r = asm.solve_variant(v, "truth", nmax); results[v] = r; tn[v] = r.t_nuc
        s = asm.stack_summary(r); save_variant(os.path.join(d, f"e3_{v}_truth.npz"), r, s, asm.times)
        print(f"{v:13s} {time.time()-t0:5.1f}s  earliest {s['earliest_t_nuc_years']:.3f} yr on {s['critical_die']}; mortal10yr {s['n_mortal_10yr']} /100yr {s['n_mortal_100yr']}")
    overlay = None
    if prov is not None:
        t0 = time.time(); rp = asm.solve_variant("full", "closure", nmax); overlay = rp.t_nuc
        save_variant(os.path.join(d, "e3_full_stackem.npz"), rp, asm.stack_summary(rp), asm.times)
        print(f"StackEM overlay {time.time()-t0:.1f}s  earliest {asm.stack_summary(rp)['earliest_t_nuc_years']:.3f} yr")
    # --- effect ratios (baseline earliest t_nuc / full earliest t_nuc, per die) --------------------
    full = results["full"]
    ratios = {}
    names = {"alone": "die-level sign-off / stack", "uniform_signoff": r"rule sign-off (105$^\circ$C) / stack", "uniform": "uniform T / 3-D field", "sigmaT_const": r"frozen $\sigma_T$ / $\sigma_T(T)$",
             "no_joule": "no self-heating / with", "no_tm": "no TM / with"}
    for v in variants:
        if v == "full": continue
        ratios[names[v]] = {dn: float(np.min(results[v].t_nuc[dn]) / np.min(full.t_nuc[dn])) for dn in dies}
    mortal_counts = {v: {dn: {h: int(np.sum(results[v].t_nuc[dn] <= h * SEC_PER_YEAR)) for h in (1, 10, 100)} for dn in dies} for v in variants}
    immortal_counts = {v: {dn: int(results[v].immortal[dn].sum()) for dn in dies} for v in variants}
    # per-rail ratio distributions (uniform vs full etc.)
    dist = {}
    for v in variants:
        if v == "full": continue
        dist[v] = {}
        for dn in dies:
            a, b = results[v].t_nuc[dn], full.t_nuc[dn]; m = np.isfinite(a) & np.isfinite(b)
            q = (a[m] / b[m]) if m.any() else np.array([np.nan])
            dist[v][dn] = dict(p5=float(np.nanpercentile(q, 5)), p50=float(np.nanpercentile(q, 50)), p95=float(np.nanpercentile(q, 95)), n=int(m.sum()))
    # cross-die alignment of every die below the top die with the top die's hot footprint (the die directly
    # under the top die is the one that sees the hot spot most; the bottom die's failures are current-selected)
    align = {dn: asm.cross_die_alignment(full, dn, dies[-1]) for dn in dies[:-1]} if len(dies) >= 2 else {}
    margins = {v: {dn: float(min(m.margin_h for m in results[v].metrics if m.die == dn)) for dn in dies} for v in variants}
    rep = dict(provider=kind, sizing=ctx.get("sizing"), thermal=asm.tf.summary(), pg=asm.pg_stats, vertical_screen=asm.vertical_screen_table(), min_margin_h=margins,
               stack={v: asm.stack_summary(results[v]) for v in variants}, effect_ratios=ratios, mortal_counts=mortal_counts,
               immortal_counts=immortal_counts, ratio_distributions=dist, cross_die_alignment=align)
    if overlay is not None:
        rep["stackem_overlay"] = asm.stack_summary(rp)
    # --- figures -----------------------------------------------------------------------------------
    def _short_role(role: str) -> str:            # "top / heat-sink side, logic die with hotspot" -> "logic"
        r = role.lower()
        for k in ("logic", "sram", "io", "memory", "hbm", "cpu", "gpu"):
            if k in r: return k.upper() if k in ("io", "sram", "hbm", "cpu", "gpu") else k
        return ""
    roles = {dd["name"]: _short_role(dd.get("role", "")) for dd in cfg["stack"]["dies"]}
    titles = {dd.name: f"{dd.name} ({roles[dd.name] + ', ' if roles.get(dd.name) else ''}{dd.total_power():.0f} W)" for dd in spec.dies}
    plot_cdf_variants({v: results[v].t_nuc for v in variants}, dies, os.path.join(d, "e3_cdf"), overlay=overlay, die_titles=titles)
    # paper version: the two practices, the residual-stress channel, the resolved stack and the surrogate
    plot_cdf_variants({v: results[v].t_nuc for v in variants if v in ("alone", "uniform_signoff", "sigmaT_const", "full")}, dies,
                      os.path.join(d, "e3_cdf_paper"), overlay=overlay, die_titles=titles)
    plot_effect_ratios(ratios, os.path.join(d, "e3_effect_ratios"))
    for v in ("full", "alone"):
        r = results[v]
        rails = {dn: [x for x in r.rails if x.die == dn] for dn in dies}
        nuc_xy = {dn: np.array([[m.nuc_x, m.nuc_y] for m in r.metrics if m.die == dn]) for dn in dies}
        plot_die_maps(spec, asm.tf, rails, r.t_nuc, nuc_xy, os.path.join(d, f"e3_diemaps_{v}"), v, T_alone=(v == "alone"))
    rails_full = {dn: [x for x in full.rails if x.die == dn] for dn in dies}
    plot_stack_exploded(spec, asm.tf, rails_full, full.t_nuc, os.path.join(d, "e3_stack3d_full"), die_roles={k: v for k, v in roles.items() if v},
                        horizon_years=asm.horizon_years)
    plot_thermal_slice(spec, asm.tf, os.path.join(d, "e3_thermal_slice"))
    for v in ("full", "alone", "uniform_signoff"):
        if v in results:
            mg = {dn: np.array([m.margin_h for m in results[v].metrics if m.die == dn]) for dn in dies}
            plot_margin_maps(spec, {dn: [x for x in results[v].rails if x.die == dn] for dn in dies}, mg,
                             os.path.join(d, f"e3_margin_{v}"), asm.horizon_years, st.VARIANT_LABEL.get(v, v))
    # stress surface of the most critical rail of the bottom die
    dn = dies[0]; k = int(np.argmin(full.t_nuc[dn])); r = rails_full[dn][k]
    tr = TreeFDM(em, n_cells=30); ns = [tr.add_node() for _ in range(r.n_seg + 1)]
    for kk in range(r.n_seg):
        tr.add_segment(ns[kk], ns[kk + 1], r.L[kk], r.A, r.j[kk], SegmentProfile(r.L[kk], r.T_L[kk], r.T_R[kk], r.T_m[kk], r.Gamma), r.sigma_T[kk])
    sol = tr.solve(times); idx = np.concatenate(tr._gidx); x = np.concatenate([tr._seg_x[kk] + r.L[:kk].sum() for kk in range(r.n_seg)])
    t_n, node, _ = sol.nucleation(em.sigma_crit)
    xn = x[np.where(idx == node)[0][0]] if node in idx else None
    plot_stress_surface(x, times, sol.sigma[:, idx], em.sigma_crit, os.path.join(d, "e3_stress_surface"),
                        f"{dn} rail {r.kind}{r.index}: most critical rail of the bottom die", t_n, xn,
                        feeds_x_m=np.concatenate([[0.0], np.cumsum(r.L)]))
    # --- hotspot-position sweep: same top-die mean temperature, different hotspot block --------------
    if not args.smoke and len(dies) >= 2:
        sweep = []; fig_entries = []
        top = cfg["stack"]["dies"][-1]
        for pos in [[0, 0], [1, 1], [2, 2], [3, 3], [0, 3]]:
            c2 = copy.deepcopy(cfg); c2["stack"]["dies"][-1]["hot"] = [[pos, top["hot"][0][1] if top.get("hot") else 5.0]]
            spec2 = stack_from_config(c2)
            if ctx.get("sizing"):
                from ..signoff import apply_sizing
                spec2 = apply_sizing(spec2, ctx["sizing"])                  # same rule-sized straps as the base case
            from ..assembler import StackAssembler
            a2 = StackAssembler(spec2, ctx["pg"], em, ctx["th"], asm.runner, times, os.path.join(d, f"hot_{pos[0]}{pos[1]}"), None,
                                truth_cells=cfg["truth"]["n_cells"]).run_fields()
            rf = a2.solve_variant("full", "truth")
            s = a2.stack_summary(rf); save_variant(os.path.join(d, f"hot_{pos[0]}{pos[1]}", "e3_full_truth.npz"), rf, s, times)
            per_die = {}
            for dn in dies[:-1]:                                   # every die below the top die
                al = a2.cross_die_alignment(rf, dn, dies[-1]); sd = s["dies"][dn]
                per_die[dn] = dict(earliest_yr=sd["earliest_t_nuc_s"] / SEC_PER_YEAR, earliest_rail=sd["earliest_rail"],
                                   mortal10=sd["mortal_within"]["10yr"], mortal100=sd["mortal_within"]["100yr"],
                                   crossing_enrichment_10yr=al["crossing_enrichment_signoff"], crossing_enrichment_100yr=al["crossing_enrichment_100yr"],
                                   tnuc_ratio_cross_over_noncross=al["tnuc_ratio_cross_over_noncross"], spearman_T_vs_logt=al["spearman_rho"],
                                   earliest_rail_crosses_hot_block=al["earliest_rail_crosses_hot_block"], share_crossing=al["share_of_rails_crossing_hot_block"])
            sweep.append(dict(hot_block=pos, top_T_mean=float(a2.tf.T_die[dies[-1]].mean()), top_T_max=float(a2.tf.T_die[dies[-1]].max()),
                              stack_earliest_yr=s["earliest_t_nuc_years"], critical_die=s["critical_die"], dies=per_die))
            under = dies[-2]; bw = spec2.size / spec2.dies[-1].nblk
            fig_entries.append(dict(pos=pos, T=a2.tf.T_die[under], size=spec2.size, rails=[x for x in rf.rails if x.die == under],
                                    t_nuc=rf.t_nuc[under], hot_rect=(pos[0] * bw, pos[1] * bw, bw), metrics=per_die))
            print("hotspot", pos, {dn: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()} for dn, m in per_die.items()})
        rep["hotspot_sweep"] = sweep
        plot_hotspot_alignment(fig_entries, dies[:-1], dies[-1], os.path.join(d, "e3_hotspot_alignment"), asm.horizon_years)
    record(cfg, "e3", "e3_summary.json", rep)


if __name__ == "__main__":
    main()
