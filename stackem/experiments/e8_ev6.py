"""
E8 - a real 3-D floorplan: HotSpot's ev6 three-die stack (examples/example4).

    python -m stackem.experiments.e8_ev6 --root DIR [--weights ...] [--current-fraction 0.15] [--smoke]

The stack (bottom -> top): L2 cache die 1, L2 cache die 2, four-core ev6 logic die,
12.4 mm x 12.76 mm each, TSV strips in the TIM layers (as shipped with HotSpot).
Every die gets its own analysed power-grid pair (the PG lattice covers the 12.4 mm x 12.4 mm
square of the die; block currents are distributed over the lattice nodes inside every
floorplan rectangle in proportion to the block's average power over the trace).

Each die's strap width is first sized by the same sign-off rule as the synthetic stacks (standalone =
"other dies off" field, earliest t_nuc = horizon x margin), then the variants are solved
(FDM truth on every rail)
    full        3-D stack temperature field of every die
    others_off  the other two dies un-powered: the die-level sign-off view
    uniform     die-average temperature
plus the StackEM overlay on the full variant (accuracy + timing on ~750 long rails).

Outputs: e8_summary.json, e8_diemaps_full / e8_diemaps_others_off, e8_cdf, e8_parity.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
import numpy as np

from ._common import base_parser, setup, record, out_dir, kernel_provider
from ..assembler import solve_rails_truth, topk_hit_rate, kendall_tau
from ..blech_screen import screen_rails
from ..closure import close_rails
from ..constants import SEC_PER_YEAR
from ..hotspot_stack import HotSpotRunner, run_ev6_example, read_flp, write_ptrace, DieSpec, StackSpec, ThermalField
from ..config import hotspot_dir
from ..korhonen_fdm import nucleation_from_history
from ..power_grid import DieGrid, embed_thermal, pg_summary, check_current_regime
from ..viz.plots import plot_cdf_variants, plot_parity
from ..viz.stack3d import plot_die_maps

EV6_DIES = [("cache1", 0, "ev6_3D_cache_1.flp"), ("cache2", 2, "ev6_3D_cache_2.flp"), ("cores", 4, "ev6_3D_core_layer.flp")]


def block_powers(workdir: str):
    """Average power of every block over the trace (W)."""
    lines = [l for l in open(os.path.join(workdir, "ev6_3D.ptrace")).read().splitlines() if l.strip()]
    names = lines[0].split(); vals = np.array([[float(v) for v in l.split()] for l in lines[1:]])
    return dict(zip(names, vals.mean(axis=0)))


def rect_sinks(grid: DieGrid, flp, powers, current_fraction, Vdd):
    """Node current sinks (A) from floorplan rectangles (name, w, h, x, y)."""
    n = grid.n; sinks = np.zeros((n, n))
    X, Y = np.meshgrid(grid.xs, grid.ys)                # [r, c] -> (x, y)
    for name, w, h, x, y in flp:
        P = powers.get(name, 0.0)
        if P <= 0:
            continue
        m = (X >= x - 1e-12) & (X < x + w - 1e-12) & (Y >= y - 1e-12) & (Y < y + h - 1e-12)
        if not m.any():                                  # thin strip narrower than the lattice pitch -> nearest row/col
            r = int(np.clip(np.round((y + 0.5 * h) / grid.pg.pitch), 0, n - 1)); m[r, :] = (X[r, :] >= x) & (X[r, :] < x + w)
        sinks[m] += P * current_fraction / Vdd / m.sum()
    return sinks


def T_at_rect(Tgrid, size_x, size_y, x, y):
    """Bilinear lookup on a HotSpot grid covering a size_x by size_y die (row 0 = top edge)."""
    rows, cols = Tgrid.shape
    fx = np.clip(np.asarray(x) / size_x * cols - 0.5, 0, cols - 1); fy = np.clip((size_y - np.asarray(y)) / size_y * rows - 0.5, 0, rows - 1)
    c0 = np.floor(fx).astype(int); r0 = np.floor(fy).astype(int); c1 = np.minimum(c0 + 1, cols - 1); r1 = np.minimum(r0 + 1, rows - 1)
    wx = fx - c0; wy = fy - r0
    return (1 - wx) * (1 - wy) * Tgrid[r0, c0] + wx * (1 - wy) * Tgrid[r0, c1] + (1 - wx) * wy * Tgrid[r1, c0] + wx * wy * Tgrid[r1, c1]


def resample_square(Tgrid, size_x, size_y, size):
    """Resample the rectangular-die grid onto a square [0,size]^2 grid of the same resolution (for embed_thermal)."""
    rows, cols = Tgrid.shape
    xs = (np.arange(cols) + 0.5) / cols * size; ys = size - (np.arange(rows) + 0.5) / rows * size
    X, Y = np.meshgrid(xs, ys)
    return T_at_rect(Tgrid, size_x, size_y, X, Y)


def main(argv=None):
    ap = base_parser("E8: ev6 real 3-D floorplan")
    ap.add_argument("--current-fraction", type=float, default=0.15, help="share of the block current carried by the analysed metal pair")
    ap.add_argument("--r-convec", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=100e-6)
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; em = ctx["em"]; th = ctx["th"]; times = ctx["times"]
    prov, kind = kernel_provider(ctx)
    d = out_dir(cfg, "e8"); runner = HotSpotRunner(hotspot_dir(cfg))
    pg = replace(ctx["pg"], pitch=args.pitch, current_fraction=args.current_fraction)
    # --- thermal fields: full and "others off" -------------------------------------------------
    ev = run_ev6_example(runner, os.path.join(d, "hotspot_full"), args.r_convec)
    powers = block_powers(ev.workdir)
    flps = {name: read_flp(os.path.join(ev.workdir, fn)) for name, _, fn in EV6_DIES}
    size_x = ev.size; size_y = max(y + h for (_, w, h, x, y) in flps["cores"]); size = size_x
    T_full = {name: ev.T_die(k) for k, (name, _, _) in enumerate(EV6_DIES)}
    T_off = {}
    for k, (name, _, fn) in enumerate(EV6_DIES):
        wd = os.path.join(d, f"hotspot_only_{name}"); os.makedirs(wd, exist_ok=True)
        for fn_ in os.listdir(ev.workdir):                            # reuse the prepared inputs of the full run
            if fn_.endswith((".flp", ".lcf", ".ptrace", ".config")):
                shutil.copy(os.path.join(ev.workdir, fn_), wd)
        own = {b[0] for b in flps[name]}
        lines = [l for l in open(os.path.join(wd, "ev6_3D.ptrace")).read().splitlines() if l.strip()]
        names = lines[0].split(); rowsv = [[float(v) for v in l.split()] for l in lines[1:]]
        vals = [[v if nm in own else 0.0 for nm, v in zip(names, row)] for row in rowsv]
        with open(os.path.join(wd, "ev6_3D.ptrace"), "w") as f:
            f.write("\t".join(names) + "\n"); f.writelines("\t".join(f"{v:.6g}" for v in row) + "\n" for row in vals)
        layers = runner.run(wd, "ev6_3D.lcf", "ev6_3D.ptrace", "ev6.config", "ev6", True)
        T_off[name] = layers[EV6_DIES[k][1]]
    # --- die specs (for the figures) and PG solutions -----------------------------------------------
    dies = []
    for name, _, _ in EV6_DIES:
        P = sum(powers.get(b[0], 0.0) for b in flps[name])
        dies.append(DieSpec(name, size=size, thickness=150e-6, nblk=1, power=np.array([[P]]), feed_pitch=400e-6, feed_kind="tsv"))
    spec = StackSpec(dies, r_convec=args.r_convec or 0.1)
    tf = ThermalField(spec, {n: resample_square(T_full[n], size_x, size_y, size) for n in T_full},
                      {n: resample_square(T_off[n], size_x, size_y, size) for n in T_off})
    # --- sign-off sizing of every die from its "others off" field (same rule as the synthetic stacks) -------------
    sizing = {}
    if not args.no_sizing:
        from ..signoff import size_strap_width
        so = cfg["signoff"]; target = float(so["horizon_years"]) * float(so["margin"]) * SEC_PER_YEAR
        for k, dd in enumerate(dies):
            def rails_of_W(W, dd=dd):
                g = DieGrid(replace(dd, rail_W=float(W)), pg)
                return embed_thermal(g, g.solve(rect_sinks(g, flps[dd.name], powers, pg.current_fraction, pg.Vdd), g.feed_mask()), tf.T_alone[dd.name], em, th, two_sided=False)
            r = size_strap_width(rails_of_W, em, times, target, float(so["W_min"]), float(so["W_max"]), 40 if args.smoke else cfg["truth"]["n_cells"])
            sizing[dd.name] = dict(W=r["W"], W_um=r["W"] * 1e6, t_earliest_years=r["t_earliest_s"] / SEC_PER_YEAR, saturated=r["saturated"])
            dies[k] = replace(dd, rail_W=r["W"])
            print(f"  sizing {dd.name}: W = {r['W']*1e6:.2f} um, standalone earliest {r['t_earliest_s']/SEC_PER_YEAR:.1f} yr{' (saturated)' if r['saturated'] else ''}", flush=True)
        spec = StackSpec(dies, r_convec=args.r_convec or 0.1); tf.spec = spec
    grids = {}; sols = {}; pgs = {}
    for dd in dies:
        g = DieGrid(dd, pg); s = g.solve(rect_sinks(g, flps[dd.name], powers, pg.current_fraction, pg.Vdd), g.feed_mask())
        grids[dd.name] = g; sols[dd.name] = s; pgs[dd.name] = pg_summary(s, g.pg); pgs[dd.name]["warnings"] = check_current_regime(pgs[dd.name])
        print(dd.name, f"P={dd.total_power():.1f} W", {k: round(v, 3) for k, v in pgs[dd.name].items() if isinstance(v, float)}, pgs[dd.name]["warnings"])
    # --- variants ----------------------------------------------------------------------------------
    nmax = 6 if args.smoke else None
    variants = {"full": lambda n: embed_thermal(grids[n], sols[n], tf.T_die[n], em, th),
                "others_off": lambda n: embed_thermal(grids[n], sols[n], tf.T_alone[n], em, th),
                "uniform": lambda n: embed_thermal(grids[n], sols[n], tf.T_die[n], em, th, uniform_T=tf.T_uniform(n))}
    if args.smoke:
        variants = {k: variants[k] for k in ("full", "others_off")}
    tn_all = {}; rails_all = {}; nuc_all = {}; rep = dict(provider=kind, sizing=sizing, pg=pgs, current_fraction=pg.current_fraction, die_size_m=[size_x, size_y],
                                                             thermal={n: dict(full_min=float(T_full[n].min()), full_max=float(T_full[n].max()), full_mean=float(T_full[n].mean()),
                                                                              others_off_mean=float(T_off[n].mean()), others_off_max=float(T_off[n].max())) for n in T_full})
    for v, fn in variants.items():
        t0 = time.time(); tn_v = {}; rails_v = {}; nuc_v = {}; summ = {}
        for dd in dies:
            rails = fn(dd.name)[:nmax] if nmax else fn(dd.name)
            imm, ss = screen_rails(rails, em, True)
            tn = np.full(len(rails), np.inf); nn = np.zeros(len(rails), int)
            todo = np.where(~imm)[0]
            if len(todo):
                _, t_, n_ = solve_rails_truth([rails[i] for i in todo], em, times, cfg["truth"]["n_cells"], True)
                tn[todo] = t_; nn[todo] = n_
            xy = np.array([[r.node_xy()[0][k], r.node_xy()[1][k]] for r, k in zip(rails, nn)])
            tn_v[dd.name] = tn; rails_v[dd.name] = rails; nuc_v[dd.name] = xy
            summ[dd.name] = dict(n_rails=len(rails), n_immortal=int(imm.sum()), earliest_years=float(tn.min() / SEC_PER_YEAR),
                                 mortal_10yr=int(np.sum(tn <= 10 * SEC_PER_YEAR)), mortal_100yr=int(np.sum(tn <= 100 * SEC_PER_YEAR)),
                                 j_max_A_per_cm2=float(max(np.abs(r.j).max() for r in rails) * 1e-4), T_max=float(max(max(r.T_L.max(), r.T_R.max()) for r in rails)))
        tn_all[v] = tn_v; rails_all[v] = rails_v; nuc_all[v] = nuc_v
        from ..power_grid import save_rails
        np.savez_compressed(os.path.join(d, f"e8_{v}_truth.npz"), times=times, **{f"tnuc_{dn}": tn_v[dn] for dn in tn_v}, **{f"nuc_xy_{dn}": nuc_v[dn] for dn in nuc_v})
        save_rails(os.path.join(d, f"e8_{v}_rails.json"), [r for dn in rails_v for r in rails_v[dn]], dict(variant=v))
        rep[v] = dict(seconds=time.time() - t0, dies=summ)
        print(f"{v:11s} {time.time()-t0:6.1f}s", {n: (round(s['earliest_years'], 3), s['mortal_10yr'], s['n_immortal']) for n, s in summ.items()})
        plot_die_maps(spec, tf, rails_v, tn_v, nuc_v, os.path.join(d, f"e8_diemaps_{v}"), f"ev6 stack - {v}", T_alone=(v == "others_off"))
    # --- StackEM overlay on the full variant ---------------------------------------------------------------
    if kind == "skn":
        t0 = time.time(); tp = {}
        for dd in dies:
            rails = rails_all["full"][dd.name]; imm, _ = screen_rails(rails, em, True)
            tn = np.full(len(rails), np.inf); todo = np.where(~imm)[0]
            for i0 in range(0, len(todo), 32):                       # batches: (32 rails x 124 segments x 2080 lags) x 4 kernels ~ 260 MB
                sub = todo[i0:i0 + 32]
                res = close_rails([rails[i] for i in sub], em, prov, times, em.sigma_crit, True, cfg["closure"]["jump_at"])
                tn[sub] = res.t_nuc
            tp[dd.name] = tn
        T = np.concatenate([tn_all["full"][dd.name] for dd in dies]); P = np.concatenate([tp[dd.name] for dd in dies])
        lab = np.concatenate([[dd.name] * len(tp[dd.name]) for dd in dies]); m = np.isfinite(T) & np.isfinite(P)
        rel = np.abs(P[m] - T[m]) / T[m]
        rep["stackem_overlay"] = dict(seconds=time.time() - t0, n_mortal=int(m.sum()), tnuc_rel_median=float(np.median(rel)) if m.any() else None,
                                     tnuc_rel_p90=float(np.percentile(rel, 90)) if m.any() else None, tnuc_rel_max=float(rel.max()) if m.any() else None,
                                     top10_hit=topk_hit_rate(T, P, min(10, len(T))), kendall_top50=kendall_tau(T, P, min(50, len(T))))
        plot_parity(T, P, os.path.join(d, "e8_parity"), labels=lab, title="ev6 stack, full 3-D")
        tn_all["stackem"] = tp
        print("StackEM overlay", rep["stackem_overlay"])
    plot_cdf_variants({v: tn_all[v] for v in variants}, [dd.name for dd in dies], os.path.join(d, "e8_cdf"),
                      overlay=tn_all.get("stackem"))
    record(cfg, "e8", "e8_summary.json", rep)


if __name__ == "__main__":
    main()
