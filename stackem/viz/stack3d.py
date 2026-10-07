"""
stackem.viz.stack3d
=====================

Three-dimensional renderings of the hybrid-bonded stack:

* ``plot_stack_exploded``  - exploded isometric view: every die drawn as a thin
  slab whose top face is textured with its temperature field, the analysed
  power rails drawn on top and coloured by their nucleation time, TSV clusters
  and bond-pad arrays between the dies, TIM / spreader / heat-sink slabs, and
  the vertical projection of the top-die hotspot onto the bottom die (the
  cross-die alignment the paper is about).
* ``plot_die_maps``        - per-die 2-D panels: temperature field with the
  rails overlaid and nucleation sites marked (the "what fails where" figure).
* ``plot_stress_surface``  - sigma(x, t) of one rail as a surface over
  (position, log time) with the sigma_crit iso-line and the nucleation point.
* ``plot_thermal_slice``   - vertical cut through the stack under the top-die
  hot spot (layer-resolved temperature image) and, for every die, T along the
  cut in the stack vs standalone: the die-to-die thermal coupling in one look.
* ``plot_hotspot_alignment`` - the hotspot-position sweep: maps of the die under
  the top die for every hot-block position + earliest t_nuc and crossing
  enrichment of every lower die.
* ``plot_margin_maps``     - per-die maps of the stress margin at the sign-off
  horizon (the sign-off view of the same result).

All functions accept the objects produced by the assembler and write vector
PDF + PNG through ``style.save``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from ..constants import SEC_PER_YEAR
from ..hotspot_stack import StackSpec, ThermalField, grid_T_at
from ..power_grid import Rail
from . import style as st


def _box(ax, x0, y0, z0, dx, dy, dz, color, alpha=1.0, edge="k", lw=0.3, **kw):
    X = [x0, x0 + dx]; Y = [y0, y0 + dy]; Z = [z0, z0 + dz]
    faces = [
        [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[0], Y[1], Z[0])],
        [(X[0], Y[0], Z[1]), (X[1], Y[0], Z[1]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
        [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[0], Z[1]), (X[0], Y[0], Z[1])],
        [(X[0], Y[1], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
        [(X[0], Y[0], Z[0]), (X[0], Y[1], Z[0]), (X[0], Y[1], Z[1]), (X[0], Y[0], Z[1])],
        [(X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[1], Y[0], Z[1])],
    ]
    pc = Poly3DCollection(faces, facecolors=color, edgecolors=edge, linewidths=lw, alpha=alpha, **kw)
    ax.add_collection3d(pc)


def _textured_top(ax, Tgrid, size, z, norm, cmap, x0=0.0, y0=0.0, stride=1, **kw):
    rows, cols = Tgrid.shape
    xs = np.linspace(0, size, cols + 1) + x0; ys = np.linspace(0, size, rows + 1) + y0
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z)
    Tplot = Tgrid[::-1, :]                       # row 0 is the top edge -> flip to y-up
    colors = plt.get_cmap(cmap)(norm(Tplot))
    surf = ax.plot_surface(X * 1e3, Y * 1e3, Z * 1e3, facecolors=colors, rstride=stride, cstride=stride, shade=False,
                           linewidth=0, antialiased=False, **kw)
    return surf


def _label3d(ax, x, y, z, s, **kw):
    """Text at a 3-D data point drawn as a 2-D artist (Text3D is ignored by the tight bounding box, so labels
    placed outside the die outline were clipped in the saved figure)."""
    from mpl_toolkits.mplot3d import proj3d
    xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return ax.text2D(xs, ys, s, transform=ax.transData, **kw)


def _fmt_earliest(t_s: float) -> str:
    return f"earliest {t_s/SEC_PER_YEAR:.2f} yr" if np.isfinite(t_s) else "immortal (> 100 yr)"


def plot_stack_exploded(spec: StackSpec, tf: ThermalField, rails: Dict[str, List[Rail]], t_nuc: Dict[str, np.ndarray],
                        path_base: str, gap: float | None = None, show_rails: bool = True, show_tsv: bool = True,
                        rail_stride: int = 1, elev: float | None = None, azim: float = -58, title: str | None = None,
                        highlight: Optional[Dict[str, Sequence[int]]] = None, horizon_years: float = 10.0,
                        die_roles: Optional[Dict[str, str]] = None):
    """Exploded 3-D stack (the paper's Fig. 1).  Every die is a slab textured with its temperature field; the analysed
    rails lie on top, coloured by nucleation time (mortal rails with a white halo so they read on the dark, cool
    regions; immortal rails faint); the top die's hot-block footprint is projected down through the stack (dashed);
    the earliest rail of every die is outlined; every die carries a label with its power, its earliest nucleation and
    the number of rails that violate the horizon.  Units drawn in mm; the layer spacing and elevation scale with
    the number of dies."""
    st.use_paper_style()
    n_die = len(spec.dies)
    size = spec.size
    gap = (0.26 + 0.05 * max(n_die - 3, 0)) * size if gap is None else gap
    elev = (24.0 + 3.0 * max(n_die - 3, 0)) if elev is None else elev
    fig = plt.figure(figsize=(st.IEEE_2COL, 3.6 + 0.5 * max(n_die - 3, 0)))
    ax = fig.add_axes([0.02, 0.0, 0.80, 1.0], projection="3d")
    ax.set_proj_type("ortho")           # orthographic: stacked layers and their labels stay aligned
    ax.computed_zorder = False          # painter's order = build order (bottom die first): mplot3d's depth sort hides
                                        # the rails of a die under its own textured surface
    zo = [0]
    def nz():
        zo[0] += 1; return zo[0]
    Tall = np.concatenate([tf.T_die[d.name].ravel() for d in spec.dies])
    tnorm = mpl.colors.Normalize(vmin=float(Tall.min()), vmax=float(Tall.max()))
    tn_norm = st.log_tnuc_norm()
    z = 0.0
    die_z = {}
    thick_draw = 0.03 * size        # exaggerated die thickness for visibility
    S = size * 1e3
    top = spec.dies[-1]; Tt = tf.T_die[top.name]
    r_, c_ = np.unravel_index(int(Tt.argmax()), Tt.shape)
    bw = size / top.nblk
    bi = min(int((c_ + 0.5) / Tt.shape[1] * top.nblk), top.nblk - 1); bj = min(int((Tt.shape[0] - r_ - 0.5) / Tt.shape[0] * top.nblk), top.nblk - 1)
    fx0, fy0 = bi * bw * 1e3, bj * bw * 1e3; fbw = bw * 1e3
    label_lines = {}; deferred = []          # 3-D-anchored texts are placed after the view and limits are final
    for k, d in enumerate(spec.dies):
        die_z[d.name] = z
        _box(ax, 0, 0, z * 1e3, S, S, thick_draw * 1e3, st.SILICON, alpha=0.95, edge="#333333", zorder=nz())
        _textured_top(ax, tf.T_die[d.name], size, z + thick_draw, tnorm, st.CMAP_T, zorder=nz())
        ztop = (z + thick_draw) * 1e3
        rl = rails[d.name]; tn = t_nuc[d.name]
        n_viol = int(np.sum(np.isfinite(tn) & (tn <= horizon_years * SEC_PER_YEAR)))
        t_min = float(np.min(tn)) if len(tn) else np.inf
        if show_rails and len(rl):
            colors = st.tnuc_colors(tn / SEC_PER_YEAR, tn_norm, immortal_color="#C8C8C8")
            segs_m, cols_m, segs_i = [], [], []
            for i in range(0, len(rl), rail_stride):
                r = rl[i]; xs, ys = r.node_xy()
                seg = [(xs[0] * 1e3, ys[0] * 1e3, ztop + 0.02 * S), (xs[-1] * 1e3, ys[-1] * 1e3, ztop + 0.02 * S)]
                if np.isfinite(tn[i]):
                    segs_m.append(seg); cols_m.append(colors[i])
                else:
                    segs_i.append(seg)
            if segs_i:      # immortal rails: faint, so the temperature texture stays readable
                ax.add_collection3d(Line3DCollection(segs_i, colors=[(1, 1, 1, 0.3)], linewidths=0.35, zorder=nz()))
            if segs_m:      # mortal rails: white halo under the coloured line
                ax.add_collection3d(Line3DCollection(segs_m, colors=[(1, 1, 1, 0.9)], linewidths=2.2, zorder=nz()))
                ax.add_collection3d(Line3DCollection(segs_m, colors=cols_m, linewidths=1.2, zorder=nz()))
            if np.isfinite(t_min):     # the earliest rail of the die: outlined in vermilion
                i0 = int(np.argmin(tn)); r = rl[i0]; xs, ys = r.node_xy()
                eseg = [[(xs[0] * 1e3, ys[0] * 1e3, ztop + 0.04 * S), (xs[-1] * 1e3, ys[-1] * 1e3, ztop + 0.04 * S)]]
                ax.add_collection3d(Line3DCollection(eseg, colors=[st.OKABE_ITO["vermilion"]], linewidths=2.8, capstyle="round", zorder=nz()))
                ax.add_collection3d(Line3DCollection(eseg, colors=["white"], linewidths=0.8, capstyle="round", zorder=nz()))
            if highlight and d.name in highlight:
                for i in highlight[d.name]:
                    r = rl[i]; xs, ys = r.node_xy()
                    ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], [ztop + 0.02 * S] * 2, color="#D55E00", lw=2.2, zorder=nz())
        # hot-block footprint of the top die, projected onto every die (white dash over a dark shadow), above the rails
        fx = np.array([fx0, fx0 + fbw, fx0 + fbw, fx0, fx0]); fy = np.array([fy0, fy0, fy0 + fbw, fy0 + fbw, fy0])
        fseg = [list(zip(fx, fy, np.full(5, ztop + 0.05 * S)))]
        ax.add_collection3d(Line3DCollection(fseg, colors=[(0, 0, 0, 0.6)], linewidths=1.8, zorder=nz()))
        ax.add_collection3d(Line3DCollection(fseg, colors=["white"], linewidths=0.9, linestyles=(0, (3, 2)), zorder=nz()))
        role = (die_roles or {}).get(d.name, "")
        head = f"{d.name}" + (f" ({role})" if role else "") + f"  {d.total_power():.0f} W"
        body = (f"earliest {t_min / SEC_PER_YEAR:.1f} yr, {n_viol} rail{'s' if n_viol != 1 else ''} < {horizon_years:.0f} yr"
                if np.isfinite(t_min) else f"immortal (> {tn_norm.vmax:.0f} yr)")
        label_lines[d.name] = (head, body, ztop)
        # vertical feeds and the bond layer up to the die above
        if show_tsv and k < n_die - 1:
            d_up = spec.dies[k + 1]; step = d_up.feed_pitch
            xs = np.arange(step / 2, size, step); ys = np.arange(step / 2, size, step)
            X, Y = np.meshgrid(xs, ys)
            z0 = ztop; z1 = (z + gap) * 1e3
            if d_up.feed_kind == "tsv":
                segs = [[(x * 1e3, y * 1e3, z0), (x * 1e3, y * 1e3, z1)] for x, y in zip(X.ravel(), Y.ravel())]
                ax.add_collection3d(Line3DCollection(segs, colors=[st.COPPER], linewidths=1.2, alpha=0.85, zorder=nz()))
            else:      # bond-pad array: drawn on the lower die only (points under the upper die would be sorted on top of it)
                ax.scatter(X.ravel() * 1e3, Y.ravel() * 1e3, np.full(X.size, z0 + 0.03 * S), s=2.0, color=st.COPPER, alpha=0.7, depthshade=False, lw=0, zorder=nz())
            zb = (z + 0.55 * gap) * 1e3
            ax.add_collection3d(Poly3DCollection([[(0, 0, zb), (S, 0, zb), (S, S, zb), (0, S, zb)]],
                                                 facecolors=st.BOND, edgecolors="#B8A67A", linewidths=0.4, alpha=0.18, zorder=nz()))
            deferred.append((S, S, zb, f"  {'hybrid bond' if d_up.feed_kind == 'pad' else 'bond + TSV'} {spec.bond_thick * 1e6:.0f} \u00b5m" if hasattr(spec, "bond_thick") else "  bond layer", "#7A6A45"))
        z += gap
    # heat-spreader / sink: an outline just above the top die (a filled plate would occlude it)
    zs = (z - 0.6 * gap) * 1e3
    ax.add_collection3d(Line3DCollection([[(0, 0, zs), (S, 0, zs), (S, S, zs), (0, S, zs), (0, 0, zs)]], colors=["#7F8B95"], linewidths=0.9, zorder=nz()))
    deferred.append((S, S, zs, "  TIM + heat spreader / sink", "#555555"))
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_zlim(0, (z - 0.35 * gap) * 1e3)
    ax.set_box_aspect((1, 1, 0.55 + 0.12 * n_die))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (mm)", labelpad=-9); ax.set_ylabel("y (mm)", labelpad=-9); ax.set_zlabel("")
    ax.set_xticks([0, S]); ax.set_yticks([0, S]); ax.set_zticks([]); ax.grid(False)
    ax.tick_params(pad=-4, labelsize=6.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    for (xx, yy, zz, txt, col) in deferred:
        _label3d(ax, xx, yy, zz, txt, fontsize=5.5, color=col, ha="left", va="center", zorder=1000)
    for d in spec.dies:                       # two-line die labels at the near-left corner
        head, body, ztop = label_lines[d.name]
        from mpl_toolkits.mplot3d import proj3d
        xs2, ys2, _ = proj3d.proj_transform(0.0, 0.0, ztop, ax.get_proj())
        ax.annotate(head + "\n" + body, (xs2, ys2), xycoords="data", xytext=(-6, 0), textcoords="offset points",
                    fontsize=6.3, color="#222222", ha="right", va="center", linespacing=1.15, zorder=1000)
    sm1 = plt.cm.ScalarMappable(norm=tnorm, cmap=st.CMAP_T); sm1.set_array([])
    cax1 = fig.add_axes([0.845, 0.56, 0.018, 0.34]); cb1 = fig.colorbar(sm1, cax=cax1); cb1.set_label("die temperature (K)", fontsize=7)
    cb1.ax.tick_params(labelsize=6.5)
    if show_rails:
        sm2 = plt.cm.ScalarMappable(norm=tn_norm, cmap=st.CMAP_TNUC); sm2.set_array([])
        cax2 = fig.add_axes([0.845, 0.10, 0.018, 0.34]); cb2 = fig.colorbar(sm2, cax=cax2)
        cb2.set_label("rail nucleation time (years)", fontsize=7); cb2.set_ticks([0.1, 1, 10, 100]); cb2.set_ticklabels(["0.1", "1", "10", "100"]); cb2.ax.tick_params(labelsize=6.5)
        cb2.ax.axhline(horizon_years, color="white", lw=1.2); cb2.ax.axhline(horizon_years, color=st.OKABE_ITO["vermilion"], lw=0.7)
        cb2.ax.text(-0.6, horizon_years, f"{horizon_years:.0f}-yr\nhorizon", transform=cb2.ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", color=st.OKABE_ITO["vermilion"])
    if title:
        ax.set_title(title, pad=2)
    st.save(fig, path_base, data=dict(**{f"T_{d.name}": tf.T_die[d.name] for d in spec.dies},
                                      **{f"t_nuc_s_{d.name}": t_nuc[d.name] for d in spec.dies},
                                      **{f"rail_xy_{d.name}": np.array([np.r_[r.node_xy()[0][[0, -1]], r.node_xy()[1][[0, -1]]] for r in rails[d.name]]) for d in spec.dies},
                                      die_size_m=size, hot_footprint_mm=[fx0, fy0, fbw]), panels=False)


def plot_die_maps(spec: StackSpec, tf: ThermalField, rails: Dict[str, List[Rail]], t_nuc: Dict[str, np.ndarray],
                  nuc_xy: Dict[str, np.ndarray], path_base: str, variant_label: str = "", T_alone: bool = False, horizon_years: float = 10.0):
    """Row of per-die panels: temperature map + rails coloured by t_nuc + nucleation sites."""
    st.use_paper_style()
    n = len(spec.dies)
    fig, axes = plt.subplots(1, n, figsize=(st.IEEE_2COL, min(2.8, 0.62 * st.IEEE_2COL / n + 0.62)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    Tall = np.concatenate([(tf.T_alone if T_alone else tf.T_die)[d.name].ravel() for d in spec.dies])
    tnorm = mpl.colors.Normalize(vmin=float(Tall.min()), vmax=float(Tall.max())); tn_norm = st.log_tnuc_norm()
    size = spec.size * 1e3
    for ax, d in zip(axes, spec.dies):
        Tg = (tf.T_alone if T_alone else tf.T_die)[d.name]
        ax.imshow(Tg, extent=[0, size, 0, size], origin="upper", cmap=st.CMAP_T, norm=tnorm, interpolation="bilinear")
        cols = st.tnuc_colors(t_nuc[d.name] / SEC_PER_YEAR, tn_norm, immortal_color="#FFFFFF")
        for r, c, t in zip(rails[d.name], cols, t_nuc[d.name]):
            xs, ys = r.node_xy()
            if np.isfinite(t) and t <= horizon_years * SEC_PER_YEAR:          # violating rails: halo + colour
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color="white", lw=1.5, alpha=0.85, solid_capstyle="butt")
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=c, lw=0.8, alpha=1.0, solid_capstyle="butt")
            elif np.isfinite(t):
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=c, lw=0.45, alpha=0.8)
            else:
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color="white", lw=0.25, alpha=0.2)
        if d.name in nuc_xy:
            pts = nuc_xy[d.name]; fin = np.isfinite(t_nuc[d.name])
            if fin.any():
                ax.scatter(pts[fin, 0] * 1e3, pts[fin, 1] * 1e3, s=6, facecolor="none", edgecolor="white", linewidths=0.5)
        ax.set_title(f"{d.name} ({d.total_power():.0f} W)  {_fmt_earliest(float(np.nanmin(t_nuc[d.name])))}", fontsize=7)
        ax.set_xlabel("x (mm)"); ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axes[0].set_ylabel("y (mm)")
    sm1 = plt.cm.ScalarMappable(norm=tnorm, cmap=st.CMAP_T); sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(norm=tn_norm, cmap=st.CMAP_TNUC); sm2.set_array([])
    fig.colorbar(sm1, ax=axes.tolist(), shrink=0.8, pad=0.01, label="die temperature (K)")
    fig.colorbar(sm2, ax=axes.tolist(), shrink=0.8, pad=0.04, label="rail $t_{nuc}$ (years)")
    if variant_label:
        fig.suptitle(variant_label, fontsize=8)
    st.save(fig, path_base, data={**{f"T_{d.name}": (tf.T_alone if T_alone else tf.T_die)[d.name] for d in spec.dies},
                                  **{f"t_nuc_s_{d.name}": t_nuc[d.name] for d in spec.dies}})


def plot_stress_surface(x_m: np.ndarray, times: np.ndarray, sigma: np.ndarray, sigma_crit: float, path_base: str,
                        title: str = "", t_nuc: float | None = None, x_nuc: float | None = None, feeds_x_m: np.ndarray | None = None,
                        j_segments: np.ndarray | None = None, seg_L_m: np.ndarray | None = None):
    """The stress history of one rail, read two ways: (a) sigma(x) at a few times up to and past nucleation, with the
    critical stress and (if given) the segment currents as a bar strip under the axis; (b) the (x, t) map with the
    sigma = sigma_crit contour and the nucleation point.  Both use the same diverging scale, saturated at
    +-1.2 sigma_crit so that the crossing is where the colour turns dark."""
    st.use_paper_style()
    x_m = np.asarray(x_m, float); sigma = np.asarray(sigma, float)
    xu, first = np.unique(np.round(x_m, 12), return_index=True)          # drop duplicated junction nodes
    x_m, sigma = xu, sigma[:, first]
    S = sigma / 1e6; sc = sigma_crit / 1e6; ty = times / SEC_PER_YEAR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(st.IEEE_2COL, 2.7), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.05, 1]))
    # (a) profiles at a few times: 0.1 yr, 1 yr, t_nuc (if any), 10 yr, 100 yr
    want = [0.1, 1.0, 10.0] if (t_nuc is not None and np.isfinite(t_nuc)) else [0.1, 1.0, 10.0, 100.0]
    if t_nuc is not None and np.isfinite(t_nuc):
        want.append(t_nuc / SEC_PER_YEAR)
    want = sorted(set(w for w in want if ty[0] <= w <= ty[-1]))
    cmap_t = plt.get_cmap("viridis")
    for i, w in enumerate(want):
        k = int(np.argmin(np.abs(np.log(ty) - np.log(w)))); c = cmap_t(0.1 + 0.8 * i / max(len(want) - 1, 1))
        is_nuc = t_nuc is not None and np.isfinite(t_nuc) and abs(w - t_nuc / SEC_PER_YEAR) < 1e-9
        ax1.plot(x_m * 1e3, S[k], color=c, lw=1.6 if is_nuc else 1.0, label=(f"t = {w:.2g} yr" + (" (nucleation)" if is_nuc else "")))
    ax1.axhline(sc, color="k", lw=0.7, ls="--"); ax1.text(0.99, sc, r"$\sigma_{crit}$", transform=mpl.transforms.blended_transform_factory(ax1.transAxes, ax1.transData), fontsize=6.5, ha="right", va="bottom")
    ax1.axhline(0, color="#BBBBBB", lw=0.5)
    if feeds_x_m is not None:
        for xf in np.asarray(feeds_x_m) * 1e3:
            ax1.axvline(xf, color="#DDDDDD", lw=0.4, zorder=0)
    if x_nuc is not None and t_nuc is not None and np.isfinite(t_nuc):
        ax1.plot([x_nuc * 1e3], [sc], marker="o", ms=4, color="k", zorder=5)
    ax1.set_xlabel("x along the rail (mm)"); ax1.set_ylabel(r"hydrostatic stress $\sigma$ (MPa)")
    ax1.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.0)
    ax1.set_title("(a) stress profiles (thin grey lines: feeds)", fontsize=7)
    # (b) (x, t) map, saturated at 1.2 sigma_crit
    vmax = 1.2 * sc; norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax2.pcolormesh(x_m * 1e3, ty, S, cmap=st.CMAP_STRESS, norm=norm, shading="auto", rasterized=True)
    ax2.set_yscale("log"); ax2.set_ylim(max(ty[0], 1e-2), ty[-1])
    ax2.contour(x_m * 1e3, ty, S, levels=[sc], colors="k", linewidths=0.8)
    ax2.set_xlabel("x along the rail (mm)"); ax2.set_ylabel("time (years)")
    if t_nuc is not None and x_nuc is not None and np.isfinite(t_nuc):
        ax2.plot([x_nuc * 1e3], [t_nuc / SEC_PER_YEAR], "o", color="k", ms=3.5, mfc="white", mew=0.8)
        import matplotlib.patheffects as pe
        ax2.annotate(f"nucleation {t_nuc/SEC_PER_YEAR:.2f} yr", (x_nuc * 1e3, t_nuc / SEC_PER_YEAR), textcoords="offset points",
                     xytext=(-8, -14) if x_nuc * 1e3 > 0.5 * x_m.max() * 1e3 else (8, -14), fontsize=6.3, ha="right" if x_nuc * 1e3 > 0.5 * x_m.max() * 1e3 else "left",
                     arrowprops=dict(arrowstyle="-", color="k", lw=0.5), path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    cb = fig.colorbar(im, ax=ax2, pad=0.02, extend="both"); cb.set_label(r"$\sigma$ (MPa), black line: $\sigma = \sigma_{crit}$", fontsize=6.5)
    ax2.set_title("(b) stress map, all times", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=8)
    st.save(fig, path_base, data=dict(x_m=x_m, times_s=times, sigma_Pa=sigma, sigma_crit_Pa=sigma_crit, t_nuc_s=t_nuc if t_nuc is not None else np.nan,
                                      x_nuc_m=x_nuc if x_nuc is not None else np.nan), panels=True)


# ----------------------------------------------------------------------------
# thermal coupling: vertical cut through the stack
# ----------------------------------------------------------------------------
def plot_thermal_slice(spec: StackSpec, tf: ThermalField, path_base: str, y_cut: float | None = None):
    """(a) layer-resolved temperature along a horizontal cut through the top-die hot spot (every HotSpot
    layer: dies, bond layers, TIM; layer heights are schematic) and (b) T(x) of every die along the same
    cut, in the stack (solid) vs standalone (dashed): the die-to-die thermal coupling of the paper."""
    st.use_paper_style()
    size = spec.size; top = spec.dies[-1]
    Tt = tf.T_die[top.name]; rows, cols = Tt.shape
    if y_cut is None:
        r_, _ = np.unravel_index(int(Tt.argmax()), Tt.shape)
        y_cut = size * (rows - r_ - 0.5) / rows
    row = int(np.clip(round(rows - y_cut / size * rows - 0.5), 0, rows - 1))
    names = []; lines = []
    for k, d in enumerate(spec.dies):
        names.append(f"{d.name} (Si)"); lines.append(tf.all_layers[2 * k][row])
        names.append("hybrid bond" if k < len(spec.dies) - 1 else "TIM"); lines.append(tf.all_layers[2 * k + 1][row])
    img = np.array(lines)                                                    # (layers, cols), bottom layer first
    x = (np.arange(cols) + 0.5) / cols * size * 1e3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(st.IEEE_2COL, 2.75), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
    vmin, vmax = float(img.min()), float(img.max())          # the slice's own range (the standalone curves live in panel b)
    im = ax1.pcolormesh(np.linspace(0, size * 1e3, cols + 1), np.arange(len(names) + 1), img, cmap=st.CMAP_T, vmin=vmin, vmax=vmax, shading="flat")
    ax1.set_yticks(np.arange(len(names)) + 0.5); ax1.set_yticklabels(names, fontsize=6.5)
    ax1.set_xlabel("x along the cut (mm)"); ax1.set_title(f"(a) vertical cut at y = {y_cut*1e3:.2f} mm through the {top.name} hot spot\n(layer heights not to scale)", fontsize=7)
    for yy in range(1, len(names)):
        ax1.axhline(yy, color="white", lw=0.4, alpha=0.6)
    fig.colorbar(im, ax=ax1, shrink=0.9, pad=0.02, label="temperature in the stack (K)")
    # hot-block footprint of the top die along x
    if top.power is not None and top.power.max() > 1.5 * top.power.mean():
        bw = size / top.nblk; hot = np.argwhere(top.power > 1.5 * top.power.mean())
        for (i, j) in hot:
            if abs((j + 0.5) * bw - y_cut) <= 0.5 * bw:
                for ax in (ax1, ax2):
                    ax.axvspan(i * bw * 1e3, (i + 1) * bw * 1e3, color="#D55E00", alpha=0.10, lw=0)
    data = dict(x_mm=x, y_cut_mm=y_cut * 1e3, layer_names=np.array(names), slice_T=img)
    xa = 0.93 * size * 1e3; ia = int(np.argmin(np.abs(x - xa)))
    import matplotlib.patheffects as pe
    for k, d in reversed(list(enumerate(spec.dies))):        # top die first: its arrow sits nearest the edge
        c = list(st.OKABE_ITO.values())[k]
        ax2.plot(x, tf.T_die[d.name][row], color=c, lw=1.2, label=f"{d.name} in stack"); data[f"T_stack_{d.name}"] = tf.T_die[d.name][row]
        if d.name in tf.T_alone:
            ax2.plot(x, tf.T_alone[d.name][row], color=c, lw=1.0, ls="--", label=f"{d.name} standalone"); data[f"T_alone_{d.name}"] = tf.T_alone[d.name][row]
            dT = float(tf.T_die[d.name][row][ia] - tf.T_alone[d.name][row][ia])
            if abs(dT) >= 3.0:            # the die-to-die heating, written on the figure
                ax2.annotate("", (xa, tf.T_die[d.name][row][ia]), (xa, tf.T_alone[d.name][row][ia]),
                             arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
                ax2.text(xa - 0.015 * size * 1e3, 0.5 * (tf.T_die[d.name][row][ia] + tf.T_alone[d.name][row][ia]), f"+{dT:.0f} K", fontsize=6, color=c, va="center", ha="right",
                         path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                xa -= 0.13 * size * 1e3; ia = int(np.argmin(np.abs(x - xa)))
    ax2.set_xlabel("x along the cut (mm)"); ax2.set_ylabel("die temperature (K)")
    ax2.legend(fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), borderaxespad=0.0, frameon=False)
    ax2.set_title("(b) along the cut: in the stack (solid) vs standalone (dashed)", fontsize=7)
    st.save(fig, path_base, data=data)


# ----------------------------------------------------------------------------
# hotspot-position sweep (cross-die alignment)
# ----------------------------------------------------------------------------
def plot_hotspot_alignment(entries: List[Dict], dies_below: Sequence[str], top_name: str, path_base: str, horizon_years: float = 10.0):
    """entries: one dict per hot-block position with keys
         pos (i,j), T (grid of the die directly under the top die), size (m), rails (list[Rail]), t_nuc (array),
         hot_rect (x0, y0, w) in m, metrics {die: {earliest_yr, crossing_enrichment_10yr, mortal10}}.
    Top row: the die under the top die for every position; bottom row: earliest t_nuc and crossing enrichment
    of every lower die vs hot-block position."""
    st.use_paper_style()
    n = len(entries); under = dies_below[-1]
    fig = plt.figure(figsize=(st.IEEE_2COL, 3.3))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.28, left=0.075, right=0.985, top=0.89, bottom=0.12)
    Tall = np.concatenate([e["T"].ravel() for e in entries]); tnorm = mpl.colors.Normalize(float(Tall.min()), float(Tall.max()))
    tn_norm = st.log_tnuc_norm()
    axm = []
    for k, e in enumerate(entries):
        ax = fig.add_subplot(gs[0, k]); axm.append(ax); size = e["size"] * 1e3
        ax.imshow(e["T"], extent=[0, size, 0, size], origin="upper", cmap=st.CMAP_T, norm=tnorm, interpolation="bilinear")
        cols = st.tnuc_colors(e["t_nuc"] / SEC_PER_YEAR, tn_norm)
        for r, c, t in zip(e["rails"], cols, e["t_nuc"]):
            xs, ys = r.node_xy()
            if np.isfinite(t) and t <= horizon_years * SEC_PER_YEAR:       # violating rails: halo + colour
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color="white", lw=1.6, alpha=0.9, solid_capstyle="butt")
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=c, lw=0.9, alpha=1.0, solid_capstyle="butt")
            elif np.isfinite(t):
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=c, lw=0.35, alpha=0.6)
        x0, y0, w = [v * 1e3 for v in e["hot_rect"]]
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + w, y0 + w, y0], color="black", lw=1.4, alpha=0.5)
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + w, y0 + w, y0], color="white", lw=0.8, ls=(0, (3, 2)))
        m = e["metrics"][under]
        ax.set_title(f"{top_name} hot block {tuple(e['pos'])}\n{under}: {m['mortal10']} mortal, enr. {m['crossing_enrichment_10yr']:.1f}", fontsize=6.5)
        ax.set_xticks([0, size]); ax.set_yticks([0, size] if k == 0 else []); ax.tick_params(labelsize=6)
        if k == 0: ax.set_ylabel("y (mm)")
        ax.set_xlabel("x (mm)", labelpad=1)
    sm = plt.cm.ScalarMappable(norm=tn_norm, cmap=st.CMAP_TNUC); sm.set_array([])
    cb = fig.colorbar(sm, ax=axm, shrink=0.75, pad=0.01, aspect=18); cb.set_label("rail $t_{nuc}$ (years)", fontsize=7); cb.ax.tick_params(labelsize=6)
    # bottom row: two wide panels
    half = max(n // 2, 1)
    gsb = gs[1, :].subgridspec(1, 2, wspace=0.32)
    axa = fig.add_subplot(gsb[0, 0]); axb = fig.add_subplot(gsb[0, 1])
    xs = np.arange(n); wbar = 0.8 / len(dies_below); labels = [str(tuple(e["pos"])) for e in entries]
    for k, dn in enumerate(dies_below):
        c = list(st.OKABE_ITO.values())[[d for d in dies_below].index(dn)]
        axa.bar(xs + (k - (len(dies_below) - 1) / 2) * wbar, [e["metrics"][dn]["earliest_yr"] for e in entries], wbar * 0.92, color=c, label=dn)
        axb.bar(xs + (k - (len(dies_below) - 1) / 2) * wbar, [e["metrics"][dn]["crossing_enrichment_10yr"] for e in entries], wbar * 0.92, color=c)
    axa.axhline(horizon_years, color="k", lw=0.6, ls=":"); axa.set_yscale("log"); axa.set_ylabel("earliest $t_{nuc}$ (yr)")
    axa.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 3.0), numticks=10))
    axa.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}")); axa.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axa.set_xticks(xs); axa.set_xticklabels(labels, fontsize=6.5); axa.set_xlabel(f"{top_name} hot-block position (i, j)"); axa.legend(fontsize=6, ncol=len(dies_below), loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
    axb.axhline(1.0, color="k", lw=0.6, ls=":"); axb.set_ylabel(f"crossing enrichment ({horizon_years:g} yr)", labelpad=2)
    axb.set_xticks(xs); axb.set_xticklabels(labels, fontsize=6.5); axb.set_xlabel(f"{top_name} hot-block position (i, j)")
    axb.margins(y=0.35)
    axb.text(0.98, 0.97, "> 1: the hot footprint\nselects the failing rails", transform=axb.transAxes, ha="right", va="top", fontsize=6, color="#444444")
    st.save(fig, path_base, data={f"{k}_{dn}": np.array([e["metrics"][dn][k] for e in entries]) for dn in dies_below for k in ("earliest_yr", "crossing_enrichment_10yr", "mortal10")}
             | {"positions": np.array([e["pos"] for e in entries])})


# ----------------------------------------------------------------------------
# sign-off view: stress margin maps
# ----------------------------------------------------------------------------
def plot_margin_maps(spec: StackSpec, rails: Dict[str, List[Rail]], margin: Dict[str, np.ndarray], path_base: str,
                     horizon_years: float = 10.0, variant_label: str = ""):
    """Per-die maps of the stress margin  m = 1 - max sigma(t <= horizon) / sigma_crit  of every rail
    (negative = violates the sign-off within the horizon).  Diverging colour map centred at zero."""
    st.use_paper_style()
    n = len(spec.dies)
    fig, axes = plt.subplots(1, n, figsize=(st.IEEE_2COL, min(2.8, 0.62 * st.IEEE_2COL / n + 0.62)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    allm = np.concatenate([np.asarray(margin[d.name], float) for d in spec.dies]); allm = allm[np.isfinite(allm)]
    lim = float(max(0.05, np.abs(allm).max())) if allm.size else 1.0
    norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim); cmap = plt.get_cmap("RdBu")
    size = spec.size * 1e3
    for ax, d in zip(axes, spec.dies):
        ax.set_facecolor("#F4F4F4"); mg = np.asarray(margin[d.name], float)
        for r, m in zip(rails[d.name], mg):
            xs, ys = r.node_xy()
            if not np.isfinite(m):
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color="#BBBBBB", lw=0.3, alpha=0.6)
            else:
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=cmap(norm(m)), lw=1.2 if m < 0 else 0.6)
        nviol = int(np.sum(mg < 0)); mn = float(np.nanmin(mg)) if np.isfinite(mg).any() else float("nan")
        ax.set_title(f"{d.name}: {nviol} rails violate, min margin {mn:+.2f}", fontsize=7)
        ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal"); ax.set_xlabel("x (mm)")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axes[0].set_ylabel("y (mm)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02, label=f"stress margin at {horizon_years:g} yr,  $1-\\max\\sigma/\\sigma_{{crit}}$")
    if variant_label:
        fig.suptitle(variant_label, fontsize=8)
    st.save(fig, path_base)
