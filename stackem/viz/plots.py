"""
stackem.viz.plots
===================

Two-dimensional result figures of the paper, all in the shared style:

    plot_cdf_variants     per-die cumulative nucleation curves of the physics variants (E3 main figure)
    plot_parity           StackEM vs FDM nucleation-time parity with density colouring (E2)
    plot_kernels          learned vs FDM kernel surfaces and error map (E1)
    plot_error_by_tau     kernel error per tau decade (E1)
    plot_ranking_hits     top-k hit rate / Kendall tau of the simplified analyses (E7)
    plot_timing           wall-clock vs number of rails, log-log (E4)
    plot_sensitivity_map  d log t_nuc / d P_block drawn on every die's floorplan (E5)
    plot_budget_trajectory  power re-allocation case study (E5)
    plot_blech_screen     Blech ratio bars of the vertical elements (BIS)
    plot_effect_ratios    effect-ratio dot plot summarising E3
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..constants import SEC_PER_YEAR
from . import style as st


def plot_cdf_variants(tnuc_by_variant: Dict[str, Dict[str, np.ndarray]], dies: Sequence[str], path_base: str,
                      overlay: Dict[str, np.ndarray] | None = None, t_range=(0.02, 300.0), horizon_lines=(10.0,),
                      series: Sequence[str] | None = None, die_titles: Dict[str, str] | None = None, annotate=("alone", "uniform_signoff", "full")):
    """Cumulative nucleation curves per die.  tnuc_by_variant[variant][die] -> t_nuc (s); overlay[die] -> the surrogate's
    prediction for 'full'.  `series` restricts the variants drawn (paper version: the two practices, the resolved stack
    and the surrogate); the count of rails nucleated at the first horizon is written next to the annotated variants."""
    st.use_paper_style()
    n = len(dies); h0 = horizon_lines[0] if horizon_lines else None
    fig, axes = plt.subplots(1, n, figsize=(st.IEEE_2COL, 2.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    grid = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), 400)
    shown = [v for v in tnuc_by_variant if series is None or v in series]
    data = {}
    for ax, d in zip(axes, dies):
        if h0 is not None:
            ax.axvspan(t_range[0], h0, color="#F3F3F3", lw=0, zorder=0)
        for v in shown:
            t = tnuc_by_variant[v][d] / SEC_PER_YEAR; data[f"{v}_{d}_years"] = t
            cnt = np.array([(t <= g).sum() for g in grid])
            ax.plot(grid, cnt, color=st.VARIANT_COLOR.get(v, "k"), lw=1.4 if v == "full" else 1.0,
                    ls="-" if v in ("full", "alone") else "--", label=st.VARIANT_LABEL.get(v, v), zorder=3 if v == "full" else 2)
        if overlay is not None and d in overlay:
            t = overlay[d] / SEC_PER_YEAR; data[f"stackem_{d}_years"] = t
            cnt = [(t <= g).sum() for g in grid]
            ax.plot(grid, cnt, color=st.VARIANT_COLOR["stackem"], lw=1.1, ls=":", label=st.VARIANT_LABEL["stackem"], zorder=4)
        for h in horizon_lines:
            ax.axvline(h, color="#999999", lw=0.6, ls=":")
        if h0 is not None:      # counts at the horizon, one per annotated variant, stacked so they never overlap
            vals = [(v, int((tnuc_by_variant[v][d] / SEC_PER_YEAR <= h0).sum())) for v in shown if v in annotate]
            ymax = max([len(tnuc_by_variant[v][d]) for v in shown] + [1])
            last_y = -1e9
            for i, (v, c) in enumerate(sorted(vals, key=lambda x: x[1])):
                y = c if c > 0 else 0
                import matplotlib.patheffects as pe
                ax.annotate(f"{c}", (h0, y), xytext=(4, 4 if c == 0 else 0), textcoords="offset points", fontsize=6, color=st.VARIANT_COLOR.get(v, "k"),
                            va="center" if c else "bottom", ha="left", fontweight="bold" if v == "full" else "normal",
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=6)
        ax.set_xscale("log"); ax.set_xlim(*t_range); ax.set_xlabel("time (years)")
        ax.set_title((die_titles or {}).get(d, d), fontsize=8)
    ymax = max(len(tnuc_by_variant[v][d]) for v in shown for d in dies)
    axes[0].set_ylim(0, ymax * 1.06)
    if h0 is not None:
        axes[-1].text(h0, ymax * 1.03, f" {h0:g}-yr horizon", fontsize=6, color="#666666", va="top", ha="left")
    axes[0].set_ylabel("rails nucleated")
    h, l = axes[-1].get_legend_handles_labels()
    ncol = 2 if len(l) <= 4 else (3 if len(l) <= 6 else 4)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02 - 0.065 * int(np.ceil(len(l) / ncol))), fontsize=6.2, frameon=False)
    st.save(fig, path_base, data=data)


def plot_parity(t_truth: np.ndarray, t_pred: np.ndarray, path_base: str, labels: np.ndarray | None = None,
                title: str = "", t_range=(0.02, 300.0), horizon_years: float | None = 10.0, band: float = 0.03,
                ref_label: str = "reference (finite-volume) $t_{nuc}$ (years)"):
    """Parity plot with a residual panel: (top) surrogate vs reference nucleation time on log axes with the +-band;
    (bottom) signed relative error vs the reference time, so the error structure (early vs late failures) is visible."""
    st.use_paper_style()
    fin = np.isfinite(t_truth) & np.isfinite(t_pred)
    x = t_truth[fin] / SEC_PER_YEAR; y = t_pred[fin] / SEC_PER_YEAR
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(st.IEEE_COL, st.IEEE_COL * 1.15), sharex=True, constrained_layout=True,
                                  gridspec_kw=dict(height_ratios=[3.2, 1.3]))
    lo, hi = t_range
    if fin.any():                                            # tighten the axes to the data (log decade margins)
        lo = max(lo, 10 ** np.floor(np.log10(min(x.min(), y.min())))); hi = min(hi, 10 ** np.ceil(np.log10(max(x.max(), y.max()))))
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, zorder=1)
    ax.fill_between([lo, hi], [lo * (1 - band), hi * (1 - band)], [lo * (1 + band), hi * (1 + band)], color="#DDDDDD", alpha=0.7, lw=0, label=f"$\\pm${band*100:g} %", zorder=0)
    rel = (y - x) / x
    cols = list(st.OKABE_ITO.values())
    if labels is not None:
        for k, lab in enumerate(np.unique(labels[fin])):
            m = labels[fin] == lab
            ax.scatter(x[m], y[m], s=8, alpha=0.8, label=str(lab), color=cols[k % 7], linewidths=0, zorder=3)
            axr.scatter(x[m], 100 * rel[m], s=6, alpha=0.8, color=cols[k % 7], linewidths=0, zorder=3)
        ax.legend(loc="upper left", fontsize=6.3, handletextpad=0.3, borderaxespad=0.3)
    else:
        ax.scatter(x, y, s=8, alpha=0.75, color=st.OKABE_ITO["blue"], linewidths=0, zorder=3)
        axr.scatter(x, 100 * rel, s=6, alpha=0.75, color=st.OKABE_ITO["blue"], linewidths=0, zorder=3)
    if horizon_years is not None and lo < horizon_years < hi:
        for a in (ax, axr):
            a.axvline(horizon_years, color="#999999", lw=0.6, ls=":", zorder=1)
        ax.axhline(horizon_years, color="#999999", lw=0.6, ls=":", zorder=1)
        ax.text(horizon_years * 0.93, hi * 0.9, f"{horizon_years:g}-yr horizon", fontsize=5.8, color="#666666", ha="right", va="top", rotation=90)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_ylabel(st.METHOD + " $t_{nuc}$ (years)")
    note = (f"n = {fin.sum()}\nmedian |err| = {np.median(np.abs(rel))*100:.2f} %\np90 |err| = {np.percentile(np.abs(rel), 90)*100:.2f} %" if fin.any()
            else "n = 0 (no rail nucleates in both solutions)")       # e.g. the few rails of a --smoke run
    ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3)
    axr.axhspan(-100 * band, 100 * band, color="#DDDDDD", alpha=0.7, lw=0, zorder=0); axr.axhline(0, color="#999999", lw=0.6, zorder=1)
    r_lim = min(max(100 * band * 1.6, float(np.abs(100 * rel).max()) * 1.15), 30.0) if fin.any() else 100 * band * 1.6
    axr.set_ylim(-r_lim, r_lim); axr.set_ylabel("error (%)"); axr.set_xlabel(ref_label)
    if title: ax.set_title(title, fontsize=8)
    st.save(fig, path_base, data=dict(t_truth_years=x, t_pred_years=y, labels=(labels[fin] if labels is not None else np.array([]))), panels=False)


def plot_kernels(xi: np.ndarray, tau: np.ndarray, truth: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], path_base: str,
                 profile_text: str = ""):
    """3 x 3 panel: truth / prediction / error of the sqrt(tau)-scaled kernels s_G, s_M, a of one held-out profile
    (the scaled quantities are what the network learns; unscaled, a(tau) ~ tau hides every early-time feature)."""
    st.use_paper_style()
    names = [("s_G", r"$s_G/\sqrt{\tau}$ (EM response)"), ("s_M", r"$s_M/\sqrt{\tau}$ (TM response)"), ("a", r"$a/\sqrt{\tau}$ (unit-flux Green)")]
    fig, axes = plt.subplots(3, 3, figsize=(st.IEEE_2COL, 5.2), constrained_layout=True)
    LT = np.log10(tau); sc = 1.0 / np.sqrt(tau)[:, None]
    for row, (k, lab) in enumerate(names):
        T, P = truth[k] * sc, pred[k] * sc
        vmax = np.abs(T).max()
        for col, (Z, ttl) in enumerate([(T, "reference FDM"), (P, "SKN"), (P - T, "SKN $-$ FDM")]):
            ax = axes[row, col]
            if col < 2:
                im = ax.pcolormesh(xi, LT, Z, cmap=st.CMAP_STRESS, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
            else:
                e = max(np.abs(Z).max(), 1e-12)
                im = ax.pcolormesh(xi, LT, Z, cmap="PuOr", vmin=-e, vmax=e, shading="auto", rasterized=True)
            cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02); cb.ax.tick_params(labelsize=6)
            if col == 2: cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()
            if row == 0: ax.set_title(ttl, fontsize=8)
            if col == 0: ax.set_ylabel(lab + "\n" + r"log$_{10}\tau$")
            if row == 2: ax.set_xlabel(r"$\xi = x/L$")
            ax.set_xticks([0, 0.5, 1])
    if profile_text:
        fig.suptitle(profile_text, fontsize=7)
    st.save(fig, path_base)


def plot_kernel_cuts(examples, path_base: str):
    """The quantities the closure actually uses: the sqrt(tau)-scaled kernels at the segment ends xi = 0 (solid) and
    xi = 1 (dashed) vs log tau for a few held-out profiles - reference FDM as lines, SKN as markers."""
    st.use_paper_style()
    names = [("s_G", r"$s_G/\sqrt{\tau}$"), ("s_M", r"$s_M/\sqrt{\tau}$"), ("a", r"$a/\sqrt{\tau}$")]
    fig, axes = plt.subplots(1, 3, figsize=(st.IEEE_2COL, 2.1), constrained_layout=True)
    for ax, (k, lab) in zip(axes, names):
        for p, (xi, tau, truth, pred, meta) in enumerate(examples[:3]):
            c = list(st.OKABE_ITO.values())[p]; LT = np.log10(tau); sc = 1.0 / np.sqrt(tau)
            for j, (col_idx, ls) in enumerate([(0, "-"), (-1, "--")]):
                ax.plot(LT, truth[k][:, col_idx] * sc, color=c, ls=ls, lw=1.0, label=(f"profile {p+1}: $T$ {meta[0]:.0f}/{meta[1]:.0f} K, $\\lambda$ = {meta[7]:.1f}" if j == 0 else None))
                ax.plot(LT[::4], (pred[k][:, col_idx] * sc)[::4], color=c, ls="none", marker="o" if j == 0 else "s", ms=2.2, mfc="none", mew=0.6)
        ax.set_xlabel(r"log$_{10}\tau$"); ax.set_ylabel(lab); ax.set_title(lab + " at both ends", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="#444444", ls="-", lw=1.0), Line2D([], [], color="#444444", ls="--", lw=1.0),
          Line2D([], [], color="#444444", ls="none", marker="o", ms=2.4, mfc="none", mew=0.6)]
    l += [r"$\xi = 0$ (reference)", r"$\xi = 1$ (reference)", "SKN"]
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=5.8, frameon=False, columnspacing=1.2)
    st.save(fig, path_base, data={f"profile{p+1}_{k}_{'xi0' if j == 0 else 'xi1'}_{who}": arr
                                  for p, (xi, tau, truth, pred, meta) in enumerate(examples[:3]) for k in ("s_G", "s_M", "a")
                                  for j, ci in enumerate((0, -1)) for who, arr in (("ref", truth[k][:, ci]), ("skn", pred[k][:, ci]))}
             | {f"profile{p+1}_log10tau": np.log10(tau) for p, (xi, tau, *_r) in enumerate(examples[:3])})


def plot_error_by_tau(tau_edges: np.ndarray, err: Dict[str, np.ndarray], path_base: str, ylabel="relative error"):
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.2))
    x = 0.5 * (np.log10(tau_edges[1:]) + np.log10(tau_edges[:-1]))
    for k, (name, e) in enumerate(err.items()):
        ax.plot(x, e, marker="o", ms=3, lw=1, color=list(st.OKABE_ITO.values())[k], label=name)
    ax.set_yscale("log"); ax.set_xlabel(r"log$_{10}\tau$ bin"); ax.set_ylabel(ylabel); ax.legend()
    st.save(fig, path_base)


def plot_error_by_tau_multi(tau_edges: np.ndarray, err_by_prov: Dict[str, Dict[str, np.ndarray]], path_base: str):
    """One panel per kernel, one line per kernel provider (full-size teacher SKN vs distilled student vs student + tabulated look-up)."""
    st.use_paper_style()
    kernels = list(next(iter(err_by_prov.values())))
    fig, axes = plt.subplots(1, len(kernels), figsize=(st.IEEE_2COL, 2.1), constrained_layout=True, sharey=True)
    x = 0.5 * (np.log10(tau_edges[1:]) + np.log10(tau_edges[:-1]))
    styles = [("k", "-", "s"), (st.OKABE_ITO["green"], "-", "D"), (st.OKABE_ITO["vermilion"], "--", "o"), (st.OKABE_ITO["blue"], ":", "^")]
    for ax, k in zip(np.atleast_1d(axes), kernels):
        for (label, e), (c, ls, mk) in zip(err_by_prov.items(), styles):
            ax.plot(x, e[k], marker=mk, ms=2.8, lw=1, color=c, ls=ls, label=label)
        ax.set_yscale("log"); ax.set_xlabel(r"log$_{10}\tau$ bin"); ax.set_title({"s_G": "$s_G$", "s_M": "$s_M$", "a": "$a$"}.get(k, k))
    np.atleast_1d(axes)[0].set_ylabel("relative error vs FDM"); np.atleast_1d(axes)[0].legend(fontsize=5.6)
    st.save(fig, path_base)


def plot_ranking_hits(results: Dict[str, Dict[str, float]], path_base: str, metrics=("top10", "top50", "kendall", "mortal10_jaccard")):
    """results[variant][metric] -> value. One bar panel per metric, variant labels only on the first panel."""
    st.use_paper_style()
    titles = {"top10": "Top-10 hit rate", "top50": "Top-50 hit rate", "kendall": r"Kendall $\tau$ (Top-50)", "mortal10_jaccard": "mortal set (10 yr)\nJaccard"}
    vs = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(st.IEEE_2COL, 0.3 * len(vs) + 0.7), constrained_layout=True, sharey=True,
                             gridspec_kw=dict(width_ratios=[1.0] * len(metrics)))
    for k, (ax, m) in enumerate(zip(np.atleast_1d(axes), metrics)):
        vals = [results[v].get(m, np.nan) for v in vs]
        ax.barh(range(len(vs)), vals, color=[st.VARIANT_COLOR.get(v, "k") for v in vs], height=0.62)
        ax.set_yticks(range(len(vs)))
        if k == 0:
            ax.set_yticklabels([st.VARIANT_LABEL.get(v, v) for v in vs], fontsize=6.3)
        ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1.0]); ax.set_title(titles.get(m, m), fontsize=7)
        ax.axvline(1.0, color="#BBBBBB", lw=0.5, ls=":")
        for i, val in enumerate(vals):
            ax.text(val + 0.03, i, f"{val:.2f}", va="center", fontsize=6)
    st.save(fig, path_base)


def plot_timing(n_rails: np.ndarray, timings: Dict[str, np.ndarray], path_base: str):
    """Wall-clock vs number of rails (log-log).  Series are matched to a fixed style by keyword so that the
    same physical thing keeps its colour across the student-engine and the teacher-engine runs."""
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.9), constrained_layout=True)

    def style_of(name):
        n = name.lower()
        if "fdm" in n:
            return ("#7F7F7F", "--", "s") if "1 core" in n else ("k", "-", "s")
        distilled = "distil" in n; tab = "tabulated" in n
        color = {(False, False): st.OKABE_ITO["vermilion"], (False, True): st.OKABE_ITO["orange"],
                 (True, False): st.OKABE_ITO["purple"], (True, True): st.OKABE_ITO["green"]}[(distilled, tab)]
        return (color, "--" if "numpy" in n else "-", "D" if tab else "o")
    for name, t in timings.items():
        c, ls, mk = style_of(name); m = np.isfinite(t)
        if m.any():
            ax.plot(n_rails[m], t[m], color=c, ls=ls, marker=mk, ms=3.2, lw=1, label=name)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of rails"); ax.set_ylabel("wall-clock time (s)")
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False); ax.grid(True, which="major", color="#EEEEEE", lw=0.5)
    st.save(fig, path_base, data=dict(n_rails=n_rails, **{k.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_").replace("+", "plus"): v for k, v in timings.items()}))


def plot_sensitivity_map(spec, sens: Dict[str, np.ndarray], block_labels: List[str], path_base: str, target_label: str,
                         unit="% per W"):
    """sens[target_die] -> vector over ALL blocks (global order). Draw one floorplan per die."""
    st.use_paper_style()
    n = len(spec.dies)
    fig, axes = plt.subplots(1, n, figsize=(st.IEEE_2COL, 2.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = max(np.abs(v).max() for v in sens.values())
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    off = 0
    for ax, d in zip(axes, spec.dies):
        nb = d.nblk
        grid = np.zeros((nb, nb))
        vec = sens[target_label] if target_label in sens else next(iter(sens.values()))
        for i in range(nb):
            for j in range(nb):
                grid[j, i] = vec[off + i * nb + j]         # row = y index (j), col = x index (i)
        off += nb * nb
        im = ax.imshow(grid, origin="lower", cmap="PiYG", norm=norm, extent=[0, d.size * 1e3, 0, d.size * 1e3])
        for i in range(nb):
            for j in range(nb):
                ax.text((i + 0.5) * d.size / nb * 1e3, (j + 0.5) * d.size / nb * 1e3, f"{grid[j, i]:+.1f}", ha="center", va="center", fontsize=5.5)
        ax.set_title(f"power blocks of {d.name}", fontsize=7); ax.set_xlabel("x (mm)")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axes[0].set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85); cb.set_label(f"d log t_nuc({target_label}) / dP  ({unit})")
    st.save(fig, path_base)


def plot_budget_trajectory(hist: List[Dict], path_base: str, verified_years: float | None = None):
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.0), constrained_layout=True)
    it = np.array([h["iter"] for h in hist]); hard = np.array([h["min_tnuc_yr"] for h in hist]); soft = np.array([10 ** h["softmin_log10_tnuc_yr"] for h in hist])
    ax.plot(it, hard, color=st.OKABE_ITO["vermilion"], marker="o", ms=2.5, lw=1, label="earliest $t_{nuc}$ (hard minimum)")
    ax.plot(it, soft, color=st.OKABE_ITO["blue"], lw=1, ls="--", label="soft minimum (objective)")
    ax.annotate(f"{hard[0]:.2f} yr", (it[0], hard[0]), xytext=(6, -9), textcoords="offset points", fontsize=6, color=st.OKABE_ITO["vermilion"])
    ax.annotate(f"{hard[-1]:.2f} yr" + (f"\n(reference: {verified_years:.2f} yr)" if verified_years else ""), (it[-1], hard[-1]), xytext=(-4, 6),
                textcoords="offset points", fontsize=6, color=st.OKABE_ITO["vermilion"], ha="right")
    ax.set_xlabel("projected-gradient iteration"); ax.set_ylabel("earliest $t_{nuc}$ (years)"); ax.legend(fontsize=6, loc="lower right")
    ax.margins(y=0.25)
    st.save(fig, path_base, data=dict(iteration=it, hard_min_years=hard, soft_min_years=soft))


def plot_blech_screen(table: List[Dict], path_base: str):
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 1.9))
    labels = [f"{r['kind']} {r['die']}" for r in table]; vals = [r["ratio"] for r in table]
    ax.barh(range(len(vals)), vals, color=[st.COPPER if r["kind"] == "TSV" else st.OKABE_ITO["orange"] for r in table], height=0.6)
    ax.axvline(1.0, color="k", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlim(1e-3, 3)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=6.5); ax.set_xlabel(r"$jL\,/\,(jL)_c$  (immortal if < 1)")
    st.save(fig, path_base)


def plot_effect_ratios(ratios: Dict[str, Dict[str, float]], path_base: str, xlim=(0.04, 15.0)):
    """ratios[effect][die] -> earliest t_nuc of the variant / of the resolved stack.  Left of 1 = the variant is
    pessimistic (predicts earlier failure), right of 1 = optimistic (misses failures)."""
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.45), constrained_layout=True)
    effects = list(ratios); dies = sorted({d for e in ratios.values() for d in e})
    ax.axvspan(xlim[0], 1.0, color="#EEF3F8", lw=0, zorder=0); ax.axvspan(1.0, xlim[1], color="#FBEFEA", lw=0, zorder=0)
    for i in range(len(effects)):
        ax.axhline(i, color="#DDDDDD", lw=0.5, zorder=1)
    for k, d in enumerate(dies):
        vals = [ratios[e].get(d, np.nan) for e in effects]
        ax.plot(vals, range(len(effects)), marker="o", ms=4.2, lw=0, color=list(st.OKABE_ITO.values())[k], label=d, zorder=3, mec="white", mew=0.4)
    for i, e in enumerate(effects):                        # label only the extreme of each row, outside the markers
        row = {d: ratios[e].get(d, np.nan) for d in dies}; row = {d: v for d, v in row.items() if np.isfinite(v)}
        if not row: continue
        dmin, dmax = min(row, key=row.get), max(row, key=row.get)
        if row[dmin] < 0.6:
            ax.annotate(f"{row[dmin]:.2f}$\\times$", (row[dmin], i), xytext=(-6, 0), textcoords="offset points", fontsize=5.4, ha="right", va="center", color=list(st.OKABE_ITO.values())[dies.index(dmin)])
        if row[dmax] > 1.5:
            ax.annotate(f"{row[dmax]:.1f}$\\times$", (row[dmax], i), xytext=(6, 0), textcoords="offset points", fontsize=5.4, ha="left", va="center", color=list(st.OKABE_ITO.values())[dies.index(dmax)])
    ax.axvline(1.0, color="#666666", lw=0.8, zorder=2)
    tr = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.93, 1.01, "$\\leftarrow$ pessimistic (false alarms)", transform=tr, fontsize=5.6, ha="right", va="bottom", color="#2F5C8C")
    ax.text(1.07, 1.01, "optimistic (misses failures) $\\rightarrow$", transform=tr, fontsize=5.6, ha="left", va="bottom", color="#8C4A2F")
    ax.set_yticks(range(len(effects))); ax.set_yticklabels(effects, fontsize=6.5); ax.set_xscale("log"); ax.set_xlim(*xlim)
    ax.set_xlabel("earliest $t_{nuc}$: variant / resolved stack")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=len(dies), handletextpad=0.2, columnspacing=1.2, frameon=False)
    ax.set_ylim(-0.6, len(effects) - 0.4)
    st.save(fig, path_base, data={f"{e}_{d}": ratios[e].get(d, np.nan) for e in effects for d in dies})


def plot_signoff_sizing(sizing: Dict[str, Dict], rule: Dict[str, Dict], target_years: float, path_base: str):
    """(a) sizing curves: earliest standalone t_nuc vs strap width W for every die (die-level sign-off: solid,
    uniform-T rule: dashed), the target line and the chosen widths; (b) the resulting widths per die."""
    st.use_paper_style()
    dies = list(sizing)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.IEEE_2COL, 2.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    for k, dn in enumerate(dies):
        c = list(st.OKABE_ITO.values())[k]
        h = np.array(sizing[dn].get("history", []), float)
        if len(h):
            h = h[np.argsort(h[:, 0])]; t = np.where(np.isfinite(h[:, 1]), h[:, 1] / SEC_PER_YEAR, 1e4)
            ax.plot(h[:, 0] * 1e6, t, marker="o", ms=2.5, lw=1, color=c, label=f"{dn} die-level sign-off")
        ax.plot([sizing[dn]["W_um"]], [sizing[dn]["t_earliest_years"]], marker="*", ms=8, color=c, mec="k", mew=0.4, lw=0)
        if dn in rule and rule[dn].get("history"):
            h = np.array(rule[dn]["history"], float); h = h[np.argsort(h[:, 0])]; t = np.where(np.isfinite(h[:, 1]), h[:, 1] / SEC_PER_YEAR, 1e4)
            ax.plot(h[:, 0] * 1e6, t, marker="s", ms=2, lw=0.9, ls="--", color=c, label=f"{dn} rule (uniform T)")
            ax.plot([rule[dn]["W_um"]], [rule[dn]["t_earliest_years"]], marker="*", ms=8, color=c, mec="k", mew=0.4, lw=0, alpha=0.6)
    ax.axhline(target_years, color="k", lw=0.7, ls=":")
    ax.text(0.02, target_years, f"target {target_years:g} yr", fontsize=6, va="bottom", transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(top=3e3)
    ax.set_xlabel(r"strap width $W$ ($\mu$m)"); ax.set_ylabel("earliest $t_{nuc}$, die standalone (yr)"); ax.legend(fontsize=5.6, ncol=2, loc="lower right")
    ax.set_title("sizing curves (stars: chosen width)", fontsize=7)
    xs = np.arange(len(dies)); w = 0.38
    ax2.bar(xs - w / 2, [sizing[dn]["W_um"] for dn in dies], w, color=[list(st.OKABE_ITO.values())[k] for k in range(len(dies))], label="die-level sign-off (standalone thermal)")
    if rule:
        ax2.bar(xs + w / 2, [rule[dn]["W_um"] for dn in dies], w, color=[list(st.OKABE_ITO.values())[k] for k in range(len(dies))], alpha=0.45, hatch="///", label="rule sign-off (uniform 105 $^\\circ$C)")
    for k, dn in enumerate(dies):
        ax2.text(xs[k] - w / 2, sizing[dn]["W_um"], f"{sizing[dn]['W_um']:.2f}", ha="center", va="bottom", fontsize=5.5)
        if dn in rule:
            ax2.text(xs[k] + w / 2, rule[dn]["W_um"], f"{rule[dn]['W_um']:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax2.set_xticks(xs); ax2.set_xticklabels(dies); ax2.set_ylabel(r"strap width $W$ ($\mu$m)"); ax2.set_yscale("log"); ax2.margins(y=0.3)
    ax2.legend(fontsize=5.6, loc="upper left"); ax2.set_title("resulting widths", fontsize=7)
    st.save(fig, path_base)


def plot_repair(spec, rails_by_die: Dict[str, list], repair: Dict[str, Dict], path_base: str, horizon_years: float = 10.0):
    """Stack-aware repair (E5 part 1).  Left: per-die maps with the rails that violate the horizon in the stack
    coloured by the widening factor that restores it (targeted repair); right: metal area added by the targeted
    repair vs a blanket widening of the whole die, in % of the die's grid metal (ratio annotated)."""
    st.use_paper_style()
    all_dies = [d.name for d in spec.dies]
    dies = [dn for dn in all_dies if repair[dn]["n_violating"] > 0] or all_dies      # maps only for dies that need repair
    n = len(dies)
    fig = plt.figure(figsize=(st.IEEE_2COL, 2.35))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.25], wspace=0.30, left=0.06, right=0.99, top=0.84, bottom=0.20)
    fmax = max([v["factor"] for dn in dies for v in repair[dn]["violating"]] + [1.5])
    norm = mpl.colors.Normalize(vmin=1.0, vmax=fmax); cmap = plt.get_cmap("YlOrRd")
    size = spec.size * 1e3; axes = []
    for k, dn in enumerate(dies):
        ax = fig.add_subplot(gs[0, k]); axes.append(ax); ax.set_facecolor("#F4F4F4")
        fac = {v["rail"]: v["factor"] for v in repair[dn]["violating"]}
        for r in rails_by_die[dn]:
            xs, ys = r.node_xy(); key = f"{r.kind}{r.index}"
            if key in fac:
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color=cmap(norm(fac[key])), lw=1.6)
            else:
                ax.plot([xs[0] * 1e3, xs[-1] * 1e3], [ys[0] * 1e3, ys[-1] * 1e3], color="#BBBBBB", lw=0.35, alpha=0.7)
        ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal")
        ax.set_title(f"{dn}: {repair[dn]['n_violating']} / {repair[dn]['n_rails']} rails\nviolate {horizon_years:g} yr", fontsize=6.5)
        ax.set_xticks([0, size]); ax.set_yticks([0, size] if k == 0 else []); ax.tick_params(labelsize=6); ax.set_xlabel("x (mm)", labelpad=1)
        if k == 0: ax.set_ylabel("y (mm)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=18, location="bottom" if n > 3 else "right"); cb.set_label("widening factor of the rail", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    dies = all_dies; n = len(dies)                                                      # the cost panel lists every die
    axb = fig.add_subplot(gs[0, -1]); xs = np.arange(n); w = 0.38
    tg = [100 * repair[dn]["area_targeted_m2"] / repair[dn]["area_grid_m2"] for dn in dies]
    bl = [100 * repair[dn]["area_blanket_m2"] / repair[dn]["area_grid_m2"] for dn in dies]
    axb.bar(xs - w / 2, tg, w, color=st.OKABE_ITO["blue"], label="targeted (rail-resolved)")
    axb.bar(xs + w / 2, bl, w, color=st.OKABE_ITO["grey"], label="blanket (whole die)")
    for k, dn in enumerate(dies):
        r = repair[dn]["area_ratio_blanket_over_targeted"]
        if r:
            axb.text(xs[k], max(tg[k], bl[k]) * 1.06, f"{r:.0f}$\\times$", ha="center", va="bottom", fontsize=6.5)
        else:
            axb.text(xs[k], 0.5, "no repair\nneeded", ha="center", va="bottom", fontsize=5.5, color="#666666")
    axb.set_xticks(xs); axb.set_xticklabels(dies); axb.set_ylabel("metal area added\n(% of the die's grid metal)")
    axb.set_ylim(0, max(bl + tg) * 1.55)                                                # head-room: ratio labels below, legend above them
    axb.legend(fontsize=5.8, loc="upper left", ncol=2, borderaxespad=0.3, handlelength=1.2, columnspacing=0.8, frameon=False)
    axb.set_title(f"cost of restoring the {horizon_years:g}-yr horizon", fontsize=7)
    st.save(fig, path_base, data={f"{dn}_{k}": v for dn in dies for k, v in (("targeted_pct", 100 * repair[dn]["area_targeted_m2"] / repair[dn]["area_grid_m2"]),
                                                                            ("blanket_pct", 100 * repair[dn]["area_blanket_m2"] / repair[dn]["area_grid_m2"]),
                                                                            ("violating", np.array([[v["factor"]] for v in repair[dn]["violating"]]).ravel()))})
