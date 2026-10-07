"""
stackem.viz.style
===================

One visual system for every figure of the paper (IEEE two-column):

* categorical identity: Okabe-Ito colour-blind-safe palette in a FIXED order
  (variant -> colour never changes between figures);
* magnitude: single perceptual sequential maps (temperature: 'inferno',
  time-to-nucleation: 'viridis_r' so that early = bright/yellow is the alarm);
* polarity: diverging 'RdBu_r' with white at zero stress (tensile red,
  compressive blue);
* thin marks, recessive axes, direct labels where possible, 8 pt type.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#7F7F7F",
}

# fixed identity of the physics variants (never re-ordered)
VARIANT_COLOR = {
    "alone": OKABE_ITO["grey"],
    "uniform_signoff": OKABE_ITO["black"],
    "uniform": OKABE_ITO["sky"],
    "sigmaT_const": OKABE_ITO["orange"],
    "no_joule": OKABE_ITO["purple"],
    "no_tm": OKABE_ITO["green"],
    "full": OKABE_ITO["vermilion"],
    "stackem": OKABE_ITO["blue"],
    "truth": OKABE_ITO["black"],
}
VARIANT_LABEL = {
    "alone": "die-level sign-off (standalone thermal)",
    "uniform_signoff": "rule sign-off (uniform 105 $^\\circ$C, const. $\\sigma_T$)",
    "uniform": "uniform die temperature",
    "sigmaT_const": r"$\sigma_T$ frozen at 2-D temperature",
    "no_joule": "no wire self-heating",
    "no_tm": "no thermomigration",
    "full": "full 3-D physics (FDM truth)",
    "stackem": "StackEM (SKN + JCX)",
    "truth": "FDM truth",
}
CMAP_T = "inferno"
CMAP_TNUC = "viridis_r"
CMAP_STRESS = "RdBu_r"
COPPER = "#B87333"
SILICON = "#8E9BA6"
BOND = "#D9C9A3"
TIM = "#C0C0C0"
SINK = "#A7B4BE"

IEEE_COL = 3.5     # inches, single column
IEEE_2COL = 7.16   # inches, double column
METHOD = "StackEM"   # the method name used in figure labels


def use_paper_style():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
    })


def save(fig, path_base: str, formats=("pdf", "png"), data: dict | None = None, panels: bool = True):
    """Write the figure (pdf + png), the data behind it (<path_base>_data.npz + <path_base>_data.json) and, for a
    multi-panel figure, every panel as its own file (<path_base>_panel<k>.pdf/png, colour bars kept with their panel).

    The data export is generic: every line, scatter, image, bar and line-collection of every axes is dumped with the
    axes' title / axis labels / scales, so that anyone can re-plot the figure from the npz without the package;
    `data` adds the plotting function's own input arrays under their names."""
    import os, json
    os.makedirs(os.path.dirname(os.path.abspath(path_base)) or ".", exist_ok=True)
    for f in formats:
        fig.savefig(f"{path_base}.{f}")
    try:
        _export_data(fig, path_base, data)
    except Exception as e:                     # the data export must never break a run
        print(f"[viz] data export failed for {path_base}: {e}")
    if panels:
        try:
            _export_panels(fig, path_base, formats)
        except Exception as e:
            print(f"[viz] panel export failed for {path_base}: {e}")
    plt.close(fig)


def _is_colorbar(ax) -> bool:
    return ax.get_label() == "<colorbar>"


def _export_data(fig, path_base: str, extra: dict | None):
    import json
    arrays = {}; index = {"axes": [], "extra": []}
    def put(key, arr):
        arrays[key] = np.asarray(arr); return key
    for k, ax in enumerate(a for a in fig.axes if not _is_colorbar(a)):
        is3d = hasattr(ax, "get_zlim")
        entry = dict(axes=k, title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                     xscale=ax.get_xscale(), yscale=ax.get_yscale(), series=[])
        if is3d:
            entry["zlabel"] = ax.get_zlabel()
        for i, ln in enumerate(ax.get_lines()):
            lab = ln.get_label()
            if is3d and hasattr(ln, "get_data_3d"):
                x, y, z = ln.get_data_3d(); entry["series"].append(dict(kind="line3d", label=lab, x=put(f"ax{k}_line{i}_x", x), y=put(f"ax{k}_line{i}_y", y), z=put(f"ax{k}_line{i}_z", z)))
            else:
                x, y = ln.get_xdata(), ln.get_ydata()
                if len(np.atleast_1d(x)) == 0: continue
                entry["series"].append(dict(kind="line", label=lab, x=put(f"ax{k}_line{i}_x", x), y=put(f"ax{k}_line{i}_y", y)))
        for i, im in enumerate(ax.images):
            entry["series"].append(dict(kind="image", label=im.get_label(), array=put(f"ax{k}_image{i}", im.get_array()), extent=list(map(float, im.get_extent()))))
        for i, coll in enumerate(ax.collections):
            lab = coll.get_label(); name = type(coll).__name__
            if hasattr(coll, "_segments3d"):
                entry["series"].append(dict(kind="segments3d", label=lab, segments=put(f"ax{k}_coll{i}_segments", np.asarray(coll._segments3d, dtype=object) if any(len(s_) != len(coll._segments3d[0]) for s_ in coll._segments3d) else np.asarray(coll._segments3d, float))))
            elif name == "PathCollection":
                off = coll.get_offsets()
                if len(off): entry["series"].append(dict(kind="scatter", label=lab, xy=put(f"ax{k}_scatter{i}_xy", np.asarray(off)), values=put(f"ax{k}_scatter{i}_values", coll.get_array()) if coll.get_array() is not None else None))
            elif name == "LineCollection":
                segs = coll.get_segments()
                if segs: entry["series"].append(dict(kind="segments", label=lab, segments=put(f"ax{k}_coll{i}_segments", np.asarray(segs, dtype=object) if any(len(s_) != len(segs[0]) for s_ in segs) else np.asarray(segs, float))))
            elif name in ("PolyCollection", "QuadMesh"):
                arr = coll.get_array()
                if arr is not None and np.size(arr): entry["series"].append(dict(kind=name.lower(), label=lab, values=put(f"ax{k}_coll{i}_values", arr)))
        for i, cont in enumerate(getattr(ax, "containers", [])):
            if type(cont).__name__ == "BarContainer":
                rects = cont.patches
                entry["series"].append(dict(kind="bars", label=cont.get_label(), x=put(f"ax{k}_bars{i}_x", [r.get_x() + r.get_width() / 2 for r in rects]),
                                            height=put(f"ax{k}_bars{i}_height", [r.get_height() for r in rects]), width=put(f"ax{k}_bars{i}_width", [r.get_width() for r in rects])))
        entry["texts"] = [dict(text=t.get_text(), xy=list(map(float, t.get_position()))) for t in ax.texts if t.get_text()]
        index["axes"].append(entry)
    for kk, v in (extra or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            index["extra"].append(dict(name=kk, value=v))
        else:
            try:
                index["extra"].append(dict(name=kk, key=put(f"extra_{kk}", v)))
            except Exception:
                index["extra"].append(dict(name=kk, value=str(v)))
    np.savez_compressed(f"{path_base}_data.npz", **{kk: v for kk, v in arrays.items()}, allow_pickle=True) if any(a.dtype == object for a in arrays.values()) else np.savez_compressed(f"{path_base}_data.npz", **arrays)
    with open(f"{path_base}_data.json", "w") as f:
        json.dump(index, f, indent=1, default=str)


def _export_panels(fig, path_base: str, formats):
    axes = [a for a in fig.axes if not _is_colorbar(a)]
    if len(axes) < 2:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cbars = [a for a in fig.axes if _is_colorbar(a)]
    boxes = []
    for a in axes:
        bb = a.get_tightbbox(r)
        if bb is None: continue
        boxes.append([a, bb])
    # attach every colour bar to the panel whose box is closest to it
    for cb in cbars:
        bc = cb.get_tightbbox(r)
        if bc is None or not boxes: continue
        j = int(np.argmin([np.hypot(bc.x0 - b.x1, (bc.y0 + bc.y1) / 2 - (b.y0 + b.y1) / 2) for _, b in boxes]))
        boxes[j][1] = mpl.transforms.Bbox.union([boxes[j][1], bc])
    for k, (a, bb) in enumerate(boxes):
        pad = 4.0                                   # points of margin in display pixels
        bbi = mpl.transforms.Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad).transformed(fig.dpi_scale_trans.inverted())
        for f in formats:
            fig.savefig(f"{path_base}_panel{k + 1}.{f}", bbox_inches=bbi)


def log_tnuc_norm(years_lo: float = 0.03, years_hi: float = 100.0):
    return mpl.colors.LogNorm(vmin=years_lo, vmax=years_hi)


def tnuc_colors(t_years: np.ndarray, norm=None, immortal_color="#DDDDDD"):
    """Map nucleation times (years) to colours; inf/NaN -> immortal grey."""
    norm = norm or log_tnuc_norm()
    cmap = plt.get_cmap(CMAP_TNUC)
    t = np.asarray(t_years, float)
    cols = np.array([cmap(norm(min(max(v, norm.vmin), norm.vmax))) if np.isfinite(v) else mpl.colors.to_rgba(immortal_color) for v in t])
    return cols
