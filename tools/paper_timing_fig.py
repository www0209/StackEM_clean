"""
Paper figure of the speed comparison: ONE engine vs the numerical references.

    python tools/paper_timing_fig.py --root ~/stackem_work/outputs [--external "COMSOL (FEM)=<root>/comsol/comsol_timing.csv"]

Reads <root>/base3/e4/e4_timing.json (written by E4 of the student-engine run), keeps
  * the configured engine  (label contains "tabulated" and "torch"; the first such series; the numpy series when PyTorch is absent),
  * FDM truth, all cores and 1 core,
  * every --external csv (columns n_rails,seconds; interpolated onto the E4 sizes),
and draws <root>/base3/e4/fig_e4_paper_timing.(pdf|png).  Internal engine variants (direct look-up,
teacher network, numpy CPU) are deliberately not drawn: the paper reports one engine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stackem.viz import style as st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", default="base3")
    ap.add_argument("--external", action="append", default=[], help="label=csv (n_rails,seconds)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = os.path.expanduser(args.root)
    with open(os.path.join(root, args.name, "e4", "e4_timing.json")) as f:
        j = json.load(f)
    sizes = np.array(j["sizes"], dtype=float); tim = {k: np.array(v, dtype=float) for k, v in j["timings"].items()}
    engine = next((k for k in tim if "tabulated" in k and "torch" in k and "full SKN" not in k), None)
    if engine is None:                                      # no PyTorch on this machine: fall back to the numpy series (sizes up to --numpy-max of E4)
        engine = next((k for k in tim if "tabulated" in k and "numpy" in k and "full SKN" not in k), None)
    if engine is None:
        raise SystemExit("no tabulated torch engine in e4_timing.json (run E4 with a tabulated-look-up config such as configs/base3.json)")
    series = [(f"{st.METHOD} ({'1 GPU' if 'CUDA' in engine else ('CPU, numpy' if 'numpy' in engine else 'CPU, torch')})", tim[engine], st.OKABE_ITO["vermilion"], "-", "o")]
    for k, v in tim.items():
        if k.startswith("FDM truth"):
            one = "1 core" in k
            series.append((f"finite-volume reference ({'1 core' if one else k.split(',')[-1].strip(' )')})", v,
                           "#7F7F7F" if one else "k", "--" if one else "-", "s"))
    for ext in args.external:
        label, path = ext.split("=", 1)
        arr = np.loadtxt(os.path.expanduser(path), delimiter=",", skiprows=1, ndmin=2)
        series.append((label, np.interp(sizes, arr[:, 0], arr[:, 1], left=np.nan, right=np.nan), st.OKABE_ITO["blue"], "-.", "^"))
    st.use_paper_style()
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.9))
    for label, t, c, ls, mk in series:
        m = np.isfinite(t)
        if m.any():
            ax.plot(sizes[m], t[m], color=c, ls=ls, marker=mk, ms=3.2, lw=1.1,
                    label=f"{label}: {t[m][-1]:.0f} s at {int(sizes[m][-1]):,} rails")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of rails"); ax.set_ylabel("wall-clock (s)")
    ax.grid(True, which="major", color="#EEEEEE", lw=0.5); ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1); ax.margins(x=0.08)
    out = args.out or os.path.join(root, args.name, "e4", "fig_e4_paper_timing")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    st.save(fig, out)
    print("wrote", out + ".pdf/.png", "| engine series:", engine)


if __name__ == "__main__":
    main()
