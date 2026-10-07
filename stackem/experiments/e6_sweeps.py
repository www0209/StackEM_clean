"""
E6 - design-space and parameter sweeps (reference solver on every rail, no network involved; the 24 rows of the
final run took 13 minutes on a 32-thread machine).

    python -m stackem.experiments.e6_sweeps --root DIR [--smoke]

Sweeps (each relative to the base configuration)
    r_convec      0.1 / 0.5 / 1.0 K/W          package cooling
    bond_k        5 / 40 / 150 W/mK            hybrid-bond layer conductivity of the HotSpot stack model only (the
                                               wire self-heating parameter ThermalParams.k_bond is NOT changed by this sweep)
    feed_pitch    200 / 400 / 800 um           vertical feed density of the bottom die (re-sized: a designer who
                                               changes the feed pitch re-runs the die-level sign-off)
    top_power     10 / 20 / 30 W               top-die power budget
    layer_order   logic die on top / at the bottom
    Ea            0.84 / 0.86 / 1.0 / 1.1 eV   activation-energy alternatives (parameter risk, re-sized)
    sigma_crit    400 / 500 / 600 MPa          (re-sized)
    signoff_margin 1.5 / 2 / 3                 margin of the die-level sign-off itself (re-sized): does the
                                               conclusion depend on the x2 of the benchmark definition?
Strap widths stay at the base sign-off sizing for the environment sweeps (the dies were designed once and the
package/thermal environment changed under them); for the design-rule and material sweeps (feed pitch, Ea,
sigma_crit, margin) the die-level sign-off sizing is redone with those parameters, because a designer sizes
with the rules and parameters they assume.  Every point reports, per die,
the earliest t_nuc standalone and in the stack and their ratio = the margin a die-level sign-off would need
to be stack-safe ("required margin", Fig. 12 second row).
Outputs e6_sweeps.csv/json and e6_sweeps.(pdf|png).
"""
from __future__ import annotations

import copy
import csv
import os
import time
import numpy as np

from ._common import base_parser, setup, record, out_dir, make_assembler
from ..config import stack_from_config, pg_from_config, em_from_config, thermal_from_config, times_from_config, hotspot_dir
from ..constants import SEC_PER_YEAR
from ..hotspot_stack import HotSpotRunner, run_stack
from ..assembler import StackAssembler
from ..viz import style as st


def run_case(cfg, tag, workdir, sizing=None, resize=False):
    spec = stack_from_config(cfg); em = em_from_config(cfg); th = thermal_from_config(cfg)
    if sizing and not resize:
        from ..signoff import apply_sizing
        spec = apply_sizing(spec, sizing)                      # strap widths fixed by the base sign-off sizing
    out = {}
    if resize:                                                 # material-parameter sweeps: the die-level sign-off is redone with those parameters
        from ..signoff import size_stack, apply_sizing
        tf0 = run_stack(spec, os.path.join(workdir, "hotspot_sizing"), HotSpotRunner(hotspot_dir(cfg)), "stack", True)
        so = cfg["signoff"]
        sz = size_stack(spec, pg_from_config(cfg), em, th, tf0, times_from_config(cfg), float(so["horizon_years"]), float(so["margin"]),
                        float(so["W_min"]), float(so["W_max"]), cfg["truth"]["n_cells"], verbose=False)
        spec = apply_sizing(spec, sz)
        for dn, v in sz.items():
            out[f"W_um_{dn}"] = v["W_um"]
    a = StackAssembler(spec, pg_from_config(cfg), em, th, HotSpotRunner(hotspot_dir(cfg)), times_from_config(cfg), workdir, None,
                       truth_cells=cfg["truth"]["n_cells"]).run_fields()
    from ..assembler import save_variant
    for v in ("full", "alone"):
        r = a.solve_variant(v, "truth"); s = a.stack_summary(r)
        save_variant(os.path.join(workdir, f"e6_{v}_truth.npz"), r, s, a.times)          # raw data of every sweep point
        for dn, m in s["dies"].items():
            out[f"{v}_{dn}_earliest_yr"] = m["earliest_t_nuc_s"] / SEC_PER_YEAR
            out[f"{v}_{dn}_mortal10"] = m["mortal_within"]["10yr"]
            out[f"{v}_{dn}_Tmax"] = m["T_max"]
    for dn in [dd.name for dd in spec.dies]:                   # margin a die-level sign-off would need to be stack-safe
        out[f"required_margin_{dn}"] = out[f"alone_{dn}_earliest_yr"] / out[f"full_{dn}_earliest_yr"]
    return out


def main(argv=None):
    ap = base_parser("E6: sweeps")
    ap.add_argument("--replot", action="store_true", help="only redraw e6_sweeps.(pdf|png) from the existing e6_sweeps.json")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; d = out_dir(cfg, "e6")
    sweeps = {
        "r_convec": [0.1, 0.5, 1.0], "bond_k": [5.0, 40.0, 150.0], "feed_pitch_bottom_um": [200, 400, 800],
        "top_power_W": [10.0, 20.0, 30.0], "layer_order": ["logic_top", "logic_bottom"],
        "Ea_eV": [0.84, 0.86, 1.0, 1.1], "sigma_crit_MPa": [400, 500, 600], "signoff_margin": [1.5, 2.0, 3.0],
    }
    titles = {"r_convec": "package\n$R_{conv}$ (K/W)", "bond_k": "bond layer\n$k$ (W/mK)", "feed_pitch_bottom_um": f"feed pitch\nof {cfg['stack']['dies'][0]['name']} ($\\mu$m)",
              "top_power_W": "top-die\npower (W)", "layer_order": "layer\norder", "Ea_eV": "$E_a$\n(eV)", "sigma_crit_MPa": "$\\sigma_{crit}$\n(MPa)",
              "signoff_margin": "die-level\nsign-off margin"}
    short = {"logic_top": "logic\ntop", "logic_bottom": "logic\nbottom"}
    if args.smoke:
        sweeps = {"r_convec": [0.3, 0.7]}
    rows = []
    if args.replot:
        import json
        rows = json.load(open(os.path.join(d, "e6_sweeps.json")))["rows"]; sweeps = {k: None for k in dict.fromkeys(r["sweep"] for r in rows)}
    for name, vals in (sweeps.items() if not args.replot else []):
        for v in vals:
            c = copy.deepcopy(cfg)
            if name == "r_convec": c["stack"]["r_convec"] = v
            elif name == "bond_k": c["stack"]["bond_k"] = v
            elif name == "feed_pitch_bottom_um": c["stack"]["dies"][0]["feed_pitch"] = v * 1e-6
            elif name == "top_power_W": c["stack"]["dies"][-1]["P"] = v
            elif name == "layer_order":
                if v == "logic_bottom":
                    c["stack"]["dies"] = list(reversed(c["stack"]["dies"]))
                    for dd, fk in zip(c["stack"]["dies"], ["tsv", "tsv", "pad"]):
                        dd["feed_kind"] = fk
            elif name == "Ea_eV": c.setdefault("em", {})["Ea_eV"] = v
            elif name == "sigma_crit_MPa": c.setdefault("em", {})["sigma_crit"] = v * 1e6
            elif name == "signoff_margin": c["signoff"]["margin"] = v
            tag = f"{name}_{v}"
            resize = name in ("feed_pitch_bottom_um", "Ea_eV", "sigma_crit_MPa", "signoff_margin")   # design rules / material assumptions: the die-level sign-off is redone with them
            t0 = time.time(); out = run_case(c, tag, os.path.join(d, tag), ctx.get("sizing"), resize); out.update(sweep=name, value=v, seconds=time.time() - t0)
            rows.append(out); print(tag, {k: round(x, 3) for k, x in out.items() if isinstance(x, float)}, flush=True)
    if not args.replot:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(d, "e6_sweeps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        record(cfg, "e6", "e6_sweeps.json", dict(rows=rows))
    # figure: earliest t_nuc per die vs parameter, one panel per sweep
    st.use_paper_style()
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    names = list(sweeps); dies = [dd["name"] for dd in cfg["stack"]["dies"]]

    def num(v):                                             # json-safe values: None -> nan, "inf" -> inf
        if v is None: return np.nan
        if isinstance(v, str): return np.inf if v == "inf" else np.nan
        return float(v)
    fig, axes = plt.subplots(2, len(names), figsize=(st.IEEE_2COL, 3.4), constrained_layout=True, squeeze=False, sharey="row")
    for col, name in enumerate(names):
        rs = [r for r in rows if r["sweep"] == name]
        xs = np.arange(len(rs)); labels = [short.get(str(r["value"]), f"{r['value']:g}" if isinstance(r["value"], (int, float)) else str(r["value"])) for r in rs]
        for k, dn in enumerate(dies):
            c = list(st.OKABE_ITO.values())[k]
            axes[0, col].plot(xs, [num(r.get(f"full_{dn}_earliest_yr")) for r in rs], marker="o", ms=3, color=c, label=dn)
            axes[1, col].plot(xs, [num(r.get(f"required_margin_{dn}")) for r in rs], marker="s", ms=3, color=c)
        axes[0, col].axhline(cfg["signoff"]["horizon_years"], color="k", lw=0.6, ls=":")
        axes[0, col].set_yscale("log"); axes[0, col].set_title(titles.get(name, name), fontsize=6.3); axes[1, col].set_yscale("log")
        axes[1, col].axhline(cfg["signoff"]["margin"], color="k", lw=0.6, ls=":")
        for ax in axes[:, col]:
            ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.8, rotation=40 if len(rs) > 3 else 0); ax.set_xlim(-0.5, len(rs) - 0.5); ax.margins(y=0.15)
        # sweep points where a die is immortal standalone (required margin undefined) are marked at the top of the panel
        for k, dn in enumerate(dies):
            ck = list(st.OKABE_ITO.values())[k]
            for xi, r in zip(xs, rs):
                if not np.isfinite(num(r.get(f"required_margin_{dn}"))):
                    axes[1, col].plot([xi + 0.12 * (k - (len(dies) - 1) / 2)], [0.95], marker="^", ms=4, mfc="none", mec=ck, mew=0.8, lw=0,
                                      transform=mpl.transforms.blended_transform_factory(axes[1, col].transData, axes[1, col].transAxes), clip_on=False)
        axes[0, col].tick_params(labelbottom=False)
    axes[0, 0].set_ylabel("earliest $t_{nuc}$ in stack (yr)"); axes[0, 0].legend(fontsize=6, loc="lower left")
    axes[1, 0].set_ylabel("required margin\n$t_{alone}/t_{stack}$")
    fig.text(0.5, -0.02, "dotted: sign-off horizon (top) and the margin the dies were sized with (bottom); re-sized sweeps: feed pitch, $E_a$, $\\sigma_{crit}$, margin\n"
                         "open triangles: die immortal standalone (required margin undefined)",
             ha="center", fontsize=6, color="#444444")
    st.save(fig, os.path.join(d, "e6_sweeps"))


if __name__ == "__main__":
    main()
