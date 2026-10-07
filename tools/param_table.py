"""
Table II of the paper: every parameter of the stack case, the EM/thermal model and the benchmark definition,
generated from the same config / constants the experiments run with (no hand-copied numbers).

    python tools/param_table.py --config configs/base3.json --root ~/stackem_work/outputs

Writes <root>/<name>/tables/table_params.tex (booktabs, IEEE two-column) and .md.  Columns:
parameter | symbol | value | swept range (E6).  Provenance is not a column (the caption / text explain the
choices: Tan-group Korhonen line for the EM set, HotSpot defaults for the package, ITRS/IRDS-style
current density and pitch for the grid, 10 yr x 2 for the sign-off target); sweep ranges show which
choices the conclusions were tested against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stackem.config import load_config, em_from_config, thermal_from_config, pg_from_config   # noqa: E402
from stackem.signoff import sizing_path, load_sizing                                              # noqa: E402

SWEEPS = {  # E6 ranges (keep in sync with experiments/e6_sweeps.py)
    "r_convec": "0.1 -- 1.0", "bond_k": "5 -- 150", "feed_pitch": "200 -- 800", "top_power": "10 -- 30",
    "Ea": "0.84 -- 1.1", "sigma_crit": "400 -- 600", "margin": "1.5 -- 3", "layer_order": "logic top / bottom",
}


def rows_for(cfg, sizing):
    em = em_from_config(cfg); th = thermal_from_config(cfg); pg = pg_from_config(cfg); st_ = cfg["stack"]; so = cfg["signoff"]
    dies = st_["dies"]
    R = []
    R.append(("__sec__", "Stack case"))
    R.append(("dies (bottom to top)", "", " / ".join(f"{d['name']} {d['P']:g} W" for d in dies), SWEEPS["top_power"] + " (top die)"))
    R.append(("die size / thickness", "", f"{dies[0]['size']*1e3:g} mm / {dies[0]['thickness']*1e6:g} $\\mu$m", ""))
    R.append(("power blocks per die", "", f"{dies[0]['nblk']}$\\times${dies[0]['nblk']}", ""))
    hot = [f"{d['name']} block {tuple(d['hot'][0][0])} $\\times${d['hot'][0][1]:g}" for d in dies if d.get("hot")]
    R.append(("hot blocks (power factor)", "", "; ".join(hot) if hot else "none", "position sweep (E3)"))
    R.append(("vertical feed pitch / kind", "", " / ".join(f"{d['feed_pitch']*1e6:g} $\\mu$m {d['feed_kind']}" for d in dies), SWEEPS["feed_pitch"] + " (bottom die)"))
    R.append(("hybrid-bond layer", "$t_b$, $k_b$", f"{st_['bond_thick']*1e6:g} $\\mu$m, {st_['bond_k']:g} W/mK", SWEEPS["bond_k"]))
    R.append(("TIM thickness", "", f"{st_['tim_thick']*1e6:g} $\\mu$m", ""))
    R.append(("ambient / package resistance", "$T_a$, $R_{conv}$", f"{st_['ambient']-273.15:g} $^\\circ$C, {st_['r_convec']:g} K/W", SWEEPS["r_convec"]))
    R.append(("layer order", "", "logic die on top (heat-sink side)", SWEEPS["layer_order"]))
    R.append(("__sec__", "Power grid (per die)"))
    R.append(("strap pitch / thickness", "$p$, $H$", f"{pg.pitch*1e6:g} $\\mu$m / {pg.H*1e6:g} $\\mu$m", ""))
    if sizing:
        R.append(("strap width (die-level sign-off sizing)", "$W$", ", ".join(f"{n} {v['W_um']:.2f}" for n, v in sizing.items()) + " $\\mu$m", "re-sized in E6"))
    else:
        R.append(("strap width", "$W$", f"{pg.W*1e6:g} $\\mu$m (before sizing)", ""))
    R.append(("supply / current fraction", "$V_{dd}$", f"{pg.Vdd:g} V / {pg.current_fraction:g}", ""))
    R.append(("Cu resistivity", "$\\rho$", f"{pg.rho*1e8:g} $\\mu\\Omega$cm", ""))
    R.append(("__sec__", "Electromigration / stress (Korhonen)"))
    R.append(("activation energy", "$E_a$", f"{em.Ea_eV:g} eV", SWEEPS["Ea"]))
    R.append(("diffusivity prefactor", "$D_0$", f"{em.D0*1e8:.2g}$\\times$10$^{{-8}}$ m$^2$/s", ""))
    R.append(("effective charge", "$Z^*$", f"{em.Z_eff:g}", ""))
    R.append(("effective bulk modulus", "$B$", f"{em.B/1e9:g} GPa", ""))
    R.append(("atomic volume", "$\\Omega$", f"{em.Omega*1e29:.3g}$\\times$10$^{{-29}}$ m$^3$", ""))
    R.append(("heat of transport (TM)", "$Q^*$", f"{em.Q_eV:g} eV", "0 (no-TM variant)"))
    R.append(("critical stress", "$\\sigma_{crit}$", f"{em.sigma_crit/1e6:g} MPa", SWEEPS["sigma_crit"]))
    R.append(("residual stress", "$\\sigma_T(T)$", f"$B\\,\\Delta\\alpha\\,(T_0 - T)$, $T_0$ = {em.T_zero_stress-273.15:g} $^\\circ$C, $\\Delta\\alpha$ = {em.dalpha*1e6:.1f} ppm/K", "frozen (variant)"))
    R.append(("__sec__", "Wire self-heating (fin model)"))
    R.append(("Cu / ILD conductivity", "$k_{cu}$, $k_{ILD}$", f"{th.k_cu:g} / {th.k_ild:g} W/mK", ""))
    R.append(("rail / ILD thickness", "$t_{cu}$, $t_{ILD}$", f"{th.t_cu*1e6:g} / {th.t_ild*1e6:g} $\\mu$m", ""))
    R.append(("thermal length (two-sided / single)", "$\\Gamma$", f"{th.Gamma(True)*1e6:.1f} / {th.Gamma(False)*1e6:.1f} $\\mu$m", ""))
    R.append(("__sec__", "Sign-off benchmark and solver"))
    R.append(("lifetime target / margin", "$t_h$, $m$", f"{so['horizon_years']:g} yr $\\times$ {so['margin']:g}", SWEEPS["margin"]))
    R.append(("rule sign-off temperature", "$T_{rule}$", f"{so['T_uniform']-273.15:g} $^\\circ$C (uniform)", ""))
    R.append(("strap width search range", "", f"{so['W_min']*1e6:g} -- {so['W_max']*1e6:g} $\\mu$m", ""))
    t = cfg["times"]
    R.append(("output time grid", "$M$", f"{t['n']} log-spaced points, 10$^{{{t['log10_start']:g}}}$ -- 10$^{{{t['log10_end']:g}}}$ s", ""))
    R.append(("reference solver cells / segment", "", f"{cfg['truth']['n_cells']}", ""))
    R.append(("HotSpot grid", "", f"{st_['grid']}$\\times${st_['grid']} per layer", ""))
    return R


def to_latex(R, name):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Parameters of the stack case, model and benchmark (" + name + r"). The last column lists the ranges every conclusion was tested against (Sec.~VI-F).}",
         r"\label{tab:params}", r"\footnotesize", r"\begin{tabular}{@{}llll@{}}", r"\toprule",
         r"parameter & symbol & value & swept range \\", r"\midrule"]
    for r in R:
        if r[0] == "__sec__":
            L.append(r"\multicolumn{4}{@{}l}{\textit{" + r[1] + r"}} \\")
        else:
            L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def to_md(R):
    L = ["| parameter | symbol | value | swept range |", "|---|---|---|---|"]
    for r in R:
        L.append(f"| **{r[1]}** | | | |" if r[0] == "__sec__" else "| " + " | ".join(r) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--root", required=True); ap.add_argument("--name", default=None)
    a = ap.parse_args(); cfg = load_config(a.config); name = a.name or cfg["name"]
    base = os.path.join(os.path.expanduser(a.root), name); sizing = load_sizing(sizing_path(base))
    R = rows_for(cfg, sizing); out = os.path.join(base, "tables"); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "table_params.tex"), "w").write(to_latex(R, name))
    open(os.path.join(out, "table_params.md"), "w").write(to_md(R))
    json.dump([list(r) for r in R], open(os.path.join(out, "table_params.json"), "w"), indent=1)
    print(to_md(R)); print("wrote", out)


if __name__ == "__main__":
    main()
