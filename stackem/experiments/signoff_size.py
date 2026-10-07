"""
Sign-off sizing of the benchmark stack (run once per case, before E2-E8).

    python -m stackem.experiments.signoff_size --root DIR [--config cfg.json] [--smoke]

Every die is simulated standalone (own package, single-sided heat loss) and the strap
width of its analysed rail pair is sized so that the die's earliest nucleation time
equals  horizon_years x margin  (config "signoff", default 10 yr x 2) with the reference
solver.  Writes <root>/<name>/signoff/signoff_sizing.json, which every later experiment
picks up automatically (the stack is then "three dies that each pass their own sign-off").
Also reports, for information, the width the uniform worst-case rule (T = signoff.T_uniform,
constant sigma_T) would give: the industrial-rule sizing is typically wider (pessimistic).
Figure signoff_sizing.(pdf|png): the sizing curves earliest t_nuc(W) of every die (both practices) and
the resulting strap widths (the benchmark definition in one picture).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
import numpy as np

from ._common import base_parser, setup, out_dir, record
from ..config import hotspot_dir
from ..hotspot_stack import HotSpotRunner, run_stack
from ..power_grid import DieGrid, embed_thermal
from ..signoff import size_stack, size_strap_width, sizing_path
from ..constants import SEC_PER_YEAR


def main(argv=None):
    ap = base_parser("sign-off sizing of the stack case")
    ap.add_argument("--margin", type=float, default=None); ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--skip-rule", action="store_true", help="skip the uniform worst-case rule sizing (information only)")
    args = ap.parse_args(argv); args.no_sizing = True          # always size from the raw config
    ctx = setup(args); cfg = ctx["cfg"]; spec = ctx["spec"]; em = ctx["em"]; th = ctx["th"]; times = ctx["times"]
    so = cfg["signoff"]; horizon = args.horizon or float(so["horizon_years"]); margin = args.margin or float(so["margin"])
    d = out_dir(cfg, "signoff")
    t0 = time.time()
    tf = run_stack(spec, os.path.join(d, "hotspot"), HotSpotRunner(hotspot_dir(cfg)), "stack", True)
    print(f"HotSpot (stack + {len(spec.dies)} standalone runs) {time.time()-t0:.0f}s")
    n_cells = 40 if args.smoke else cfg["truth"]["n_cells"]
    t0 = time.time()
    sizing = size_stack(spec, ctx["pg"], em, th, tf, times, horizon, margin, float(so["W_min"]), float(so["W_max"]), n_cells)
    print(f"sizing {time.time()-t0:.0f}s")
    rule = {}
    if not args.skip_rule:
        T_rule = float(so["T_uniform"]); em_c = replace(em, sigma_T_model="constant", sigma_T_const=float(em.sigma_T(T_rule)))
        for die in spec.dies:
            def rails_of_W(W, die=die):
                g = DieGrid(replace(die, rail_W=float(W)), ctx["pg"])
                return embed_thermal(g, g.solve_from_blocks(), tf.T_die[die.name], em_c, th, two_sided=False, uniform_T=T_rule)
            r = size_strap_width(rails_of_W, em_c, times, horizon * margin * SEC_PER_YEAR, float(so["W_min"]), float(so["W_max"]), n_cells)
            rule[die.name] = dict(W_um=r["W"] * 1e6, t_earliest_years=r["t_earliest_s"] / SEC_PER_YEAR, saturated=r["saturated"], history=r["history"])
            print(f"  uniform {T_rule-273.15:.0f} C rule would size {die.name} at W = {r['W']*1e6:.2f} um  (3-D-aware standalone: {sizing[die.name]['W_um']:.2f} um)")
    payload = dict(horizon_years=horizon, margin=margin, rule=dict(kind="standalone die, own HotSpot field, single-sided heat loss, reference solver"),
                   dies=sizing, uniform_rule_sizing=rule, thermal=tf.summary(), n_cells=n_cells)
    os.makedirs(os.path.dirname(sizing_path(out_dir(cfg))), exist_ok=True)
    json.dump(payload, open(sizing_path(out_dir(cfg)), "w"), indent=1)
    print(f"[saved] {sizing_path(out_dir(cfg))}")
    from ..viz.plots import plot_signoff_sizing
    plot_signoff_sizing(sizing, rule, horizon * margin, os.path.join(d, "signoff_sizing"))


if __name__ == "__main__":
    main()
