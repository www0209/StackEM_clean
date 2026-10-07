"""
COMSOL large-scale cross-check, step 1: export the REAL rails of a stack (the rails that E2 solved) with their
reference and StackEM nucleation times, so that COMSOL can solve exactly the same rails.

    python comsol/export_rails.py --e2 ~/stackem_work/outputs/base3/e2 \
        --out ~/stackem_work/outputs/comsol [--max-rails 0] [--mortal-only]

Reads   e2_full_truth_rails.json  (per-rail geometry, currents, thermal profile, sigma_T of every segment)
        e2_full_truth.json        (reference t_nuc per rail)      e2_full_pred.json  (StackEM t_nuc per rail)
Writes  rails.csv         one row per segment: rail, k, L_m, j_A_per_m2, T_L_K, T_R_K, T_m_K, Gamma_m, sigma_T_Pa
        rails_tnuc.csv    one row per rail: rail, die, kind, index, n_seg, t_nuc_fdm_s, t_nuc_stackem_s
        constants.json    the EM constants the COMSOL variables need (D0, Ea, B, Omega, rho, Z*, Q*, sigma_crit)
Then comsol/livelink_rails.m solves every rail in COMSOL and comsol/compare_rails.py closes the loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stackem.constants import default_em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True, help="the e2/ directory of the student-engine run, e.g. <root>/base3/e2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rails", type=int, default=0, help="0 = every rail; N = the N earliest-failing rails (ranking by the reference)")
    ap.add_argument("--mortal-only", action="store_true", help="skip rails that never nucleate (immortal in the reference)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    geo = json.load(open(os.path.join(a.e2, "e2_full_truth_rails.json")))["rails"]
    tru = {(r["die"], r["kind"], r["index"]): r for r in json.load(open(os.path.join(a.e2, "e2_full_truth.json")))["rails"]}
    pre = {(r["die"], r["kind"], r["index"]): r for r in json.load(open(os.path.join(a.e2, "e2_full_pred.json")))["rails"]}
    rows = []
    for g in geo:
        key = (g["die"], g["kind"], g["index"]); t = float(tru[key]["t_nuc_s"]); p = float(pre[key]["t_nuc_s"]) if key in pre else float("inf")
        if a.mortal_only and not (t < float("inf")):
            continue
        rows.append((t, g, p))
    rows.sort(key=lambda x: x[0])
    if a.max_rails > 0:
        rows = rows[:a.max_rails]
    em = default_em()
    with open(os.path.join(a.out, "rails.csv"), "w", newline="") as f, open(os.path.join(a.out, "rails_tnuc.csv"), "w", newline="") as g:
        w = csv.writer(f); w.writerow(["rail", "k", "L_m", "j_A_per_m2", "T_L_K", "T_R_K", "T_m_K", "Gamma_m", "sigma_T_Pa"])
        v = csv.writer(g); v.writerow(["rail", "die", "kind", "index", "n_seg", "t_nuc_fdm_s", "t_nuc_stackem_s"])
        for i, (t, r, p) in enumerate(rows):
            S = len(r["L"])
            for k in range(S):
                w.writerow([i, k, f"{r['L'][k]:.9e}", f"{r['j'][k]:.9e}", f"{r['T_L'][k]:.6f}", f"{r['T_R'][k]:.6f}", f"{r['T_m'][k]:.6f}", f"{r['Gamma']:.9e}", f"{r['sigma_T'][k]:.6e}"])
            v.writerow([i, r["die"], r["kind"], r["index"], S, f"{t:.9e}", f"{p:.9e}"])
    json.dump(dict(D0=em.D0, Ea_eV=em.Ea_eV, B=em.B, Omega=em.Omega, rho=em.rho, Z_eff=em.Z_eff, Q_eV=em.Q_eV, sigma_crit=em.sigma_crit,
                   note="T(xi) = T_L + (T_R - T_L) xi + T_m (1 - cosh((xi - 0.5) lam) / cosh(0.5 lam)), lam = L / Gamma; "
                        "kappa = D0 exp(-Ea/kT) B Omega / kT; G = e Z_eff rho j / Omega; M = Q T'(x) / (Omega T); "
                        "d sigma/dt = d/dx [ kappa (sigma_x - G - M) ], flux kappa (sigma_x - G - M) = 0 at the rail ends (Z_eff = +10)"),
              open(os.path.join(a.out, "constants.json"), "w"), indent=1)
    n_seg = [len(r["L"]) for _, r, _ in rows]
    print(f"wrote {len(rows)} rails ({sum(n_seg)} segments, {min(n_seg)}-{max(n_seg)} per rail) to {a.out}")


if __name__ == "__main__":
    main()
