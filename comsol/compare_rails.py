"""
COMSOL large-scale cross-check, step 3: COMSOL vs reference vs StackEM on the same rails, plus the COMSOL
timing curve for the speed figure.

    python comsol/compare_rails.py --dir ~/stackem_work/outputs/comsol \
        [--sizes 82,246,1000,5000,20000] [--parallel 32]

Reads   rails_tnuc.csv (reference + StackEM), comsol_rails_result.csv (livelink_rails.m)
Writes  comsol_rails_compare.json     error statistics (COMSOL vs reference, StackEM vs COMSOL, StackEM vs reference)
        comsol_timing.csv             n_rails, seconds  = n_rails x mean COMSOL seconds per rail (sequential; the
                                      --parallel column gives the optimistic bound with that many independent COMSOL
                                      processes) -> tools/paper_timing_fig.py --external "COMSOL (FEM)=..."
        comsol_rails_parity.(pdf|png) parity StackEM vs COMSOL
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
Y = 365.25 * 86400.0
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
ap.add_argument("--sizes", default="82,246,1000,5000,20000"); ap.add_argument("--parallel", type=int, default=32)
a = ap.parse_args()
ref = {int(r["rail"]): r for r in csv.DictReader(open(os.path.join(a.dir, "rails_tnuc.csv")))}
com = {int(r["rail"]): r for r in csv.DictReader(open(os.path.join(a.dir, "comsol_rails_result.csv")))}
ids = sorted(set(ref) & set(com))
tf = np.array([float(ref[i]["t_nuc_fdm_s"]) for i in ids]); ts = np.array([float(ref[i]["t_nuc_stackem_s"]) for i in ids])
tc = np.array([float(com[i]["t_nuc_comsol_s"]) for i in ids]); sec = np.array([float(com[i]["seconds_total"]) for i in ids])
nseg = np.array([int(com[i]["n_seg"]) for i in ids])
m = np.isfinite(tf) & np.isfinite(tc) & np.isfinite(ts)
def stats(x, y):
    r = np.abs(x[m] - y[m]) / y[m]
    return dict(n=int(m.sum()), median=float(np.median(r)), p90=float(np.percentile(r, 90)), max=float(r.max()))
out = dict(comsol_vs_reference=stats(tc, tf), stackem_vs_comsol=stats(ts, tc), stackem_vs_reference=stats(ts, tf),
           mortality_agreement_comsol_reference=float(np.mean(np.isfinite(tc) == np.isfinite(tf))),
           comsol_seconds_per_rail=dict(mean=float(sec.mean()), median=float(np.median(sec)), per_segment=float(sec.sum() / nseg.sum())),
           n_rails_timed=len(ids), parallel_bound=a.parallel)
for k, v in out.items(): print(k, v)
json.dump(out, open(os.path.join(a.dir, "comsol_rails_compare.json"), "w"), indent=1)
sizes = [int(s) for s in a.sizes.split(",")]
with open(os.path.join(a.dir, "comsol_timing.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n_rails", "seconds", f"seconds_{a.parallel}_processes"])
    for n in sizes: w.writerow([n, f"{n * sec.mean():.1f}", f"{n * sec.mean() / a.parallel:.1f}"])
print("comsol_timing.csv:", [(n, round(n * sec.mean())) for n in sizes], f"(sequential; /{a.parallel} for the parallel bound)")
from stackem.viz.plots import plot_parity
plot_parity(tc, ts, os.path.join(a.dir, "comsol_rails_parity"), title="StackEM vs COMSOL", ref_label="COMSOL (FEM) $t_{nuc}$ (years)")
print("wrote", os.path.join(a.dir, "comsol_rails_parity.pdf/.png"))
