"""
Collect every figure and every summary JSON of one case into one folder and write an index.

    python tools/collect_figures.py --root ~/stackem_work/outputs [--name base3]

Creates <root>/<name>/figs_index/  with  fig_<exp>_<name>.pdf/png, the JSON results as <exp>_<file>.json, and
INDEX.md mapping each file to the paper figure/table it feeds.  The folder only holds COPIES; it can be deleted
and rebuilt at any time.  (JSON copies carry the experiment folder as a prefix; originally they were copied under
their bare names, so that skn/train_summary.json and skn_pde/train_summary.json overwrote each other.)
"""
import argparse, glob, json, os, shutil

MAP = {  # file stem -> (paper item, caption gist)
    "e0_kernels_vs_series": ("Fig. S1", "FDM kernels vs closed-form series"),
    "e0_closure_convergence": ("Fig. S2", "junction closure error vs time steps"),
    "e1_kernels": ("Fig. 4", "learned kernels vs FDM on held-out profiles"),
    "e1_kernel_cuts": ("Fig. 4a", "scaled kernels at the segment ends vs tau (FDM lines, SKN markers)"),
    "framework": ("Fig. 2", "framework block diagram"),
    "signoff_sizing": ("Fig. 5", "benchmark definition: sizing curves and strap widths"),
    "e3_thermal_slice": ("Fig. 6", "vertical cut: in-stack vs standalone temperature"),
    "e3_margin": ("Fig. 8b", "stress-margin maps at the sign-off horizon"),
    "e3_hotspot_alignment": ("Fig. 9", "hot-block position sweep: alignment of the lower dies"),
    "e5_repair": ("Fig. 12", "targeted vs blanket repair"),
    "e1_error_by_tau": ("Fig. 4b", "kernel error vs dimensionless time"),
    "e2_parity_full": ("Fig. 5", "t_nuc parity, full 3-D variant"),
    "e2_parity_alone": ("Fig. S3", "t_nuc parity, standalone dies"),
    "e2_parity_uniform": ("Fig. S3", "t_nuc parity, uniform temperature"),
    "e3_stack3d_full": ("Fig. 1", "exploded stack, rails coloured by t_nuc"),
    "e3_diemaps_full": ("Fig. 6", "per-die maps, full physics"),
    "e3_diemaps_alone": ("Fig. 6", "per-die maps, standalone"),
    "e3_cdf": ("Fig. 7", "cumulative nucleation curves per variant"),
    "e3_effect_ratios": ("Fig. 8", "optimism factors of the simplifications"),
    "e3_stress_surface": ("Fig. 3", "sigma(x,t) of the critical rail"),
    "e7_ranking": ("Fig. 9", "top-k / Kendall of every simplification"),
    "e7_confusion": ("Fig. 9b", "mortal-set confusion within 10 yr"),
    "e4_timing": ("Fig. 10", "wall-clock scaling"),
    "e5_budget_trajectory": ("Fig. 11", "power re-allocation trajectory"),
    "e5_diemaps_optimized": ("Fig. 11b", "die maps after re-allocation"),
    "e6_sweeps": ("Fig. 12", "design-space sweeps"),
    "e8_diemaps_full": ("Fig. 13", "ev6 real floorplan, full"),
    "e8_diemaps_others_off": ("Fig. 13", "ev6 real floorplan, others off"),
    "e8_cdf": ("Fig. 13b", "ev6 cumulative curves"),
    "e8_parity": ("Fig. S4", "ev6 StackEM parity"),
    "train_curves": ("Fig. S5", "SKN training curves"),
}

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--name", default="base3")
a = ap.parse_args(); base = os.path.join(os.path.expanduser(a.root), a.name); out = os.path.join(base, "figs_index"); os.makedirs(out, exist_ok=True)
lines = ["# Figure / table index (StackEM run '%s')" % a.name, "", "| file | paper item | content | source |", "|---|---|---|---|"]
# figures (pdf + png), their single panels (*_panelN.*) and the data behind them (*_data.npz + *_data.json, written by
# viz.style.save: every plotted series with the axes' labels, so that the figures can be re-plotted without the package)
data_dir = os.path.join(out, "figure_data"); os.makedirs(data_dir, exist_ok=True); n_data = 0
for f in sorted(glob.glob(os.path.join(base, "*", "*.pdf")) + glob.glob(os.path.join(base, "*", "*.png"))):
    if os.path.basename(os.path.dirname(f)) == "figs_index": continue
    stem = os.path.splitext(os.path.basename(f))[0]; exp = os.path.basename(os.path.dirname(f))
    key = next((k for k in MAP if stem.startswith(k)), None)
    dst = os.path.join(out, f"fig_{exp}_{os.path.basename(f)}"); shutil.copy(f, dst)
    item, cap = MAP.get(key, ("-", ""))
    if "_panel" in stem: item, cap = item + " (panel)", cap
    lines.append(f"| {os.path.basename(dst)} | {item} | {cap} | {os.path.relpath(f, base)} |")
for f in sorted(glob.glob(os.path.join(base, "*", "*_data.npz")) + glob.glob(os.path.join(base, "*", "*_data.json"))):
    if os.path.basename(os.path.dirname(f)) == "figs_index": continue
    exp = os.path.basename(os.path.dirname(f)); shutil.copy(f, os.path.join(data_dir, f"{exp}_{os.path.basename(f)}")); n_data += 1
lines.append(f"| figure_data/ | data | {n_data} files: the arrays behind every figure (npz) + axes/series index (json) | */*_data.* |")
for f in sorted(glob.glob(os.path.join(base, "*", "*.json"))):
    if os.path.basename(os.path.dirname(f)) == "figs_index": continue      # a previous index run copied these here
    if f.endswith(("_summary.json", "_validation.json", "_timing.json", "_ranking.json", "_sweeps.json", "_eval.json", "_accuracy.json", "_repair.json", "_sizing.json", ".json")):
        exp = os.path.basename(os.path.dirname(f)); dst = f"{exp}_{os.path.basename(f)}"
        shutil.copy(f, os.path.join(out, dst)); lines.append(f"| {dst} | table source | JSON metrics | {os.path.relpath(f, base)} |")
for f in glob.glob(os.path.join(base, "tables", "*")):
    shutil.copy(f, os.path.join(out, os.path.basename(f))); lines.append(f"| {os.path.basename(f)} | Table II | parameter table | {os.path.relpath(f, base)} |")
open(os.path.join(out, "INDEX.md"), "w").write("\n".join(lines) + "\n"); print("wrote", out, len(lines) - 4, "entries")
