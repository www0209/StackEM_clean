#!/usr/bin/env python3
"""Compare a fresh output root with the released results in reference_results/.

    python tools/check_results.py                                   # ~/stackem_work/outputs vs <repo>/reference_results
    python tools/check_results.py --root <output root> [--ref <reference dir>] [--retrained] [-v]

For every summary JSON that exists in BOTH trees the script walks the two files in parallel and compares
every number.  How strictly a number is compared depends on where it comes from:

    REF   produced by the reference solver, HotSpot or the power-grid solve (deterministic numerics)
            integers (rail counts, mortal counts, iterations)   must be EQUAL
            floats                                              relative tolerance --ref-tol (default 1e-3;
                                                                5e-3 for the earliest t_nuc recorded by the sizing, 5e-2 for E0)
    SUR   produced by the surrogate (SKN + closure): accuracy metrics, predicted t_nuc, rankings
            floats     |new - ref| <= --sur-tol * |ref| + --sur-abs   (defaults 0.02 and 2e-4)
            integers   must be equal
          with --retrained (the networks were trained again, so they are different networks):
            floats     same order of magnitude (factor 3) or |new - ref| <= 0.02
            integers   |new - ref| <= max(2, 10 %)
    TIME  wall-clock seconds, device names, worker counts, commands, dates: never compared

Use the default mode after scripts/run_with_released_weights.sh and --retrained after scripts/run_all.sh.
Output: one line per experiment file with PASS / DIFF, the number of compared values and the largest deviation;
-v lists every deviating value.  Files that exist only in the reference are listed as "not run".
Exit status 1 when at least one file is DIFF.  Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# summary files that are compared (basename -> default class of their numbers, regexes of key paths that are SUR)
FILES = {
    "e0_validation.json":        ("REF", []),
    "e1_summary.json":           ("SUR", []),
    "e2_summary.json":           ("REF", [r"/metrics/", r"/stack_pred/"]),
    "e3_summary.json":           ("REF", [r"^stackem_overlay/"]),
    "e4_timing.json":            ("REF", []),
    "e5_repair.json":            ("SUR", []),
    "e5_summary.json":           ("SUR", []),
    "e6_sweeps.json":            ("REF", []),
    "e7_ranking.json":           ("REF", [r"^results/stackem/"]),
    "e8_summary.json":           ("REF", [r"^stackem_overlay/"]),
    "signoff_sizing.json":       ("REF", []),
    "ibmpg_summary.json":        ("REF", [r"^tnuc_rel_", r"^top10_hit", r"^kendall_top50", r"^n_mortal_pred", r"^mortality_agreement"]),
    "comsol_rails_compare.json": ("REF", [r"^stackem_"]),
}
# keys whose whole subtree is never compared (timings, provenance, optimiser traces, plot-only data)
SKIP_KEYS = {"_meta", "timings", "history", "trajectory", "cmd", "time", "device", "workers", "truth_workers", "e3_dir", "spice",
             "closure_backend", "kernel_evals_by_engine", "parallel_bound", "comsol_seconds_per_rail", "warnings"}
SKIP_KEY_RE = re.compile(r"^seconds|_seconds$|^seconds_")
# short strings that carry a result (everything else that is text is ignored)
STRING_KEYS = {"critical_die", "variant", "provider", "main_provider", "closure_mode", "kind", "die", "earliest_rail"}
# E0 reports error magnitudes of the solver itself; they move with library versions, so they get a wider band
FILE_REF_TOL = {"e0_validation.json": 0.05}
# The released sizing files were produced with 64 output times, a fresh run sizes with the 32 of configs/base3.json.
# The strap widths W agree to all digits; only the recorded earliest t_nuc at that width moves by up to 1e-3.
KEY_REF_TOL = {"t_earliest_s": 5e-3, "t_earliest_years": 5e-3}
NOISE = 1e-9          # two values below this magnitude are both "numerical zero"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)                      # Python's json accepts the bare Infinity / NaN that some files contain


def num(v):
    """float for numbers and for the strings 'inf' / '-inf' / 'nan' that the experiments write; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v.lower() in ("inf", "-inf", "nan", "infinity", "-infinity"):
        return float(v.lower().replace("infinity", "inf"))
    return None


class Report:
    def __init__(self):
        self.n = 0; self.diffs = []          # (path, ref, new, class, deviation)
        self.maxdev = 0.0                    # largest finite relative deviation among all compared numbers

    def add(self, ok, path, ref, new, cls, dev):
        self.n += 1
        if isinstance(dev, float) and math.isfinite(dev):
            self.maxdev = max(self.maxdev, dev)
        if not ok:
            self.diffs.append((path, ref, new, cls, dev))


def compare_leaf(rep, path, key, ref, new, cls, opt, fname):
    a, b = num(ref), num(new)
    if a is None or b is None:
        if isinstance(ref, bool) or isinstance(new, bool):
            rep.add(ref == new, path, ref, new, cls, float("inf") if ref != new else 0.0)
        elif isinstance(ref, str) and key in STRING_KEYS:
            rep.add(ref == new, path, ref, new, cls, float("inf") if ref != new else 0.0)
        elif (ref is None) != (new is None):
            rep.add(False, path, ref, new, cls, float("inf"))
        return
    if math.isnan(a) or math.isnan(b):
        rep.add(math.isnan(a) == math.isnan(b), path, ref, new, cls, float("inf")); return
    if math.isinf(a) or math.isinf(b):
        rep.add(a == b, path, ref, new, cls, float("inf") if a != b else 0.0); return
    is_int = isinstance(ref, int) and isinstance(new, int)
    dev = abs(b - a) / abs(a) if a != 0 else abs(b - a)
    if is_int:
        if cls == "SUR" and opt.retrained:
            ok = abs(b - a) <= max(2.0, 0.10 * abs(a))
        else:
            ok = a == b
    elif cls == "REF":
        tol = max(FILE_REF_TOL.get(fname, opt.ref_tol), KEY_REF_TOL.get(key, 0.0))
        ok = abs(b - a) <= tol * abs(a) or (abs(a) < NOISE and abs(b) < NOISE)
    elif opt.retrained:
        ok = abs(b - a) <= 0.02 or (a * b > 0 and 1 / 3 <= b / a <= 3) or (abs(a) < NOISE and abs(b) < NOISE)
    else:
        ok = abs(b - a) <= opt.sur_tol * abs(a) + opt.sur_abs
    rep.add(ok, path, ref, new, cls, dev)


def walk(rep, ref, new, path, default_cls, sur_res, opt, fname, key=""):
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            rep.add(False, path, "dict", type(new).__name__, default_cls, float("inf")); return
        for k, v in ref.items():
            if k in SKIP_KEYS or SKIP_KEY_RE.search(k):
                continue
            p = f"{path}/{k}" if path else k
            if k not in new:
                rep.add(False, p, "present", "missing", default_cls, float("inf")); continue
            walk(rep, v, new[k], p, default_cls, sur_res, opt, fname, k)
        return
    if isinstance(ref, list):
        if not isinstance(new, list) or len(new) != len(ref):
            rep.add(False, path, f"list[{len(ref)}]", f"list[{len(new)}]" if isinstance(new, list) else type(new).__name__, default_cls, float("inf")); return
        for i, (x, y) in enumerate(zip(ref, new)):
            walk(rep, x, y, f"{path}/{i}", default_cls, sur_res, opt, fname, key)
        return
    cls = "SUR" if default_cls == "SUR" or any(r.search(path + "/") for r in sur_res) else "REF"
    compare_leaf(rep, path, key, ref, new, cls, opt, fname)


def find_files(ref_root):
    out = []
    for d, dirs, files in os.walk(ref_root):
        dirs.sort()
        depth = os.path.relpath(d, ref_root).count(os.sep)
        if depth >= 2:                       # <case>/<experiment>/ is the deepest level that holds summaries
            dirs[:] = []
        for f in sorted(files):
            if f in FILES:
                out.append(os.path.relpath(os.path.join(d, f), ref_root))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(os.environ.get("STACKEM_WORK", os.path.expanduser("~/stackem_work")), "outputs"), help="fresh output root")
    ap.add_argument("--ref", default=os.path.join(REPO, "reference_results"), help="released results")
    ap.add_argument("--retrained", action="store_true", help="the networks were trained again: loose comparison of the surrogate numbers")
    ap.add_argument("--ref-tol", type=float, default=1e-3, help="relative tolerance of reference-solver floats")
    ap.add_argument("--sur-tol", type=float, default=0.02, help="relative tolerance of surrogate floats (released weights)")
    ap.add_argument("--sur-abs", type=float, default=2e-4, help="absolute tolerance added for surrogate floats")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every deviating value")
    opt = ap.parse_args()
    root, ref_root = os.path.expanduser(opt.root), os.path.expanduser(opt.ref)
    if not os.path.isdir(ref_root):
        print(f"reference directory not found: {ref_root}"); return 2
    if not os.path.isdir(root):
        print(f"output root not found: {root}"); return 2
    files = find_files(ref_root)
    print(f"fresh results : {root}\nreference     : {ref_root}\nmode          : {'retrained networks (loose surrogate tolerances)' if opt.retrained else 'released weights'}\n")
    print(f"{'experiment file':46s} {'status':8s} {'values':>7s} {'differ':>7s}  worst deviation")
    n_pass = n_diff = n_missing = 0
    for rel in files:
        new_path = os.path.join(root, rel)
        if not os.path.exists(new_path):
            print(f"{rel:46s} {'not run':8s}"); n_missing += 1; continue
        fname = os.path.basename(rel); default_cls, sur = FILES[fname]
        rep = Report()
        try:
            walk(rep, load(os.path.join(ref_root, rel)), load(new_path), "", default_cls, [re.compile(s) for s in sur], opt, fname)
        except Exception as e:                                   # unreadable file: report, keep going
            print(f"{rel:46s} {'ERROR':8s} {type(e).__name__}: {e}"); n_diff += 1; continue
        if rep.diffs:
            n_diff += 1
            p, a, b, cls, dev = max(rep.diffs, key=lambda d: d[4])
            print(f"{rel:46s} {'DIFF':8s} {rep.n:7d} {len(rep.diffs):7d}  {p}: ref {a!r} new {b!r} [{cls}]")
            if opt.verbose:
                for p, a, b, cls, dev in rep.diffs:
                    print(f"      {cls} {p}: ref {a!r}  new {b!r}  rel.dev {dev:.3g}")
        else:
            n_pass += 1
            print(f"{rel:46s} {'PASS':8s} {rep.n:7d} {0:7d}  largest relative deviation {rep.maxdev:.1e}")
    print(f"\n{n_pass} PASS, {n_diff} DIFF, {n_missing} not run (present only in the reference)")
    print("timings are never compared; E4 timing files are only checked for the same sizes and settings")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())
