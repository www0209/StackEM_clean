"""
Side-by-side comparison of the two engines on the same stack, the same sizing and the same reference solver:

    teacher engine   6x256 SKN, direct kernel look-up, M = 64      results in <root>/base3_teacher
    student engine   4x128 SKN, tabulated look-up, M = 32          results in <root>/base3   (the engine of every other result)

    python tools/compare_engines.py --root ~/stackem_work/outputs [--student base3] [--teacher base3_teacher]

Reads the summary JSONs of both runs (E1, E2, E4, E5, E8 and, where present, E3 overlay / E7; distillation summary) and writes
    <root>/<teacher>/comparison/comparison_teacher_student.md / .tex / .json   (table: accuracy and cost of both engines)
    <root>/<teacher>/comparison/compare_timing.(pdf|png)                       (both E4 curves on one axis)
    <root>/<teacher>/comparison/compare_accuracy.(pdf|png)                     (t_nuc error per die, both engines)
Everything is read from files; nothing is recomputed.  A "-" means that the run did not produce that quantity
(the teacher-engine comparison runs E1, E2, E4, E5 and E8 only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stackem.viz import style as st        # noqa: E402


def _load_json(root, name, sub, fn):
    p = os.path.join(root, name, sub, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{100*x:.{nd}f} %" if pct else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--student", default="base3", help="case directory of the student-engine run")
    ap.add_argument("--teacher", default="base3_teacher", help="case directory of the teacher-engine run")
    a = ap.parse_args(); root = os.path.expanduser(a.root)
    out = os.path.join(root, a.teacher, "comparison"); os.makedirs(out, exist_ok=True)
    _load = _load_json

    def load(which, _unused, sub, fn):                     # which: "P" = teacher-engine run, "D" = student-engine run, "B" = shared base case
        return _load(root, {"P": a.teacher, "D": a.student, "B": a.student}[which], sub, fn)
    P, D, name = "P", "D", None
    rows = []                                              # (quantity, teacher engine, student engine)
    # ---- network / closure configuration ---------------------------------------------------------------------
    cfgP = load(P, name, "e2", "config_used.json") or {}; cfgD = load(D, name, "e2", "config_used.json") or {}
    dist = load(D, name, "skn_distill", "distill_summary.json") or {}
    trP = load("B", name, "skn", "train_summary.json") or {}                     # the teacher weights live in the base case
    rows += [("__sec__", "Configuration"),
             ("kernel network", f"teacher SKN, {trP.get('n_params', '-'):,} params" if trP.get("n_params") else "teacher SKN", f"student, {dist.get('student', {}).get('n_params', '-'):,} params ({dist.get('compression', float('nan')):.0f}x smaller)" if dist else "-"),
             ("kernel look-up", cfgP.get("closure", {}).get("mode", "direct"), f"{cfgD.get('closure', {}).get('mode', '-')} ({cfgD.get('closure', {}).get('n_tab', '-')} pts / segment)"),
             ("output times M", str(cfgP.get("times", {}).get("n", "-")), str(cfgD.get("times", {}).get("n", "-")))]
    # ---- E1 kernel accuracy ---------------------------------------------------------------------------------
    e1P = load(P, name, "e1", "e1_summary.json") or {}; e1D = load(D, name, "e1", "e1_summary.json") or {}
    rows.append(("__sec__", "E1 kernel accuracy vs FDM (held-out profiles, median rel-L2)"))
    for k, lab in (("s_G", "$s_G$"), ("s_M", "$s_M$"), ("a", "$a$")):
        rows.append((lab, fmt(e1P.get("val", {}).get(k, {}).get("rel_l2_median"), True, 3), fmt(e1D.get("val", {}).get(k, {}).get("rel_l2_median"), True, 3)))
    # ---- E2 rail accuracy -----------------------------------------------------------------------------------
    e2P = load(P, name, "e2", "e2_summary.json") or {}; e2D = load(D, name, "e2", "e2_summary.json") or {}
    rows.append(("__sec__", "E2 $t_{nuc}$ error vs FDM truth, full 3-D variant (median / p90)"))
    dies = list((e2P.get("full", {}).get("metrics") or e2D.get("full", {}).get("metrics") or {}).keys())
    acc = {}
    for dn in dies:
        mP = e2P.get("full", {}).get("metrics", {}).get(dn, {}); mD = e2D.get("full", {}).get("metrics", {}).get(dn, {})
        acc[dn] = (mP.get("tnuc_rel_median"), mP.get("tnuc_rel_p90"), mD.get("tnuc_rel_median"), mD.get("tnuc_rel_p90"))
        rows.append((dn, f"{fmt(mP.get('tnuc_rel_median'), True)} / {fmt(mP.get('tnuc_rel_p90'), True)}", f"{fmt(mD.get('tnuc_rel_median'), True)} / {fmt(mD.get('tnuc_rel_p90'), True)}"))
    agP = [e2P.get("full", {}).get("metrics", {}).get(dn, {}).get("mortality_agreement") for dn in dies]
    agD = [e2D.get("full", {}).get("metrics", {}).get(dn, {}).get("mortality_agreement") for dn in dies]
    rows.append(("mortality agreement (min over dies)", fmt(min([x for x in agP if x is not None] or [float("nan")]), True, 1), fmt(min([x for x in agD if x is not None] or [float("nan")]), True, 1)))
    rows.append(("closure time, 246 rails (numpy, s)", fmt(e2P.get("full", {}).get("seconds_closure"), nd=1), fmt(e2D.get("full", {}).get("seconds_closure"), nd=1)))
    # ---- E3 overlay / E7 ------------------------------------------------------------------------------------
    e3P = load(P, name, "e3", "e3_summary.json") or {}; e3D = load(D, name, "e3", "e3_summary.json") or {}
    rows.append(("__sec__", "E3 stack (truth earliest $t_{nuc}$ vs surrogate overlay)"))
    rows.append(("truth earliest (yr)", fmt(e3P.get("stack", {}).get("full", {}).get("earliest_t_nuc_years"), nd=3), fmt(e3D.get("stack", {}).get("full", {}).get("earliest_t_nuc_years"), nd=3)))
    rows.append(("overlay earliest (yr)", fmt(e3P.get("stackem_overlay", {}).get("earliest_t_nuc_years"), nd=3), fmt(e3D.get("stackem_overlay", {}).get("earliest_t_nuc_years"), nd=3)))
    e7P = load(P, name, "e7", "e7_ranking.json") or {}; e7D = load(D, name, "e7", "e7_ranking.json") or {}
    sP = e7P.get("results", {}).get("stackem", {}); sD = e7D.get("results", {}).get("stackem", {})
    rows.append(("E7 surrogate: Top-50 hit / Kendall / mortal-set Jaccard", f"{fmt(sP.get('top50'))} / {fmt(sP.get('kendall'))} / {fmt(sP.get('mortal10_jaccard'))}", f"{fmt(sD.get('top50'))} / {fmt(sD.get('kendall'))} / {fmt(sD.get('mortal10_jaccard'))}"))
    # ---- E5 --------------------------------------------------------------------------------------------------
    e5P = load(P, name, "e5", "e5_summary.json") or {}; e5D = load(D, name, "e5", "e5_summary.json") or {}
    rows.append(("__sec__", "E5 differentiability and budgeting"))

    def fd_dev(e5):
        fd = e5.get("finite_difference_check", [])
        v = [abs(f["autograd"] - f["finite_diff"]) / max(abs(f["finite_diff"]), 1e-9) for f in fd if abs(f["finite_diff"]) > 1e-2]
        return max(v) if v else None
    rows.append(("autograd vs finite difference (max rel. dev., |s| > 0.01/W)", fmt(fd_dev(e5P), True, 2), fmt(fd_dev(e5D), True, 2)))
    rows.append(("earliest $t_{nuc}$ after power re-allocation (truth, yr)", fmt(e5P.get("verification", {}).get("optimized", {}).get("earliest_t_nuc_years"), nd=3), fmt(e5D.get("verification", {}).get("optimized", {}).get("earliest_t_nuc_years"), nd=3)))
    # ---- E8 --------------------------------------------------------------------------------------------------
    e8P = load(P, name, "e8", "e8_summary.json") or {}; e8D = load(D, name, "e8", "e8_summary.json") or {}
    oP = e8P.get("stackem_overlay", {}); oD = e8D.get("stackem_overlay", {})
    if oP or oD:
        rows.append(("__sec__", "E8 ev6 stack (surrogate vs truth)"))
        for k, lab in (("tnuc_rel_median", "t_nuc error median"), ("tnuc_rel_p90", "t_nuc error p90")):
            rows.append((lab, fmt(oP.get(k), True), fmt(oD.get(k), True)))
    # ---- E4 --------------------------------------------------------------------------------------------------
    e4P = load(P, name, "e4", "e4_timing.json") or {}; e4D = load(D, name, "e4", "e4_timing.json") or {}
    rows.append(("__sec__", "E4 wall-clock for 20 000 rails (s)"))

    def at(e4, key_sub, n=20000):
        if not e4: return None
        sizes = e4["sizes"]
        for k, v in e4["timings"].items():
            if all(s in k.lower() for s in key_sub):
                if n in sizes:
                    x = v[sizes.index(n)]; return None if x is None else float(x)
        return None
    rows.append(("FDM truth, all cores", fmt(at(e4P, ["fdm"]) if at(e4P, ["fdm", "cores"]) is None else at(e4P, ["fdm", "cores"]), nd=1), fmt(at(e4D, ["fdm", "cores"]), nd=1)))
    rows.append(("surrogate, torch (the configured path)", fmt(at(e4P, ["torch"]), nd=1), fmt(at(e4D, ["distilled", "tabulated", "torch"]) or at(e4D, ["tabulated", "torch"]), nd=1)))
    rows.append(("full SKN, direct look-up, torch (same machine)", fmt(at(e4P, ["torch"]), nd=1), fmt(at(e4D, ["full skn", "direct", "torch"]), nd=1)))
    # ---- write ------------------------------------------------------------------------------------------------
    md = ["| quantity | teacher engine (6x256, direct, M = 64) | student engine (4x128, tabulated, M = 32) |", "|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering", r"\caption{Teacher engine vs student engine: same physics, same reference, same stack.}", r"\label{tab:distill}", r"\footnotesize",
           r"\begin{tabular}{@{}lll@{}}", r"\toprule", r"quantity & teacher engine & student engine \\", r"\midrule"]
    for r in rows:
        if r[0] == "__sec__":
            md.append(f"| **{r[1]}** | | |"); tex.append(r"\multicolumn{3}{@{}l}{\textit{" + r[1] + r"}} \\")
        else:
            md.append(f"| {r[0]} | {r[1]} | {r[2]} |"); tex.append(" & ".join(r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "comparison_teacher_student.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(out, "comparison_teacher_student.tex"), "w").write("\n".join(tex) + "\n")
    json.dump([list(r) for r in rows], open(os.path.join(out, "comparison_teacher_student.json"), "w"), indent=1)
    print("\n".join(md))
    # ---- figures ------------------------------------------------------------------------------------------------
    st.use_paper_style()
    import matplotlib.pyplot as plt
    if e4P and e4D:
        fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.6))
        series = [("FDM truth (CPU, all cores)", e4P, ["fdm", "cores"], "k", "-", "s"), ("FDM truth (CPU, 1 core)", e4D, ["fdm", "1 core"], "#7F7F7F", "--", "s"),
                  ("full SKN, direct look-up (torch)", e4P, ["torch"], st.OKABE_ITO["vermilion"], "-", "o"),
                  ("full SKN, tabulated look-up (torch)", e4D, ["full skn", "tabulated", "torch"], st.OKABE_ITO["orange"], "-", "D"),
                  ("distilled SKN, direct look-up (torch)", e4D, ["distilled", "direct", "torch"], st.OKABE_ITO["purple"], "-", "o"),
                  ("distilled SKN, tabulated look-up (torch)", e4D, ["distilled", "tabulated", "torch"], st.OKABE_ITO["green"], "-", "D"),
                  ("distilled SKN, tabulated look-up (numpy)", e4D, ["distilled", "tabulated", "numpy"], st.OKABE_ITO["green"], "--", "D")]
        for lab, e4, subs, c, ls, mk in series:
            sizes = np.array(e4["sizes"], float)
            key = next((k for k in e4["timings"] if all(s in k.lower() for s in subs)), None)
            if key is None and subs == ["fdm", "cores"]:
                key = next((k for k in e4["timings"] if "fdm" in k.lower() and "1 core" not in k.lower()), None)
            if key is None: continue
            v = np.array([np.nan if x is None else x for x in e4["timings"][key]], float); m = np.isfinite(v)
            if m.any(): ax.plot(sizes[m], v[m], color=c, ls=ls, marker=mk, ms=3.2, lw=1, label=lab + (" [teacher-engine run]" if e4 is e4P else ""))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of rails"); ax.set_ylabel("wall-clock time (s)"); ax.legend(fontsize=5.2, loc="lower right"); ax.grid(True, color="#EEEEEE", lw=0.5)
        ax.set_title(f"M = {e4P.get('M', cfgP.get('times', {}).get('n', '?'))} (teacher engine) vs M = {e4D.get('M', '?')} (student engine)", fontsize=7)
        st.save(fig, os.path.join(out, "compare_timing"))
    if acc:
        fig, ax = plt.subplots(figsize=(st.IEEE_COL, 2.2)); xs = np.arange(len(dies)); w = 0.38
        ax.bar(xs - w / 2, [100 * (acc[d][0] or np.nan) for d in dies], w, color=st.OKABE_ITO["vermilion"], label="teacher engine (median)")
        ax.bar(xs + w / 2, [100 * (acc[d][2] or np.nan) for d in dies], w, color=st.OKABE_ITO["green"], label="student engine (median)")
        ax.plot(xs - w / 2, [100 * (acc[d][1] or np.nan) for d in dies], "k_", ms=9, mew=1.2, label="p90"); ax.plot(xs + w / 2, [100 * (acc[d][3] or np.nan) for d in dies], "k_", ms=9, mew=1.2)
        ax.set_xticks(xs); ax.set_xticklabels(dies); ax.set_ylabel("$t_{nuc}$ error vs FDM truth (%)"); ax.legend(fontsize=6); ax.margins(y=0.25)
        st.save(fig, os.path.join(out, "compare_accuracy"))
    print("wrote", out)


if __name__ == "__main__":
    main()
