"""
E1 - accuracy of the Segment Kernel Network on held-out kernel surfaces.

    python -m stackem.experiments.e1_skn_eval --root DIR [--weights skn_best.npz]

Metrics (per kernel s_G, s_M, a): rel-L2 over the full (xi, tau) surface, max
relative error of the end values (xi = 0, 1, the ones the closure uses), error
per tau decade, and the same on the out-of-range (OOD) profile set.
Figures: e1_kernels_profile_<set> (scaled surfaces), e1_kernel_cuts_<set> (end values vs tau), e1_error_by_tau_<set>.
"""
from __future__ import annotations

import os
import numpy as np

from ._common import base_parser, setup, record, out_dir, base_dir
from ..closure import KernelProvider
from ..skn_model import SKNWeights
from ..skn_numpy import SKNNumpyProvider
from ..viz.plots import plot_kernels, plot_error_by_tau, plot_kernel_cuts, plot_error_by_tau_multi


def eval_surfaces(prov: KernelProvider, path: str):
    d = np.load(path)
    xi, tau, META = d["xi"], d["tau"], d["META"]
    out = {k: dict(rel_l2=[], end_max_rel=[]) for k in ("s_G", "s_M", "a")}
    edges = np.logspace(-6, 1, 8)
    per_dec = {k: np.zeros(len(edges) - 1) for k in out}; cnt = np.zeros(len(edges) - 1)
    XI, TAU = np.meshgrid(xi, tau)
    examples = []
    for p in range(len(META)):
        theta, r, rhoJ, lam = META[p, 4:8]
        desc = np.tile([theta, r, rhoJ, lam], (XI.size, 1))
        pred = prov.evaluate(desc, XI.T.ravel(), TAU.T.ravel())          # xi-major order: runs of equal (desc, xi) for the tabulated provider
        truth = dict(s_G=d["s_G"][p], s_M=d["s_M"][p], a=d["a"][p])
        pr = {k: pred[k].reshape(XI.T.shape).T for k in truth}
        for k in truth:
            T, P = truth[k], pr[k]
            out[k]["rel_l2"].append(float(np.linalg.norm(P - T) / (np.linalg.norm(T) + 1e-30)))
            ends = np.abs(P[:, [0, -1]] - T[:, [0, -1]]) / (np.abs(T[:, [0, -1]]) + 1e-3 * np.abs(T).max())
            out[k]["end_max_rel"].append(float(ends.max()))
            for b in range(len(edges) - 1):
                m = (tau >= edges[b]) & (tau < edges[b + 1])
                if m.any():
                    per_dec[k][b] += float(np.linalg.norm(P[m] - T[m]) / (np.linalg.norm(T[m]) + 1e-30))
        cnt += 1
        if p < 3:
            examples.append((xi, tau, truth, pr, META[p]))
    summ = {k: dict(rel_l2_median=float(np.median(v["rel_l2"])), rel_l2_p90=float(np.percentile(v["rel_l2"], 90)),
                    rel_l2_max=float(np.max(v["rel_l2"])), end_max_rel_median=float(np.median(v["end_max_rel"])))
            for k, v in out.items()}
    return summ, {k: per_dec[k] / np.maximum(cnt, 1) for k in per_dec}, edges, examples


def main(argv=None):
    ap = base_parser("E1: SKN kernel accuracy")
    args = ap.parse_args(argv); args.no_sizing = True          # kernel accuracy does not depend on any stack or strap width
    ctx = setup(args); cfg = ctx["cfg"]
    d = out_dir(cfg, "e1"); kd = base_dir(cfg, "kernels")     # runs with a base_case share the base case's kernel data
    from ..closure_tab import TabulatedProvider
    main_prov = SKNNumpyProvider(SKNWeights.load(ctx["weights"]))
    sk = cfg.get("skn", {})
    # three providers (student, student + table, teacher) only for a student-engine config, i.e. one whose skn.weights and
    # skn.teacher differ; a teacher-engine config evaluates exactly one network, also when --weights points at another
    # full-size network such as the PDE-residual ablation
    distilled = (sk.get("weights") != sk.get("teacher") and os.path.exists(ctx["teacher"])
                 and os.path.abspath(ctx["teacher"]) != os.path.abspath(ctx["weights"]))
    provs = {}
    if distilled:                                       # student engine: student (direct + tabulated) and the full-size teacher
        provs["distilled SKN"] = main_prov
        provs[f"distilled SKN, tabulated ({ctx['n_tab']} pts)"] = TabulatedProvider(main_prov, ctx["n_tab"])
        provs["full SKN"] = SKNNumpyProvider(SKNWeights.load(ctx["teacher"]))
        main_label = f"distilled SKN, tabulated ({ctx['n_tab']} pts)" if ctx["closure_mode"] == "tabulated" else "distilled SKN"
    else:
        provs["SKN"] = main_prov; main_label = "SKN"
        if ctx["closure_mode"] == "tabulated":
            provs[f"SKN, tabulated ({ctx['n_tab']} pts)"] = TabulatedProvider(main_prov, ctx["n_tab"]); main_label = f"SKN, tabulated ({ctx['n_tab']} pts)"
    rep = {"providers": list(provs), "main_provider": main_label}
    for tag in ("val", "ood"):
        path = os.path.join(kd, f"kernels_{tag}_full.npz")
        if not os.path.exists(path):
            print("missing", path); continue
        per_prov = {}; per_dec_all = {}
        for label, prov in provs.items():
            summ, per_dec, edges, ex = eval_surfaces(prov, path)
            per_prov[label] = summ; per_dec_all[label] = per_dec
            print(tag, label, summ)
            if label == main_label and ex:
                xi, tau, truth, pr, meta = ex[0]
                plot_kernels(xi, tau, truth, pr, os.path.join(d, f"e1_kernels_profile_{tag}"),
                             profile_text=f"held-out profile: $T_L$ = {meta[0]:.0f} K, $T_R$ = {meta[1]:.0f} K, $T_m$ = {meta[2]:.1f} K, $L/\\Gamma$ = {meta[7]:.1f}")
                plot_kernel_cuts(ex, os.path.join(d, f"e1_kernel_cuts_{tag}"))
        rep[tag] = per_prov[main_label]; rep[f"{tag}_by_provider"] = per_prov
        if len(provs) == 1:
            plot_error_by_tau(edges, per_dec_all[main_label], os.path.join(d, f"e1_error_by_tau_{tag}"))
        else:
            plot_error_by_tau_multi(edges, per_dec_all, os.path.join(d, f"e1_error_by_tau_{tag}"))
    record(cfg, "e1", "e1_summary.json", rep)


if __name__ == "__main__":
    main()
