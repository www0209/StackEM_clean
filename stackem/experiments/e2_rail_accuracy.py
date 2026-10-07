"""
E2 - rail-level accuracy of StackEM (SKN kernels + junction closure) against the
full finite-volume tree solver on every rail of the stack, and zero-shot use on
the standalone-die ("alone") and uniform-temperature ("uniform") variants (the SKN is never trained on any stack).

    python -m stackem.experiments.e2_rail_accuracy --root DIR [--weights ...] [--smoke]

Metrics: t_nuc relative error (median / p90 / max) over mortal rails, node-stress
rel-L2 over the histories, mortal/immortal agreement, Top-k critical-rail hit
rate; wall-clock of both methods.  Figures: e2_parity_<variant>.(pdf|png) for full, alone, uniform.
--smoke: 8 rails per die, variant "full" only.
"""
from __future__ import annotations

import os
import time
import numpy as np

from ._common import base_parser, setup, record, out_dir, make_assembler, kernel_provider
from ..assembler import topk_hit_rate, kendall_tau, save_variant
from ..constants import SEC_PER_YEAR
from ..viz.plots import plot_parity


def compare(res_pred, res_truth, dies):
    rep = {}; all_t = []; all_p = []; all_lab = []
    for dname in dies:
        tp, tt = res_pred.t_nuc[dname], res_truth.t_nuc[dname]
        sp, stt = res_pred.sigma_nodes[dname], res_truth.sigma_nodes[dname]
        mortal = np.isfinite(tt) & np.isfinite(tp)
        rel = np.abs(tp[mortal] - tt[mortal]) / tt[mortal]
        # node-stress error over the rails that were actually closed (Blech-immortal rails carry the same analytic
        # steady state in both results and would pull the median to zero)
        imm = np.asarray(res_truth.immortal[dname], bool) if getattr(res_truth, "immortal", None) is not None else np.zeros(len(tt), bool)
        ok = np.isfinite(sp).all(axis=(1, 2)) & np.isfinite(stt).all(axis=(1, 2)) & ~imm
        if not ok.any():
            ok = np.isfinite(sp).all(axis=(1, 2)) & np.isfinite(stt).all(axis=(1, 2))
        l2 = np.linalg.norm((sp - stt)[ok].reshape(ok.sum(), -1), axis=1) / np.linalg.norm(stt[ok].reshape(ok.sum(), -1), axis=1)
        agree = float(np.mean(np.isfinite(tp) == np.isfinite(tt)))
        rep[dname] = dict(n=int(len(tt)), n_mortal=int(mortal.sum()), tnuc_rel_median=float(np.median(rel)) if mortal.any() else None,
                          tnuc_rel_p90=float(np.percentile(rel, 90)) if mortal.any() else None, tnuc_rel_max=float(rel.max()) if mortal.any() else None,
                          sigma_relL2_median=float(np.median(l2)), sigma_relL2_p90=float(np.percentile(l2, 90)),
                          mortality_agreement=agree, top10_hit=topk_hit_rate(tt, tp, min(10, len(tt))),
                          kendall_top50=kendall_tau(tt, tp, min(50, len(tt))))
        all_t.append(tt); all_p.append(tp); all_lab.append(np.array([dname] * len(tt)))
    return rep, np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_lab)


def main(argv=None):
    ap = base_parser("E2: StackEM vs FDM on every rail")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]
    prov, kind = kernel_provider(ctx)
    asm = make_assembler(ctx, "e2", prov).run_fields()
    d = out_dir(cfg, "e2")
    nmax = 8 if args.smoke else None
    rep = dict(provider=kind, pg=asm.pg_stats)
    dies = [dd.name for dd in ctx["spec"].dies]
    for variant in (["full"] if args.smoke else ["full", "alone", "uniform"]):
        t0 = time.time(); rp = asm.solve_variant(variant, "closure", nmax); tp_ = time.time() - t0
        t0 = time.time(); rt = asm.solve_variant(variant, "truth", nmax); tt_ = time.time() - t0
        cmp_, T, P, lab = compare(rp, rt, dies)
        rep[variant] = dict(metrics=cmp_, seconds_closure=tp_, seconds_truth=tt_, stack_pred=asm.stack_summary(rp), stack_truth=asm.stack_summary(rt))
        plot_parity(T, P, os.path.join(d, f"e2_parity_{variant}"), labels=lab, title=f"{variant}")
        save_variant(os.path.join(d, f"e2_{variant}_pred.npz"), rp, asm.stack_summary(rp), asm.times)
        save_variant(os.path.join(d, f"e2_{variant}_truth.npz"), rt, asm.stack_summary(rt), asm.times)
        print(variant, {k: (v["tnuc_rel_median"], v["sigma_relL2_median"], v["top10_hit"]) for k, v in cmp_.items()}, f"closure {tp_:.1f}s truth {tt_:.1f}s")
    record(cfg, "e2", "e2_summary.json", rep)


if __name__ == "__main__":
    main()
