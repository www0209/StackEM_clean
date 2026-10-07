"""
E7 - "does the physics change the decision?": ranking value of every simplification.

    python -m stackem.experiments.e7_ranking --root DIR [--e3-dir ...]

Reads the per-variant truth results written by E3 (e3_<variant>_truth.npz) and asks,
for every simplified physics (alone / uniform / sigmaT_const / no_joule / no_tm) and
for the StackEM overlay (e3_full_stackem.npz):

    * Top-k hit rate of the critical-rail set (k = 10, 50) w.r.t. the full 3-D truth (plus a 1 %-tolerant
      Top-10 hit rate: near-ties at the Top-10 boundary are counted as hits, see top10_boundary_gap_rel);
    * Kendall tau of the Top-50 ordering;
    * the "design decision" metrics: does the simplified model pick the same critical die,
      the same earliest rail, and the same mortal set within 10 yr (Jaccard)?
    * the optimism factor: earliest t_nuc(simplified) / earliest t_nuc(full) per die.

Figures: e7_ranking.(pdf|png) (dot plots), e7_confusion.(pdf|png) (mortal-set confusion within 10 yr).
"""
from __future__ import annotations

import json
import os
import numpy as np

from ._common import base_parser, setup, record, out_dir
from ..assembler import topk_hit_rate, kendall_tau
from ..constants import SEC_PER_YEAR
from ..viz.plots import plot_ranking_hits
from ..viz import style as st


def load_variant(path):
    z = np.load(path)
    dies = sorted({k.split("_", 1)[1] for k in z.files if k.startswith("tnuc_")})
    tn = {d: z[f"tnuc_{d}"] for d in dies}
    meta = json.load(open(path.replace(".npz", ".json")))
    return tn, meta, dies


def main(argv=None):
    ap = base_parser("E7: ranking value of the physics")
    ap.add_argument("--e3-dir", default=None, help="directory with the E3 results (default <root>/<name>/e3)")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]
    e3 = os.path.expanduser(args.e3_dir) if args.e3_dir else out_dir(cfg, "e3"); d = out_dir(cfg, "e7")
    full, meta_full, dies = load_variant(os.path.join(e3, "e3_full_truth.npz"))
    order = [dd["name"] for dd in cfg["stack"]["dies"] if dd["name"] in dies]
    t_full = np.concatenate([full[dn] for dn in order])
    die_of = np.concatenate([[dn] * len(full[dn]) for dn in order])
    cand = {"alone": "e3_alone_truth.npz", "uniform_signoff": "e3_uniform_signoff_truth.npz", "uniform": "e3_uniform_truth.npz", "sigmaT_const": "e3_sigmaT_const_truth.npz",
            "no_joule": "e3_no_joule_truth.npz", "no_tm": "e3_no_tm_truth.npz", "stackem": "e3_full_stackem.npz"}
    results = {}; rows = []
    k10 = min(10, len(t_full)); k50 = min(50, len(t_full))
    mortal_full = t_full <= 10 * SEC_PER_YEAR
    ts = np.sort(t_full)
    # how tight is the Top-10 boundary of the truth?  (rails 10 and 11 differ by this relative gap; a surrogate with
    # ~0.2 % t_nuc error necessarily swaps near-ties, which is why the tolerant hit rate is also reported)
    boundary_gap = float((ts[k10] - ts[k10 - 1]) / ts[k10 - 1]) if len(ts) > k10 else float("nan")

    def topk_tol(t_ref, t_test, k, tol=0.01):
        thr = np.sort(t_ref)[k - 1] * (1.0 + tol); b = np.argsort(t_test)[:k]
        return float(np.mean(t_ref[b] <= thr))

    for v, fn in cand.items():
        p = os.path.join(e3, fn)
        if not os.path.exists(p):
            print(f"[skip] {v}: {p} not found"); continue
        tn, meta, _ = load_variant(p)
        t_v = np.concatenate([tn[dn] for dn in order])
        mortal_v = t_v <= 10 * SEC_PER_YEAR
        inter = np.sum(mortal_full & mortal_v); union = np.sum(mortal_full | mortal_v)
        res = dict(top10=topk_hit_rate(t_full, t_v, k10), top10_tol1pct=topk_tol(t_full, t_v, k10, 0.01),
                   top50=topk_hit_rate(t_full, t_v, k50), kendall=kendall_tau(t_full, t_v, k50),
                   same_critical_die=bool(die_of[np.argmin(t_full)] == die_of[np.argmin(t_v)]),
                   same_earliest_rail=bool(np.argmin(t_full) == np.argmin(t_v)),
                   mortal10_jaccard=float(inter / union) if union else 1.0,
                   mortal10_missed=int(np.sum(mortal_full & ~mortal_v)), mortal10_false=int(np.sum(~mortal_full & mortal_v)),
                   optimism_factor={dn: float(np.min(tn[dn]) / np.min(full[dn])) for dn in order},
                   earliest_years={dn: float(np.min(tn[dn]) / SEC_PER_YEAR) for dn in order})
        results[v] = res; rows.append(dict(variant=v, **{k: val for k, val in res.items() if not isinstance(val, dict)}))
        print(v, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in res.items() if not isinstance(val, dict)})
    plot_ranking_hits(results, os.path.join(d, "e7_ranking"))
    # confusion figure (mortal within 10 yr) - one row per variant
    st.use_paper_style()
    import matplotlib.pyplot as plt
    vs = list(results)
    fig, ax = plt.subplots(figsize=(st.IEEE_COL, 0.35 * len(vs) + 0.8), constrained_layout=True)
    for i, v in enumerate(vs):
        r = results[v]; hit = int(np.sum(mortal_full)) - r["mortal10_missed"]
        ax.barh(i, hit, color=st.OKABE_ITO["green"], height=0.6, label="detected" if i == 0 else None)
        ax.barh(i, r["mortal10_missed"], left=hit, color=st.OKABE_ITO["vermilion"], height=0.6, label="missed (optimistic)" if i == 0 else None)
        ax.barh(i, r["mortal10_false"], left=hit + r["mortal10_missed"], color=st.OKABE_ITO["orange"], height=0.6, label="false alarm" if i == 0 else None)
    ax.set_yticks(range(len(vs))); ax.set_yticklabels([st.VARIANT_LABEL.get(v, v) for v in vs], fontsize=6)
    ax.set_xlabel("rails mortal within 10 yr"); ax.legend(fontsize=6, loc="lower right")
    st.save(fig, os.path.join(d, "e7_confusion"))
    record(cfg, "e7", "e7_ranking.json", dict(results=results, n_rails=int(len(t_full)), n_mortal10_full=int(mortal_full.sum()),
                                              top10_boundary_gap_rel=boundary_gap, e3_dir=e3, full_summary=meta_full["summary"]))
    print(f"truth Top-10 boundary: rails #10 and #11 differ by {boundary_gap*100:.2f} % in t_nuc")


if __name__ == "__main__":
    main()
