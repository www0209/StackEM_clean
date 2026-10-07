"""
Rails from an IBM power-grid benchmark netlist (ibmpg1 ... ibmpg6), solved by the reference solver and by StackEM.

    python tools/run_ibmpg.py --spice ~/stackem_work/ibmpg/ibmpg1.spice --config configs/base3.json \
        [--root ~/stackem_work/outputs] [--hotspot ~/stackem_work/HotSpot] [--weights <skn_student.npz>] [--die D3] [--synthetic]

Defaults = the final experiments: every rail (--max-rails 0), every rail sized to j_max = 1e6 A/cm^2, at least 3 segments,
the student weights of <root>/base3 and the engine of the config (tabulated look-up, M = 32).  The closure runs on the
GPU when one is available and on the CPU (numpy) otherwise; the accuracy numbers do not depend on that choice.

Steps: parse + MNA DC solve of the whole benchmark -> straight multi-segment rails per metal layer
(A = rho L / R) -> temperature from the base3 HotSpot field of one die projected on the netlist bounding
box -> Blech screen -> FDM truth and StackEM closure on the mortal rails -> accuracy + timing.
--synthetic writes and uses a small ibmpg-format grid (for a functional test without the download).
Outputs <root>/ibmpg/<netlist name>/ibmpg_summary.json, ibmpg_parity.(pdf|png), ibmpg_rails.npz.
CHECK the printed "coordinate unit" line of every new netlist: the benchmarks state no unit and the die size
it implies must be plausible (use --unit 1e-9 etc. to override the heuristic).
"""
import argparse, os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stackem.config import load_config, em_from_config, thermal_from_config, times_from_config, stack_from_config, hotspot_dir
from stackem.ibmpg import parse_spice, solve_dc, extract_rails, write_synthetic_benchmark
from stackem.hotspot_stack import HotSpotRunner, run_stack
from stackem.blech_screen import screen_rails
from stackem.assembler import solve_rails_truth, topk_hit_rate, kendall_tau
from stackem.closure import close_rails
from stackem.skn_model import SKNWeights
from stackem.skn_numpy import SKNNumpyProvider
from stackem.constants import SEC_PER_YEAR
from stackem.viz.plots import plot_parity

def _truth_job(args):
    idx_b, rl, em_, times_, cells_ = args
    _, t_, _ = solve_rails_truth(rl, em_, times_, cells_, True, adaptive_cells=True)
    return idx_b, t_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spice"); ap.add_argument("--root", default=None, help="output root (default: paths.root_out of the config)")
    ap.add_argument("--hotspot", default=None, help="HotSpot directory (default: paths.hotspot_dir of the config)")
    ap.add_argument("--weights", default=None, help="SKN weights (default: <root>/base3/<config skn.weights>, i.e. the student)")
    ap.add_argument("--config", default=None); ap.add_argument("--die", default="D3")
    ap.add_argument("--max-rails", type=int, default=0, help="keep the N rails with the largest current; 0 = all rails")
    # ^ originally 400 (and 2000 in the first run script); the final experiments used 0 = all rails (see docs)
    ap.add_argument("--unit", default="auto", help="coordinate unit of the netlist (m), e.g. 1e-6, or 'auto': a heuristic on the die span and the median segment length "
                                                   "(result in the final runs: ibmpg1/2/4/5: 1 um, ibmpg3/6: 1 nm)")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--no-thermal", action="store_true", help="uniform temperature (--uniform-T) instead of the HotSpot field")
    ap.add_argument("--uniform-T", type=float, default=353.0, help="K, the temperature used with --no-thermal (EMSpice-style assumption: 373 K)")
    ap.add_argument("--sigma-T0", type=float, default=None, help="Pa: override the residual stress of every segment with a constant (EMSpice-style assumption: 0)")
    ap.add_argument("--tag", default="", help="suffix of the output sub-directory, e.g. _uniform373 for the model-assumption variant")
    ap.add_argument("--strap-width-um", type=float, default=None, help="assumed physical strap width (um); the ibmpg resistors are coarsened buses, so for the EM current density use e.g. 4 um (M5) with --strap-thickness-um 1")
    ap.add_argument("--strap-thickness-um", type=float, default=1.0)
    ap.add_argument("--min-segments", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="numpy closure even when a GPU is available")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="processes for the reference solver")
    ap.add_argument("--size-to-jmax", type=float, default=1e6, help="A/cm^2: size every rail's strap width so that its peak current density equals this EM design limit (the rail-level analogue of the paper's sign-off sizing; typical rule 1e6)")
    # ^ --size-to-jmax: originally None (no sizing); the final experiments used 1e6 (see docs).  Pass 0 to switch the sizing off.
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.root: cfg["paths"]["root_out"] = a.root
    if a.hotspot: cfg["paths"]["hotspot_dir"] = a.hotspot
    root = os.path.expanduser(cfg["paths"]["root_out"])
    if not a.weights:
        a.weights = os.path.join(root, cfg.get("base_case") or cfg["name"], cfg.get("skn", {}).get("weights", "skn_distill/skn_student.npz"))
    a.weights = os.path.expanduser(a.weights)
    em, th, times = em_from_config(cfg), thermal_from_config(cfg), times_from_config(cfg)
    spice = os.path.expanduser(a.spice) if a.spice else None
    # one sub-directory per benchmark: <root>/ibmpg/ibmpg1, <root>/ibmpg/ibmpg2, ... (<root>/ibmpg/synthetic for --synthetic)
    d = os.path.join(root, "ibmpg", (os.path.splitext(os.path.basename(spice))[0] if spice and not a.synthetic else "synthetic") + a.tag); os.makedirs(d, exist_ok=True)
    if a.synthetic or not spice:
        spice = write_synthetic_benchmark(os.path.join(d, "synthetic_ibmpg_format.spice"), n=24, pitch=50000); print("synthetic netlist", spice)
    t0 = time.time(); net = parse_spice(spice); Vn, Ir = solve_dc(net); print(f"parsed {len(net.nodes)} nodes, {len(net.R)} resistors; DC solve {time.time()-t0:.1f}s; Vmin {Vn[1:].min():.4f} V", flush=True)
    if str(a.unit).lower() == "auto":                                  # the benchmarks do not share one coordinate unit
        d_ = []
        for (na, nb, _, _) in net.R[:200000]:
            ca, cb = net.coords.get(na), net.coords.get(nb)
            if ca and cb and ca[0] == cb[0] and (ca[1] == cb[1]) != (ca[2] == cb[2]):
                d_.append(abs(ca[1] - cb[1]) + abs(ca[2] - cb[2]))
        med = float(np.median(d_)) if d_ else 100.0
        span = max(max(c[1] for c in net.coords.values()), max(c[2] for c in net.coords.values()))
        a.unit = 1e-6
        while span * a.unit > 4e-2: a.unit /= 10.0                      # the benchmarks state no unit: take the largest power of ten
        while span * a.unit <= 4e-3 and med * a.unit < 1e-5: a.unit *= 10.0   # that keeps the die within 40 mm (final runs: ibmpg1/2/4/5 -> 1 um, ibmpg3/6 -> 1 nm)
        a.unit = 10.0 ** round(np.log10(a.unit))
        print(f"coordinate unit: auto -> {a.unit:g} m (median collinear segment {med:g} units = {med*a.unit*1e6:.0f} um; "
              f"die {max(c[1] for c in net.coords.values())*a.unit*1e3:.1f} x {max(c[2] for c in net.coords.values())*a.unit*1e3:.1f} mm)", flush=True)
    else:
        a.unit = float(a.unit)
    Tgrid = None
    if not a.no_thermal:
        tf = run_stack(stack_from_config(cfg), os.path.join(d, "hotspot"), HotSpotRunner(hotspot_dir(cfg)), "stack", False)
        Tgrid = tf.T_die[a.die]; print(f"temperature field of {a.die}: {Tgrid.min():.1f}-{Tgrid.max():.1f} K projected on the netlist")
    W_as = a.strap_width_um * 1e-6 if a.strap_width_um else None
    t0 = time.time(); rails = extract_rails(net, Vn, Ir, em, th, Tgrid, a.uniform_T, a.unit, a.strap_thickness_um * 1e-6, True, a.min_segments, W_as or 1e-6)
    print(f"  [rails] {len(rails)} straight rails extracted ({time.time()-t0:.1f}s)", flush=True)
    if a.sigma_T0 is not None:                                       # model-assumption variant: constant initial stress (EMSpice: 0)
        from dataclasses import replace as _rep0
        rails = [_rep0(r, sigma_T=np.full_like(r.sigma_T, a.sigma_T0)) for r in rails]; print(f"residual stress overridden: sigma_T = {a.sigma_T0:g} Pa on every segment")
    if a.size_to_jmax:                                              # rail-level sign-off sizing: W = I_max / (j_max H)
        from dataclasses import replace as _rep
        H = a.strap_thickness_um * 1e-6; sized = []
        for r in rails:
            I_max = float(np.abs(r.j).max() * r.W * r.H)            # current through the rail (A)
            W_new = max(I_max / (a.size_to_jmax * 1e4 * H), 0.2e-6)
            f = (r.W * r.H) / (W_new * H)
            sized.append(_rep(r, W=W_new, j=r.j * f, T_m=th.T_m(np.abs(r.j * f), True)))
        rails = sized; print(f"rails sized to j_max = {a.size_to_jmax:g} A/cm^2: W median {np.median([r.W for r in rails])*1e6:.2f} um", flush=True)
    # optional check of our DC solve against the official solution file (ibmpgN.solution next to the netlist)
    sol_path = os.path.splitext(spice)[0] + ".solution"
    if os.path.exists(sol_path):
        ref = {}
        for line in open(sol_path):
            t = line.split()
            if len(t) == 2 and t[0] in net.nodes:
                try: ref[net.nodes[t[0]]] = float(t[1])
                except ValueError: pass
        if ref:
            ids = np.array(list(ref)); dv = np.abs(Vn[ids] - np.array([ref[i] for i in ids]))
            print(f"DC solve vs official solution: {len(ids)} nodes, max |dV| = {dv.max()*1e3:.3f} mV, mean {dv.mean()*1e3:.3f} mV")
    print(f"{len(rails)} straight rails extracted (>= {a.min_segments} segments); by net: " + ", ".join(f"{k}: {v}" for k, v in sorted(__import__('collections').Counter(r.die for r in rails).items())))
    rails.sort(key=lambda r: -float(np.abs(r.j).max() * r.W * r.H))
    if a.max_rails > 0: rails = rails[:a.max_rails]                      # the rails carrying the largest current (0 = every rail)
    Ls = np.concatenate([r.L for r in rails]) * 1e6
    print(f"{len(rails)} rails kept ({'all' if a.max_rails <= 0 else 'top by current'}); segments per rail median {int(np.median([r.n_seg for r in rails]))}, segment length median {np.median(Ls):.0f} um (p10 {np.percentile(Ls,10):.0f}, p90 {np.percentile(Ls,90):.0f}); "
          f"j_max {max(np.abs(r.j).max() for r in rails)*1e-4:.3g} A/cm^2, j p50 {np.median(np.concatenate([np.abs(r.j) for r in rails]))*1e-4:.3g}")
    t0 = time.time(); imm, ss = screen_rails(rails, em, True); todo = np.where(~imm)[0]; print(f"immortal by Blech screen: {imm.sum()} / {len(rails)} ({time.time()-t0:.1f}s)", flush=True)
    rep = dict(spice=os.path.basename(spice), n_nodes=len(net.nodes), n_resistors=len(net.R), n_rails=len(rails), n_immortal=int(imm.sum()), die=a.die, unit_m=a.unit, thermal=('uniform %g K' % a.uniform_T) if a.no_thermal else 'HotSpot stack field', sigma_T0=a.sigma_T0,
               strap_width_um=a.strap_width_um, strap_thickness_um=a.strap_thickness_um, size_to_jmax=a.size_to_jmax,
               W_um_median=float(np.median([r.W for r in rails]) * 1e6), segment_length_um_median=float(np.median(Ls)),
               j_max_A_per_cm2=float(max(np.abs(r.j).max() for r in rails) * 1e-4))
    if len(todo):
        from stackem.closure_tab import wrap_provider
        sub = [rails[i] for i in todo]
        groups = {}
        for i, r in enumerate(sub):
            groups.setdefault(r.n_seg, []).append(i)
        tt = np.full(len(sub), np.inf); tp = np.full(len(sub), np.inf)
        t0 = time.time()
        # reference solver: rails with the same segment count form a batch; batches are spread over the cores
        jobs = []
        for S, idx in groups.items():
            for i0 in range(0, len(idx), 16):
                jobs.append((idx[i0:i0 + 16], [sub[i] for i in idx[i0:i0 + 16]]))
        n_workers = max(1, min(a.workers, len(jobs)))
        print(f"reference solver: {len(sub)} rails in {len(jobs)} batches on {n_workers} workers ...", flush=True)
        if n_workers > 1:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                for idx_b, t_ in pool.imap_unordered(_truth_job, [(idx_b, rl, em, times, cfg["truth"]["n_cells"]) for idx_b, rl in jobs]):
                    tt[idx_b] = t_
        else:
            for idx_b, rl in jobs:
                _, t_, _ = solve_rails_truth(rl, em, times, cfg["truth"]["n_cells"], True, adaptive_cells=True); tt[idx_b] = t_
        rep["seconds_truth"] = time.time() - t0; rep["truth_workers"] = n_workers
        from stackem.assembler import cells_for_length
        nc = int(cfg["truth"]["n_cells"]); levels = [nc, 4 * nc, 16 * nc]            # 60 / 240 / 960 with the default 60 cells
        print(f"reference solver: {rep['seconds_truth']:.1f}s on {n_workers} workers, cells per segment adaptive in the longest segment ({nc} / {4*nc} / {16*nc} for <=0.3 / <=3 / >3 mm): "
              f"{np.bincount([levels.index(max(nc, cells_for_length(float(r.L.max()), nc))) for r in sub], minlength=3).tolist()} rails")
        mode, n_tab = cfg["closure"].get("mode", "direct"), int(cfg["closure"].get("n_tab", 192))
        use_torch = False
        if not a.cpu:
            try:
                import torch; use_torch = torch.cuda.is_available()
            except Exception:
                use_torch = False
        t0 = time.time()
        if use_torch:                                                            # the deployed engine: batched closure on the GPU
            import torch
            from stackem.closure_torch import SKNTorchProvider, close_rails_torch, rails_to_tensors
            dev = "cuda"; tprov = wrap_provider(SKNTorchProvider(SKNWeights.load(a.weights), dev), mode, n_tab)
            tt_ = torch.as_tensor(times, dtype=torch.float64, device=dev)
            with torch.no_grad():
                for S, idx in groups.items():
                    bs = max(8, min(1024, int(2e7 // (S * len(times) ** 2 // 2 + 1))))   # keep the (B,S,P) tensors around 1e7 elements
                    for i0 in range(0, len(idx), bs):
                        b = idx[i0:i0 + bs]; T = rails_to_tensors([sub[i] for i in b], dev)
                        out = close_rails_torch(T["L"], T["j"], T["T_L"], T["T_R"], T["T_m"], T["Gamma"], T["sigma_T"], em, tprov, tt_, True, cfg["closure"]["jump_at"])
                        tn_ = out["t_nuc"].cpu().numpy(); tn_[tn_ >= 0.99 * float(times[-1]) * 1e3] = np.inf; tp[b] = tn_
                torch.cuda.synchronize()
            rep["closure_backend"] = "torch/cuda"
        else:
            prov = wrap_provider(SKNNumpyProvider(SKNWeights.load(a.weights)), mode, n_tab)
            for S, idx in groups.items():
                for i0 in range(0, len(idx), 32):
                    b = idx[i0:i0 + 32]
                    tp[b] = close_rails([sub[i] for i in b], em, prov, times, em.sigma_crit, True, cfg["closure"]["jump_at"]).t_nuc
            rep["closure_backend"] = "numpy/cpu"
        rep["seconds_closure"] = time.time() - t0
        print(f"closure ({rep['closure_backend']}, {mode}): {rep['seconds_closure']:.1f}s for {len(sub)} rails")
        m = np.isfinite(tt) & np.isfinite(tp); rel = np.abs(tp[m] - tt[m]) / tt[m]
        rep.update(n_mortal_truth=int(np.isfinite(tt).sum()), n_mortal_pred=int(np.isfinite(tp).sum()), mortality_agreement=float(np.mean(np.isfinite(tt) == np.isfinite(tp))),
                   earliest_years=float(tt.min() / SEC_PER_YEAR), n_mortal_10yr=int(np.sum(tt <= 10 * SEC_PER_YEAR)),
                   tnuc_rel_median=float(np.median(rel)) if m.any() else None, tnuc_rel_p90=float(np.percentile(rel, 90)) if m.any() else None,
                   top10_hit=topk_hit_rate(tt, tp, min(10, len(tt))), kendall_top50=kendall_tau(tt, tp, min(50, len(tt))))
        plot_parity(tt, tp, os.path.join(d, "ibmpg_parity"), title=f"IBM PG rails ({os.path.basename(spice)})")
        # ---- diagnostics: per-rail dump + error breakdown by rail geometry (which rails does the closure get wrong?) ----
        nseg = np.array([r.n_seg for r in sub]); Lmin = np.array([r.L.min() for r in sub]) * 1e6; Lmax = np.array([r.L.max() for r in sub]) * 1e6
        Lmed = np.array([np.median(r.L) for r in sub]) * 1e6; jmx = np.array([np.abs(r.j).max() for r in sub]) * 1e-4
        Tmean = np.array([0.5 * (r.T_L + r.T_R).mean() for r in sub]); Tmmax = np.array([np.abs(r.T_m).max() for r in sub])
        nsign = np.array([int((np.sign(r.j[1:]) != np.sign(r.j[:-1])).sum()) for r in sub])
        np.savez(os.path.join(d, "ibmpg_rails.npz"), t_truth=tt, t_pred=tp, n_seg=nseg, L_min_um=Lmin, L_max_um=Lmax, L_med_um=Lmed,
                 j_max_A_per_cm2=jmx, T_mean=Tmean, T_m_max=Tmmax, n_sign_changes=nsign, W_um=np.array([r.W for r in sub]) * 1e6)
        relf = np.full(len(sub), np.nan); relf[m] = rel
        def bucket(name, v, edges):
            print(f"  error by {name}:")
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                k = m & (v >= lo_) & (v < hi_)
                if k.any():
                    print(f"    [{lo_:g}, {hi_:g}): n={k.sum():5d}  median {np.median(relf[k]):.3%}  p90 {np.percentile(relf[k], 90):.3%}  max {relf[k].max():.2%}")
        bucket("n_seg", nseg, [3, 5, 9, 17, 33, 65, 10**9])
        bucket("min segment length (um)", Lmin, [0, 5, 10, 20, 50, 100, 10**9])
        bucket("max segment length (um)", Lmax, [0, 100, 300, 1000, 3000, 10**9])
        bucket("sign changes of j along the rail", nsign, [0, 1, 2, 4, 10**9])
        worst = np.argsort(-np.nan_to_num(relf, nan=-1))[:10]
        print("  10 worst rails (truth yr, pred yr, n_seg, L_min/med/max um, j_max A/cm^2, T_m max K, sign changes):")
        for i in worst:
            print(f"    {tt[i]/SEC_PER_YEAR:9.3f} {tp[i]/SEC_PER_YEAR:9.3f} {nseg[i]:4d} {Lmin[i]:7.1f}/{Lmed[i]:7.1f}/{Lmax[i]:8.1f} {jmx[i]:9.3g} {Tmmax[i]:6.2f} {nsign[i]:3d}")
    json.dump(rep, open(os.path.join(d, "ibmpg_summary.json"), "w"), indent=1); print(rep)


if __name__ == "__main__":      # required: the worker processes of the reference solver import this file
    main()
