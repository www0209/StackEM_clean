"""
E4 - wall-clock scaling with the number of rails.

    python -m stackem.experiments.e4_scaling --root DIR --teacher <skn_best.npz> --ablate-modes [--sizes 82,246,1000,5000,20000] [--smoke]

Compared
    FDM truth (multi-process, all CPU cores; and one core)   exact reference solver
    SKN + JCX closure, numpy on CPU                           the torch-free path
    SKN + JCX closure, torch (CUDA if available)              the intended deployment path
    --teacher / --ablate-modes add the full-size network and the other look-up mode (both were on in the final run)
Larger sizes are produced by replicating the base-stack rails with random
+-20 % current and +-5 K temperature perturbations (same segment count).
External baselines measured on the same machine can be appended through
--external name=path.csv with columns n_rails,seconds.  Figure: e4_timing.(pdf|png).
Timings are single measurements and depend on the machine and its load.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from ._common import base_parser, setup, record, out_dir, make_assembler, kernel_provider, torch_available
from ..assembler import solve_rails_truth
from ..closure import close_rails
from ..viz.plots import plot_timing


def _truth_worker(args):
    rails, em, times, cells = args
    solve_rails_truth(rails, em, times, cells)
    return len(rails)


def replicate(rails, n, seed=0):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        for r in rails:
            f = rng.uniform(0.8, 1.2, r.n_seg); dT = rng.uniform(-5, 5)
            out.append(replace(r, j=r.j * f, T_L=r.T_L + dT, T_R=r.T_R + dT))
            if len(out) >= n:
                break
    return out


def main(argv=None):
    ap = base_parser("E4: scaling")
    ap.add_argument("--sizes", type=str, default="82,246,1000,5000,20000")      # the sizes of the final run (a later script edit added 100000; not used for any result)
    ap.add_argument("--numpy-max", type=int, default=2000, help="largest size timed with the numpy closure")
    ap.add_argument("--truth-max", type=int, default=20000)
    ap.add_argument("--single-core-max", type=int, default=20000, help="largest size also timed with the FDM on ONE core (per-core reference)")
    # ^ originally 1000; the final experiments used 20000 for the student engine (see docs).  The teacher-engine comparison keeps 1000 (scripts/run_all.sh).
    ap.add_argument("--external", action="append", default=[], help="name=csv with columns n_rails,seconds")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--torch-bs", type=int, default=64, help="rails per torch closure batch (64 rails x 40 segments x M^2/2 lags ~ 5e6 kernel points)")
    ap.add_argument("--tab-bs", type=int, default=1024, help="rails per torch batch for the tabulated look-up (its per-rail memory is small; larger batches amortise the M-step march)")
    ap.add_argument("--teacher", default=None, help="also time the full-size SKN (teacher), direct and tabulated, for the ablation")
    ap.add_argument("--ablate-modes", action="store_true", help="time the configured network in BOTH look-up modes")
    ap.add_argument("--profile", action="store_true", help="print the torch closure's time split (kernel tables vs time march) per engine and size")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; em = ctx["em"]; times = ctx["times"]
    mode = ctx["closure_mode"]; n_tab = ctx["n_tab"]
    prov, kind = kernel_provider(ctx)
    asm = make_assembler(ctx, "e4", prov).run_fields()
    rails_by_die, _ = asm.rails("full")
    base = [r for d in ctx["spec"].dies for r in rails_by_die[d.name]]
    sizes = [int(s) for s in args.sizes.split(",")] if not args.smoke else [24, 82]
    net_name = "SKN" if not args.teacher or os.path.abspath(args.teacher) == os.path.abspath(ctx["weights"]) else "distilled SKN"
    mode_name = {"direct": "direct look-up", "tabulated": f"tabulated look-up ({n_tab} pts/segment)"}[mode]
    # ---- the engines to time: label -> (backend, provider factory) ------------------------------------------
    from ..closure_tab import wrap_provider
    from ..skn_model import SKNWeights
    engines = {}
    if kind == "skn":
        engines[f"{net_name} + JCX, {mode_name} (numpy, CPU)"] = ("numpy", prov, None)
    dev = "cpu"
    if torch_available() and kind == "skn":
        import torch
        from ..closure_torch import SKNTorchProvider, close_rails_torch, rails_to_tensors
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tt = torch.as_tensor(times, dtype=torch.float64, device=dev)
        w_main = SKNWeights.load(ctx["weights"])
        engines[f"{net_name} + JCX, {mode_name} (torch, {dev.upper()})"] = ("torch", wrap_provider(SKNTorchProvider(w_main, dev), mode, n_tab), tt)
        if args.ablate_modes:
            other = "tabulated" if mode == "direct" else "direct"
            on = {"direct": "direct look-up", "tabulated": f"tabulated look-up ({n_tab} pts/segment)"}[other]
            engines[f"{net_name} + JCX, {on} (torch, {dev.upper()})"] = ("torch", wrap_provider(SKNTorchProvider(w_main, dev), other, n_tab), tt)
        if args.teacher and net_name != "SKN":
            w_t = SKNWeights.load(os.path.expanduser(args.teacher))
            engines[f"full SKN + JCX, direct look-up (torch, {dev.upper()})"] = ("torch", SKNTorchProvider(w_t, dev), tt)
            engines[f"full SKN + JCX, tabulated look-up ({n_tab} pts/segment) (torch, {dev.upper()})"] = ("torch", wrap_provider(SKNTorchProvider(w_t, dev), "tabulated", n_tab), tt)
    if args.profile:
        import stackem.closure_torch as _ct
        _ct.PROFILE = {}
    t_fdm, t_fdm1 = [], []; t_eng = {k: [] for k in engines}
    for n in sizes:
        rails = replicate(base, n)
        if n <= args.truth_max:                                # FDM truth, all cores
            t0 = time.time()
            chunks = [rails[i::args.workers] for i in range(args.workers)]
            with Pool(args.workers) as pool:
                pool.map(_truth_worker, [(c, em, times, cfg["truth"]["n_cells"]) for c in chunks if c])
            t_fdm.append(time.time() - t0)
        else:
            t_fdm.append(np.nan)
        if n <= args.single_core_max:                          # the same solver on one core (per-core reference)
            t0 = time.time(); _truth_worker((rails, em, times, cfg["truth"]["n_cells"])); t_fdm1.append(time.time() - t0)
        else:
            t_fdm1.append(np.nan)
        for label, (backend, engine, tt_) in engines.items():
            if backend == "numpy":
                if n <= args.numpy_max:
                    t0 = time.time()
                    for i in range(0, n, 256):
                        close_rails(rails[i:i + 256], em, engine, times)
                    t_eng[label].append(time.time() - t0)
                else:
                    t_eng[label].append(np.nan)
            else:
                import torch
                from ..closure_torch import close_rails_torch, rails_to_tensors
                T = rails_to_tensors(rails[:1], dev)
                with torch.no_grad():
                    close_rails_torch(T["L"], T["j"], T["T_L"], T["T_R"], T["T_m"], T["Gamma"], T["sigma_T"], em, engine, tt_)  # warm-up
                    if dev == "cuda": torch.cuda.synchronize()
                    if args.profile:
                        import stackem.closure_torch as _ct; _ct.PROFILE = {}
                    t0 = time.time(); bs = args.torch_bs if "direct" in label else max(args.torch_bs, args.tab_bs)
                    for i in range(0, n, bs):
                        T = rails_to_tensors(rails[i:i + bs], dev)
                        close_rails_torch(T["L"], T["j"], T["T_L"], T["T_R"], T["T_m"], T["Gamma"], T["sigma_T"], em, engine, tt_)
                        del T
                    if dev == "cuda": torch.cuda.synchronize(); torch.cuda.empty_cache()
                t_eng[label].append(time.time() - t0)
                if args.profile:
                    import stackem.closure_torch as _ct
                    print(f"    [profile] n={n} {label}: " + ", ".join(f"{k} {v:.1f}s" for k, v in (_ct.PROFILE or {}).items()), flush=True)
                    _ct.PROFILE = {}
        print(f"n={n:6d}: FDM({args.workers} cores) {t_fdm[-1]:.1f}s  FDM(1 core) {t_fdm1[-1]:.1f}s  " + "  ".join(f"{k}: {v[-1]:.1f}s" for k, v in t_eng.items()), flush=True)
    timings = {f"FDM truth (CPU, {args.workers} cores)": np.array(t_fdm), "FDM truth (CPU, 1 core)": np.array(t_fdm1)}
    timings.update({k: np.array(v) for k, v in t_eng.items()})
    for ext in args.external:
        name, path = ext.split("=", 1)
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        timings[name] = np.interp(sizes, arr[:, 0], arr[:, 1], left=np.nan, right=np.nan)
    d = out_dir(cfg, "e4")
    plot_timing(np.array(sizes), timings, os.path.join(d, "e4_timing"))
    evals = {k: getattr(e[1], "n_base_evals", None) for k, e in engines.items()}
    record(cfg, "e4", "e4_timing.json", dict(sizes=sizes, timings={k: v.tolist() for k, v in timings.items()}, device=dev, workers=args.workers,
                                              provider=kind, closure_mode=mode, n_tab=n_tab, M=int(len(times)), kernel_evals_by_engine=evals))


if __name__ == "__main__":
    main()
