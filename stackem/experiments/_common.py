"""Shared helpers of the experiment scripts (argument parsing, assembler construction,
kernel-provider selection, result recording)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict
import numpy as np

from ..config import (load_config, stack_from_config, pg_from_config, em_from_config, thermal_from_config,
                      times_from_config, out_dir, hotspot_dir, save_config, base_case, base_dir)
from ..constants import verify_constants
from ..hotspot_stack import HotSpotRunner
from ..assembler import StackAssembler


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="JSON config; several files separated by commas are merged left to right (defaults: stackem.config.DEFAULT_CONFIG)")
    ap.add_argument("--root", default=None, help="override paths.root_out")
    ap.add_argument("--hotspot", default=None, help="override paths.hotspot_dir")
    ap.add_argument("--weights", default=None, help="trained SKN weights (.npz); default <root>/<base case>/<config skn.weights>")
    ap.add_argument("--out-name", default=None, help="write the outputs to <root>/<out-name>/ instead of <root>/<name>/; the sign-off sizing, kernel data "
                                                     "and default weights are still read from the base case (used by the teacher-engine and ablation runs)")
    ap.add_argument("--base-case", default=None, help="override the config's base_case: the case directory that holds the sizing / kernel data / weights")
    ap.add_argument("--smoke", action="store_true", help="reduced problem size for a quick functional test")
    ap.add_argument("--no-sizing", action="store_true", help="ignore the sign-off sizing file (use PGParams.W for every die)")
    ap.add_argument("--fdm-provider", action="store_true", help="allow the exact FDM kernel provider when no SKN weights exist (validation only: it caches 2 MB of kernels per unique segment and needs tens of GB on a stack)")
    return ap


def setup(args) -> Dict[str, Any]:
    cfg = load_config(args.config)
    if args.root:
        cfg["paths"]["root_out"] = args.root
    if args.hotspot:
        cfg["paths"]["hotspot_dir"] = args.hotspot
    if getattr(args, "base_case", None):
        cfg["base_case"] = args.base_case
    if getattr(args, "out_name", None):                      # e.g. ablation_pde: a separate output tree that shares the base case's sizing
        cfg["base_case"] = base_case(cfg); cfg["name"] = args.out_name
    verify_constants(em_from_config(cfg), thermal_from_config(cfg))
    # the Cu resistivity is stored three times (EM driving force, power-grid solve, Joule heating); they must agree
    rhos = dict(em=em_from_config(cfg).rho, pg=pg_from_config(cfg).rho, thermal=thermal_from_config(cfg).rho_ref)
    if max(rhos.values()) - min(rhos.values()) > 1e-12 * max(rhos.values()):
        print(f"[warn] inconsistent Cu resistivity in the config: {rhos} (em.rho, pg.rho and thermal.rho_ref should be equal)")
    ctx = dict(cfg=cfg, spec=stack_from_config(cfg), pg=pg_from_config(cfg), em=em_from_config(cfg),
               th=thermal_from_config(cfg), times=times_from_config(cfg))
    sk = cfg.get("skn", {})
    ctx["weights"] = os.path.expanduser(args.weights) if args.weights else os.path.join(base_dir(cfg), sk.get("weights", "skn_distill/skn_student.npz"))
    ctx["teacher"] = os.path.join(base_dir(cfg), sk.get("teacher", "skn/skn_best.npz"))     # the full-size SKN (== weights in the teacher-engine config)
    ctx["closure_mode"] = cfg["closure"].get("mode", "direct"); ctx["n_tab"] = int(cfg["closure"].get("n_tab", 192))
    ctx["allow_fdm_provider"] = bool(getattr(args, "fdm_provider", False))
    # sign-off sizing (stackem.signoff): if the case has been rule-sized, every experiment uses those strap widths
    from ..signoff import sizing_path, load_sizing, apply_sizing
    sizing = load_sizing(sizing_path(base_dir(cfg)))                 # runs with a base_case share the base case's sizing
    if sizing is not None and not getattr(args, "no_sizing", False):
        ctx["spec"] = apply_sizing(ctx["spec"], sizing); ctx["sizing"] = sizing
        print("[sign-off sizing] strap widths: " + ", ".join(f"{d}={v['W']*1e6:.2f} um" for d, v in sizing.items()))
    else:
        ctx["sizing"] = None
        if not getattr(args, "no_sizing", False):
            print("[warn] no sign-off sizing found (run stackem.experiments.signoff_size first); using PGParams.W for every die")
    return ctx


def make_assembler(ctx, subdir: str, provider=None) -> StackAssembler:
    cfg = ctx["cfg"]
    runner = HotSpotRunner(hotspot_dir(cfg))
    so = cfg.get("signoff", {})
    return StackAssembler(ctx["spec"], ctx["pg"], ctx["em"], ctx["th"], runner, ctx["times"], out_dir(cfg, subdir), provider,
                          jump_at=cfg["closure"]["jump_at"], with_tm=cfg["closure"]["with_tm"], truth_cells=cfg["truth"]["n_cells"],
                          horizon_years=float(so.get("horizon_years", 10.0)), T_signoff=float(so.get("T_uniform", 378.15)))


def kernel_provider(ctx, prefer: str = "skn", weights: str | None = None):
    """SKN provider if the weights exist, otherwise the exact FDM provider (slow, for validation).
    The provider is wrapped according to config closure.mode ('direct' or 'tabulated', see closure_tab)."""
    from ..closure import FDMKernelProvider
    from ..closure_tab import wrap_provider
    w = weights or ctx["weights"]
    if prefer == "skn" and os.path.exists(w):
        from ..skn_model import SKNWeights
        from ..skn_numpy import SKNNumpyProvider
        prov = SKNNumpyProvider(SKNWeights.load(w))
        return wrap_provider(prov, ctx.get("closure_mode", "direct"), ctx.get("n_tab", 192)), "skn"
    if not ctx.get("allow_fdm_provider", False):
        raise FileNotFoundError(f"SKN weights not found at {w}. Pass --weights <path to skn_best.npz / skn_student.npz> (a second stack such as quad4 "
                                f"uses base3's weights), or --fdm-provider to use the exact FDM kernels for a small validation run.")
    print(f"[warn] SKN weights not found at {w}; using the exact FDM kernel provider (slow, memory-hungry)")
    return wrap_provider(FDMKernelProvider(ctx["em"]), ctx.get("closure_mode", "direct"), ctx.get("n_tab", 192)), "fdm"


def record(cfg, subdir: str, name: str, payload: Dict[str, Any]):
    d = out_dir(cfg, subdir)
    payload = dict(payload)
    payload["_meta"] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                            machine=platform.machine(), cmd=" ".join(sys.argv), config_name=cfg["name"])
    path = os.path.join(d, name)
    json.dump(_jsonable(payload), open(path, "w"), indent=1)
    save_config(cfg, os.path.join(d, "config_used.json"))            # provenance: the exact configuration of this run
    print(f"[saved] {path}")
    return path


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
    return o


def torch_available() -> bool:
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self) -> float:
        now = time.time(); d = now - self.t; self.t = now
        return d
