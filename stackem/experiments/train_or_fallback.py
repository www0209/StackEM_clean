"""
Train the SKN: torch trainer when PyTorch is available, sklearn fallback otherwise.

    python -m stackem.experiments.train_or_fallback --root DIR [--epochs 300] [--w-pde 0] [--out-subdir skn] [--force-sklearn]

Reads  <root>/<base case>/kernels/kernels_train.npz
Writes <root>/<name>/<out-subdir>/skn_best.npz (+ logs, curves, summary).

The teacher of the final experiments was trained on data only (w_pde = 0, the default).
The network with the PDE-residual term is a separate ablation with its own output folder:

    python -m stackem.experiments.train_or_fallback --config configs/base3_teacher.json --out-name ablation_pde \
        --w-pde 0.05 --out-subdir skn_pde                      # -> <root>/ablation_pde/skn_pde/skn_best.npz

The sklearn fallback (no PyTorch installed) only serves functional tests; it is not used for any reported number.
"""
from __future__ import annotations

import os
import json
import numpy as np

from ._common import base_parser, setup, out_dir, base_dir, torch_available, record


def main(argv=None):
    ap = base_parser("train the Segment Kernel Network")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--w-pde", type=float, default=0.0, help="weight of the PDE-residual loss; 0 = data only (the final teacher); 0.05 = the ablation network")
    ap.add_argument("--out-subdir", default="skn", help="sub-directory of <root>/<name>/ for the weights: skn (teacher) or skn_pde (PDE-residual ablation)")
    ap.add_argument("--force-sklearn", action="store_true")
    ap.add_argument("--extra", type=str, default="", help="extra args passed to train_skn (quoted)")
    args = ap.parse_args(argv); args.no_sizing = True          # training does not depend on any stack or strap width
    ctx = setup(args); cfg = ctx["cfg"]
    data = os.path.join(base_dir(cfg, "kernels"), "kernels_train.npz")
    out = out_dir(cfg, args.out_subdir)
    if torch_available() and not args.force_sklearn:
        from ..train_skn import main as train_main
        argv2 = ["--data", data, "--out", out, "--epochs", str(8 if args.smoke else args.epochs), "--w-pde", str(args.w_pde)]
        if args.smoke:
            argv2 += ["--hidden", "128,128,128,128", "--batch", "4096"]
        if args.extra:
            argv2 += args.extra.split()
        train_main(argv2)
        if "--out" in args.extra.split():                    # an explicit --out inside --extra wins (train_skn parses the last one)
            out = os.path.expanduser(args.extra.split()[args.extra.split().index("--out") + 1])
    else:
        from ..kernel_dataset import load_dataset
        from ..skn_numpy import train_sklearn_fallback
        X, Y, AUX, META = load_dataset(data)
        w = train_sklearn_fallback(X, Y, max_iter=60 if args.smoke else 400)
        w.save(os.path.join(out, "skn_best.npz"))
        record(cfg, args.out_subdir, "train_summary.json", dict(backend="sklearn_fallback", rows=int(len(X)), w_pde=None,
                                                                 note="sklearn fallback: data-only regression, the PDE-residual term is not available without PyTorch"))
    print("SKN weights:", os.path.join(out, "skn_best.npz"))      # the directory actually written (originally always printed <name>/skn/)


if __name__ == "__main__":
    main()
