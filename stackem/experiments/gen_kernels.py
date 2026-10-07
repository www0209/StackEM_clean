"""
Kernel dataset generation (training data of the SKN) with the reference solver.

    python -m stackem.experiments.gen_kernels --root DIR [--n-profiles 120000] [--workers 16]
    python -m stackem.experiments.gen_kernels --root DIR --smoke        # 2 000 profiles
    python -m stackem.experiments.gen_kernels --root DIR --val-only     # only the two held-out sets that E1 needs
                                                                        # (for runs that use the released weights)

Outputs  <root>/<name>/kernels/kernels_train.npz      (X, Y, AUX, META, profile_id)
         <root>/<name>/kernels/kernels_val_full.npz   400 held-out full kernel surfaces (E1)
         <root>/<name>/kernels/kernels_ood_full.npz   100 out-of-range profiles (E1 OOD)
"""
from __future__ import annotations

import os
from dataclasses import replace
import numpy as np

from ._common import base_parser, setup, record, out_dir
from ..kernel_dataset import SamplingRanges, generate_dataset, generate_validation_kernels


def main(argv=None):
    ap = base_parser("generate the SKN kernel dataset with the reference solver")
    ap.add_argument("--n-profiles", type=int, default=120000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--points", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--val-only", action="store_true", help="skip kernels_train.npz; write only kernels_val_full.npz and kernels_ood_full.npz (same seeds as a full run)")
    args = ap.parse_args(argv); args.no_sizing = True          # the kernel data do not depend on any stack or strap width
    ctx = setup(args); cfg = ctx["cfg"]; em = ctx["em"]
    d = out_dir(cfg, "kernels")
    rg = SamplingRanges(points_per_profile=args.points)
    n = 2000 if args.smoke else args.n_profiles
    if not args.val_only:
        generate_dataset(os.path.join(d, "kernels_train.npz"), n, rg, em, seed=args.seed, n_workers=args.workers, chunk=max(200, n // 20))
    generate_validation_kernels(os.path.join(d, "kernels_val_full.npz"), 40 if args.smoke else 400, rg, em, seed=args.seed + 1)
    ood = replace(rg, T_bar=(450.0, 480.0), T_m=(40.0, 60.0), dT=(-20.0, 20.0))
    generate_validation_kernels(os.path.join(d, "kernels_ood_full.npz"), 10 if args.smoke else 100, ood, em, seed=args.seed + 2)
    record(cfg, "kernels", "dataset_info_val_only.json" if args.val_only else "dataset_info.json", dict(n_profiles=0 if args.val_only else n, ranges=rg.__dict__, ood_ranges=ood.__dict__, em=em.to_dict()))


if __name__ == "__main__":
    main()
