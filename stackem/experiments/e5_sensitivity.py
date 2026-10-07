"""
E5 - differentiable stack-level EM: sensitivities and reliability-aware power budgeting.

    python -m stackem.experiments.e5_sensitivity --root DIR [--weights ...] [--smoke]

Requires PyTorch (autograd through SKN + closure).  Steps
  1. exact linear maps  T_node(P), j_seg(P)  from unit-power HotSpot / PG runs;
  2. d log t_nuc / dP_block for the k most critical rails of every die (autograd);
  3. finite-difference validation of three entries with the numpy closure;
  4. cross-die sensitivity maps (which top-die block shortens the bottom die's life);
  5. projected-gradient re-allocation of block powers (per-die totals fixed) maximising
     the earliest nucleation time, verified afterwards with HotSpot + FDM truth.
Part 1 (no torch needed): stack-aware repair - the rails that violate the sign-off horizon in the
stack are widened individually (targeted) vs the whole die widened by the largest factor (blanket);
metal-area cost of both, verified with the reference solver -> e5_repair.json.
Figures: e5_repair (violating rails + metal cost), e5_sensmap_<die>, e5_budget_trajectory, e5_diemaps_optimized.
"""
from __future__ import annotations

import os
import time
import numpy as np

from ._common import base_parser, setup, record, out_dir, make_assembler, kernel_provider, torch_available
from ..constants import SEC_PER_YEAR
from ..sensitivity import build_linear_maps, block_power_vector, StackSensitivity, finite_difference_check, optimize_power_budget
from ..viz.plots import plot_sensitivity_map, plot_budget_trajectory, plot_repair
from ..viz.stack3d import plot_die_maps


def main(argv=None):
    ap = base_parser("E5: differentiable sensitivities and power budgeting")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    ctx = setup(args); cfg = ctx["cfg"]; spec = ctx["spec"]
    prov, kind = kernel_provider(ctx)
    assert kind == "skn", "E5 needs trained SKN weights"
    asm = make_assembler(ctx, "e5", prov).run_fields()
    d = out_dir(cfg, "e5")
    # --- part 1 (numpy only): stack-aware repair, targeted vs blanket strap widening ---------------------------
    from ..sensitivity import targeted_vs_blanket_widening
    horizon = float(cfg["signoff"]["horizon_years"])
    print(f"stack-aware repair: widening to restore the {horizon:g}-yr horizon")
    repair = targeted_vs_blanket_widening(asm, prov, horizon)
    record(cfg, "e5", "e5_repair.json", dict(horizon_years=horizon, sizing=ctx.get("sizing"), dies=repair))
    plot_repair(spec, asm.rails("full")[0], repair, os.path.join(d, "e5_repair"), horizon)
    if not torch_available():
        print("E5 part 2 (autograd sensitivities, power budgeting) needs PyTorch; skipping."); return
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    t0 = time.time(); maps = build_linear_maps(asm, d); print(f"linear maps built in {time.time()-t0:.0f}s ({len(maps.block_labels)} blocks)")
    P0 = block_power_vector(spec)
    sens = StackSensitivity(maps, ctx["em"], ctx["th"], ctx["weights"], ctx["times"], dev, cfg["closure"]["with_tm"], cfg["closure"]["jump_at"], ctx["th"].two_sided,
                            ctx["closure_mode"], ctx["n_tab"])
    # baseline t_nuc of every rail (torch)
    with torch.no_grad():
        base = {dn: v.detach().cpu().numpy() for dn, v in sens.forward(torch.as_tensor(P0, dtype=torch.float64, device=dev)).items()}
    k = 2 if args.smoke else args.topk
    targets = {dn: np.argsort(v)[:k].tolist() for dn, v in base.items()}
    t0 = time.time(); grads = sens.gradients(P0, targets); print(f"gradients for {sum(len(v) for v in targets.values())} rails in {time.time()-t0:.0f}s")
    rep = dict(block_labels=maps.block_labels, P0=P0.tolist(), targets=targets, device=dev, repair=repair,
               base_earliest_years={dn: float(v.min() / SEC_PER_YEAR) for dn, v in base.items()},
               grads_dlogt_dP_per_W={dn: g.tolist() for dn, g in grads.items()})
    # finite-difference validation
    fd = []
    for dn, idx in targets.items():
        ri = idx[0]; g = grads[dn][0]
        for b in [int(np.argmax(np.abs(g))), int(np.argmin(np.abs(g) + (np.abs(g) == 0) * 1e9))][:2]:
            delta = 0.01 * max(P0[b], 0.05)                       # central difference, 1 % of the block power
            t_m, t_p = finite_difference_check(asm, P0, b, -delta, dn, ri, maps, prov)[1], finite_difference_check(asm, P0, b, delta, dn, ri, maps, prov)[1]
            fd.append(dict(die=dn, rail=ri, block=maps.block_labels[b], autograd=float(g[b]), finite_diff=float(np.log(t_p / t_m) / (2 * delta)), delta_W=delta))
            print("FD check", fd[-1])
        if args.smoke: break
    rep["finite_difference_check"] = fd
    # cross-die sensitivity maps: most critical rail of every die vs all blocks
    for dn, idx in targets.items():
        plot_sensitivity_map(spec, {f"{dn} rail#{idx[0]}": grads[dn][0] * 100.0}, maps.block_labels,
                             os.path.join(d, f"e5_sensmap_{dn}"), f"{dn} rail#{idx[0]}")
    # power budgeting
    if not args.smoke:
        P_opt, hist = optimize_power_budget(sens, P0, maps.die_block_slices, n_iter=args.iters)
        plot_budget_trajectory(hist, os.path.join(d, "e5_budget_trajectory"))
        rep["optimized_P"] = P_opt.tolist(); rep["trajectory"] = [dict(h, P=None) for h in hist]
        # verify with HotSpot + FDM truth on the optimised power map
        from ..hotspot_stack import make_power_map
        from dataclasses import replace
        spec2 = replace(spec, dies=[replace(dd, power=P_opt[maps.die_block_slices[dd.name]].reshape(dd.nblk, dd.nblk)) for dd in spec.dies])
        from ..assembler import StackAssembler
        a2 = StackAssembler(spec2, ctx["pg"], ctx["em"], ctx["th"], asm.runner, ctx["times"], os.path.join(d, "opt_verify"), None,
                            truth_cells=cfg["truth"]["n_cells"]).run_fields()
        r_opt = a2.solve_variant("full", "truth"); r_base = asm.solve_variant("full", "truth")
        rep["verification"] = dict(base=asm.stack_summary(r_base), optimized=a2.stack_summary(r_opt))
        dies = [dd.name for dd in spec.dies]
        rails = {dn: [x for x in r_opt.rails if x.die == dn] for dn in dies}
        nuc = {dn: np.array([[m.nuc_x, m.nuc_y] for m in r_opt.metrics if m.die == dn]) for dn in dies}
        plot_die_maps(spec2, a2.tf, rails, r_opt.t_nuc, nuc, os.path.join(d, "e5_diemaps_optimized"), "optimised block powers (FDM truth)")
        print("earliest t_nuc: base %.3f yr -> optimised %.3f yr" % (rep["verification"]["base"]["earliest_t_nuc_years"], rep["verification"]["optimized"]["earliest_t_nuc_years"]))
    record(cfg, "e5", "e5_summary.json", rep)


if __name__ == "__main__":
    main()
