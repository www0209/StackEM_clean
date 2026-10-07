"""
stackem.train_distill
=======================

Knowledge distillation of the Segment Kernel Network into a small student (PyTorch).

    python -m stackem.train_distill --data <kernels.npz> --teacher <skn_best.npz> --out <dir> [--epochs 150]

The teacher (6 x 256, 343,811 parameters) is accurate but expensive to evaluate at the 10^7-10^9
kernel points a stack asks for.  The student (default 4 x 128 with 16 Fourier features, 54,915
parameters, 6.3x smaller) is trained on

  * the reference kernel data (FDM targets, the same rows the teacher was trained on), and
  * teacher-labelled points: the same descriptor distribution, log tau uniform over the trained
    range, and xi drawn from {0, 1} with probability ``--end-fraction`` (the closure only ever
    evaluates the segment ends), uniform in [0, 1] otherwise;

with the standard per-target standardised MSE.  Validation is against the held-out FDM
profiles (never the teacher), so the reported accuracy is against the physics, not the teacher.
The exported weights use the same container as the teacher and drop into every provider.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import numpy as np

from .kernel_dataset import FEATURES, TARGETS
from .skn_model import SKNConfig, SKNWeights, forward_numpy
from .train_skn import split_by_profile, rel_l2


def main(argv=None):
    ap = argparse.ArgumentParser(description="Distil the SKN into a small student network")
    ap.add_argument("--data", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)        # originally 60; the final experiments used 150 (see docs)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--lr-min", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--hidden", type=str, default="128,128,128,128")   # originally "64,64,64,64"; the final experiments used "128,128,128,128" (see docs)
    ap.add_argument("--fourier", type=int, default=16); ap.add_argument("--fourier-scale", type=float, default=2.0)
    ap.add_argument("--n-synthetic", type=int, default=8_000_000, help="teacher-labelled points added to the FDM rows")
    ap.add_argument("--end-fraction", type=float, default=0.7, help="share of synthetic points at xi in {0, 1}")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=30)       # originally 15; the final experiments used 30 (see docs)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args(argv)

    import torch
    from .skn_model import SKN
    args.out = os.path.expanduser(args.out); args.data = os.path.expanduser(args.data); args.teacher = os.path.expanduser(args.teacher)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data, allow_pickle=False); X, Y, pid = d["X"], d["Y"], d["profile_id"]
    tr, va = split_by_profile(pid, args.val_fraction, args.seed)
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    end_va = (Xva[:, 0] <= 1e-6) | (Xva[:, 0] >= 1 - 1e-6)
    tw = SKNWeights.load(args.teacher); teacher = SKN.from_weights(tw).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    # ---- teacher-labelled synthetic points -------------------------------------------------------------------
    n_syn = args.n_synthetic
    src = rng.integers(0, len(Xtr), n_syn)
    Xs = Xtr[src].copy()                                                # descriptors of real profiles
    lt_lo, lt_hi = float(X[:, 1].min()), float(X[:, 1].max())
    Xs[:, 1] = rng.uniform(lt_lo, lt_hi, n_syn)
    ends = rng.random(n_syn) < args.end_fraction
    Xs[:, 0] = np.where(ends, rng.integers(0, 2, n_syn).astype(np.float32), rng.uniform(0, 1, n_syn).astype(np.float32))
    with torch.no_grad():
        Ys = np.concatenate([teacher(torch.as_tensor(Xs[i:i + 262144], dtype=torch.float32, device=dev)).cpu().numpy() for i in range(0, n_syn, 262144)])
    Xall = np.concatenate([Xtr, Xs]).astype(np.float32); Yall = np.concatenate([Ytr, Ys]).astype(np.float32)
    print(f"device {dev}; FDM rows {len(Xtr):,} + teacher-labelled {n_syn:,} = {len(Xall):,}; val rows {len(Xva):,} ({end_va.sum():,} at the ends)")
    with torch.no_grad():
        pt = np.concatenate([teacher(torch.as_tensor(Xva[i:i + 65536], dtype=torch.float32, device=dev)).cpu().numpy() for i in range(0, len(Xva), 65536)])
    rl_teacher = rel_l2(pt, Yva); rl_teacher_end = rel_l2(pt[end_va], Yva[end_va])
    print(f"teacher ({n_teacher:,} params): val rel-L2 {np.round(rl_teacher, 5)}, at the ends {np.round(rl_teacher_end, 5)}")

    cfg = SKNConfig(n_fourier=args.fourier, fourier_scale=args.fourier_scale, hidden=[int(h) for h in args.hidden.split(",")], activation="silu", seed=args.seed)
    model = SKN(cfg, Xall.mean(0), Xall.std(0) + 1e-8, Yall.mean(0), Yall.std(0) + 1e-8).to(dev)
    n_student = sum(p.numel() for p in model.parameters())
    print(f"student parameters: {n_student:,}  ({n_teacher / n_student:.1f}x smaller)")
    Xt = torch.as_tensor(Xall, device=dev); Yt = torch.as_tensor(Yall, device=dev); Xv = torch.as_tensor(Xva, dtype=torch.float32, device=dev)
    ysd = model.y_sd
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches = int(math.ceil(len(Xt) / args.batch)); total = args.epochs * n_batches

    def lr_at(step):
        if step < args.warmup * n_batches:
            return args.lr * (step + 1) / (args.warmup * n_batches)
        p = (step - args.warmup * n_batches) / max(1, total - args.warmup * n_batches)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * p))

    log_path = os.path.join(args.out, "train_log.csv"); logf = open(log_path, "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "lr", "train_mse", "val_rel_l2_sG", "val_rel_l2_sM", "val_rel_l2_a", "val_end_rel_l2_sG", "val_end_rel_l2_sM", "val_end_rel_l2_a", "seconds"])
    best = np.inf; best_epoch = -1; step = 0; t0 = time.time(); bad = 0
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt), device=dev); tot = 0.0
        for bi in range(n_batches):
            idx = perm[bi * args.batch:(bi + 1) * args.batch]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            pred = model(Xt[idx]); loss = (((pred - Yt[idx]) / ysd) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); step += 1
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xv[i:i + 65536]) for i in range(0, len(Xv), 65536)]).cpu().numpy()
        rl = rel_l2(pv, Yva); rle = rel_l2(pv[end_va], Yva[end_va]); worst = float(rl.max()); el = time.time() - t0
        logw.writerow([ep, lr_at(step), tot / n_batches, *rl.tolist(), *rle.tolist(), el]); logf.flush()
        print(f"epoch {ep:3d} lr {lr_at(step):.2e} mse {tot/n_batches:.3e} val rel-L2 [sG {rl[0]:.2e} sM {rl[1]:.2e} a {rl[2]:.2e}] ends [{rle[0]:.2e} {rle[1]:.2e} {rle[2]:.2e}] {el:.0f}s", flush=True)
        if worst < best:
            best, best_epoch, bad = worst, ep, 0; model.export().save(os.path.join(args.out, "skn_student.npz"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best {best:.3e} @ {best_epoch})"); break
    logf.close()
    w = SKNWeights.load(os.path.join(args.out, "skn_student.npz"))
    p_n = forward_numpy(w, Xva[:4096])
    with torch.no_grad():
        p_t = SKN.from_weights(w).to(dev).eval()(Xv[:4096]).cpu().numpy()
    pv = forward_numpy(w, Xva); rl = rel_l2(pv, Yva); rle = rel_l2(pv[end_va], Yva[end_va])
    summary = dict(student=dict(n_params=n_student, val_rel_l2=rl.tolist(), val_end_rel_l2=rle.tolist(), best_epoch=best_epoch, hidden=cfg.hidden, n_fourier=cfg.n_fourier),
                   teacher=dict(n_params=n_teacher, val_rel_l2=rl_teacher.tolist(), val_end_rel_l2=rl_teacher_end.tolist(), path=args.teacher),
                   compression=n_teacher / n_student, torch_numpy_max_diff=float(np.abs(p_t - p_n).max()), rows=dict(fdm=int(len(Xtr)), synthetic=int(n_syn), val=int(len(Xva))),
                   args=vars(args), features=FEATURES, targets=TARGETS)
    json.dump(summary, open(os.path.join(args.out, "distill_summary.json"), "w"), indent=2)
    print(f"done. student val rel-L2 {np.round(rl, 5)} (teacher {np.round(rl_teacher, 5)}); {n_teacher/n_student:.1f}x fewer parameters")
    try:
        from .viz import style as st
        import matplotlib.pyplot as plt
        st.use_paper_style()
        rows = list(csv.DictReader(open(log_path))); ep_ = [int(r["epoch"]) for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(st.IEEE_2COL, 2.3), constrained_layout=True)
        ax[0].semilogy(ep_, [float(r["train_mse"]) for r in rows], color=st.OKABE_ITO["blue"]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train MSE (standardised)")
        for k, c, lab in zip(["val_rel_l2_sG", "val_rel_l2_sM", "val_rel_l2_a"], [st.OKABE_ITO["blue"], st.OKABE_ITO["orange"], st.OKABE_ITO["green"]], ["$s_G$", "$s_M$", "$a$"]):
            ax[1].semilogy(ep_, [float(r[k]) for r in rows], label=f"student {lab}", color=c)
        for v, c in zip(rl_teacher, [st.OKABE_ITO["blue"], st.OKABE_ITO["orange"], st.OKABE_ITO["green"]]):
            ax[1].axhline(v, color=c, ls=":", lw=0.8)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation rel-L2 vs FDM"); ax[1].legend(fontsize=6, title="dotted: teacher", title_fontsize=6)
        ax[1].set_title(f"student {n_student:,} vs teacher {n_teacher:,} parameters", fontsize=7)
        st.save(fig, os.path.join(args.out, "distill_curves"))
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
