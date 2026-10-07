"""
stackem.train_skn
===================

Trainer of the Segment Kernel Network (PyTorch).

    python -m stackem.train_skn --data <kernels.npz> --out <dir> [--epochs 300] [--w-pde 0]

Features
--------
* split by *profile* (all points of a profile go to the same split);
* per-feature / per-target standardisation stored inside the weights;
* AdamW + cosine learning-rate schedule with warm-up, gradient clipping;
* optional PDE-residual term (``--w-pde``), evaluated on a random subset of every
  batch (second-order autograd).  It is OFF by default (w_pde = 0): the teacher of the
  final experiments is trained on reference-solver data only; w_pde = 0.05 is the ablation network;
* early stopping on the validation rel-L2 of the worst target;
* CSV log, loss curve figure, checkpoint export to the framework-independent
  ``.npz`` weight container (evaluated by numpy on any machine);
* deterministic seeds; mixed precision off by default (kernels need fp32).

Measured on the final run (120,000 profiles, 300 epochs, one RTX 4070 Ti SUPER): 1345 s of training,
best validation rel-L2 of the worst target 1.8e-3.  With the PDE-residual term (w_pde = 0.05) the same
300 epochs took 11080 s and reached 2.9e-3.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import numpy as np

from .kernel_dataset import load_dataset, FEATURES, TARGETS
from .skn_model import SKNConfig, SKNWeights, forward_numpy


def split_by_profile(profile_id: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.unique(profile_id)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val].tolist())
    is_val = np.fromiter((p in val_ids for p in profile_id), bool, len(profile_id))
    return ~is_val, is_val


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=0) / (np.linalg.norm(truth, axis=0) + 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Segment Kernel Network")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--w-pde", type=float, default=0.0, help="physics residual weight relative to its initial magnitude (0 = data only; 0.05 = 5 %% of the initial PDE loss)")
    ap.add_argument("--pde-points", type=int, default=1024, help="collocation points per batch for the residual")
    ap.add_argument("--hidden", type=str, default="256,256,256,256,256,256")
    ap.add_argument("--fourier", type=int, default=24)
    ap.add_argument("--fourier-scale", type=float, default=2.0)
    ap.add_argument("--act", type=str, default="silu")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--init", type=str, default=None, help="warm start from a .npz weight file (transfer, E6)")
    ap.add_argument("--max-rows", type=int, default=0, help="subsample the training rows (data-efficiency study)")
    args = ap.parse_args(argv)

    import torch
    from .skn_model import SKN, pde_residual
    args.out = os.path.expanduser(args.out); args.data = os.path.expanduser(args.data)
    if args.init:
        args.init = os.path.expanduser(args.init)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data, allow_pickle=False)
    X, Y, AUX, pid = d["X"], d["Y"], d["AUX"], d["profile_id"]
    tr, va = split_by_profile(pid, args.val_fraction, args.seed)
    Xtr, Ytr, Atr = X[tr], Y[tr], AUX[tr]
    Xva, Yva = X[va], Y[va]
    if args.max_rows and args.max_rows < len(Xtr):
        keep = np.random.default_rng(args.seed).choice(len(Xtr), args.max_rows, replace=False)
        Xtr, Ytr, Atr = Xtr[keep], Ytr[keep], Atr[keep]
    print(f"device {dev}; train rows {len(Xtr):,}  val rows {len(Xva):,}  (profiles {len(np.unique(pid)):,})")

    cfg = SKNConfig(n_fourier=args.fourier, fourier_scale=args.fourier_scale,
                    hidden=[int(h) for h in args.hidden.split(",")], activation=args.act, seed=args.seed)
    if args.init:
        w0 = SKNWeights.load(args.init)
        model = SKN.from_weights(w0).to(dev)
        print(f"warm start from {args.init}")
    else:
        model = SKN(cfg, Xtr.mean(0), Xtr.std(0) + 1e-8, Ytr.mean(0), Ytr.std(0) + 1e-8).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SKN parameters: {n_params:,}")

    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    At = torch.as_tensor(Atr, dtype=torch.float32, device=dev)
    Xv = torch.as_tensor(Xva, dtype=torch.float32, device=dev)
    ysd = model.y_sd; ymu = model.y_mu
    tw = torch.as_tensor(cfg.target_weights, dtype=torch.float32, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches = int(math.ceil(len(Xt) / args.batch))
    total_steps = args.epochs * n_batches

    def lr_at(step):
        if step < args.warmup * n_batches:
            return args.lr * (step + 1) / (args.warmup * n_batches)
        p = (step - args.warmup * n_batches) / max(1, total_steps - args.warmup * n_batches)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * p))

    log_path = os.path.join(args.out, "train_log.csv")
    logf = open(log_path, "w", newline=""); logw = csv.writer(logf)
    logw.writerow(["epoch", "lr", "train_mse", "train_pde", "val_rel_l2_sG", "val_rel_l2_sM", "val_rel_l2_a", "seconds"])
    best = np.inf; best_epoch = -1; step = 0; t_start = time.time(); bad = 0
    pde_ref = None            # PDE loss at the first step: w_pde is a weight RELATIVE to it (w_pde = 0.05 -> 5 % of its initial size)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        tot_mse = 0.0; tot_pde = 0.0
        for bi in range(n_batches):
            idx = perm[bi * args.batch:(bi + 1) * args.batch]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            mse = (((pred - yb) / ysd) ** 2 * tw).mean()
            loss = mse
            if args.w_pde > 0:
                sub = idx[: args.pde_points]
                r = pde_residual(model, Xt[sub], At[sub])
                pde = (r ** 2).mean()
                if pde_ref is None:
                    pde_ref = max(pde.item(), 1e-12); print(f"PDE residual reference (first batch): {pde_ref:.3e}")
                loss = loss + args.w_pde * pde / pde_ref
                tot_pde += pde.item() / pde_ref
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_mse += mse.item(); step += 1
        # validation
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xv[i:i + 65536]) for i in range(0, len(Xv), 65536)]).cpu().numpy()
        rl = rel_l2(pv, Yva)
        worst = float(rl.max())
        el = time.time() - t_start
        logw.writerow([ep, lr_at(step), tot_mse / n_batches, tot_pde / n_batches, *rl.tolist(), el]); logf.flush()
        print(f"epoch {ep:4d} lr {lr_at(step):.2e} mse {tot_mse/n_batches:.3e} pde {tot_pde/max(1,n_batches):.3e} "
              f"val rel-L2 [sG {rl[0]:.2e} sM {rl[1]:.2e} a {rl[2]:.2e}] {el:.0f}s", flush=True)
        if worst < best:
            best, best_epoch, bad = worst, ep, 0
            model.export().save(os.path.join(args.out, "skn_best.npz"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best {best:.3e} @ {best_epoch})"); break
    logf.close()
    model.export().save(os.path.join(args.out, "skn_last.npz"))
    # numpy/torch consistency check on the exported weights
    w = SKNWeights.load(os.path.join(args.out, "skn_best.npz"))
    m2 = SKN.from_weights(w).to(dev).eval()
    with torch.no_grad():
        p_t = m2(Xv[:4096]).cpu().numpy()
    p_n = forward_numpy(w, Xva[:4096])
    diff = float(np.abs(p_t - p_n).max())
    summary = dict(best_val_worst_rel_l2=best, best_epoch=best_epoch, n_params=n_params, torch_numpy_max_diff=diff,
                   train_rows=int(len(Xtr)), val_rows=int(len(Xva)), args=vars(args), features=FEATURES, targets=TARGETS)
    json.dump(summary, open(os.path.join(args.out, "train_summary.json"), "w"), indent=2)
    print(f"done. best val worst-target rel-L2 {best:.3e} at epoch {best_epoch}; torch-vs-numpy max diff {diff:.2e}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(log_path)))
        ep_ = [int(r["epoch"]) for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].semilogy(ep_, [float(r["train_mse"]) for r in rows], color="#1f5f8b"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train MSE (standardised)")
        for k, c in zip(["val_rel_l2_sG", "val_rel_l2_sM", "val_rel_l2_a"], ["#0072B2", "#E69F00", "#009E73"]):
            ax[1].semilogy(ep_, [float(r[k]) for r in rows], label=k.replace("val_rel_l2_", ""), color=c)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation rel-L2"); ax[1].legend(frameon=False)
        for a in ax: a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "train_curves.png"), dpi=200); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
