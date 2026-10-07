"""
stackem.kernel_dataset
========================

Generation of the Segment Kernel Network (SKN) training data with the
reference finite-volume solver.

Every training *profile* is a segment thermal state described by the
dimensionless descriptors (theta, r, rho_J, lam).  For each profile the three
kernels are computed on a dense (xi, tau) grid and then sub-sampled:

    s_G(xi,tau)   EM response      (unit source, blocked ends)         stored as s_G / sqrt(tau)
    s_M(xi,tau)   TM response      (source M_hat(xi), blocked ends)   stored as s_M / (m_amp sqrt(tau))
    a(xi,tau)     unit-flux Green  (flux 1 at xi=0)                    stored as a / sqrt(tau)

with  m_amp = theta (|r| + rho_J lam tanh(lam/2)) + 1e-3  (the TM source amplitude).  The
sqrt(tau) scaling is essential: the junction closure divides by the same-end
kernel values at the smallest time lags, where the kernels behave like
-2 sqrt(tau/(pi kappa_hat_end)); predicting the scaled quantity keeps the
*relative* accuracy uniform down to tau = 1e-6 (an unscaled network with 1 %
absolute error there has 1000 % relative error and the closure diverges).  The ends xi = 0 and xi = 1 are always included in the
sample because the junction closure evaluates the kernels there.

For the optional PDE-residual loss (ablation only), kappa_hat, dkappa_hat/dxi and
the sources at every sampled point are stored as well.

Feature vector (network input):   [xi, log10 tau, theta, r, rho_J, lam]
Target vector:                    [s_G, s_M / m_amp, a] / sqrt(tau)

The sampling ranges below cover every segment the stack pipeline can produce
(T_bar 300-450 K, |dT| <= 12 K across a segment, Joule bump <= 40 K,
L/Gamma from 0.3 to 130, tau from 1e-6 to 10) with margin.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from typing import Dict, List, Tuple
import numpy as np

from .constants import EMParams, default_em, K_B_EV
from .thermal_profile import SegmentProfile
from .korhonen_fdm import solve_segment_hat, default_tau_grid, chebyshev_mesh

FEATURES = ["xi", "log10_tau", "theta", "r", "rho_J", "lam"]
# lam range the SKN is trained on (SamplingRanges.log10_lam).  The network providers clamp lam to this range:
# for lam -> infinity the Joule boundary layer (width Gamma) becomes thin against the segment and the kernels
# converge to their lam-independent limit (exactly so for T_m = 0, where lam does not enter the physics at all),
# whereas an MLP fed lam = 400 (a 3-mm segment) extrapolates far outside its inputs and returns garbage.
LAM_CLAMP = (10 ** -0.5, 10 ** 2.1)
TARGETS = ["s_G_over_sqrt_tau", "s_M_scaled_over_sqrt_tau", "a_over_sqrt_tau"]


@dataclass
class SamplingRanges:
    T_bar: Tuple[float, float] = (300.0, 450.0)
    dT: Tuple[float, float] = (-12.0, 12.0)
    T_m: Tuple[float, float] = (0.0, 40.0)
    p_zero_Tm: float = 0.3                    # fraction of profiles without Joule bump
    log10_lam: Tuple[float, float] = (-0.5, 2.1)   # lam in [0.32, 126]
    log10_tau: Tuple[float, float] = (-6.0, 1.0)
    n_cells: int = 200
    n_tau_grid: int = 96
    points_per_profile: int = 128
    n_end_points: int = 32                    # of which at xi = 0 or xi = 1


def m_amp_of(theta, r, rhoJ, lam):
    """Amplitude of the dimensionless TM source M_hat = L T'/T from the descriptors:
    dT/T_bar ~ theta r  and  max|dT_Joule/dx| L / T_bar ~ theta rho_J lam tanh(lam/2)."""
    lam = np.asarray(lam, float)
    return theta * (np.abs(r) + np.abs(rhoJ) * lam * np.tanh(0.5 * lam)) + 1e-3


def sample_profile(rng: np.random.Generator, rg: SamplingRanges, em: EMParams, L_ref: float = 100e-6) -> SegmentProfile:
    Tb = rng.uniform(*rg.T_bar)
    dT = rng.uniform(*rg.dT)
    Tm = 0.0 if rng.random() < rg.p_zero_Tm else rng.uniform(*rg.T_m)
    lam = 10 ** rng.uniform(*rg.log10_lam)
    return SegmentProfile(L=L_ref, T_L=Tb - 0.5 * dT, T_R=Tb + 0.5 * dT, T_m=Tm, Gamma=L_ref / lam)


def kernels_on_grid(prof: SegmentProfile, em: EMParams, tau_grid: np.ndarray, n_cells: int):
    kh = lambda xi: prof.kappa_hat(xi, em)
    sG = solve_segment_hat(kh, 1.0, 0.0, 0.0, tau_grid, 0.0, n_cells)
    sM = solve_segment_hat(kh, lambda xi: prof.M_hat(xi, em), 0.0, 0.0, tau_grid, 0.0, n_cells)
    a = solve_segment_hat(kh, 0.0, 1.0, 0.0, tau_grid, 0.0, n_cells)
    return sG.xi, sG.s, sM.s, a.s


def _profile_rows(args):
    """Worker: one profile -> (X, Y, aux) arrays with points_per_profile rows."""
    seed, rg_dict, em_dict = args
    rg = SamplingRanges(**rg_dict); em = EMParams(**em_dict)
    rng = np.random.default_rng(seed)
    prof = sample_profile(rng, rg, em)
    theta, r, rhoJ, lam, _ = prof.descriptors(em)
    tau_grid = default_tau_grid(10 ** rg.log10_tau[0], 10 ** rg.log10_tau[1], rg.n_tau_grid)
    xi, sG, sM, a = kernels_on_grid(prof, em, tau_grid, rg.n_cells)
    P = rg.points_per_profile; E = rg.n_end_points
    # sample (xi index, tau index): interior points uniform in the Chebyshev index (dense near ends)
    it = rng.integers(0, len(tau_grid), P)
    ix = rng.integers(0, len(xi), P)
    ix[:E // 2] = 0; ix[E // 2:E] = len(xi) - 1
    ma = m_amp_of(theta, r, rhoJ, lam)
    X = np.stack([xi[ix], np.log10(tau_grid[it]), np.full(P, theta), np.full(P, r), np.full(P, rhoJ), np.full(P, lam)], 1)
    st = np.sqrt(tau_grid[it])
    Y = np.stack([sG[it, ix] / st, sM[it, ix] / (ma * st), a[it, ix] / st], 1)
    # aux for the physics residual: kappa_hat, dkappa_hat/dxi, M_hat at the points
    kh = prof.kappa_hat(xi[ix], em)
    h = 1e-4
    dkh = (prof.kappa_hat(np.clip(xi[ix] + h, 0, 1), em) - prof.kappa_hat(np.clip(xi[ix] - h, 0, 1), em)) / (
        np.clip(xi[ix] + h, 0, 1) - np.clip(xi[ix] - h, 0, 1))
    Mh = prof.M_hat(xi[ix], em) / ma
    AUX = np.stack([kh, dkh, Mh], 1)
    meta = np.array([prof.T_L, prof.T_R, prof.T_m, prof.Gamma, theta, r, rhoJ, lam])
    return X.astype(np.float32), Y.astype(np.float32), AUX.astype(np.float32), meta.astype(np.float64)


def generate_dataset(out_path: str, n_profiles: int, rg: SamplingRanges | None = None, em: EMParams | None = None,
                     seed: int = 0, n_workers: int = 8, chunk: int = 2000, verbose: bool = True) -> str:
    """Generate n_profiles profiles in parallel and store an .npz with X, Y, AUX, META and the ranges."""
    rg = rg or SamplingRanges(); em = em or default_em()
    seeds = [seed * 1_000_003 + k for k in range(n_profiles)]
    args = [(s, asdict(rg), em.to_dict()) for s in seeds]
    Xs, Ys, As, Ms = [], [], [], []
    t0 = time.time()
    with Pool(n_workers) as pool:
        for k, (X, Y, A, Mt) in enumerate(pool.imap(_profile_rows, args, chunksize=max(1, chunk // 50))):
            Xs.append(X); Ys.append(Y); As.append(A); Ms.append(Mt)
            if verbose and (k + 1) % chunk == 0:
                el = time.time() - t0
                print(f"  {k+1}/{n_profiles} profiles, {el:.0f} s elapsed, eta {el/(k+1)*(n_profiles-k-1):.0f} s", flush=True)
    X = np.concatenate(Xs); Y = np.concatenate(Ys); AUX = np.concatenate(As); META = np.stack(Ms)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, AUX=AUX, META=META, features=np.array(FEATURES), targets=np.array(TARGETS),
                        ranges=np.array([str(asdict(rg))]), em=np.array([str(em.to_dict())]),
                        profile_id=np.repeat(np.arange(n_profiles), rg.points_per_profile))
    if verbose:
        print(f"saved {out_path}: X {X.shape}, Y {Y.shape}, {time.time()-t0:.0f} s")
    return out_path


def generate_validation_kernels(out_path: str, n_profiles: int, rg: SamplingRanges | None = None,
                                em: EMParams | None = None, seed: int = 12345, n_tau: int = 48, n_cells: int = 200) -> str:
    """Full kernels (dense xi, tau) for a few hundred held-out profiles: used by E1 to
    report errors on complete kernel surfaces (not just sampled points)."""
    rg = rg or SamplingRanges(); em = em or default_em()
    rng = np.random.default_rng(seed)
    tau_grid = default_tau_grid(10 ** rg.log10_tau[0], 10 ** rg.log10_tau[1], n_tau)
    metas, SG, SM, A = [], [], [], []
    for _ in range(n_profiles):
        prof = sample_profile(rng, rg, em)
        theta, r, rhoJ, lam, _ = prof.descriptors(em)
        xi, sG, sM, a = kernels_on_grid(prof, em, tau_grid, n_cells)
        metas.append([prof.T_L, prof.T_R, prof.T_m, prof.Gamma, theta, r, rhoJ, lam])
        SG.append(sG); SM.append(sM); A.append(a)
    np.savez_compressed(out_path, xi=xi, tau=tau_grid, META=np.array(metas), s_G=np.array(SG), s_M=np.array(SM), a=np.array(A))
    return out_path


def load_dataset(path: str):
    d = np.load(path, allow_pickle=False)
    return d["X"], d["Y"], d["AUX"], d["META"]
