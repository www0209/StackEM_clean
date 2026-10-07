"""
Unit tests of the StackEM package (no HotSpot, no torch, no pytest required):

    python -m unittest stackem.tests.test_core -v

Every test is a small, fast version of the validation experiments (E0/E2); the
thresholds are the ones the paper relies on.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from ..constants import default_em, default_thermal, verify_constants, EMParams, SEC_PER_YEAR
from ..thermal_profile import SegmentProfile, descriptors_to_profile
from ..korhonen_fdm import solve_segment_hat, default_tau_grid, TreeFDM, nucleation_from_history, single_segment_physical
from ..analytic import s_G_blocked_series, a_unit_flux_series
from ..closure import FDMKernelProvider, close_rails, close_tree_general, TreeSpec
from ..power_grid import Rail, DieGrid, PGParams, embed_thermal, pg_summary
from ..hotspot_stack import DieSpec, make_power_map
from ..blech_screen import rail_steady_state, screen_rails
from ..config import load_config, stack_from_config, pg_from_config, em_from_config, thermal_from_config, times_from_config
from ..skn_model import SKNConfig, SKNWeights, forward_numpy
from ..skn_numpy import SKNNumpyProvider
from ..kernel_dataset import SamplingRanges, sample_profile, kernels_on_grid, FEATURES, TARGETS
from ..ibmpg import parse_spice, solve_dc, extract_rails, write_synthetic_benchmark


def _chain_rail(em, n_seg=6, L=100e-6, j=8e9, T=360.0, dT=4.0, Tm=3.0, Gam=3.0e-5, W=2e-6, H=1e-6):
    Ln = np.full(n_seg, L); jj = np.full(n_seg, j) * np.linspace(1.0, 0.4, n_seg)
    Tn = T + np.linspace(-dT, dT, n_seg + 1)
    Tm_ = np.full(n_seg, Tm); Tbar = 0.5 * (Tn[:-1] + Tn[1:])
    return Rail("D", "H", 0, 0.0, 0.0, Ln, W, H, jj, Tn[:-1], Tn[1:], Tm_, Gam, np.asarray(em.sigma_T(Tbar)), np.zeros(n_seg + 1, bool))


class TestConstants(unittest.TestCase):
    def test_verify(self):
        rep = verify_constants(default_em(), default_thermal())
        self.assertIn("kappa_353K_m2_per_s", rep)

    def test_atomic_volume_matches_lattice(self):
        em = default_em()
        a = 3.615e-10
        self.assertAlmostEqual(em.Omega / (a ** 3 / 4), 1.0, delta=0.01)   # fcc, 4 atoms per cell

    def test_blech_product_band(self):
        em = default_em()
        jl = em.blech_critical_product(353.0) * 1e-4 * 1e2                  # A/m -> A/cm
        self.assertTrue(500 < jl < 5000, jl)

    def test_sigma_T_below_crit(self):
        em = default_em()
        for T in (300.0, 353.0, 400.0):
            self.assertLess(em.sigma_T(T), em.sigma_crit)


class TestKernels(unittest.TestCase):
    def test_fdm_vs_series(self):
        tau = default_tau_grid(1e-6, 3.0, 24)
        k = solve_segment_hat(lambda x: np.ones_like(x), 1.0, 0.0, 0.0, tau, 0.0, 200)
        a = solve_segment_hat(lambda x: np.ones_like(x), 0.0, 1.0, 0.0, tau, 0.0, 200)
        self.assertLess(np.abs(k.s - s_G_blocked_series(k.xi, tau)).max(), 1e-4)
        self.assertLess(np.abs(a.s - a_unit_flux_series(a.xi, tau)).max(), 1e-4)

    def test_profile_descriptor_roundtrip(self):
        em = default_em()
        p = SegmentProfile(120e-6, 355.0, 362.0, 6.0, 2.5e-5)
        th, r, rhoJ, lam, kb = p.descriptors(em)
        p2 = descriptors_to_profile(th, r, rhoJ, lam, p.L, em)
        self.assertAlmostEqual(p2.T_L, p.T_L, places=6); self.assertAlmostEqual(p2.T_R, p.T_R, places=6)
        self.assertAlmostEqual(p2.T_m, p.T_m, places=6)

    def test_mass_conservation_blocked_tree(self):
        em = default_em(); tr = TreeFDM(em, n_cells=30)
        n = [tr.add_node() for _ in range(4)]
        for k in range(3):
            tr.add_segment(n[k], n[k + 1], 100e-6, 2e-12, 5e9 * (1 - 0.3 * k), SegmentProfile(100e-6, 350.0, 356.0, 2.0, 3e-5), 50e6)
        t = np.logspace(4, 9, 20); sol = tr.solve(t)
        m = tr.total_stress_integral(sol)
        self.assertLess(np.abs(m - m[0]).max() / max(abs(m[0]), 1e-30), 1e-8)

    def test_kernel_dataset_rows(self):
        em = default_em(); rng = np.random.default_rng(0)
        prof = sample_profile(rng, SamplingRanges(), em)
        tau = default_tau_grid(1e-6, 1.0, 8)
        xi, sG, sM, a = kernels_on_grid(prof, em, tau, 60)
        for arr in (sG, sM, a):
            self.assertEqual(arr.shape, (len(tau), len(xi))); self.assertTrue(np.isfinite(arr).all())


class TestClosure(unittest.TestCase):
    def setUp(self):
        self.em = default_em(); self.prov = FDMKernelProvider(self.em, n_cells=120)
        self.times = np.logspace(4.0, 9.0, 40)

    def test_closure_vs_tree_fdm(self):
        r = _chain_rail(self.em)
        res = close_rails([r], self.em, self.prov, self.times, self.em.sigma_crit, True, "mid")
        tr = TreeFDM(self.em, n_cells=60); ns = [tr.add_node() for _ in range(r.n_seg + 1)]
        for k in range(r.n_seg):
            tr.add_segment(ns[k], ns[k + 1], r.L[k], r.A, r.j[k], SegmentProfile(r.L[k], r.T_L[k], r.T_R[k], r.T_m[k], r.Gamma), r.sigma_T[k])
        sol = tr.solve(self.times); idx = [tr._gidx[k][0] for k in range(r.n_seg)] + [tr._gidx[-1][-1]]
        truth = sol.sigma[:, idx]
        err = np.linalg.norm(res.sigma_nodes[0] - truth) / np.linalg.norm(truth)
        self.assertLess(err, 2e-2, err)

    def test_tabulated_lookup_matches_direct(self):
        """closure_tab.TabulatedProvider reproduces the wrapped provider (interpolation error only) with far fewer evaluations."""
        from ..closure_tab import TabulatedProvider
        r = _chain_rail(self.em)
        ref = close_rails([r], self.em, self.prov, self.times, self.em.sigma_crit, True, "mid")
        tp = TabulatedProvider(self.prov, 192)
        res = close_rails([r], self.em, tp, self.times, self.em.sigma_crit, True, "mid")
        err = np.abs(res.sigma_nodes[0] - ref.sigma_nodes[0]).max() / np.abs(ref.sigma_nodes[0]).max()
        self.assertLess(err, 2e-3, err)
        self.assertLess(tp.n_base_evals, 0.5 * tp.n_points)          # M = 40 -> 820 lags per segment vs 192 table points
        if np.isfinite(ref.t_nuc[0]):
            self.assertLess(abs(res.t_nuc[0] - ref.t_nuc[0]) / ref.t_nuc[0], 2e-3)

    def test_general_tree_equals_chain(self):
        r = _chain_rail(self.em, n_seg=4)
        res = close_rails([r], self.em, self.prov, self.times, None, True, "mid")
        S = r.n_seg
        tree = TreeSpec(S + 1, np.arange(S), np.arange(1, S + 1), r.L, np.full(S, r.A), r.j, r.T_L, r.T_R, r.T_m, r.Gamma, r.sigma_T)
        sig_ends, tn, node = close_tree_general(tree, self.em, self.prov, self.times, None, True, "mid")
        chain_nodes = np.concatenate([sig_ends[:, :, 0], sig_ends[:, -1:, 1]], axis=1)
        self.assertLess(np.abs(chain_nodes - res.sigma_nodes[0]).max() / np.abs(res.sigma_nodes[0]).max(), 1e-6)

    def test_blech_screen_consistency(self):
        # a rail whose steady-state maximum is below sigma_crit must never nucleate in the FDM either
        em = self.em
        r = _chain_rail(em, j=5e7, T=340.0, dT=0.5, Tm=0.0)   # 5e3 A/cm^2, 600 um: jL below the Blech product
        imm, smax = screen_rails([r], em, True)
        self.assertTrue(imm[0]); self.assertLess(smax[0], em.sigma_crit)
        res = close_rails([r], em, self.prov, np.logspace(4, 10.5, 30), em.sigma_crit, True, "mid")
        self.assertTrue(np.isinf(res.t_nuc[0]))


class TestPowerGrid(unittest.TestCase):
    def test_superposition_and_kcl(self):
        die = DieSpec("D", size=1e-3, nblk=2, power=make_power_map(1.0, 2, [((1, 1), 3.0)]), feed_pitch=250e-6)
        pg = PGParams(pitch=50e-6, W=2e-6, H=1e-6, rho=3e-8, Vdd=0.8, current_fraction=0.4, R_feed=0.0)
        g = DieGrid(die, pg); sol = g.solve_from_blocks()
        # KCL: total current from the feeds equals total sink current
        I_sink = sol.sink_A.sum()
        s = pg_summary(sol, pg)
        self.assertAlmostEqual(s["total_sink_A"], I_sink, delta=1e-9 * max(1.0, I_sink))
        Sh, Sv = g.unit_power_current_responses()
        Ih = np.tensordot(die.power.ravel(), Sh, axes=1)
        self.assertLess(np.abs(Ih - sol.I_h).max(), 1e-9 * np.abs(sol.I_h).max())

    def test_embed_thermal_shapes(self):
        die = DieSpec("D", size=1e-3, nblk=2, power=make_power_map(1.0, 2), feed_pitch=250e-6)
        pg = PGParams(pitch=100e-6, W=2e-6, H=1e-6, rho=3e-8, Vdd=0.8, current_fraction=0.4, R_feed=0.0)
        g = DieGrid(die, pg); sol = g.solve_from_blocks()
        T = 350 + 10 * np.random.default_rng(0).random((16, 16))
        rails = embed_thermal(g, sol, T, default_em(), default_thermal())
        self.assertEqual(len(rails), 2 * g.n)
        for r in rails:
            self.assertEqual(r.n_seg, g.n - 1); self.assertTrue(np.all(r.T_m >= 0))


class TestSKN(unittest.TestCase):
    def test_random_weights_forward_and_provider(self):
        cfg = SKNConfig(hidden=[32, 32], n_fourier=8)
        w = SKNWeights.random(cfg, seed=1)
        X = np.random.default_rng(0).random((5, len(FEATURES)))
        Y = forward_numpy(w, X)
        self.assertEqual(Y.shape, (5, len(TARGETS)))
        prov = SKNNumpyProvider(w)
        out = prov.evaluate(np.array([[0.02, 0.0, 0.0, 3.0]]), np.array([0.0]), np.array([1e-3]))
        for k in ("a", "s_G", "s_M"):
            self.assertTrue(np.isfinite(out[k]).all())
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.npz"); w.save(p); w2 = SKNWeights.load(p)
            self.assertLess(np.abs(forward_numpy(w2, X) - Y).max(), 1e-12)


class TestLambdaClamp(unittest.TestCase):
    def test_provider_clamps_lambda_to_trained_range(self):
        """Segments longer than ~0.9 mm (lam > 126 at Gamma = 6.9 um) lie outside the SKN's input range; the providers
        clamp lam so the network is never evaluated there (the kernels are lam-independent for T_m = 0)."""
        from ..kernel_dataset import LAM_CLAMP
        w = SKNWeights.random(SKNConfig(hidden=[32, 32], n_fourier=8), seed=2); prov = SKNNumpyProvider(w)
        tau = np.array([1e-4, 1e-2, 1.0]); xi = np.zeros(3)
        a = prov.evaluate(np.tile([0.03, 0.1, 0.0, LAM_CLAMP[1]], (3, 1)), xi, tau)
        b = prov.evaluate(np.tile([0.03, 0.1, 0.0, 435.0], (3, 1)), xi, tau)
        for k in ("a", "s_G"):
            self.assertLess(np.abs(a[k] - b[k]).max(), 1e-12)


class TestIBMPG(unittest.TestCase):
    def test_synthetic_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_synthetic_benchmark(os.path.join(td, "syn.spice"), n=8)
            net = parse_spice(p); Vn, Ir = solve_dc(net)
            self.assertTrue(np.all(Vn[1:] <= 1.0 + 1e-9))
            rails = extract_rails(net, Vn, Ir, default_em(), default_thermal(), None, 353.0, 1e-9)
            self.assertGreater(len(rails), 0)
            for r in rails:
                self.assertTrue(np.isfinite(r.j).all()); self.assertGreaterEqual(r.n_seg, 3)


class TestConfig(unittest.TestCase):
    def test_default_config_roundtrip(self):
        cfg = load_config(None)
        spec = stack_from_config(cfg); pg = pg_from_config(cfg); em = em_from_config(cfg); th = thermal_from_config(cfg)
        t = times_from_config(cfg)
        self.assertEqual(len(spec.dies), 3); self.assertGreater(t[-1], t[0])
        self.assertIn("Gamma_um", verify_constants(em, th))
        for d in spec.dies:
            self.assertAlmostEqual(d.total_power(), next(x["P"] for x in cfg["stack"]["dies"] if x["name"] == d.name), places=9)

    def test_shipped_configs(self):
        """configs/base3.json spells out DEFAULT_CONFIG; the other files only change what their comment says."""
        import json
        from ..config import DEFAULT_CONFIG, base_case
        cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs")
        if not os.path.isdir(cdir):
            self.skipTest("configs/ not next to the package")
        strip = lambda c: {k: v for k, v in c.items() if k not in ("_comment", "name", "base_case")}
        base = load_config(os.path.join(cdir, "base3.json"))
        self.assertEqual(strip(base), strip(DEFAULT_CONFIG)); self.assertEqual(base["name"], "base3")
        self.assertEqual(base["closure"]["mode"], "tabulated"); self.assertEqual(base["times"]["n"], 32)
        self.assertEqual(base["skn"]["weights"], "skn_distill/skn_student.npz")
        t = load_config(os.path.join(cdir, "base3_teacher.json"))
        self.assertEqual(base_case(t), "base3"); self.assertEqual(t["name"], "base3_teacher")
        for k in ("stack", "pg", "em", "thermal", "signoff", "truth"):          # same stack, same sizing rule, same reference
            self.assertEqual(t[k], base[k])
        self.assertEqual((t["closure"]["mode"], t["times"]["n"], t["skn"]["weights"]), ("direct", 64, "skn/skn_best.npz"))
        d = load_config(os.path.join(cdir, "dense3.json")); self.assertEqual(d["stack"], base["stack"]); self.assertEqual(d["pg"]["pitch"], 50e-6)
        q = load_config(os.path.join(cdir, "quad4.json")); self.assertEqual(len(q["stack"]["dies"]), 4)
        s = load_config(os.path.join(cdir, "quad4.json") + "," + os.path.join(cdir, "smoke.json"))   # merged left to right
        self.assertEqual((len(s["stack"]["dies"]), s["stack"]["grid"], s["truth"]["n_cells"]), (4, 32, 40))

    def test_nucleation_interpolation(self):
        t = np.logspace(4, 8, 30); sig = np.zeros((30, 3)); sig[:, 1] = np.linspace(0, 600e6, 30)
        tn, node, _ = nucleation_from_history(t, sig, 500e6)
        self.assertEqual(node, 1); self.assertTrue(t[0] < tn < t[-1])


class TestSignoff(unittest.TestCase):
    def test_per_die_strap_width(self):
        from dataclasses import replace
        die = DieSpec("D", size=1e-3, nblk=2, power=make_power_map(1.0, 2), feed_pitch=250e-6)
        pg = PGParams(pitch=100e-6, W=2e-6, H=1e-6, rho=3e-8, Vdd=0.8, current_fraction=0.4, R_feed=0.0)
        g1 = DieGrid(die, pg); g2 = DieGrid(replace(die, rail_W=4e-6), pg)
        self.assertAlmostEqual(g2.pg.A / g1.pg.A, 2.0)
        s1, s2 = g1.solve_from_blocks(), g2.solve_from_blocks()
        self.assertLess(np.abs(s1.I_h - s2.I_h).max(), 1e-12 * np.abs(s1.I_h).max())     # currents independent of W ...
        self.assertAlmostEqual(np.abs(s1.j_h(g1.pg.A)).max() / np.abs(s2.j_h(g2.pg.A)).max(), 2.0, places=9)   # ... j halves

    def test_size_strap_width_bisection(self):
        from ..signoff import size_strap_width, earliest_t_nuc
        from ..constants import SEC_PER_YEAR
        em = default_em(); times = np.logspace(4, 9.5, 40)
        base = _chain_rail(em, n_seg=4, j=4e9, T=345.0, dT=1.0, Tm=0.0)

        def rails_of_W(W):
            from dataclasses import replace
            return [replace(base, W=W, j=base.j * (2e-6 / W))]
        target = 10 * SEC_PER_YEAR
        r = size_strap_width(rails_of_W, em, times, target, 0.5e-6, 40e-6, n_cells=30)
        self.assertFalse(r["saturated"]); self.assertGreaterEqual(r["t_earliest_s"], target)
        # 2 % below the sized width the target must be missed (minimality)
        self.assertLess(earliest_t_nuc(rails_of_W(r["W"] / 1.05), em, times, 30), target)


if __name__ == "__main__":
    unittest.main()
