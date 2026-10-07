"""
stackem.constants
===================

Physical constants and Cu-interconnect material parameters used everywhere in
StackEM, each with its provenance and a self-check.  All quantities are SI
(m, s, K, Pa, A/m^2, J) unless the name says otherwise.

Why this file exists
--------------------
Two earlier code bases carried parameter errors that would have
silently corrupted every downstream number:

* the released HierPINN-EM code (2022): ``e = 1.69e-19 C`` (elementary charge is
  1.602176634e-19 C, a 5.5 % error in the EM driving force G).
* an earlier in-house draft: ``Omega = 8.78e-30 m^3`` (the Cu atomic volume is 1.18e-29 m^3,
  a 26 % error in G *and* in kappa).

Every parameter below is therefore (a) traced to a source, (b) cross-checked
by an independent derivation where one exists, and (c) guarded by
``verify_constants()`` which is executed by the test-suite and by every
experiment script at start-up.

Parameter set
-------------
The EM parameter set follows the UC-Riverside (Tan group) Korhonen line
(EMSpice, HierPINN-EM, PostPINN-EM) so that our numbers are directly
comparable with the closest prior work.  Alternatives used by other groups
are listed in ``ALTERNATIVE_SETS`` for reference only (no code reads that table;
the E6 sweep varies Ea and sigma_crit through its own explicit lists).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict

# --------------------------------------------------------------------------
# Fundamental constants (CODATA 2018, exact by definition of the SI)
# --------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19      # C      elementary charge
K_B = 1.380649e-23              # J/K    Boltzmann constant
K_B_EV = 8.617333262e-5         # eV/K   Boltzmann constant in eV/K
N_A = 6.02214076e23             # 1/mol  Avogadro constant
SEC_PER_YEAR = 365.25 * 86400.0  # s     Julian year

# --------------------------------------------------------------------------
# Copper crystallography (independent derivation of the atomic volume)
# --------------------------------------------------------------------------
CU_LATTICE_A = 3.615e-10        # m   FCC lattice constant of Cu at 293 K (Kittel)
CU_ATOMS_PER_CELL = 4
CU_MOLAR_VOLUME = 7.11e-6       # m^3/mol   (63.546 g/mol / 8.96 g/cm^3)
OMEGA_FROM_LATTICE = CU_LATTICE_A ** 3 / CU_ATOMS_PER_CELL      # 1.181e-29
OMEGA_FROM_MOLAR = CU_MOLAR_VOLUME / N_A                        # 1.181e-29


@dataclass
class EMParams:
    """Korhonen-model parameters for a Cu dual-damascene interconnect.

    Attributes
    ----------
    Z_eff : effective charge number (dimensionless). Tan-group line uses 10.
    Ea_eV : activation energy of the dominant (interface/grain-boundary)
            diffusion path.  0.86 eV is the value quoted in the EMSpice/HierPINN
            papers (their code uses 0.84).  Literature range 0.8-1.1 eV.
    D0    : pre-exponential diffusivity (m^2/s).  7.5e-8 with Ea=0.86 eV gives
            kappa(353 K) ~ 1e-17 m^2/s, i.e. months-to-years nucleation at
            1e6 A/cm^2, consistent with EMSpice-3 case studies.
    B     : effective bulk modulus of the confined line (Pa).  1e11 is the
            standard value in the Korhonen literature (Korhonen 1993; Tan group).
    Omega : atomic volume (m^3).  1.18e-29 for Cu (derived above).
    rho   : Cu line resistivity at operating temperature (Ohm m).  3e-8 accounts
            for size effects in sub-100 nm lines (bulk 1.68e-8 at 293 K).
    Q_eV  : heat of transport |Q*| for thermomigration (eV).  0.09 eV (Chen &
            Tan, TCAD'21 EM-TM; magnitude also used by Pathak et al., 2011,
            as -0.0867 eV under the opposite sign convention).  Sign convention
            here: atoms migrate from hot to cold, i.e. the TM force is
            -Q*/(Omega T) dT/dx and enters the flux bracket with the same sign
            as the EM term (see korhonen_fdm.py docstring).
    sigma_crit : void-nucleation critical hydrostatic stress (Pa).  500 MPa
            (Tan group; literature 400-600 MPa).
    T_zero_stress : zero-stress (anneal) temperature (K) for the thermal
            residual stress model sigma_T(T) = B*dalpha*(T_zs - T).
            250 C = 523 K is a typical Cu BEOL anneal / capping temperature.
    dalpha : CTE mismatch alpha_Cu - alpha_Si (1/K): 16.5e-6 - 2.6e-6.
    """
    Z_eff: float = 10.0
    Ea_eV: float = 0.86
    D0: float = 7.5e-8
    B: float = 1.0e11
    Omega: float = 1.181e-29
    rho: float = 3.0e-8
    Q_eV: float = 0.09
    sigma_crit: float = 5.0e8
    T_zero_stress: float = 523.15
    dalpha: float = (16.5 - 2.6) * 1e-6
    sigma_T_model: str = "linear"      # "linear" (temperature dependent) | "constant"
    sigma_T_const: float = 2.36e8      # used when sigma_T_model == "constant" (value at 353 K)
    name: str = "tan_group_line"

    # ---- derived quantities -------------------------------------------
    @property
    def Ea_J(self) -> float:
        return self.Ea_eV * E_CHARGE

    @property
    def Q_J(self) -> float:
        return self.Q_eV * E_CHARGE

    def D_a(self, T):
        """Atomic diffusivity D_a(T) = D0 exp(-Ea/kT)  [m^2/s]. Vectorised."""
        import numpy as np
        T = np.asarray(T, dtype=float)
        return self.D0 * np.exp(-self.Ea_eV / (K_B_EV * T))

    def kappa(self, T):
        """Stress diffusivity kappa(T) = D_a B Omega / (k_B T)  [m^2/s]."""
        import numpy as np
        T = np.asarray(T, dtype=float)
        return self.D_a(T) * self.B * self.Omega / (K_B * T)

    def G_of_j(self, j):
        """EM driving stress gradient G = e Z* rho j / Omega  [Pa/m] (sign of j kept)."""
        return E_CHARGE * self.Z_eff * self.rho * j / self.Omega

    def M_of(self, T, dTdx):
        """Thermomigration stress gradient M = Q*/(Omega T) dT/dx  [Pa/m]."""
        return self.Q_J / (self.Omega * T) * dTdx

    def sigma_T(self, T):
        """Thermal residual (initial) hydrostatic stress at operating temperature T [Pa].

        linear model:  sigma_T = B * dalpha * (T_zs - T), clipped at >= 0.
        constant model: sigma_T_const.
        """
        import numpy as np
        T = np.asarray(T, dtype=float)
        if self.sigma_T_model == "constant":
            return np.full_like(T, self.sigma_T_const)
        return np.clip(self.B * self.dalpha * (self.T_zero_stress - T), 0.0, None)

    @property
    def dsigmaT_dT(self) -> float:
        """d sigma_T / dT  [Pa/K]  (negative: heating relaxes the tensile stress)."""
        return -self.B * self.dalpha if self.sigma_T_model == "linear" else 0.0

    def blech_critical_product(self, T=353.0):
        """Blech critical (j L) product for a single blocked segment:
        (jL)_c = 2 Omega (sigma_crit - sigma_T(T)) / (e Z* rho)   [A/m]."""
        return 2.0 * self.Omega * (self.sigma_crit - float(self.sigma_T(T))) / (E_CHARGE * self.Z_eff * self.rho)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThermalParams:
    """Wire self-heating (fin-equation) parameters.

    T(x) along a segment obeys  k_cu t_cu T'' - (T - T_sub(x)) * h_eff + j^2 rho t_cu = 0,
    giving the thermal characteristic length  Gamma^2 = k_cu t_cu / h_eff  and the
    Joule temperature rise amplitude  T_m = j^2 rho Gamma^2 / k_cu  (Chen & Tan,
    TCAD 2016; EMSpice 2.1/3 eq. (4), whose cos/sin should read cosh/sinh).

    h_eff is the heat-loss conductance per unit area to the surroundings.  For a
    standalone die the wire loses heat only downward through the ILD stack:
        h_eff = k_ILD / t_ILD.
    In a hybrid-bonded stack a top-metal rail also loses heat upward through the
    bonding layer into the neighbouring die:
        h_eff = k_ILD/t_ILD + k_bond/t_bond.
    """
    k_cu: float = 400.0        # W/(m K)  Cu thermal conductivity (bulk 401)
    k_ild: float = 1.4         # W/(m K)  SiO2-like ILD (low-k would be ~1.0)
    t_cu: float = 1.0e-6       # m        rail thickness
    t_ild: float = 4.0e-6      # m        dielectric between rail and Si substrate
    k_bond: float = 40.0       # W/(m K)  effective vertical conductivity of a hybrid-bond layer (~9 % Cu pad area)
    t_bond: float = 5.0e-6     # m        bonding layer thickness
    two_sided: bool = True     # True in a bonded stack, False for a standalone die
    rho_ref: float = 3.0e-8    # Ohm m    (same rho as EMParams.rho)

    def h_eff(self, two_sided: bool | None = None) -> float:
        ts = self.two_sided if two_sided is None else two_sided
        h = self.k_ild / self.t_ild
        if ts:
            h += self.k_bond / self.t_bond
        return h

    def Gamma(self, two_sided: bool | None = None) -> float:
        """Thermal characteristic length [m]."""
        return math.sqrt(self.k_cu * self.t_cu / self.h_eff(two_sided))

    def T_m(self, j, two_sided: bool | None = None):
        """Joule temperature-rise amplitude [K] for current density j [A/m^2]."""
        import numpy as np
        j = np.asarray(j, dtype=float)
        return j ** 2 * self.rho_ref * self.Gamma(two_sided) ** 2 / self.k_cu


# Alternative parameter sets found in the literature.  NOTE: this table is documentation only - no code reads it
# (the E6 sweeps in experiments/e6_sweeps.py carry their own explicit Ea / sigma_crit lists).
ALTERNATIVE_SETS: Dict[str, Dict] = {
    "tan_group_line": dict(Ea_eV=0.86, D0=7.5e-8, Z_eff=10.0),
    "hierpinn_code": dict(Ea_eV=0.84, D0=7.5e-8, Z_eff=10.0),
    "hou_najm_todaes23": dict(Ea_eV=1.1, D0=5.2e-5, Z_eff=10.0),
    "pathak2011_Z4": dict(Ea_eV=0.86, D0=7.5e-8, Z_eff=4.0),
}


def default_em() -> EMParams:
    return EMParams()


def default_thermal() -> ThermalParams:
    return ThermalParams()


def verify_constants(em: EMParams | None = None, th: ThermalParams | None = None, verbose: bool = False) -> Dict:
    """Self-check the parameter set. Raises AssertionError on any inconsistency.

    Checks
    ------
    1. Omega agrees with two independent derivations to < 0.5 %.
    2. e and k_B are the CODATA-2018 exact values.
    3. kappa(353 K) is within the physically sensible band 1e-19 .. 1e-15 m^2/s.
    4. Blech critical product at 353 K lies in the literature band 500-5000 A/cm.
    5. sigma_T(353 K) < sigma_crit (otherwise every line nucleates at t=0).
    6. Gamma in the 1-100 um band and T_m(1e10 A/m^2) < 60 K.
    """
    em = em or default_em()
    th = th or default_thermal()
    rep = {}
    assert abs(em.Omega - OMEGA_FROM_LATTICE) / OMEGA_FROM_LATTICE < 5e-3, "Omega inconsistent with FCC lattice"
    assert abs(em.Omega - OMEGA_FROM_MOLAR) / OMEGA_FROM_MOLAR < 5e-3, "Omega inconsistent with molar volume"
    assert abs(E_CHARGE - 1.602176634e-19) < 1e-30 and abs(K_B - 1.380649e-23) < 1e-35
    k353 = float(em.kappa(353.0))
    rep["kappa_353K_m2_per_s"] = k353
    assert 1e-19 < k353 < 1e-15, f"kappa(353K)={k353:.3e} outside sensible band"
    jl = em.blech_critical_product(353.0)
    rep["blech_jL_crit_353K_A_per_cm"] = jl / 100.0
    assert 500.0 < jl / 100.0 < 5000.0, f"(jL)_c={jl/100:.0f} A/cm outside literature band"
    sT = float(em.sigma_T(353.0))
    rep["sigma_T_353K_MPa"] = sT / 1e6
    assert sT < em.sigma_crit, "sigma_T must be below sigma_crit"
    gam = th.Gamma()
    rep["Gamma_um"] = gam * 1e6
    assert 1e-6 < gam < 1e-4, f"Gamma={gam:.2e} m outside 1-100 um"
    tm = float(th.T_m(1e10))
    rep["T_m_at_1e10_K"] = tm
    assert tm < 60.0, "Joule rise at 1e6 A/cm^2 unreasonably large"
    rep["G_at_1e10_Pa_per_m"] = em.G_of_j(1e10)
    rep["dsigmaT_dT_MPa_per_K"] = em.dsigmaT_dT / 1e6
    if verbose:
        for k, v in rep.items():
            print(f"  {k:32s} {v:.4g}")
    return rep


if __name__ == "__main__":
    print("StackEM constants self-check")
    verify_constants(verbose=True)
    print("OK")
