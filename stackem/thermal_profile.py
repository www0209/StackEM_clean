"""
stackem.thermal_profile
=========================

Per-segment temperature profile, stress-diffusivity profile and thermomigration
source, plus the *similarity-scaled* profile descriptors that the Segment
Kernel Network (SKN) takes as input.

Segment temperature model
-------------------------
A segment of length L runs between two vias (x = 0 and x = L).  The
substrate temperature under the segment is taken linear between the two via
temperatures T_L, T_R (from the 3-D thermal field).  On top of that, wire
self-heating adds the fin-equation bump

    T(x) = T_sub(x) + T_m [ 1 - cosh((x - L/2)/Gamma) / cosh(L/(2 Gamma)) ],

which vanishes at the vias (vias are assumed to be at substrate temperature,
the standard assumption of Chen & Tan TCAD'16 / EMSpice) and saturates to T_m
in the middle of long segments (L >> Gamma).

Dimensionless profile descriptors (inputs of the SKN)
----------------------------------------------------
    r     = ln kappa(T_R) / kappa(T_L)                 end-to-end asymmetry
    rho_J = ln kappa(T_bar + T_m) / kappa(T_bar)       Joule bump strength
    lam   = L / Gamma                                  bump shape
    theta = k_B T_bar / Ea                             absolute-temperature scale
                                                        (needed only by the TM source)
with T_bar = (T_L + T_R)/2 and kappa_bar := kappa(T_bar) the time scale used in
tau = kappa_bar t / L^2.  For fixed Ea the map (T_bar, dT, T_m, lam) <->
(theta, r, rho_J, lam) is one-to-one, so the descriptors lose no information.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import EMParams, K_B_EV


@dataclass
class SegmentProfile:
    """Physical description of one segment's thermal state."""
    L: float          # m
    T_L: float        # K  via temperature at x = 0
    T_R: float        # K  via temperature at x = L
    T_m: float = 0.0  # K  Joule bump amplitude
    Gamma: float = 3.0e-5  # m thermal characteristic length

    @property
    def T_bar(self) -> float:
        return 0.5 * (self.T_L + self.T_R)

    @property
    def lam(self) -> float:
        return self.L / self.Gamma

    def T_of_xi(self, xi):
        """Temperature at dimensionless position xi = x/L (vectorised)."""
        xi = np.asarray(xi, dtype=float)
        T_sub = self.T_L + (self.T_R - self.T_L) * xi
        if self.T_m == 0.0 or self.lam == 0.0:
            return T_sub
        lam = self.lam
        # cosh((xi-1/2) lam) / cosh(lam/2) written with non-positive exponents only (no overflow for lam > 700,
        # i.e. millimetre-long segments in a bonded stack)
        ratio = (np.exp((xi - 1.0) * lam) + np.exp(-xi * lam)) / (1.0 + np.exp(-lam))
        return T_sub + self.T_m * (1.0 - ratio)

    def dTdx_of_xi(self, xi):
        """dT/dx [K/m] at xi."""
        xi = np.asarray(xi, dtype=float)
        d = (self.T_R - self.T_L) / self.L * np.ones_like(xi)
        if self.T_m != 0.0 and self.lam != 0.0:
            lam = self.lam
            ratio = (np.exp((xi - 1.0) * lam) - np.exp(-xi * lam)) / (1.0 + np.exp(-lam))   # sinh((xi-1/2)lam)/cosh(lam/2)
            d = d - self.T_m * (lam / self.L) * ratio
        return d

    def descriptors(self, em: EMParams):
        """Return (theta, r, rho_J, lam, kappa_bar)."""
        Tb = self.T_bar
        kb = float(em.kappa(Tb))
        r = float(np.log(em.kappa(self.T_R) / em.kappa(self.T_L)))
        rhoJ = float(np.log(em.kappa(Tb + self.T_m) / kb))
        theta = K_B_EV * Tb / em.Ea_eV
        return theta, r, rhoJ, self.lam, kb

    def kappa_hat(self, xi, em: EMParams):
        """kappa(x)/kappa_bar at xi."""
        return em.kappa(self.T_of_xi(xi)) / em.kappa(self.T_bar)

    def M_hat(self, xi, em: EMParams):
        """Dimensionless TM source  M_hat = L * T'(x) / T(x)  (so that M = Q*/(Omega L) * M_hat)."""
        return self.L * self.dTdx_of_xi(xi) / self.T_of_xi(xi)


def descriptors_to_profile(theta: float, r: float, rhoJ: float, lam: float, L: float,
                           em: EMParams, Gamma: float | None = None) -> SegmentProfile:
    """Invert the descriptor map (for a fixed Ea).  Used by tests and by the
    dataset generator to reconstruct the physical profile from sampled descriptors.

    T_bar = theta * Ea / k_B.  r and rho_J are inverted with a few Newton steps on
    ln kappa(T) = ln D0 - Ea/(kT) + ln(B Omega/(k T)), which is monotone in T.
    """
    Tb = theta * em.Ea_eV / K_B_EV

    def lnk(T):
        return np.log(em.kappa(T))

    def solve_T(target, T0):
        T = float(T0)
        for _ in range(30):
            f = lnk(T) - target
            dfdT = em.Ea_eV / (K_B_EV * T * T) - 1.0 / T
            T -= f / dfdT
        return T

    base = lnk(Tb)
    # end temperatures: ln k(T_R) - ln k(T_L) = r with (T_L+T_R)/2 = Tb -> solve for dT
    dT = 0.0
    for _ in range(30):
        f = lnk(Tb + 0.5 * dT) - lnk(Tb - 0.5 * dT) - r
        d = 0.5 * (em.Ea_eV / (K_B_EV * (Tb + 0.5 * dT) ** 2) - 1.0 / (Tb + 0.5 * dT)) \
            + 0.5 * (em.Ea_eV / (K_B_EV * (Tb - 0.5 * dT) ** 2) - 1.0 / (Tb - 0.5 * dT))
        dT -= f / d
    T_m = solve_T(base + rhoJ, Tb) - Tb
    Gamma = L / lam if Gamma is None else Gamma
    return SegmentProfile(L=L, T_L=Tb - 0.5 * dT, T_R=Tb + 0.5 * dT, T_m=T_m, Gamma=Gamma)
