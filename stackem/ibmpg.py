"""
stackem.ibmpg
===============

Reader / DC solver for the IBM Power Grid Benchmarks (Nassif, SLIP 2008;
ibmpg1-ibmpg6, SPICE-like text files) and extraction of straight rails for
the EM flow (benchmark tier B3: "real, irregular industrial grid").

File format (as distributed)
----------------------------
    R<name> <node1> <node2> <value>      resistors (segments and vias)
    V<name> <node1> <node2> <value>      voltage sources (pads / supply)
    I<name> <node1> <node2> <value>      current sources (cell currents)
    .end
Node names encode the metal layer and the coordinates:  n<layer>_<x>_<y>
(some files add suffixes; node "0" is ground).  Vias join two layers at the
same (x, y).

What we extract
---------------
* the DC solution (MNA) of the whole benchmark: node voltages, branch currents;
* for every metal layer, the maximal straight runs of collinear nodes joined by
  segment resistors -> a straight multi-segment rail; segment length L from
  the coordinates, cross-section A from  A = rho L / R  (the benchmark gives R,
  not W/H), current density j = I / A along +x / +y;
* temperatures by projecting a StackEM die temperature field (HotSpot) onto
  the benchmark's bounding box (the benchmark carries no thermal information).
Only rails with >= 2 segments and a physically sensible A (10 % - 1000 % of
the median) are kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import EMParams, ThermalParams
from .hotspot_stack import grid_T_at
from .power_grid import Rail

_NODE = re.compile(r"^n(\d+)_(\d+)_(\d+)")


@dataclass
class PGNetlist:
    nodes: Dict[str, int]
    R: List[Tuple[int, int, float, str]]      # (a, b, value, name)
    V: List[Tuple[int, int, float]]
    I: List[Tuple[int, int, float]]
    coords: Dict[int, Tuple[int, int, int]]   # node id -> (layer, x, y)


def parse_spice(path: str) -> PGNetlist:
    nodes: Dict[str, int] = {"0": 0}
    coords: Dict[int, Tuple[int, int, int]] = {}
    R, V, I = [], [], []

    def nid(name):
        if name not in nodes:
            nodes[name] = len(nodes)
            m = _NODE.match(name)
            if m:
                coords[nodes[name]] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return nodes[name]

    import sys, time
    t0 = time.time(); n_lines = 0
    with open(path) as fh:
        lines = fh.readlines()
    print(f"  [parse] {len(lines)} lines read ({time.time()-t0:.1f}s)", flush=True)
    for line in lines:
        n_lines += 1
        if n_lines % 500000 == 0:
            print(f"  [parse] {n_lines} lines, {len(nodes)} nodes, {len(R)} R, {len(V)} V, {len(I)} I ({time.time()-t0:.1f}s)", flush=True)
        t = line.split()
        if not t or t[0].startswith(("*", ".")):
            continue
        key = t[0][0].upper()
        if key == "R" and len(t) >= 4:
            R.append((nid(t[1]), nid(t[2]), float(t[3]), t[0]))
        elif key == "V" and len(t) >= 4:
            V.append((nid(t[1]), nid(t[2]), float(t[3])))
        elif key == "I" and len(t) >= 4:
            I.append((nid(t[1]), nid(t[2]), float(t[3])))
    return PGNetlist(nodes, R, V, I, coords)


def solve_dc(net: PGNetlist) -> Tuple[np.ndarray, np.ndarray]:
    """DC solve; returns (node voltages (N,), resistor currents a->b (nR,)).

    Voltage sources to ground (the pad sources of the IBM benchmarks: ibmpg3 has ~10^5 of them) are treated as known
    node voltages and eliminated, so the system that is factorised is the symmetric positive-definite conductance
    matrix of the free nodes only.  The MNA saddle-point form with one extra row per source is kept for sources that
    are not tied to ground (it made SuperLU crawl for hours on ibmpg3)."""
    import time
    t0 = time.time(); tick = lambda msg: print(f"  [dc] {msg} ({time.time()-t0:.1f}s)", flush=True)
    N = len(net.nodes)
    Ra = np.array([a for a, b, r, _ in net.R], int); Rb = np.array([b for a, b, r, _ in net.R], int)
    Rr = np.maximum(np.array([r for a, b, r, _ in net.R], float), 1e-9); g = 1.0 / Rr
    rows = np.concatenate([Ra, Rb, Ra, Rb]); cols = np.concatenate([Ra, Rb, Rb, Ra]); vals = np.concatenate([g, g, -g, -g])
    G = sp.csc_matrix((vals, (rows, cols)), shape=(N, N)); tick(f"conductance matrix {N} x {N}, {G.nnz} nnz")
    rhs = np.zeros(N)
    for a, b, cur in net.I:                       # current flows from a to b through the source
        rhs[a] -= cur; rhs[b] += cur
    known = np.zeros(N, bool); Vk = np.zeros(N); known[0] = True          # node 0 = ground
    floating = []
    for a, b, v in net.V:
        if b == 0: known[a] = True; Vk[a] = v
        elif a == 0: known[b] = True; Vk[b] = -v
        else: floating.append((a, b, v))
    tick(f"{len(net.V)} voltage sources: {int(known.sum()) - 1} to ground (eliminated), {len(floating)} floating; {len(net.I)} current sources")
    if floating:                                                           # general MNA (rare)
        nV = len(floating)
        rows2 = [rows, np.zeros(1, int)]; cols2 = [cols, np.zeros(1, int)]; vals2 = [vals, np.ones(1)]
        keep = rows != 0; rows2[0], cols2[0], vals2[0] = rows[keep], cols[keep], vals[keep]
        for k, (a, b, v) in enumerate(floating):
            rows2.append(np.array([a, N + k, b, N + k])); cols2.append(np.array([N + k, a, N + k, b])); vals2.append(np.array([1.0, 1.0, -1.0, -1.0]))
        rhs2 = np.concatenate([rhs, [v for _, _, v in floating]]); rhs2[0] = 0.0
        for a in np.where(known)[0]:                                       # grounded sources as constraints too
            if a == 0: continue
            rows2.append(np.array([a])); cols2.append(np.array([a])); vals2.append(np.array([1e12])); rhs2[a] += 1e12 * Vk[a]
        M = sp.csc_matrix((np.concatenate(vals2), (np.concatenate(rows2), np.concatenate(cols2))), shape=(N + nV, N + nV))
        Vn = spla.splu(M).solve(rhs2)[:N]
    else:
        free = np.where(~known)[0]; kn = np.where(known)[0]
        Gff = G[free][:, free].tocsc(); Gfk = G[free][:, kn]; tick(f"reduced system {len(free)} unknowns, {Gff.nnz} nnz")
        b = rhs[free] - Gfk @ Vk[kn]
        lu = spla.splu(Gff); tick(f"LU factorised, {lu.L.nnz + lu.U.nnz} nnz in the factors")
        Vn = Vk.copy(); Vn[free] = lu.solve(b); tick("solved")
    Ir = (Vn[Ra] - Vn[Rb]) / Rr
    return Vn, Ir


def extract_rails(net: PGNetlist, Vn: np.ndarray, Ir: np.ndarray, em: EMParams, th: ThermalParams,
                  Tgrid: np.ndarray | None = None, T_uniform: float = 353.0, unit: float = 1e-9,
                  H: float = 1e-6, two_sided: bool = True, min_segments: int = 3, W_assumed: float | None = None) -> List[Rail]:
    """Straight rails from collinear segment resistors of every net (net index = metal layer + polarity in ibmpg).

    unit      : coordinate unit of the benchmark (ibmpg1-6: 1 unit = 1 um -> 21 mm die for ibmpg1)
    H         : metal thickness
    W_assumed : strap width (m).  The ibmpg resistors are *coarsened* buses (a 1.9 mm segment has 1.3 Ohm, i.e.
                A = rho L / R would be 4e4 um^2), so the cross-section for the EM current density is taken from an
                assumed physical strap (W_assumed x H) and only the DC current comes from the benchmark; with
                W_assumed = None the cross-section is derived from R (only sensible for non-coarsened netlists)."""
    # segment resistors = both nodes on the same layer and collinear
    seg = {}
    for idx, (a, b, r, name) in enumerate(net.R):
        ca, cb = net.coords.get(a), net.coords.get(b)
        if ca is None or cb is None or ca[0] != cb[0]:
            continue
        if ca[1] == cb[1] and ca[2] != cb[2]:
            key = ("V", ca[0], ca[1]); lo, hi = sorted([(ca[2], a), (cb[2], b)])
        elif ca[2] == cb[2] and ca[1] != cb[1]:
            key = ("H", ca[0], ca[2]); lo, hi = sorted([(ca[1], a), (cb[1], b)])
        else:
            continue
        seg.setdefault(key, []).append((lo[0], hi[0], lo[1], hi[1], r, idx))
    rails = []
    rho = em.rho
    bbox = np.array([[c[1], c[2]] for c in net.coords.values()], float) * unit
    x_min, y_min = bbox.min(0); x_max, y_max = bbox.max(0)
    size = max(x_max - x_min, y_max - y_min) + 1e-12
    for (kind, layer, line_coord), items in seg.items():
        items.sort()
        # split into maximal chains of consecutive segments (shared nodes)
        chain = []
        for it in items:
            if chain and chain[-1][1] != it[0]:
                _emit(net, chain, kind, layer, line_coord, rails, Ir, rho, unit, H, em, th, Tgrid, T_uniform, x_min, y_min, size, two_sided, min_segments, W_assumed)
                chain = []
            chain.append(it)
        if chain:
            _emit(net, chain, kind, layer, line_coord, rails, Ir, rho, unit, H, em, th, Tgrid, T_uniform, x_min, y_min, size, two_sided, min_segments, W_assumed)
    return rails


def _merge_short(chain, unit, min_len_m):
    """Merge segments shorter than min_len_m into their longer neighbour (via stubs / duplicated nodes in the netlists
    produce 1-2 um 'segments' next to 50-100 um ones; below the trained lambda range and physically not a rail
    segment).  The merged segment keeps the current of the longer part."""
    out = []
    for it in chain:
        if out and (it[1] - it[0]) * unit < min_len_m:
            lo, hi, a, b, r, idx = out[-1]; out[-1] = (lo, it[1], a, it[3], r + it[4], idx)          # absorb into the previous
        elif out and (out[-1][1] - out[-1][0]) * unit < min_len_m:
            lo, hi, a, b, r, idx = out[-1]; out[-1] = (lo, it[1], a, it[3], r + it[4], it[5])         # previous was short: absorb it into this one
        else:
            out.append(it)
    return out


def _emit(net, chain, kind, layer, line_coord, rails, Ir, rho, unit, H, em, th, Tgrid, T_uniform, x_min, y_min, size, two_sided, min_segments, W_assumed=None, min_seg_len=2e-6):
    chain = _merge_short(chain, unit, min_seg_len)
    if len(chain) < min_segments:
        return
    L = np.array([(c[1] - c[0]) * unit for c in chain])
    R = np.array([c[4] for c in chain])
    if W_assumed is None:
        A = rho * L / np.maximum(R, 1e-9)
        Amed = np.median(A)
        if not (np.all(A > 0.1 * Amed) and np.all(A < 10 * Amed)):
            return
        A_rail = float(np.median(A)); W = A_rail / H
    else:
        W = float(W_assumed); A_rail = W * H
    # Ir is the current a->b as listed in the file; the rail runs from the low to the high coordinate
    j = np.array([Ir[c[5]] * (1.0 if net.R[c[5]][0] == c[2] else -1.0) for c in chain]) / A_rail
    coords = np.array([c[0] for c in chain] + [chain[-1][1]], float) * unit
    if kind == "H":
        x0, y0 = coords[0], line_coord * unit; xs = coords; ys = np.full_like(coords, y0)
    else:
        x0, y0 = line_coord * unit, coords[0]; xs = np.full_like(coords, x0); ys = coords
    if Tgrid is not None:
        Tn = grid_T_at(Tgrid, size, xs - x_min, ys - y_min)
    else:
        Tn = np.full(len(coords), T_uniform)
    Gam = th.Gamma(two_sided)
    Tm = th.T_m(np.abs(j), two_sided)
    lam = L / Gam
    Tbar = 0.5 * (Tn[:-1] + Tn[1:]) + Tm * (1 - np.tanh(0.5 * lam) / (0.5 * lam))
    rails.append(Rail(f"M{layer}", kind, int(line_coord), float(x0), float(y0), L, W, H, j, Tn[:-1], Tn[1:], Tm, Gam,
                      np.asarray(em.sigma_T(Tbar)), np.zeros(len(coords), bool)))


def write_synthetic_benchmark(path: str, n: int = 12, pitch: int = 100000, layers: int = 2, R_seg: float = 1.5,
                              R_via: float = 0.05, Vdd: float = 1.0, I_cell: float = 5e-4, seed: int = 0):
    """Small ibmpg-format grid for tests (two metal layers, pads at the corners)."""
    rng = np.random.default_rng(seed)
    lines = ["* synthetic ibmpg-format grid (stackem tests)"]
    k = 0
    for l in range(1, layers + 1):
        for i in range(n):
            for jj in range(n):
                if l % 2 == 1 and jj + 1 < n:      # odd layers horizontal (x varies)
                    lines.append(f"R{k} n{l}_{i*pitch}_{jj*pitch} n{l}_{i*pitch}_{(jj+1)*pitch} {R_seg}"); k += 1
                if l % 2 == 0 and i + 1 < n:       # even layers vertical (y varies)
                    lines.append(f"R{k} n{l}_{i*pitch}_{jj*pitch} n{l}_{(i+1)*pitch}_{jj*pitch} {R_seg}"); k += 1
                if l < layers:
                    lines.append(f"Rvia{k} n{l}_{i*pitch}_{jj*pitch} n{l+1}_{i*pitch}_{jj*pitch} {R_via}"); k += 1
    for (i, jj) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1), (n // 2, n // 2)]:
        lines.append(f"V{i}_{jj} n{layers}_{i*pitch}_{jj*pitch} 0 {Vdd}")
    for i in range(n):
        for jj in range(n):
            lines.append(f"I{i}_{jj} n1_{i*pitch}_{jj*pitch} 0 {I_cell * rng.uniform(0.5, 1.5):.6e}")
    lines.append(".end")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
