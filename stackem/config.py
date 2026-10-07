"""
stackem.config
================

JSON configuration -> dataclasses (StackSpec, PGParams, EMParams, ThermalParams,
time grid).  One config file describes one stack case; sweeps generate
variants programmatically (experiments/e6_sweeps.py).

A config file only needs the keys that differ from ``DEFAULT_CONFIG`` (the base3 case with the
final engine).  ``--config a.json,b.json`` merges several files left to right (later files win).

Where results go and where shared inputs come from
    <root>/<name>/...            every output of the case (``paths.root_out`` and ``name``)
    <root>/<base_case>/...       the sign-off sizing, the kernel dataset and the SKN weights that the
                                 case re-uses.  ``base_case`` defaults to ``name``; it is set when a
                                 second run of the SAME stack (another engine, another network) must
                                 share the sizing of the first one, e.g. configs/base3_teacher.json.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List
import numpy as np

from .constants import EMParams, ThermalParams, SEC_PER_YEAR
from .hotspot_stack import DieSpec, StackSpec, make_power_map
from .power_grid import PGParams

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "base3",
    "paths": {
        "hotspot_dir": "~/stackem_work/HotSpot",
        "root_out": "~/stackem_work/outputs",
    },
    "stack": {
        "dies": [
            {"name": "D1", "size": 4e-3, "thickness": 50e-6, "nblk": 4, "P": 2.0, "hot": [[[3, 3], 4.0]],
             "feed_pitch": 400e-6, "feed_kind": "tsv", "role": "bottom / package side, low-power IO-memory die"},
            {"name": "D2", "size": 4e-3, "thickness": 50e-6, "nblk": 4, "P": 4.0, "hot": None,
             "feed_pitch": 400e-6, "feed_kind": "tsv", "role": "middle die, SRAM-like"},
            {"name": "D3", "size": 4e-3, "thickness": 50e-6, "nblk": 4, "P": 20.0, "hot": [[[1, 1], 5.0]],
             "feed_pitch": 200e-6, "feed_kind": "pad", "role": "top / heat-sink side, logic die with hotspot"},
        ],
        "bond_thick": 5e-6, "bond_k": 40.0, "tim_thick": 20e-6,
        "ambient": 318.15, "r_convec": 0.5, "grid": 64, "detailed_3d": False,
    },
    "pg": {"pitch": 100e-6, "W": 2e-6, "H": 1e-6, "rho": 3e-8, "Vdd": 0.8, "current_fraction": 0.4, "R_feed": 0.0},
    "em": {},
    "thermal": {"two_sided": True},
    "times": {"log10_start": 4.0, "log10_end": 9.5, "n": 32},
    # ^ originally "n": 64; the final experiments used 32 output times (see docs).  The teacher-engine
    #   comparison (configs/base3_teacher.json) keeps 64.
    "horizons_years": [1.0, 10.0, 100.0],
    "closure": {"jump_at": "mid", "with_tm": True, "mode": "tabulated", "n_tab": 192},   # mode: direct | tabulated
    # ^ originally "mode": "direct"; the final experiments used the tabulated look-up, 192 points (see docs)
    "skn": {"weights": "skn_distill/skn_student.npz", "teacher": "skn/skn_best.npz"},    # relative to <root>/<base_case>/
    # ^ originally "weights": "skn/skn_best.npz" (the 6x256 teacher); the final experiments used the 4x128 student (see docs)
    "signoff": {"horizon_years": 10.0, "margin": 2.0, "T_uniform": 378.15, "W_min": 0.5e-6, "W_max": 40e-6},
    "truth": {"n_cells": 60},
}


def load_config(path: str | None) -> Dict[str, Any]:
    """Defaults, updated by the JSON file(s) in ``path`` (a single path or a comma-separated list, merged left to right)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for p in (path.split(",") if path else []):
        if p.strip():
            with open(os.path.expanduser(p.strip())) as f:
                _deep_update(cfg, json.load(f))
    return cfg


def base_case(cfg: Dict[str, Any]) -> str:
    """Name of the case whose sizing / kernel data / weights this run re-uses (defaults to the case itself)."""
    return cfg.get("base_case") or cfg["name"]


def base_dir(cfg: Dict[str, Any], *sub) -> str:
    """<root>/<base_case>/<sub...> (created if missing)."""
    return out_dir(dict(cfg, name=base_case(cfg)), *sub)


def _deep_update(base: Dict, upd: Dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def save_config(cfg: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(cfg, open(path, "w"), indent=2)


def stack_from_config(cfg: Dict[str, Any]) -> StackSpec:
    s = cfg["stack"]
    dies = []
    for d in s["dies"]:
        hot = [((int(h[0][0]), int(h[0][1])), float(h[1])) for h in d["hot"]] if d.get("hot") else None
        pm = make_power_map(float(d["P"]), int(d["nblk"]), hot)
        dies.append(DieSpec(name=d["name"], size=float(d["size"]), thickness=float(d["thickness"]), nblk=int(d["nblk"]),
                            power=pm, feed_pitch=float(d["feed_pitch"]), feed_kind=d.get("feed_kind", "tsv"),
                            tsv_pitch=float(d.get("tsv_pitch", 40e-6)), tsv_diameter=float(d.get("tsv_diameter", 5e-6)),
                            rail_W=(float(d["W"]) if d.get("W") else None)))
    return StackSpec(dies=dies, bond_thick=float(s["bond_thick"]), bond_k=float(s["bond_k"]), tim_thick=float(s["tim_thick"]),
                     ambient=float(s["ambient"]), r_convec=float(s["r_convec"]), grid_rows=int(s["grid"]), grid_cols=int(s["grid"]),
                     detailed_3d=bool(s.get("detailed_3d", False)))


def pg_from_config(cfg) -> PGParams:
    return PGParams(**{k: float(v) for k, v in cfg["pg"].items()})


def em_from_config(cfg) -> EMParams:
    return EMParams(**cfg.get("em", {}))


def thermal_from_config(cfg) -> ThermalParams:
    return ThermalParams(**cfg.get("thermal", {}))


def times_from_config(cfg) -> np.ndarray:
    t = cfg["times"]
    return np.logspace(float(t["log10_start"]), float(t["log10_end"]), int(t["n"]))


def out_dir(cfg, *sub) -> str:
    p = os.path.expanduser(cfg["paths"]["root_out"])
    p = os.path.join(p, cfg["name"], *sub)
    os.makedirs(p, exist_ok=True)
    return p


def hotspot_dir(cfg) -> str:
    return os.path.expanduser(cfg["paths"]["hotspot_dir"])
