"""
stackem.hotspot_stack
=======================

Thermal Field Embedding (TFE), part 1: the 3-D die-level temperature field of a
hybrid-bonded stack computed with HotSpot 7 (grid model, .lcf layer stack).

What this module does
---------------------
* Describes an N-die stack (bottom die = package side = layer 0, top die next
  to the TIM / heat spreader) with per-die block power maps.
* Writes the HotSpot input files (.lcf, .flp, .ptrace, .config), runs the
  solver, and parses the per-layer grid temperature.
* Runs every die *standalone* (die + TIM + spreader + sink) for the
  "2-D analysis" baseline of experiment E3a.
* Samples the temperature at arbitrary (x, y) positions of a die (bilinear
  interpolation) - this is what the power-grid module uses to attach via
  temperatures to every rail segment.
* Exploits the linearity of the HotSpot thermal network to build the exact
  sensitivity  dT(x,y,die)/dP_block  by unit-power runs (used by the
  differentiable stack assembler, experiment E5).
* Loads HotSpot's own 3-D example (examples/example4, the Boston University
  ev6 six-layer stack) as the "real floorplan" benchmark B2.

Coordinate conventions
----------------------
HotSpot floorplans use (x to the right, y upward) with the origin at the
bottom-left corner.  The .grid.steady file lists rows top-to-bottom, i.e.
row 0 is the strip at y = size (top edge).  ``grid_T_at`` handles this.
"""
from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# HotSpot material defaults (README / example configs)
SI_CP = 1.75e6          # J/(m^3 K)
SI_RES = 0.01           # (m K)/W  -> k = 100 W/(m K)
TIM_CP = 4.0e6
TIM_RES = 0.25          # k = 4 W/(m K)
BOND_CP = 1.6e6         # hybrid-bond layer (SiO2 + Cu pads)


@dataclass
class DieSpec:
    name: str
    size: float = 4e-3                       # m (square die)
    thickness: float = 50e-6                 # m (thinned)
    nblk: int = 4                            # nblk x nblk power blocks
    power: np.ndarray | None = None          # (nblk, nblk) W, index [i (x), j (y)]
    feed_pitch: float = 400e-6               # m  vertical feed (TSV cluster / pad) pitch for the PG
    feed_kind: str = "tsv"                   # "tsv" | "pad"
    tsv_pitch: float = 40e-6                 # m  TSV pitch inside a feed cluster (visualisation / BIS)
    tsv_diameter: float = 5e-6               # m
    tsv_height: float | None = None          # defaults to die thickness
    rail_W: float | None = None              # m  strap width of this die's analysed rails (None -> PGParams.W);
                                             #    set by the sign-off sizing rule (stackem.signoff)

    def total_power(self) -> float:
        return float(np.sum(self.power))


@dataclass
class StackSpec:
    dies: List[DieSpec]                      # bottom -> top
    bond_thick: float = 5e-6
    bond_k: float = 40.0                     # effective vertical conductivity of the hybrid-bond layer
    tim_thick: float = 20e-6
    ambient: float = 318.15
    r_convec: float = 0.5                    # K/W  (compact air-cooled package)
    spreader_size: float = 12e-3
    spreader_thick: float = 1e-3
    sink_size: float = 30e-3
    sink_thick: float = 6.9e-3
    grid_rows: int = 64
    grid_cols: int = 64
    detailed_3d: bool = False

    @property
    def size(self) -> float:
        return self.dies[0].size

    def die_layer_index(self) -> Dict[str, int]:
        """lcf layer index of each die (die k -> layer 2k)."""
        return {d.name: 2 * k for k, d in enumerate(self.dies)}


def make_power_map(P_total: float, nblk: int, hot: Sequence[Tuple[Tuple[int, int], float]] | None = None,
                   seed: int | None = None, jitter: float = 0.0) -> np.ndarray:
    """Block power map (W): uniform P_total/nblk^2, optionally with hot blocks
    ((i,j), factor) and multiplicative log-normal jitter; renormalised to P_total."""
    pm = np.full((nblk, nblk), P_total / nblk ** 2)
    if hot:
        for (i, j), f in hot:
            pm[i, j] *= f
    if jitter > 0:
        rng = np.random.default_rng(seed)
        pm *= np.exp(rng.normal(0.0, jitter, pm.shape))
    return pm * (P_total / pm.sum())


# ----------------------------------------------------------------------------
# input writers
# ----------------------------------------------------------------------------
def write_block_flp(path: str, die: DieSpec):
    bw = die.size / die.nblk
    with open(path, "w") as f:
        f.write("# StackEM block floorplan: name width height left-x bottom-y\n")
        for i, j in itertools.product(range(die.nblk), range(die.nblk)):
            f.write(f"{die.name}_{i}_{j}\t{bw:.7e}\t{bw:.7e}\t{i*bw:.7e}\t{j*bw:.7e}\n")


def write_uniform_flp(path: str, size: float, name: str):
    with open(path, "w") as f:
        f.write(f"{name}\t{size:.7e}\t{size:.7e}\t0\t0\n")


def write_lcf(path: str, layers: List[dict]):
    with open(path, "w") as f:
        f.write("# StackEM layer configuration (layer 0 = farthest from heat sink)\n")
        for k, L in enumerate(layers):
            f.write(f"\n# {L.get('comment','')}\n{k}\n{L['lateral']}\n{L['power']}\n{L['cp']}\n{L['res']}\n{L['thick']}\n{L['flp']}\n")


def write_ptrace(path: str, names: Sequence[str], values: Sequence[float]):
    with open(path, "w") as f:
        f.write("\t".join(names) + "\n")
        f.write("\t".join(f"{v:.7f}" for v in values) + "\n")


def write_config(path: str, template: str, overrides: Dict[str, object]):
    txt = open(template).read()
    for k, v in overrides.items():
        txt, n = re.subn(rf"(-{k}\s+)\S+", lambda m: f"{m.group(1)}{v}", txt, count=1)
        if n == 0:
            txt += f"\n\t\t-{k}\t\t{v}\n"
    open(path, "w").write(txt)


def block_names(die: DieSpec) -> List[str]:
    return [f"{die.name}_{i}_{j}" for i, j in itertools.product(range(die.nblk), range(die.nblk))]


def block_values(die: DieSpec, pm: np.ndarray | None = None) -> List[float]:
    pm = die.power if pm is None else pm
    return [float(pm[i, j]) for i, j in itertools.product(range(die.nblk), range(die.nblk))]


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
class HotSpotRunner:
    def __init__(self, hotspot_dir: str):
        self.dir = os.path.abspath(os.path.expanduser(hotspot_dir))
        self.exe = os.path.join(self.dir, "hotspot")
        self.template = os.path.join(self.dir, "examples", "example4", "example.config")
        if not os.path.exists(self.exe):
            raise FileNotFoundError(f"HotSpot executable not found: {self.exe}  (clone uvahotspot/HotSpot and run make)")
        if not os.path.exists(self.template):
            raise FileNotFoundError(f"HotSpot example config not found: {self.template}")

    def run(self, workdir: str, lcf: str, ptrace: str, config: str, tag: str, detailed_3d: bool = False,
            rows: int = 64, cols: int = 64) -> Dict[int, np.ndarray]:
        cmd = [self.exe, "-c", config, "-p", ptrace, "-grid_layer_file", lcf, "-model_type", "grid",
               "-steady_file", f"{tag}.steady", "-grid_steady_file", f"{tag}.grid.steady"]
        if detailed_3d:
            cmd += ["-detailed_3D", "on"]
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"HotSpot failed ({' '.join(cmd)}):\n{r.stdout}\n{r.stderr}")
        return read_grid_steady(os.path.join(workdir, f"{tag}.grid.steady"), rows, cols)


def read_grid_steady(path: str, rows: int = 64, cols: int = 64) -> Dict[int, np.ndarray]:
    """{layer: (rows, cols) K}. Row 0 = top edge of the chip (y = size)."""
    layers: Dict[int, np.ndarray] = {}
    cur = None
    for line in open(path):
        m = re.match(r"Layer (\d+):", line)
        if m:
            cur = int(m.group(1)); layers[cur] = np.zeros(rows * cols); continue
        if cur is not None and line.strip():
            k, v = line.split(); layers[cur][int(k)] = float(v)
    return {k: v.reshape(rows, cols) for k, v in layers.items()}


def read_steady(path: str) -> Dict[str, float]:
    out = {}
    for line in open(path):
        if line.strip():
            k, v = line.split(); out[k] = float(v)
    return out


def grid_T_at(Tgrid: np.ndarray, size: float, x, y):
    """Bilinear interpolation of a HotSpot layer grid at physical (x, y) in [0,size].
    Row 0 corresponds to the top edge (y = size).  Vectorised."""
    rows, cols = Tgrid.shape
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    fx = np.clip(x / size * cols - 0.5, 0, cols - 1)
    fy = np.clip((size - y) / size * rows - 0.5, 0, rows - 1)
    c0 = np.floor(fx).astype(int); r0 = np.floor(fy).astype(int)
    c1 = np.minimum(c0 + 1, cols - 1); r1 = np.minimum(r0 + 1, rows - 1)
    wx = fx - c0; wy = fy - r0
    return ((1 - wx) * (1 - wy) * Tgrid[r0, c0] + wx * (1 - wy) * Tgrid[r0, c1]
            + (1 - wx) * wy * Tgrid[r1, c0] + wx * wy * Tgrid[r1, c1])


# ----------------------------------------------------------------------------
# stack builder
# ----------------------------------------------------------------------------
@dataclass
class ThermalField:
    """Result of a stack run: per-die temperature grids (K) + metadata."""
    spec: StackSpec
    T_die: Dict[str, np.ndarray]                 # die name -> (rows, cols)
    T_alone: Dict[str, np.ndarray] = field(default_factory=dict)
    all_layers: Dict[int, np.ndarray] = field(default_factory=dict)
    steady: Dict[str, float] = field(default_factory=dict)

    def T_uniform(self, name: str) -> float:
        return float(self.T_die[name].mean())

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for d in self.spec.dies:
            Tg = self.T_die[d.name]
            out[d.name] = dict(min=float(Tg.min()), max=float(Tg.max()), mean=float(Tg.mean()))
            if d.name in self.T_alone:
                Ta = self.T_alone[d.name]
                out[d.name].update(alone_min=float(Ta.min()), alone_max=float(Ta.max()), alone_mean=float(Ta.mean()))
        return out


def _config_overrides(spec: StackSpec) -> Dict[str, object]:
    return {
        "ambient": spec.ambient, "r_convec": spec.r_convec,
        "s_sink": spec.sink_size, "t_sink": spec.sink_thick,
        "s_spreader": spec.spreader_size, "t_spreader": spec.spreader_thick,
        "t_interface": spec.tim_thick, "k_interface": 1.0 / TIM_RES,
        "grid_rows": spec.grid_rows, "grid_cols": spec.grid_cols,
        "model_type": "grid",
    }


def build_stack_inputs(spec: StackSpec, workdir: str, runner: HotSpotRunner, tag: str = "stack",
                       power_override: Dict[str, np.ndarray] | None = None) -> Tuple[str, str, str]:
    """Write lcf/flp/ptrace/config for the full stack. Returns (lcf, ptrace, config) paths."""
    os.makedirs(workdir, exist_ok=True)
    size = spec.size
    for d in spec.dies:
        write_block_flp(os.path.join(workdir, f"{d.name}.flp"), d)
    write_uniform_flp(os.path.join(workdir, "bond.flp"), size, "BOND")
    write_uniform_flp(os.path.join(workdir, "tim.flp"), size, "TIM")
    layers = []
    for k, d in enumerate(spec.dies):
        layers.append(dict(lateral="Y", power="Y", cp=f"{SI_CP:.4e}", res=f"{SI_RES}", thick=f"{d.thickness:.4e}",
                           flp=f"{d.name}.flp", comment=f"die {d.name} (Si, thinned)"))
        if k < len(spec.dies) - 1:
            layers.append(dict(lateral="Y", power="N", cp=f"{BOND_CP:.4e}", res=f"{1.0/spec.bond_k:.6f}",
                               thick=f"{spec.bond_thick:.4e}", flp="bond.flp", comment="hybrid-bond layer"))
        else:
            layers.append(dict(lateral="Y", power="N", cp=f"{TIM_CP:.4e}", res=f"{TIM_RES}",
                               thick=f"{spec.tim_thick:.4e}", flp="tim.flp", comment="TIM under spreader"))
    lcf = os.path.join(workdir, f"{tag}.lcf"); write_lcf(lcf, layers)
    names, vals = [], []
    for d in spec.dies:
        pm = None if power_override is None else power_override.get(d.name)
        names += block_names(d); vals += block_values(d, pm)
    ptrace = os.path.join(workdir, f"{tag}.ptrace"); write_ptrace(ptrace, names, vals)
    config = os.path.join(workdir, f"{tag}.config"); write_config(config, runner.template, _config_overrides(spec))
    return lcf, ptrace, config


def run_stack(spec: StackSpec, workdir: str, runner: HotSpotRunner, tag: str = "stack",
              with_alone: bool = True, power_override: Dict[str, np.ndarray] | None = None) -> ThermalField:
    lcf, ptrace, config = build_stack_inputs(spec, workdir, runner, tag, power_override)
    layers = runner.run(workdir, os.path.basename(lcf), os.path.basename(ptrace), os.path.basename(config), tag,
                        spec.detailed_3d, spec.grid_rows, spec.grid_cols)
    li = spec.die_layer_index()
    T_die = {name: layers[k] for name, k in li.items()}
    tf = ThermalField(spec=spec, T_die=T_die, all_layers=layers,
                      steady=read_steady(os.path.join(workdir, f"{tag}.steady")))
    if with_alone:
        for d in spec.dies:
            tf.T_alone[d.name] = run_die_alone(spec, d, workdir, runner, f"{tag}_{d.name}_alone", power_override)
    return tf


def run_die_alone(spec: StackSpec, die: DieSpec, workdir: str, runner: HotSpotRunner, tag: str,
                  power_override: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    """The same die as a standalone 2-D chip (die + TIM + spreader + sink, same package)."""
    layers = [dict(lateral="Y", power="Y", cp=f"{SI_CP:.4e}", res=f"{SI_RES}", thick=f"{die.thickness:.4e}",
                   flp=f"{die.name}.flp", comment=f"die {die.name} alone"),
              dict(lateral="Y", power="N", cp=f"{TIM_CP:.4e}", res=f"{TIM_RES}", thick=f"{spec.tim_thick:.4e}",
                   flp="tim.flp", comment="TIM")]
    lcf = os.path.join(workdir, f"{tag}.lcf"); write_lcf(lcf, layers)
    pm = None if power_override is None else power_override.get(die.name)
    ptrace = os.path.join(workdir, f"{tag}.ptrace"); write_ptrace(ptrace, block_names(die), block_values(die, pm))
    config = os.path.join(workdir, f"{tag}.config"); write_config(config, runner.template, _config_overrides(spec))
    out = runner.run(workdir, os.path.basename(lcf), os.path.basename(ptrace), os.path.basename(config), tag,
                     False, spec.grid_rows, spec.grid_cols)
    return out[0]


def unit_power_responses(spec: StackSpec, workdir: str, runner: HotSpotRunner, tag: str = "unit") -> Dict[str, np.ndarray]:
    """Exact linear thermal sensitivity: for every block b of every die run the stack
    with 1 W in that block only.  Returns {die_name: R} with R of shape
    (n_blocks_total, rows, cols) such that  T_die = T_amb + sum_b P_b R[b]  (verified
    by tests: HotSpot's grid model is linear in power).  Block ordering follows
    ``block_names`` concatenated over dies."""
    li = spec.die_layer_index()
    names = []
    for d in spec.dies:
        names += [(d.name, n) for n in block_names(d)]
    R = {d.name: np.zeros((len(names), spec.grid_rows, spec.grid_cols)) for d in spec.dies}
    for b, (dn, bn) in enumerate(names):
        po = {d.name: np.zeros((d.nblk, d.nblk)) for d in spec.dies}
        i, j = map(int, bn.split("_")[-2:])
        po[dn][i, j] = 1.0
        lcf, ptrace, config = build_stack_inputs(spec, workdir, runner, f"{tag}_{b}", po)
        layers = runner.run(workdir, os.path.basename(lcf), os.path.basename(ptrace), os.path.basename(config),
                            f"{tag}_{b}", spec.detailed_3d, spec.grid_rows, spec.grid_cols)
        for d in spec.dies:
            R[d.name][b] = layers[li[d.name]] - spec.ambient
    return R


# ----------------------------------------------------------------------------
# HotSpot's own 3-D benchmark (ev6, examples/example4)
# ----------------------------------------------------------------------------
@dataclass
class Ev6Stack:
    """The six-layer ev6 example shipped with HotSpot (BU Coskun group):
    layer 0 Si (L2 cache 1), 1 TIM+TSV, 2 Si (L2 cache 2), 3 TIM+TSV, 4 Si (4 ev6 cores), 5 TIM.
    Die size 12.4 mm x 12.4 mm (from the floorplans)."""
    workdir: str
    layers: Dict[int, np.ndarray]
    size: float
    die_layers: Tuple[int, ...] = (0, 2, 4)
    flps: Dict[int, List[Tuple[str, float, float, float, float]]] = field(default_factory=dict)

    def T_die(self, k: int) -> np.ndarray:
        return self.layers[self.die_layers[k]]


def read_flp(path: str) -> List[Tuple[str, float, float, float, float]]:
    out = []
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            out.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def run_ev6_example(runner: HotSpotRunner, workdir: str, r_convec: float | None = None,
                    power_scale: float = 1.0, rows: int = 64, cols: int = 64) -> Ev6Stack:
    """Run HotSpot's example4 (real 3-D floorplan) with optional package / power overrides."""
    src = os.path.join(runner.dir, "examples", "example4")
    os.makedirs(workdir, exist_ok=True)
    for fn in os.listdir(src):
        if fn.endswith((".flp", ".lcf", ".ptrace", ".config")):
            shutil.copy(os.path.join(src, fn), workdir)
    ov = {"grid_rows": rows, "grid_cols": cols, "model_type": "grid"}
    if r_convec is not None:
        ov["r_convec"] = r_convec
    write_config(os.path.join(workdir, "ev6.config"), os.path.join(src, "example.config"), ov)
    if power_scale != 1.0:
        lines = open(os.path.join(workdir, "ev6_3D.ptrace")).read().splitlines()
        vals = [float(v) * power_scale for v in lines[1].split()]
        write_ptrace(os.path.join(workdir, "ev6_3D.ptrace"), lines[0].split(), vals)
    layers = runner.run(workdir, "ev6_3D.lcf", "ev6_3D.ptrace", "ev6.config", "ev6", True, rows, cols)
    flps = {}
    lcf_lines = [l.strip() for l in open(os.path.join(workdir, "ev6_3D.lcf")) if l.strip() and not l.startswith("#")]
    for k in range(0, len(lcf_lines), 7):
        flps[int(lcf_lines[k])] = read_flp(os.path.join(workdir, lcf_lines[k + 6]))
    size = max(x + w for (_, w, h, x, y) in flps[4])
    return Ev6Stack(workdir=workdir, layers=layers, size=size, flps=flps)
