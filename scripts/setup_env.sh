#!/usr/bin/env bash
# One-time environment setup of StackEM on Ubuntu 22.04 / 24.04 (native or WSL2).  About 10 minutes, needs internet.
#
#     bash scripts/setup_env.sh 2>&1 | tee setup_env.log
#
# Creates
#     ~/stackem_work/            work root (override with STACKEM_WORK)
#     ~/stackem_work/venv        Python virtual environment with the pinned package versions
#     ~/stackem_work/HotSpot     HotSpot thermal simulator, built from third_party/HotSpot-master.zip
#     ~/stackem_work/outputs     results of the runs (never written into the repository)
#
# Options (environment variables)
#     STACKEM_WORK=<dir>          work root, default ~/stackem_work
#     STACKEM_PYTHON_BIN=<exe>    interpreter for the venv, default python3.11 if present, else python3
#     STACKEM_TORCH=auto|cuda|cpu|none   which PyTorch build to install (auto: cuda when nvidia-smi exists)
#     STACKEM_NO_APT=1            do not call apt-get (packages already present or no sudo rights)
#     STACKEM_UNPINNED=1          install the newest package versions instead of the pinned ones
#     STACKEM_VENV_ARGS=<args>    extra arguments of "python -m venv", e.g. --system-site-packages on a machine without internet
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${STACKEM_WORK:-$HOME/stackem_work}"
mkdir -p "$WORK/outputs"

echo "== 1/5 system packages =="
# HotSpot is built with its default Makefile options (-DSUPERLU=0): no SuperLU / BLAS / LAPACK is needed.
if [ -z "${STACKEM_NO_APT:-}" ] && command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -y
  $SUDO apt-get install -y build-essential python3 python3-venv python3-dev unzip bzip2 wget tmux
else
  echo "   apt-get skipped; required commands: gcc make unzip bzip2 wget python3 (with the venv module)"
fi
for c in gcc make unzip; do command -v "$c" >/dev/null 2>&1 || { echo "missing command: $c"; exit 1; }; done

echo "== 2/5 python environment =="
# The released results were produced with Python 3.11.16.  Deactivate conda first (conda deactivate), otherwise
# the venv is created on top of the conda interpreter.
PYBIN="${STACKEM_PYTHON_BIN:-}"
if [ -z "$PYBIN" ]; then
  if command -v python3.11 >/dev/null 2>&1; then PYBIN=python3.11
  else
    # no python3.11 on the PATH: use one from a conda environment if there is one, else whatever python3 is
    for c in "$HOME"/miniconda3/envs/*/bin/python3.11 "$HOME"/anaconda3/envs/*/bin/python3.11 "$HOME"/miniforge3/envs/*/bin/python3.11; do
      [ -x "$c" ] && { PYBIN="$c"; break; }
    done
    [ -z "$PYBIN" ] && PYBIN=python3
  fi
fi
echo "   interpreter: $PYBIN ($($PYBIN --version 2>&1))"
case "$($PYBIN --version 2>&1)" in *" 3.11."*) ;; *) echo "   [warn] not Python 3.11: the pinned versions may be unavailable for this interpreter (then use STACKEM_UNPINNED=1); results were only produced with 3.11";; esac
[ -d "$WORK/venv" ] || "$PYBIN" -m venv ${STACKEM_VENV_ARGS:-} "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip || echo "   [warn] pip could not be upgraded (no network?)"
if [ -n "${STACKEM_UNPINNED:-}" ]; then
  python -m pip install numpy scipy matplotlib scikit-learn pandas
else
  python -m pip install -r "$REPO/scripts/requirements.txt"
fi

echo "== 3/5 PyTorch =="
# Released results: torch 2.14.0 built for CUDA 13.0 (torch.__version__ = 2.14.0+cu130) on one RTX 4070 Ti SUPER.
# A GPU is optional: without one the CPU build is installed; training and the E4/E5 torch paths then run on the CPU.
TORCH_MODE="${STACKEM_TORCH:-auto}"
if [ "$TORCH_MODE" = auto ]; then if command -v nvidia-smi >/dev/null 2>&1; then TORCH_MODE=cuda; else TORCH_MODE=cpu; fi; fi
TORCH_SPEC="torch==2.14.0"; [ -n "${STACKEM_UNPINNED:-}" ] && TORCH_SPEC="torch"
case "$TORCH_MODE" in
  cuda) python -m pip install "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cu130 \
          || python -m pip install "$TORCH_SPEC" \
          || echo "   [warn] PyTorch (CUDA) could not be installed" ;;
  cpu)  python -m pip install "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cpu \
          || python -m pip install "$TORCH_SPEC" \
          || echo "   [warn] PyTorch (CPU) could not be installed" ;;
  none) echo "   PyTorch not installed on request (only the numpy paths and the sklearn fallback trainer will work)" ;;
  *)    echo "unknown STACKEM_TORCH=$TORCH_MODE"; exit 1 ;;
esac
python - <<'PY'
import sys, numpy, scipy, matplotlib, sklearn, pandas
print("   python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__,
      "| scikit-learn", sklearn.__version__, "| pandas", pandas.__version__)
try:
    import torch
    print("   torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
except Exception as e:
    print("   [warn] torch not importable:", e)
PY

echo "== 4/5 HotSpot (built from third_party/HotSpot-master.zip) =="
if [ ! -x "$WORK/HotSpot/hotspot" ]; then
  rm -rf "$WORK/HotSpot" "$WORK/HotSpot-master"
  unzip -q "$REPO/third_party/HotSpot-master.zip" -d "$WORK"
  mv "$WORK/HotSpot-master" "$WORK/HotSpot"
  # default Makefile options, the same as the released results: gcc -O3 -DVERBOSE=1 -DMATHACCEL=0 -DSUPERLU=0
  ( cd "$WORK/HotSpot" && make -j"$(nproc)" > make.log 2>&1 ) || { echo "HotSpot build failed, see $WORK/HotSpot/make.log"; exit 1; }
fi
test -x "$WORK/HotSpot/hotspot" && echo "   HotSpot built: $WORK/HotSpot/hotspot"
echo "   sha256 $(sha256sum "$WORK/HotSpot/hotspot" | cut -d' ' -f1)"
echo "   (released results: 4b4954cd8eff2201955be7c16de5e99b1294d9427153bf6371e075924f9438fc; equal only for the same compiler version)"
# quick functional check: example1 (2-D ev6 floorplan, about 1 s)
CHK="$(mktemp -d)"
( cd "$WORK/HotSpot/examples/example1" && ../../hotspot -c example.config -f ev6.flp -p gcc.ptrace -materials_file example.materials \
    -model_type block -steady_file "$CHK/ev6.steady" >/dev/null 2>&1 && test -s "$CHK/ev6.steady" && echo "   HotSpot example1 runs" ) \
  || { echo "   [error] HotSpot quick check failed"; exit 1; }
rm -rf "$CHK"
test -f "$WORK/HotSpot/examples/example4/example.config" || { echo "   [error] examples/example4/example.config missing (the runner uses it as its template)"; exit 1; }

echo "== 5/5 unit tests =="
cd "$REPO"
python -m unittest stackem.tests.test_core
echo
echo "== done =="
echo "Activate the environment in every new terminal:   source $WORK/venv/bin/activate"
echo "Then, from the repository root:                    bash scripts/run_smoke.sh"
