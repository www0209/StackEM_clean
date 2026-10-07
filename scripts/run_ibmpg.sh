#!/usr/bin/env bash
# IBM power-grid benchmarks only: all rails of every netlist, student engine (the ibmpg group of scripts/run_all.sh).
#
#     bash scripts/download_ibmpg.sh                       # once: ibmpg1 ... ibmpg6 into ~/stackem_work/ibmpg
#     bash scripts/run_ibmpg.sh                            # every ibmpgN.spice found there
#     bash scripts/run_ibmpg.sh ~/stackem_work/ibmpg/ibmpg1.spice ~/stackem_work/ibmpg/ibmpg2.spice     # a selection
#
# Needs <root>/base3/skn_distill/skn_student.npz (from run_all.sh or run_with_released_weights.sh) and HotSpot.
# Results: <root>/ibmpg/ibmpgN/ibmpg_summary.json.  The closure runs on the GPU when PyTorch sees one, else on the CPU;
# check the "closure (torch/cuda ...)" or "closure (numpy/cpu ...)" line of the log.  Before a GPU run verify
#     python -c "import torch; print(torch.cuda.is_available())"
# Look at the "coordinate unit" line of every netlist: the die size it prints must be plausible.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ $# -gt 0 ] && export STACKEM_IBMPG="$*"
STACKEM_ONLY=ibmpg exec bash scripts/run_all.sh
