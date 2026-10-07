#!/usr/bin/env bash
# Functional smoke test: the SAME ordered steps as scripts/run_all.sh at reduced size, to check that the environment,
# the paths and HotSpot work.  Outputs go to ~/stackem_work/outputs_smoke (never into the outputs of a real run).
#
#     source ~/stackem_work/venv/bin/activate && bash scripts/run_smoke.sh
#
# Reduced sizes: 2000 kernel profiles, teacher 8 epochs (4x128), student 3 epochs (2x32), 32x32 HotSpot grid,
# 40 reference cells, 8-10 rails per die in E2/E3, E4 sizes 24 and 82, a synthetic IBM-format netlist.
# The numbers of a smoke run have no physical value; only "all executed steps succeeded" in the last lines matters.
# Duration: dominated by the single-threaded HotSpot runs of the ev6 stack (E8, twice) and by the three sign-off
# sizings; measured 45 to 70 minutes on a 2-core machine without GPU (the faster figure with an idle machine); a
# workstation is several times faster.
# Without PyTorch the teacher is trained by the sklearn fallback, the student is a copy of it, and the parts of
# E4/E5 that need PyTorch are skipped by the experiments themselves.
# All controls of run_all.sh apply (STACKEM_SKIP, STACKEM_ONLY, STACKEM_ROOT, ...).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STACKEM_MODE=smoke exec bash scripts/run_all.sh
