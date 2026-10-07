#!/usr/bin/env bash
# The complete flow of scripts/run_all.sh WITHOUT generating the kernel training set and WITHOUT any training:
# the three released networks of weights/ are copied into the output root first, then every experiment runs.
#
#     source ~/stackem_work/venv/bin/activate && bash scripts/run_with_released_weights.sh
#
#     weights/skn_teacher.npz      ->  <root>/base3/skn/skn_best.npz                 teacher, 6x256, data-only loss
#     weights/skn_student.npz      ->  <root>/base3/skn_distill/skn_student.npz      student, 4x128 (the engine)
#     weights/skn_teacher_pde.npz  ->  <root>/ablation_pde/skn_pde/skn_best.npz      teacher with the PDE residual (ablation)
#
# Skipped steps: kernels, train_teacher, distill, ablation.train.  E1 needs the two held-out kernel sets; they are
# taken from reference_results/base3/kernels/ when the released data archive has been unpacked there, otherwise the
# step kernels_val regenerates just those two sets with the reference solver (500 profiles, no training set).
# All controls of run_all.sh apply; a STACKEM_SKIP given by the caller is added to the list above.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
WORK="${STACKEM_WORK:-$HOME/stackem_work}"
ROOT="${STACKEM_ROOT:-$WORK/outputs}"
for f in skn_teacher.npz skn_student.npz skn_teacher_pde.npz; do
  [ -f "weights/$f" ] || { echo "missing weights/$f"; exit 1; }
done
if [ -f weights/SHA256SUMS ] && command -v sha256sum >/dev/null 2>&1; then
  ( cd weights && sha256sum -c SHA256SUMS ) || { echo "weights do not match weights/SHA256SUMS"; exit 1; }
fi
put() { mkdir -p "$(dirname "$2")"; cp "$1" "$2"; echo "  copied $1 -> $2"; }
put weights/skn_teacher.npz      "$ROOT/base3/skn/skn_best.npz"
put weights/skn_student.npz      "$ROOT/base3/skn_distill/skn_student.npz"
put weights/skn_teacher_pde.npz  "$ROOT/ablation_pde/skn_pde/skn_best.npz"
# training records of the released networks (summaries, logs, curves): used by the comparison table for the parameter counts
records() {   # records <released folder> <output folder>
  [ -d "$1" ] || return 0
  mkdir -p "$2"
  for f in "$1"/*; do case "$f" in *.npz) ;; *) [ -f "$f" ] && cp "$f" "$2/"; esac; done
  echo "  copied training records $1 -> $2"
}
records reference_results/base3/skn             "$ROOT/base3/skn"
records reference_results/base3/skn_distill     "$ROOT/base3/skn_distill"
records reference_results/ablation_pde/skn_pde  "$ROOT/ablation_pde/skn_pde"
# released kernel data, when unpacked into reference_results/base3/kernels/
for f in kernels_val_full.npz kernels_ood_full.npz kernels_train.npz; do
  if [ -f "reference_results/base3/kernels/$f" ] && [ ! -f "$ROOT/base3/kernels/$f" ]; then put "reference_results/base3/kernels/$f" "$ROOT/base3/kernels/$f"; fi
done
export STACKEM_SKIP="kernels,train_teacher,distill,ablation.train${STACKEM_SKIP:+,$STACKEM_SKIP}"
exec bash scripts/run_all.sh
