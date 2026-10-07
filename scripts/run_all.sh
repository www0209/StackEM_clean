#!/usr/bin/env bash
# StackEM - the complete flow of the final experiments, as ONE ordered list of named steps.
#
#     source ~/stackem_work/venv/bin/activate
#     cd <repository root>
#     bash scripts/run_all.sh                                   # everything from scratch (data, training, all experiments)
#     bash scripts/run_with_released_weights.sh                 # the same experiments with the weights of this repository
#     bash scripts/run_smoke.sh                                 # reduced sizes, minutes, to test the installation
#
# Run long jobs inside tmux and run ONE job at a time (E4 is a timing experiment; a second job on the same GPU or
# CPU makes its numbers meaningless).  Everything printed is also written to <root>/logs/run_all_<time>.log, and a
# table with the seconds of every step to <root>/logs/steps_<time>.tsv.
#
# Steps, in order (a name before the dot is a GROUP: naming the group selects all of its steps)
#     tests                      unit tests
#     kernels                    kernel dataset: 120000 profiles from the reference solver        -> base3/kernels
#     kernels_val                only when the held-out sets are missing (released-weights runs): the two sets E1 needs
#     train_teacher              teacher SKN 6x256, 300 epochs, data-only loss (w_pde = 0)        -> base3/skn
#     distill                    student SKN 4x128, 150 epochs, distilled from the teacher        -> base3/skn_distill
#     e1                         kernel accuracy of student, student + table, teacher
#     signoff                    sign-off sizing of base3 (reference solver only)                 -> base3/signoff
#     e0 e2 e3 e7 e4 e5 e6 e8    experiments on base3 with the student engine
#     quad4.signoff quad4.e2 quad4.e3 quad4.e7        second stack, base3's student
#     dense3.signoff dense3.e2 dense3.e3 dense3.e7    50-um strap pitch, base3's student
#     teacher.e1 teacher.e2 teacher.e4 teacher.e5 teacher.e8 teacher.compare
#                                the teacher engine on base3 (same sizing) and the comparison    -> base3_teacher
#     ablation.train ablation.e1 ablation.e2
#                                teacher trained WITH the PDE residual (w_pde = 0.05)             -> ablation_pde
#     ibmpg.<netlist>            every ibmpgN.spice found in STACKEM_IBMPG_DIR, all rails        -> ibmpg/ibmpgN
#     comsol.export comsol.compare   export the base3 rails for COMSOL; compare when its result file is present -> comsol
#     tables.base3 tables.quad4 tables.dense3         parameter tables
#     figs.framework figs.timing figs.index_base3 figs.index_quad4 figs.index_dense3
#
# Controls (environment variables)
#     STACKEM_SKIP="kernels,train_teacher,ablation"   skip these steps or groups (an empty value skips nothing)
#     STACKEM_ONLY="e2,quad4"                         run only these steps or groups (overrides STACKEM_SKIP)
#     STACKEM_WORK       work root, default ~/stackem_work
#     STACKEM_ROOT       output root, default <work>/outputs   (smoke mode: <work>/outputs_smoke)
#     STACKEM_HOTSPOT    HotSpot directory, default <work>/HotSpot
#     STACKEM_IBMPG_DIR  folder with ibmpgN.spice (+ ibmpgN.solution), default <work>/ibmpg ; see scripts/download_ibmpg.sh
#     STACKEM_IBMPG      explicit, space-separated list of netlists (overrides the folder scan)
#     STACKEM_NPROFILES  kernel profiles, default 120000        STACKEM_EPOCHS  teacher epochs, default 300
#     STACKEM_MODE       full (default) | smoke
#     STACKEM_PYTHON     interpreter, default python
#     STACKEM_NO_LOG=1   do not write the log file
#
# A failing step does not stop the run: it is reported as FAILED in the final table, the remaining steps still run,
# and the script exits with status 1.  Re-run only the failed ones with STACKEM_ONLY.
export PYTHONUNBUFFERED=1          # progress lines reach the log immediately
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
WORK="${STACKEM_WORK:-$HOME/stackem_work}"
MODE="${STACKEM_MODE:-full}"
HS="${STACKEM_HOTSPOT:-$WORK/HotSpot}"
IBMDIR="${STACKEM_IBMPG_DIR:-$WORK/ibmpg}"
PY="${STACKEM_PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ "$MODE" = smoke ]; then
  ROOT="${STACKEM_ROOT:-$WORK/outputs_smoke}"
  BASE=smoke; SMK="--smoke"
  CFG="configs/smoke.json"
  CFG_Q="configs/quad4.json,configs/smoke.json";   X_Q="--out-name quad4 --base-case quad4"
  CFG_D="configs/dense3.json,configs/smoke.json";  X_D="--out-name dense3 --base-case dense3"
  CFG_T="configs/smoke.json,configs/base3_teacher.json"; TEACHER_OUT="smoke_teacher"; X_T="--base-case smoke --out-name $TEACHER_OUT"
  X_A="--base-case smoke --out-name ablation_pde"
  DISTILL_ARGS="--epochs 3 --n-synthetic 200000 --hidden 32,32 --fourier 8"
  E4_STUDENT=""; E4_TEACHER=""
elif [ "$MODE" = full ]; then
  ROOT="${STACKEM_ROOT:-$WORK/outputs}"
  BASE=base3; SMK=""
  CFG="configs/base3.json"
  CFG_Q="configs/quad4.json";   X_Q=""
  CFG_D="configs/dense3.json";  X_D=""
  CFG_T="configs/base3_teacher.json"; TEACHER_OUT="base3_teacher"; X_T=""
  X_A="--out-name ablation_pde"
  DISTILL_ARGS=""                                   # the defaults of train_distill ARE the final settings: 4x128, 150 epochs, patience 30
  # E4: the sizes and options of the final run.  Student engine: one-core reference up to 20000 rails (the default of
  # e4_scaling).  Teacher engine: one-core reference up to 1000 rails only, as in its final run.
  E4_STUDENT="--sizes ${STACKEM_E4_SIZES:-82,246,1000,5000,20000}"
  E4_TEACHER="--sizes ${STACKEM_E4_SIZES:-82,246,1000,5000,20000} --single-core-max 1000"
else
  echo "unknown STACKEM_MODE=$MODE (use full or smoke)"; exit 2
fi
NPROF="${STACKEM_NPROFILES:-120000}"; EPOCHS="${STACKEM_EPOCHS:-300}"
W_TEACHER="$ROOT/$BASE/skn/skn_best.npz"
W_STUDENT="$ROOT/$BASE/skn_distill/skn_student.npz"
W_PDE="$ROOT/ablation_pde/skn_pde/skn_best.npz"
COMMON="--root $ROOT --hotspot $HS"
A="--config $CFG $COMMON $SMK"                                         # base case, student engine
AQ="--config $CFG_Q $X_Q $COMMON --weights $W_STUDENT $SMK"           # quad4 has no network of its own
AD="--config $CFG_D $X_D $COMMON --weights $W_STUDENT $SMK"           # dense3 neither
AT="--config $CFG_T $X_T $COMMON $SMK"                                 # teacher engine on the base case
AA="--config $CFG_T $X_A $COMMON $SMK"                                 # PDE-residual ablation (teacher-engine settings)

STAMP="$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$ROOT/logs"
if [ -z "${STACKEM_NO_LOG:-}" ]; then
  LOG="$ROOT/logs/run_all_$STAMP.log"
  exec > >(tee -a "$LOG") 2>&1
fi
echo "StackEM run_all: mode $MODE | output root $ROOT | HotSpot $HS | python $($PY --version 2>&1)"
echo "skip: '${STACKEM_SKIP-}'   only: '${STACKEM_ONLY-}'"

# ---- step runner ------------------------------------------------------------------------------------------------
SKIP=",${STACKEM_SKIP-},"; ONLY=",${STACKEM_ONLY-},"
STEP_NAME=(); STEP_SEC=(); STEP_STATE=(); FAILED=()
wanted() {                              # $1 = step name; a group name (text before the first dot) selects the whole group
  local n="$1" g="${1%%.*}"
  if [ "$ONLY" != ",," ]; then [[ "$ONLY" == *",$n,"* || "$ONLY" == *",$g,"* ]]; return; fi
  [[ "$SKIP" != *",$n,"* && "$SKIP" != *",$g,"* ]]
}
note() { STEP_NAME+=("$1"); STEP_SEC+=("$2"); STEP_STATE+=("$3"); }
run() {                                 # run <step name> <command...>
  local name="$1"; shift
  if ! wanted "$name"; then echo "[skip] $name"; note "$name" 0 skipped; return 0; fi
  echo; echo "=================== $(date '+%F %T')  $name ==================="
  local t0=$SECONDS rc=0
  "$@" || rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then note "$name" "$dt" ok; echo "[done] $name (${dt}s)"
  else note "$name" "$dt" "FAILED($rc)"; FAILED+=("$name"); echo "[FAILED] $name (exit $rc, ${dt}s)"; fi
  return 0
}

# ---- helper steps ---------------------------------------------------------------------------------------------------
kernels_val() {
  if [ -f "$ROOT/$BASE/kernels/kernels_val_full.npz" ] && [ -f "$ROOT/$BASE/kernels/kernels_ood_full.npz" ]; then
    echo "held-out kernel sets present, nothing to do"
  else
    $PY -m stackem.experiments.gen_kernels $A --val-only
  fi
}
distill() {
  if $PY -c "import torch" 2>/dev/null; then
    $PY -m stackem.train_distill --data "$ROOT/$BASE/kernels/kernels_train.npz" --teacher "$W_TEACHER" --out "$ROOT/$BASE/skn_distill" $DISTILL_ARGS
  elif [ "$MODE" = smoke ]; then
    echo "[warn] PyTorch is not installed: the distillation cannot run.  Smoke mode copies the teacher as the student so that the remaining steps can be exercised."
    mkdir -p "$ROOT/$BASE/skn_distill" && cp "$W_TEACHER" "$W_STUDENT"
  else
    echo "PyTorch is required for the distillation (see scripts/setup_env.sh)"; return 1
  fi
}
ablation_train() {
  $PY -m stackem.experiments.train_or_fallback $AA --epochs "$EPOCHS" --w-pde 0.05 --out-subdir skn_pde
}
comsol_compare() {
  if [ -f "$ROOT/comsol/comsol_rails_result.csv" ]; then
    $PY comsol/compare_rails.py --dir "$ROOT/comsol"
  else
    echo "no $ROOT/comsol/comsol_rails_result.csv: solve the exported rails in COMSOL first (comsol/COMSOL_MODEL_BUILD.md), then re-run this step"
  fi
}
timing_fig() {
  if [ -f "$ROOT/comsol/comsol_timing.csv" ]; then
    $PY tools/paper_timing_fig.py --root "$ROOT" --name "$BASE" --external "COMSOL (FEM)=$ROOT/comsol/comsol_timing.csv"
  else
    $PY tools/paper_timing_fig.py --root "$ROOT" --name "$BASE"
  fi
}

# ---- 1. tests, data, networks ----------------------------------------------------------------------------------------
run tests          $PY -m unittest stackem.tests.test_core
run kernels        $PY -m stackem.experiments.gen_kernels $A --n-profiles "$NPROF"
run kernels_val    kernels_val
run train_teacher  $PY -m stackem.experiments.train_or_fallback $A --epochs "$EPOCHS" --w-pde 0
run distill        distill

# ---- 2. base case, student engine --------------------------------------------------------------------------------------
run e1       $PY -m stackem.experiments.e1_skn_eval       $A
run signoff  $PY -m stackem.experiments.signoff_size      $A
run e0       $PY -m stackem.experiments.e0_validate       $A
run e2       $PY -m stackem.experiments.e2_rail_accuracy  $A
run e3       $PY -m stackem.experiments.e3_decomposition  $A
run e7       $PY -m stackem.experiments.e7_ranking        $A
run e4       $PY -m stackem.experiments.e4_scaling        $A --teacher "$W_TEACHER" --ablate-modes $E4_STUDENT
run e5       $PY -m stackem.experiments.e5_sensitivity    $A
run e6       $PY -m stackem.experiments.e6_sweeps         $A
run e8       $PY -m stackem.experiments.e8_ev6            $A

# ---- 3. the other two stacks (own sizing, base case's student) ------------------------------------------------------------
run quad4.signoff   $PY -m stackem.experiments.signoff_size      $AQ
run quad4.e2        $PY -m stackem.experiments.e2_rail_accuracy  $AQ
run quad4.e3        $PY -m stackem.experiments.e3_decomposition  $AQ
run quad4.e7        $PY -m stackem.experiments.e7_ranking        $AQ
run dense3.signoff  $PY -m stackem.experiments.signoff_size      $AD
run dense3.e2       $PY -m stackem.experiments.e2_rail_accuracy  $AD
run dense3.e3       $PY -m stackem.experiments.e3_decomposition  $AD
run dense3.e7       $PY -m stackem.experiments.e7_ranking        $AD

# ---- 4. teacher engine on the base case (same sizing, teacher weights) and the comparison -----------------------------------
run teacher.e1       $PY -m stackem.experiments.e1_skn_eval       $AT
run teacher.e2       $PY -m stackem.experiments.e2_rail_accuracy  $AT
run teacher.e4       $PY -m stackem.experiments.e4_scaling        $AT $E4_TEACHER
run teacher.e5       $PY -m stackem.experiments.e5_sensitivity    $AT
run teacher.e8       $PY -m stackem.experiments.e8_ev6            $AT
run teacher.compare  $PY tools/compare_engines.py --root "$ROOT" --student "$BASE" --teacher "$TEACHER_OUT"

# ---- 5. PDE-residual ablation: an explicit, separate network; the teacher above is never overwritten --------------------------
run ablation.train  ablation_train
run ablation.e1     $PY -m stackem.experiments.e1_skn_eval       $AA --weights "$W_PDE"
run ablation.e2     $PY -m stackem.experiments.e2_rail_accuracy  $AA --weights "$W_PDE"

# ---- 6. IBM power-grid benchmarks: all rails of every netlist present, student engine, GPU closure when available -----------------
if [ "$MODE" = smoke ]; then
  run ibmpg.synthetic $PY tools/run_ibmpg.py --synthetic --config "$CFG" $COMMON --weights "$W_STUDENT"
else
  NETS="${STACKEM_IBMPG:-}"
  if [ -z "$NETS" ]; then for n in 1 2 3 4 5 6; do [ -f "$IBMDIR/ibmpg$n.spice" ] && NETS="$NETS $IBMDIR/ibmpg$n.spice"; done; fi
  if [ -z "$NETS" ]; then
    echo "[skip] ibmpg: no netlist in $IBMDIR (run scripts/download_ibmpg.sh first)"; note ibmpg 0 "skipped(no netlist)"
  else
    for NET in $NETS; do
      run "ibmpg.$(basename "$NET" .spice)" $PY tools/run_ibmpg.py --spice "$NET" --config "$CFG" $COMMON --weights "$W_STUDENT" \
          --unit auto --size-to-jmax 1e6 --max-rails 0 --min-segments 3
    done
  fi
fi

# ---- 7. COMSOL cross-check (the COMSOL solve itself is a manual step on a machine with COMSOL) ------------------------------------
run comsol.export   $PY comsol/export_rails.py --e2 "$ROOT/$BASE/e2" --out "$ROOT/comsol"
run comsol.compare  comsol_compare

# ---- 8. tables and figures ------------------------------------------------------------------------------------------------------------
run tables.$BASE      $PY tools/param_table.py --config "$CFG"   --root "$ROOT" --name "$BASE"
run tables.quad4      $PY tools/param_table.py --config "$CFG_Q" --root "$ROOT" --name quad4
run tables.dense3     $PY tools/param_table.py --config "$CFG_D" --root "$ROOT" --name dense3
run figs.framework    $PY tools/framework_figure.py --out "$ROOT/$BASE/figs/framework"
run figs.timing       timing_fig
run figs.index_$BASE  $PY tools/collect_figures.py --root "$ROOT" --name "$BASE"
run figs.index_quad4  $PY tools/collect_figures.py --root "$ROOT" --name quad4
run figs.index_dense3 $PY tools/collect_figures.py --root "$ROOT" --name dense3

# ---- summary ------------------------------------------------------------------------------------------------------------------------------
echo; echo "=================== $(date '+%F %T')  summary ==================="
TSV="$ROOT/logs/steps_$STAMP.tsv"; printf "step\tseconds\tstate\n" > "$TSV"; TOTAL=0
printf "%-22s %9s  %s\n" step seconds state
for i in "${!STEP_NAME[@]}"; do
  printf "%-22s %9s  %s\n" "${STEP_NAME[$i]}" "${STEP_SEC[$i]}" "${STEP_STATE[$i]}"
  printf "%s\t%s\t%s\n" "${STEP_NAME[$i]}" "${STEP_SEC[$i]}" "${STEP_STATE[$i]}" >> "$TSV"
  TOTAL=$((TOTAL + STEP_SEC[$i]))
done
printf "%-22s %9s\n" total "$TOTAL"
echo "step table: $TSV"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "StackEM flow finished: $ROOT (all executed steps succeeded)"
  echo "compare with the released results:  $PY tools/check_results.py --root $ROOT"
else
  echo "StackEM flow finished: $ROOT  -- FAILED steps: ${FAILED[*]}"
  echo "re-run them with:  STACKEM_ONLY=$(IFS=,; echo "${FAILED[*]}")  and the same script you started (run_all.sh, run_with_released_weights.sh or run_smoke.sh)"
  exit 1
fi
