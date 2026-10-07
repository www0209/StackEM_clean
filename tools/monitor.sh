#!/usr/bin/env bash
# Live monitor of a StackEM run - open it in a second terminal window:
#     bash tools/monitor.sh [case, default base3] [log, default the newest <root>/logs/run_all_*.log] [output root, default ~/stackem_work/outputs]
# Refreshes every 20 s: current step, finished experiments (newest first), SKN training progress,
# GPU / CPU / disk, and the last log lines.  Ctrl+C to leave (the run itself is unaffected).
NAME="${1:-base3}"
OUT="${3:-${STACKEM_ROOT:-$HOME/stackem_work/outputs}}"
LOG="${2:-$(ls -t "$OUT"/logs/run_all_*.log 2>/dev/null | head -1)}"
ROOT="$OUT/$NAME"
while true; do
  clear
  echo "StackEM monitor  $(date '+%F %T')   case: $NAME   log: $LOG"
  echo "------------------------------------------------------------------------------------------"
  echo "current step : $(grep -E '^=+ .* =+$' "$LOG" 2>/dev/null | tail -1 | sed 's/=//g')"
  echo "started      : $(grep -m1 -E '^=+ .* =+$' "$LOG" 2>/dev/null | awk '{print $2, $3}')"
  echo
  echo "== finished outputs (newest first) =="
  ls -lt --time-style='+%H:%M' "$ROOT"/*/*.json 2>/dev/null | grep -v config_used | grep -v _rails | head -12 | awk '{print "  " $6, $7}' | sed "s#$ROOT/##"
  echo
  echo "== SKN training (last epochs) =="
  for T in skn skn_distill; do
    if [ -f "$ROOT/$T/train_log.csv" ]; then echo "  [$T]"; head -1 "$ROOT/$T/train_log.csv"; tail -3 "$ROOT/$T/train_log.csv"; fi
  done
  echo
  echo "== kernel dataset ==  $(ls -la --block-size=M "$ROOT"/kernels/*.npz 2>/dev/null | awk '{print $5, $9}' | sed "s#$ROOT/##" | tr '\n' ' ')"
  echo "== GPU ==  $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
  echo "== CPU ==  load $(cut -d' ' -f1-3 /proc/loadavg) (of $(nproc) cores)    == RAM ==  $(free -g | awk '/Mem/{print $3"/"$2" GB"}')    == disk ==  $(df -h "$HOME" | awk 'NR==2{print $4" free"}')"
  echo
  echo "== log tail =="
  tail -n 14 "$LOG" 2>/dev/null | cut -c1-160
  sleep 20
done
