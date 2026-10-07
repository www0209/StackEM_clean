#!/usr/bin/env bash
# Download the IBM power-grid benchmarks ibmpg1 ... ibmpg6 (netlist + official DC solution) and check their MD5 sums.
#
#     bash scripts/download_ibmpg.sh [target folder, default ~/stackem_work/ibmpg]
#
# Source: the public benchmark page (S. R. Nassif, "Power grid analysis benchmarks", 2008).
# The .solution files are the official DC solutions; tools/run_ibmpg.py uses them to check its own DC solve.
# Size: about 76 MB to download, about 610 MB unpacked (the .bz2 files are kept).  wget -c resumes an interrupted download.
set -euo pipefail
DEST="${1:-${STACKEM_IBMPG_DIR:-${STACKEM_WORK:-$HOME/stackem_work}/ibmpg}}"
URL="https://web.ece.ucsb.edu/~lip/PGBenchmarks/ibmpg"
mkdir -p "$DEST"; cd "$DEST"
for n in 1 2 3 4 5 6; do
  for f in spice solution; do
    if [ -f "ibmpg$n.$f" ]; then echo "ibmpg$n.$f present"; continue; fi
    wget -c "$URL/ibmpg$n.$f.bz2"
    bunzip2 -k -f "ibmpg$n.$f.bz2"
  done
done
# MD5 sums published with the benchmarks
cat > MD5SUMS.txt <<'MD5'
f6867bbc87cd15fa05c9ccb58554e2c9  ibmpg1.solution
033949515514232397464ac8304fea59  ibmpg1.spice
d82e8ff62c8d2acf74966563ec59db5b  ibmpg2.solution
488f77156a52a186e75b411c580d9191  ibmpg2.spice
0f3ca69542ce402a86b878a1701378a3  ibmpg3.solution
58c9816dac766ae0ba0fb275c4665601  ibmpg3.spice
81e36c320e8417fece3ee32e0c4adf07  ibmpg4.solution
71c3ee8062430b11861a97874a6929b1  ibmpg4.spice
bf3f90c98c0784d88ea25acf6e76bd9a  ibmpg5.solution
719b6993c38add21fa259b2a06899e36  ibmpg5.spice
faa959d1ea1d4537fc618ba303fb0ab7  ibmpg6.solution
e03444e4c4252fcd938b4c2f1265d67a  ibmpg6.spice
MD5
md5sum -c MD5SUMS.txt
echo "benchmarks ready in $DEST"
