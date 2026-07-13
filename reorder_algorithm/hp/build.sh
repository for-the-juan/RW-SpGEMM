#!/usr/bin/env bash
set -euo pipefail
root="${MATRIX_PARTITION_UTILITY_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/Matrix-Partitioning-Utility}"
patoh_home="${HP_PARTITION_HOME:-$root/PaToH}"
converter_home="$root/Converter"

if [ ! -x "$patoh_home/a.out" ]; then
  gcc "$patoh_home/main.c" "$patoh_home/libpatoh_linux.a" -lm -o "$patoh_home/a.out"
fi

if [ ! -x "$converter_home/PartitiontoReorderConverter" ]; then
  make -C "$converter_home"
fi

[ -x "$patoh_home/a.out" ] || { echo "hp: missing $patoh_home/a.out"; exit 42; }
[ -x "$converter_home/PartitiontoReorderConverter" ] || { echo "hp: missing $converter_home/PartitiontoReorderConverter"; exit 42; }
echo "hp: found AE partition workflow"
