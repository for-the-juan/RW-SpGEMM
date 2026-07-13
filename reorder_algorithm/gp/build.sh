#!/usr/bin/env bash
set -euo pipefail
root="${MATRIX_PARTITION_UTILITY_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/Matrix-Partitioning-Utility}"
metis_install="${METIS_INSTALL_HOME:-/home/hangcheng.dong/PPoPP27/metis-5.1.0/install}"
metis_home="${GP_PARTITION_HOME:-$root/METIS}"
converter_home="$root/Converter"

if [ ! -x "$metis_home/run" ]; then
  g++ -g "$metis_home/mmio.c" "$metis_home/main.cpp" -o "$metis_home/run" -std=c++11 \
    -I"$metis_install/include" -L"$metis_install/lib" -lmetis
fi

if [ ! -x "$converter_home/PartitiontoReorderConverter" ]; then
  make -C "$converter_home"
fi

[ -x "$metis_home/run" ] || { echo "gp: missing $metis_home/run"; exit 42; }
[ -x "$converter_home/PartitiontoReorderConverter" ] || { echo "gp: missing $converter_home/PartitiontoReorderConverter"; exit 42; }
echo "gp: found AE partition workflow"
