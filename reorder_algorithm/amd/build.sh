#!/usr/bin/env bash
set -euo pipefail
home="${SPARSEBASE_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/SparseBase}"
bin="$home/build/examples/amd_order/amd_order"
if [ -x "$bin" ]; then
  echo "amd: found $bin"
  exit 0
fi
suitesparse_home="${SUITESPARSE_HOME:-/home/hangcheng.dong/PPoPP27/SuiteSparse-5.13.0/install}"
if [ -d "$home" ] && [ -f "$suitesparse_home/include/amd.h" ] && [ -e "$suitesparse_home/lib/libamd.so" ]; then
  cmake -S "$home" -B "$home/build" -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF \
    -DUSE_AMD_ORDER=ON -DAMD_LIB_DIR="$suitesparse_home/lib" -DAMD_INC_DIR="$suitesparse_home/include"
  cmake --build "$home/build" --target amd_order -j"$(nproc)"
fi
[ -x "$bin" ] || {
  echo "amd: missing $bin"
  echo "Build SuiteSparse AMD and SparseBase with USE_AMD_ORDER=ON, then rerun."
  exit 42
}
echo "amd: found $bin"
