#!/usr/bin/env bash
set -euo pipefail
home="${SPARSEBASE_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/SparseBase}"
bin="$home/build/examples/gray_order/gray_order"
if [ -x "$bin" ]; then
  echo "gray: found $bin"
  exit 0
fi
if [ -d "$home" ]; then
  cmake -S "$home" -B "$home/build" -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
  cmake --build "$home/build" --target gray_order -j"$(nproc)"
fi
[ -x "$bin" ] || { echo "gray: missing $bin"; exit 42; }
