#!/usr/bin/env bash
set -euo pipefail
home="${SPARSEBASE_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/SparseBase}"
bin="$home/build/examples/degree_order/degree_order_mtx"
if [ -x "$bin" ]; then
  echo "degree: found $bin"
  exit 0
fi
if [ -d "$home" ]; then
  cmake -S "$home" -B "$home/build" -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
  cmake --build "$home/build" --target degree_order_mtx -j"$(nproc)"
fi
[ -x "$bin" ] || { echo "degree: missing $bin"; exit 42; }
echo "degree: found $bin"
