#!/usr/bin/env bash
set -euo pipefail
home="${SPARSEBASE_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/SparseBase}"
bin="$home/build/examples/slashburn_order/slashburn_order"
if [ -x "$bin" ]; then
  echo "slashburn: found $bin"
  exit 0
fi
if [ -d "$home" ]; then
  cmake -S "$home" -B "$home/build" -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
  cmake --build "$home/build" --target slashburn_order -j"$(nproc)"
fi
[ -x "$bin" ] || { echo "slashburn: missing $bin"; exit 42; }
