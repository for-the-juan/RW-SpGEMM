#!/usr/bin/env bash
set -euo pipefail
home="${RABBIT_HOME:-/home/hangcheng.dong/PPoPP27/reordering-spgemm/rabbit_order}"
converter="$home/converter/mtx_to_el"
reorder="$home/demo/reorder"
if [ ! -x "$converter" ]; then
  g++ "$home/converter/mtx_to_el.cc" "$home/converter/mmio.c" -o "$converter" -std=c++11
fi
if [ ! -x "$reorder" ]; then
  make -C "$home/demo"
fi
[ -x "$converter" ] && [ -x "$reorder" ] || { echo "rabbit: missing converter or reorder binary"; exit 42; }
echo "rabbit: found converter and reorder"
