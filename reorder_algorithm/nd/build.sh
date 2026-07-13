#!/usr/bin/env bash
set -euo pipefail
if command -v mtxreorder >/dev/null 2>&1; then
  echo "nd: found mtxreorder"
  exit 0
fi
libmtx_home="${LIBMTX_HOME:-/home/hangcheng.dong/PPoPP27/libmtx}"
if [ -x "$libmtx_home/install/bin/mtxreorder" ]; then
  echo "nd: found $libmtx_home/install/bin/mtxreorder"
  exit 0
fi
echo "nd: missing mtxreorder from libmtx in PATH or $libmtx_home/install/bin"
exit 42
