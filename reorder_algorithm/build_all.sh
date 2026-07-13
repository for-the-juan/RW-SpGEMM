#!/usr/bin/env bash
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
status=0
for dir in "$root"/*; do
  [ -d "$dir" ] || continue
  [ -x "$dir/build.sh" ] || continue
  algo="$(basename "$dir")"
  echo "== build/check $algo =="
  "$dir/build.sh"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "algorithm $algo unavailable, status=$rc"
    status=1
  fi
done
exit $status
