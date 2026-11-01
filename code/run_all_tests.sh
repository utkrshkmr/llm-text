#!/usr/bin/env bash
# Run the core tests and every lab in its own pytest process.
# (Triton's CPU interpreter must be configured before triton is first imported,
#  so labs that use it cannot share a process with labs that import triton normally.)
#   LAB_IMPL=solution ./run_all_tests.sh      # reference solutions
#   ./run_all_tests.sh                        # your starter implementations
set -u
cd "$(dirname "$0")"
status=0
python -m pytest -q -p no:cacheprovider tests | tail -1 || status=1
for d in labs/lab*/ projects/*/; do
  [ -d "$d" ] || continue
  ls "$d"/test_*.py >/dev/null 2>&1 || continue
  printf "%-28s " "$(basename "$d")"
  python -m pytest -q -p no:cacheprovider "$d" 2>&1 | tail -1
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
done
exit $status
