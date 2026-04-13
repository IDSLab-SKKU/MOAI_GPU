#!/usr/bin/env bash
# Run isolated micro-benchmarks in test.cu (MOAI_BENCH_MODE): bootstrapping_test, ct_ct_matrix_mul_test.
# Parses kernel timings from stdout; no single_layer, no keys env required.
#
# Note: bootstrapping_test / ct_ct_matrix_mul_test use fixed parameters in C++ and do not read MOAI_ALPHA.
#       To study alpha vs cost, use single_layer (default test without MOAI_BENCH_MODE) or extend those tests.
#
# Usage:
#   ./src/scripts/bench_alpha_performance.sh
#   MOAI_BENCH_MODES="boot" ./src/scripts/bench_alpha_performance.sh
#   MOAI_BENCH_MODES="ct_ct" MOAI_BENCH_OMP_THREADS=4 ./src/scripts/bench_alpha_performance.sh
#
# Env:
#   MOAI_BUILD_DIR       - build dir with `test` (default: <repo>/build)
#   MOAI_TEST_EXE        - override path to test binary
#   MOAI_BENCH_MODES     - space-separated: boot ct_ct (default: both)
#   MOAI_BENCH_OMP_THREADS - OMP_NUM_THREADS (default: 8)
#   MOAI_BENCH_OUT       - TSV path (default: <repo>/bench_micro_boot_ct_ct.tsv)
#   MOAI_BENCH_REPEAT    - repeat each mode this many times (default: 1)
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR="${MOAI_BUILD_DIR:-${REPO_ROOT}/build}"
TEST_EXE="${MOAI_TEST_EXE:-${BUILD_DIR}/test}"
OUT="${MOAI_BENCH_OUT:-${REPO_ROOT}/bench_micro_boot_ct_ct.tsv}"
MODES="${MOAI_BENCH_MODES:-boot ct_ct}"
OMP_CAP="${MOAI_BENCH_OMP_THREADS:-8}"
REPEAT="${MOAI_BENCH_REPEAT:-1}"

if [[ ! -f "$TEST_EXE" ]]; then
  echo "error: test binary not found: $TEST_EXE" >&2
  echo "  cmake --build $BUILD_DIR" >&2
  exit 1
fi

export OMP_NUM_THREADS="$OMP_CAP"

parse_boot_sec() {
  local log=$1
  grep "Bootstrapping took:" "$log" 2>/dev/null | head -1 \
    | sed -n 's/.*Bootstrapping took:[[:space:]]*\([0-9.]*\).*/\1/p'
}

parse_ct_ct_colpack() {
  local log=$1
  grep "column packing Ct-Ct matrix multiplication time = " "$log" 2>/dev/null | head -1 \
    | sed -n 's/.*column packing Ct-Ct matrix multiplication time = \([0-9.]*\).*/\1/p'
}

parse_ct_ct_diag() {
  local log=$1
  grep "diag packing Ct-Ct matrix multiplication time = " "$log" 2>/dev/null | head -1 \
    | sed -n 's/.*diag packing Ct-Ct matrix multiplication time = \([0-9.]*\).*/\1/p'
}

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

echo -e "run\tmode\texit_code\twall_sec\tboot_sec\tct_ct_colpack_sec\tct_ct_diag_sec" >"$OUT"

run=0
for ((i = 1; i <= REPEAT; i++)); do
  for mode in $MODES; do
    run=$((run + 1))
    echo "=== run $run MOAI_BENCH_MODE=$mode OMP_NUM_THREADS=$OMP_CAP ===" >&2
    : >"$tmp"
    t0=$(date +%s)
    set +e
    (
      cd "$BUILD_DIR"
      export MOAI_BENCH_MODE="$mode"
      export OMP_NUM_THREADS="$OMP_CAP"
      exec "$TEST_EXE"
    ) >"$tmp" 2>&1
    ec=$?
    set -e
    t1=$(date +%s)
    wall=$((t1 - t0))

    boot=""
    col=""
    diag=""
    if [[ "$mode" == "boot" ]]; then
      boot=$(parse_boot_sec "$tmp")
    elif [[ "$mode" == "ct_ct" ]]; then
      col=$(parse_ct_ct_colpack "$tmp")
      diag=$(parse_ct_ct_diag "$tmp")
    fi

    echo -e "${run}\t${mode}\t${ec}\t${wall}\t${boot}\t${col}\t${diag}" >>"$OUT"

    if [[ $ec -ne 0 ]]; then
      echo "  failed (exit $ec); log: ${BUILD_DIR}/bench_micro_last.log" >&2
      cp "$tmp" "${BUILD_DIR}/bench_micro_last.log"
    else
      echo "  boot=${boot:-n/a} colpack=${col:-n/a} diag=${diag:-n/a} wall=${wall}s" >&2
    fi
  done
done

echo "Wrote $OUT" >&2
