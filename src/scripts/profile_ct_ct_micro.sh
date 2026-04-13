#!/usr/bin/env bash
# Profile MOAI micro-benchmark: MOAI_BENCH_MODE=ct_ct -> ct_ct_matrix_mul_test()
# Artifacts default to <repo>/output/ct_ct/ (nsys-rep, kern_sum TSV, index, *_clean.tsv, charts — same folder)
#
# Requires: Nsight Systems (`nsys` on PATH), built `test` binary, Python 3.
#
# Env:
#   MOAI_BUILD_DIR      - default: <repo>/build (must contain ./test and copied src/data)
#   MOAI_TEST_EXE       - override path to test binary
#   MOAI_PROFILE_OUT    - output basename prefix (default: ct_ct_micro under MOAI_PROFILE_DIR)
#   MOAI_PROFILE_DIR    - directory for .nsys-rep, TSV, logs (default: <repo>/output/ct_ct)
#   MOAI_KERN_EXPORT    - if 1 (default), run export_ct_pt_kern_sum.py on kern_sum TSVs (same dir as TSV)
#   MOAI_PROFILE_REPEAT - number of profile runs (default: 1)
#   MOAI_NSYS_TRACE     - nsys --trace= (default: cuda,nvtx,osrt)
#   MOAI_NSYS_STATS_FILTER_NVTX - if 1 (default), emit two filtered kernel summaries from the same .nsys-rep:
#       moai:ct_ct_matrix_mul_colpacking  -> *.kern_sum.nvtx_col.tsv
#       moai:ct_ct_matrix_mul_diagpacking -> *.kern_sum.nvtx_diag.tsv
#       (requires MOAI built with NVTX / CUDA::nvtx3)
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR="${MOAI_BUILD_DIR:-${REPO_ROOT}/build}"
TEST_EXE="${MOAI_TEST_EXE:-${BUILD_DIR}/test}"
OUT_DIR="${MOAI_PROFILE_DIR:-${REPO_ROOT}/output/ct_ct}"
OUT_BASE="${MOAI_PROFILE_OUT:-ct_ct_micro}"
REPEAT="${MOAI_PROFILE_REPEAT:-1}"
TRACE="${MOAI_NSYS_TRACE:-cuda,nvtx,osrt}"
FILTER_NVTX="${MOAI_NSYS_STATS_FILTER_NVTX:-1}"
KERN_EXPORT="${MOAI_KERN_EXPORT:-1}"
PARSE_PY="${MOAI_PARSE_SCRIPT:-${SCRIPT_DIR}/parse_nsys_cuda_kern_sum.py}"
EXPORT_PY="${MOAI_KERN_EXPORT_SCRIPT:-${SCRIPT_DIR}/export_ct_pt_kern_sum.py}"

if [[ ! -x "$TEST_EXE" ]]; then
  echo "error: test binary not found or not executable: $TEST_EXE" >&2
  echo "  hint: CONDA_PREFIX=... cmake -S $REPO_ROOT -B $BUILD_DIR && cmake --build $BUILD_DIR --target test" >&2
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not on PATH (install Nsight Systems)" >&2
  exit 1
fi

if [[ ! -f "$PARSE_PY" ]]; then
  echo "error: parser script missing: $PARSE_PY" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

echo -e "run\tnsys_rep\tkern_sum_tsv\tparse_full\tparse_nvtx_col\tparse_nvtx_diag" >"${OUT_DIR}/${OUT_BASE}_index.tsv"

for ((i = 1; i <= REPEAT; i++)); do
  stem="${OUT_DIR}/${OUT_BASE}_run${i}"
  rep="${stem}.nsys-rep"
  sum_full="${stem}.kern_sum.tsv"
  sum_nvtx_col="${stem}.kern_sum.nvtx_col.tsv"
  sum_nvtx_diag="${stem}.kern_sum.nvtx_diag.tsv"
  log="${stem}.stdout.log"

  echo "=== run $i: nsys profile -> $rep ===" >&2
  set +e
  (
    cd "$BUILD_DIR"
    export MOAI_BENCH_MODE=ct_ct
    exec nsys profile --force-overwrite=true -o "$stem" --trace="$TRACE" "$TEST_EXE"
  ) >"$log" 2>&1
  ec=$?
  set -e
  if [[ $ec -ne 0 ]]; then
    echo "error: nsys profile failed (exit $ec); see $log" >&2
    cp "$log" "${OUT_DIR}/${OUT_BASE}_last_fail.log"
    exit "$ec"
  fi

  rm -f "${stem}.sqlite"

  nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv "$rep" >"$sum_full"

  parse_full=""
  if parse_out=$(python3 "$PARSE_PY" "$sum_full"); then
    parse_full=$(echo "$parse_out" | tr '\n' ';')
  else
    parse_full="parse_failed"
  fi

  parse_col="n/a"
  parse_diag="n/a"
  if [[ "$FILTER_NVTX" == "1" ]]; then
    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:ct_ct_matrix_mul_colpacking" "$rep" >"$sum_nvtx_col" 2>"$tmp"
    fc=$?
    set -e
    if [[ $fc -eq 0 ]] && [[ -s "$sum_nvtx_col" ]]; then
      if p2=$(python3 "$PARSE_PY" "$sum_nvtx_col"); then
        parse_col=$(echo "$p2" | tr '\n' ';')
      else
        parse_col="parse_failed"
      fi
    else
      parse_col="nvtx_filter_unavailable_or_empty"
      if [[ -s "$tmp" ]]; then
        head -3 "$tmp" >&2
      fi
    fi

    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:ct_ct_matrix_mul_diagpacking" "$rep" >"$sum_nvtx_diag" 2>"$tmp"
    fd=$?
    set -e
    if [[ $fd -eq 0 ]] && [[ -s "$sum_nvtx_diag" ]]; then
      if p3=$(python3 "$PARSE_PY" "$sum_nvtx_diag"); then
        parse_diag=$(echo "$p3" | tr '\n' ';')
      else
        parse_diag="parse_failed"
      fi
    else
      parse_diag="nvtx_filter_unavailable_or_empty"
      if [[ -s "$tmp" ]]; then
        head -3 "$tmp" >&2
      fi
    fi
  fi

  echo -e "${i}\t${rep}\t${sum_full}\t${parse_full}\t${parse_col}\t${parse_diag}" >>"${OUT_DIR}/${OUT_BASE}_index.tsv"
  echo "  wrote $rep ; $sum_full" >&2

  if [[ "$KERN_EXPORT" == "1" ]] && [[ -f "$EXPORT_PY" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_full" || true
      if [[ -s "$sum_nvtx_col" ]]; then
        python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_col" || true
      fi
      if [[ -s "$sum_nvtx_diag" ]]; then
        python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_diag" || true
      fi
    fi
  fi
done

echo "Index: ${OUT_DIR}/${OUT_BASE}_index.tsv" >&2
echo "Open GUI: nsys-ui ${OUT_DIR}/${OUT_BASE}_run1.nsys-rep" >&2
