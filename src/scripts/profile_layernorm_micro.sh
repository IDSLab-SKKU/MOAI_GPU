#!/usr/bin/env bash
# Profile MOAI micro-benchmark: MOAI_BENCH_MODE=layernorm -> layernorm_test()
# Artifacts default to <repo>/output/layernorm/
#
# NVTX: moai:layernorm (see test_layernorm.cuh). Requires MOAI_HAVE_NVTX build.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR="${MOAI_BUILD_DIR:-${REPO_ROOT}/build}"
TEST_EXE="${MOAI_TEST_EXE:-${BUILD_DIR}/test}"
OUT_DIR="${MOAI_PROFILE_DIR:-${REPO_ROOT}/output/layernorm}"
OUT_BASE="${MOAI_PROFILE_OUT:-layernorm_micro}"
REPEAT="${MOAI_PROFILE_REPEAT:-1}"
TRACE="${MOAI_NSYS_TRACE:-cuda,nvtx,osrt}"
FILTER_NVTX="${MOAI_NSYS_STATS_FILTER_NVTX:-1}"
KERN_EXPORT="${MOAI_KERN_EXPORT:-1}"
PARSE_PY="${MOAI_PARSE_SCRIPT:-${SCRIPT_DIR}/parse_nsys_cuda_kern_sum.py}"
EXPORT_PY="${MOAI_KERN_EXPORT_SCRIPT:-${SCRIPT_DIR}/export_ct_pt_kern_sum.py}"

if [[ ! -x "$TEST_EXE" ]]; then
  echo "error: test binary not found or not executable: $TEST_EXE" >&2
  exit 1
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not on PATH" >&2
  exit 1
fi
if [[ ! -f "$PARSE_PY" ]]; then
  echo "error: parser script missing: $PARSE_PY" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

echo -e "run\tnsys_rep\tkern_sum_tsv\tparse_full\tparse_nvtx" >"${OUT_DIR}/${OUT_BASE}_index.tsv"

for ((i = 1; i <= REPEAT; i++)); do
  stem="${OUT_DIR}/${OUT_BASE}_run${i}"
  rep="${stem}.nsys-rep"
  sum_full="${stem}.kern_sum.tsv"
  sum_nvtx="${stem}.kern_sum.nvtx.tsv"
  log="${stem}.stdout.log"

  echo "=== run $i: nsys profile -> $rep ===" >&2
  set +e
  (cd "$BUILD_DIR" && export MOAI_BENCH_MODE=layernorm && exec nsys profile --force-overwrite=true -o "$stem" --trace="$TRACE" "$TEST_EXE") >"$log" 2>&1
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

  parse_nv="n/a"
  if [[ "$FILTER_NVTX" == "1" ]]; then
    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:layernorm" "$rep" >"$sum_nvtx" 2>"$tmp"
    fc=$?
    set -e
    if [[ $fc -eq 0 ]] && [[ -s "$sum_nvtx" ]]; then
      if p2=$(python3 "$PARSE_PY" "$sum_nvtx"); then
        parse_nv=$(echo "$p2" | tr '\n' ';')
      else
        parse_nv="parse_failed"
      fi
    else
      parse_nv="nvtx_filter_unavailable_or_empty"
      [[ -s "$tmp" ]] && head -3 "$tmp" >&2
    fi
  fi

  echo -e "${i}\t${rep}\t${sum_full}\t${parse_full}\t${parse_nv}" >>"${OUT_DIR}/${OUT_BASE}_index.tsv"
  if [[ "$KERN_EXPORT" == "1" ]] && [[ -f "$EXPORT_PY" ]] && command -v python3 >/dev/null 2>&1; then
    python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_full" || true
    [[ -s "$sum_nvtx" ]] && python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx" || true
  fi
done

echo "Index: ${OUT_DIR}/${OUT_BASE}_index.tsv" >&2
echo "Open GUI: nsys-ui ${OUT_DIR}/${OUT_BASE}_run1.nsys-rep" >&2
