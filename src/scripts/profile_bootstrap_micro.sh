#!/usr/bin/env bash
# Profile MOAI micro-benchmark: MOAI_BENCH_MODE=bootstrap_micro -> bootstrapping_test()
# Artifacts default to <repo>/output/bootstrap/
#
# NVTX (MOAI_HAVE_NVTX, see bootstrapping.cuh):
#   moai:bootstrap_prepare — prepare_mod_polynomial + key/LT setup through generate_LT_coefficient_3
#   moai:bootstrap_3       — single bootstrap_3(rtn, cipher) call
#
# Env: same family as profile_softmax_micro.sh (MOAI_BUILD_DIR, MOAI_PROFILE_DIR, MOAI_KERN_EXPORT, …)
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR="${MOAI_BUILD_DIR:-${REPO_ROOT}/build}"
TEST_EXE="${MOAI_TEST_EXE:-${BUILD_DIR}/test}"
OUT_DIR="${MOAI_PROFILE_DIR:-${REPO_ROOT}/output/bootstrap}"
OUT_BASE="${MOAI_PROFILE_OUT:-bootstrap_micro}"
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

echo -e "run\tnsys_rep\tkern_sum_tsv\tparse_full\tparse_nvtx_prepare\tparse_nvtx_bootstrap3" >"${OUT_DIR}/${OUT_BASE}_index.tsv"

for ((i = 1; i <= REPEAT; i++)); do
  stem="${OUT_DIR}/${OUT_BASE}_run${i}"
  rep="${stem}.nsys-rep"
  sum_full="${stem}.kern_sum.tsv"
  sum_nvtx_prep="${stem}.kern_sum.nvtx_prepare.tsv"
  sum_nvtx_b3="${stem}.kern_sum.nvtx_bootstrap3.tsv"
  log="${stem}.stdout.log"

  echo "=== run $i: nsys profile -> $rep ===" >&2
  set +e
  (
    cd "$BUILD_DIR"
    export MOAI_BENCH_MODE=bootstrap_micro
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

  parse_prep="n/a"
  parse_b3="n/a"
  if [[ "$FILTER_NVTX" == "1" ]]; then
    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:bootstrap_prepare" "$rep" >"$sum_nvtx_prep" 2>"$tmp"
    fc=$?
    set -e
    if [[ $fc -eq 0 ]] && [[ -s "$sum_nvtx_prep" ]]; then
      if p2=$(python3 "$PARSE_PY" "$sum_nvtx_prep"); then
        parse_prep=$(echo "$p2" | tr '\n' ';')
      else
        parse_prep="parse_failed"
      fi
    else
      parse_prep="nvtx_filter_unavailable_or_empty"
      [[ -s "$tmp" ]] && head -3 "$tmp" >&2
    fi

    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:bootstrap_3" "$rep" >"$sum_nvtx_b3" 2>"$tmp"
    fd=$?
    set -e
    if [[ $fd -eq 0 ]] && [[ -s "$sum_nvtx_b3" ]]; then
      if p3=$(python3 "$PARSE_PY" "$sum_nvtx_b3"); then
        parse_b3=$(echo "$p3" | tr '\n' ';')
      else
        parse_b3="parse_failed"
      fi
    else
      parse_b3="nvtx_filter_unavailable_or_empty"
      [[ -s "$tmp" ]] && head -3 "$tmp" >&2
    fi
  fi

  echo -e "${i}\t${rep}\t${sum_full}\t${parse_full}\t${parse_prep}\t${parse_b3}" >>"${OUT_DIR}/${OUT_BASE}_index.tsv"

  if [[ "$KERN_EXPORT" == "1" ]] && [[ -f "$EXPORT_PY" ]] && command -v python3 >/dev/null 2>&1; then
    python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_full" || true
    [[ -s "$sum_nvtx_prep" ]] && python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_prep" || true
    [[ -s "$sum_nvtx_b3" ]] && python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_b3" || true
  fi
done

echo "Index: ${OUT_DIR}/${OUT_BASE}_index.tsv" >&2
echo "Open GUI: nsys-ui ${OUT_DIR}/${OUT_BASE}_run1.nsys-rep" >&2
