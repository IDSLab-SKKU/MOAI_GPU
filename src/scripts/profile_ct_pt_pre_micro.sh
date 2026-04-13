#!/usr/bin/env bash
# Profile MOAI micro-benchmark: MOAI_BENCH_MODE=ct_pt_pre -> ct_pt_matrix_mul_w_preprocess_test()
# Uses pre-encoded plaintext weights ecd_w and NVTX around ct_pt_matrix_mul(...) only (not W encode).
# Artifacts default to <repo>/output/ct_pt_pre/
#
# NVTX (test_ct_pt_matrix_mul.cuh, MOAI_HAVE_NVTX):
#   moai:ct_pt_pre_encode_w           — scalar loop encoder.encode -> ecd_w (IFFT-heavy)
#   moai:ct_pt_matrix_mul_pre_encoded — ct_pt_matrix_mul only (multiply_plain / rescale, no new encode)
#
# Env: same family as profile_ct_pt_micro.sh (MOAI_BUILD_DIR, MOAI_PROFILE_DIR, MOAI_KERN_EXPORT, …)
#
#   MOAI_CT_PT_PRE_MICRO - default 1: shrink num_X/num_col/col_W in the test so ecd_w fits GPU memory
#                         (768*64 full encodes OOM on typical cards). Set to 0 for full original sizes.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR="${MOAI_BUILD_DIR:-${REPO_ROOT}/build}"
TEST_EXE="${MOAI_TEST_EXE:-${BUILD_DIR}/test}"
OUT_DIR="${MOAI_PROFILE_DIR:-${REPO_ROOT}/output/ct_pt_pre}"
OUT_BASE="${MOAI_PROFILE_OUT:-ct_pt_pre_micro}"
REPEAT="${MOAI_PROFILE_REPEAT:-1}"
TRACE="${MOAI_NSYS_TRACE:-cuda,nvtx,osrt}"
FILTER_MUL="${MOAI_NSYS_STATS_FILTER_NVTX:-1}"
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

echo -e "run\tnsys_rep\tkern_sum_tsv\tparse_full\tparse_nvtx_encode_w\tparse_nvtx_mul" >"${OUT_DIR}/${OUT_BASE}_index.tsv"

for ((i = 1; i <= REPEAT; i++)); do
  stem="${OUT_DIR}/${OUT_BASE}_run${i}"
  rep="${stem}.nsys-rep"
  sum_full="${stem}.kern_sum.tsv"
  sum_nvtx_enc="${stem}.kern_sum.nvtx_encode_w.tsv"
  sum_nvtx_mul="${stem}.kern_sum.pre_enc_nvtx.tsv"
  log="${stem}.stdout.log"

  echo "=== run $i: nsys profile -> $rep ===" >&2
  set +e
  (
    cd "$BUILD_DIR"
    export MOAI_BENCH_MODE=ct_pt_pre
    export MOAI_CT_PT_PRE_MICRO="${MOAI_CT_PT_PRE_MICRO:-1}"
    exec nsys profile --force-overwrite=true -o "$stem" --trace="$TRACE" "$TEST_EXE"
  ) >"$log" 2>&1
  ec=$?
  set -e
  if [[ $ec -ne 0 ]]; then
    echo "error: nsys profile failed (exit $ec); see $log" >&2
    cp "$log" "${OUT_DIR}/${OUT_BASE}_last_fail.log"
    exit "$ec"
  fi

  # Stale ${stem}.sqlite causes "does not contain NVTX data" for --filter-nvtx / nvtx_sum.
  rm -f "${stem}.sqlite"

  nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv "$rep" >"$sum_full"

  parse_full=""
  if parse_out=$(python3 "$PARSE_PY" "$sum_full"); then
    parse_full=$(echo "$parse_out" | tr '\n' ';')
  else
    parse_full="parse_failed"
  fi

  parse_enc="n/a"
  parse_mul="n/a"
  if [[ "$FILTER_MUL" == "1" ]]; then
    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:ct_pt_pre_encode_w" "$rep" >"$sum_nvtx_enc" 2>"$tmp"
    fe=$?
    set -e
    if [[ $fe -eq 0 ]] && python3 "$PARSE_PY" "$sum_nvtx_enc" &>/dev/null; then
      parse_enc=$(python3 "$PARSE_PY" "$sum_nvtx_enc" | tr '\n' ';')
    else
      parse_enc="nvtx_filter_unavailable_or_empty"
      [[ -s "$tmp" ]] && head -3 "$tmp" >&2
    fi

    set +e
    nsys stats --report cuda_gpu_kern_sum --force-export=true --format tsv \
      --filter-nvtx "moai:ct_pt_matrix_mul_pre_encoded" "$rep" >"$sum_nvtx_mul" 2>"$tmp"
    fm=$?
    set -e
    if [[ $fm -eq 0 ]] && python3 "$PARSE_PY" "$sum_nvtx_mul" &>/dev/null; then
      parse_mul=$(python3 "$PARSE_PY" "$sum_nvtx_mul" | tr '\n' ';')
    else
      parse_mul="nvtx_filter_unavailable_or_empty"
      [[ -s "$tmp" ]] && head -3 "$tmp" >&2
    fi
  fi

  echo -e "${i}\t${rep}\t${sum_full}\t${parse_full}\t${parse_enc}\t${parse_mul}" >>"${OUT_DIR}/${OUT_BASE}_index.tsv"
  echo "  wrote $rep ; $sum_full" >&2

  if [[ "$KERN_EXPORT" == "1" ]] && [[ -f "$EXPORT_PY" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_full" || true
      if python3 "$PARSE_PY" "$sum_nvtx_enc" &>/dev/null; then
        python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_enc" || true
      fi
      if python3 "$PARSE_PY" "$sum_nvtx_mul" &>/dev/null; then
        python3 "$EXPORT_PY" --out-dir "$OUT_DIR" "$sum_nvtx_mul" || true
      fi
    fi
  fi
done

echo "Index: ${OUT_DIR}/${OUT_BASE}_index.tsv" >&2
echo "Open GUI: nsys-ui ${OUT_DIR}/${OUT_BASE}_run1.nsys-rep" >&2
