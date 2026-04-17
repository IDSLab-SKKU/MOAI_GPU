#!/usr/bin/env bash
set -euo pipefail

# Bench CT×PT MatMul projections (QKV/out/FC1/FC2) under single-layer CKKS params,
# comparing legacy scalar-encode (slot-vector) vs v2 scalar-encode (fast-path).
#
# Outputs under:
#   output/ct_pt_proj_compare/
#     run_<op>_<mode>.txt
#     summary.json
#
# Plotting is intentionally separated (optional):
#   python3 src/scripts/plot_ct_pt_proj_compare.py
#
# Usage:
#   bash src/scripts/bench_ct_pt_proj_compare.sh
#
# Optional env knobs:
# - MOAI_CT_PT_PROJ_MODE=legacy|v2|both   (default: both)
# - MOAI_CT_PT_PROJ_OPS="qkv,out,fc1,fc2" (default: all four)
# - FC2: MOAI_CT_PT_FC2_MODE=full_vram|chunk_vram|stream (default: chunk_vram)
# - FC2: MOAI_CT_PT_FC2_CHUNK=<n> (used when mode=chunk_vram)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MOAI_BUILD_DIR:-${ROOT}/build}"
OUT_DIR="${ROOT}/output/ct_pt_proj_compare"
mkdir -p "${OUT_DIR}"

cmake --build "${BUILD_DIR}" --target test -j"$(nproc)"

run_one () {
  local op="$1"      # qkv|out|fc1|fc2
  local mode="$2"    # legacy|v2
  local legacy="$3"  # 0/1
  local log="${OUT_DIR}/run_${op}_${mode}.txt"
  echo "[ct_pt_proj_compare] run op=${op} mode=${mode} legacy=${legacy}"
  (
    cd "${BUILD_DIR}"
    export MOAI_BENCH_MODE=ct_pt_proj_compare
    export MOAI_CT_PT_PROJ_OP="${op}"
    # Keep stdout on terminal so run logs include timings; otherwise test.cu redirects stdout
    # under build/output/test_logs/ and only prints a short stderr notice.
    export MOAI_TEST_OUTPUT_DISABLE=1
    if [[ "${legacy}" == "1" ]]; then
      export MOAI_SCALAR_ENCODE_LEGACY_VEC=1
    else
      unset MOAI_SCALAR_ENCODE_LEGACY_VEC || true
    fi
    ./test
  ) |& tee "${log}"
}

mode="${MOAI_CT_PT_PROJ_MODE:-both}"
case "${mode}" in
  legacy|v2|both) ;;
  *) echo "ERROR: MOAI_CT_PT_PROJ_MODE must be legacy|v2|both (got '${mode}')" >&2; exit 2 ;;
esac

ops_csv="${MOAI_CT_PT_PROJ_OPS:-qkv,out,fc1,fc2}"
IFS=',' read -r -a ops <<< "${ops_csv}"

for op in "${ops[@]}"; do
  op="$(echo "${op}" | tr '[:upper:]' '[:lower:]' | xargs)"
  [[ -z "${op}" ]] && continue
  if [[ "${mode}" == "legacy" || "${mode}" == "both" ]]; then
    run_one "${op}" "legacy" "1"
  fi
  if [[ "${mode}" == "v2" || "${mode}" == "both" ]]; then
    run_one "${op}" "v2" "0"
  fi
done

MOAI_GPU_REPO_ROOT="${ROOT}" python3 - <<'PY'
import json, re
from pathlib import Path
import os

root = Path(os.environ.get("MOAI_GPU_REPO_ROOT", os.getcwd())).resolve()
out = root / "output" / "ct_pt_proj_compare"
out.mkdir(parents=True, exist_ok=True)

def parse_run(p: Path):
  txt = p.read_text(encoding="utf-8", errors="ignore")
  # lines like: [CT_PT_PROJ] fc1 time_s=...
  pat = re.compile(r"\[CT_PT_PROJ\]\s+([A-Za-z0-9_]+)\s+time_s=([0-9.]+)")
  d = {}
  for m in pat.finditer(txt):
    d[m.group(1)] = float(m.group(2))
  # QKV_proj total line
  m = re.search(r"\[CT_PT_PROJ\]\s+QKV_proj\s+time_s=([0-9.]+)", txt)
  if m: d["QKV_proj"] = float(m.group(1))
  return d

ops = ["qkv","out","fc1","fc2"]
modes = ["legacy","v2"]
data = {m: {} for m in modes}
for op in ops:
  for m in modes:
    p = out / f"run_{op}_{m}.txt"
    data[m][op] = parse_run(p) if p.exists() else {}

proj_key = {"qkv": "QKV_proj", "out": "out_proj", "fc1": "fc1", "fc2": "fc2_compute"}
meta = {
  # Must match the projection bench config in C++ (ct_pt_proj_matmul_bench_single_layer_compare).
  # num_row is fixed at 128; num_X is configured to fully pack sparse_slots: 256*128=32768.
  "num_X": 256,
  "num_row": 128,
}
summary = {"meta": meta, "runs": data, "speedup": {}}
for op in ops:
  l = data["legacy"][op].get(proj_key[op])
  n = data["v2"][op].get(proj_key[op])
  summary["speedup"][op] = (l / n) if (l and n) else None

(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Wrote", out / "summary.json")
PY

echo "[ct_pt_proj_compare] done: ${OUT_DIR}"

