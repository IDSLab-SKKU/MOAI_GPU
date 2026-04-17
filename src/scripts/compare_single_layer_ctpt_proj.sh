#!/usr/bin/env bash
set -euo pipefail

# Compare single-layer inference CT×PT-heavy stages between:
# - legacy scalar encode (full slot-vector encode; includes IFFT/NTT work)
# - v2 scalar encode (uniform scalar fast-path; no IFFT/NTT, broadcast-plain in mul/add/sub)
#
# Outputs:
# - output/single_layer_compare/summary.json
# - output/single_layer_compare/compare_total.png
# - output/single_layer_compare/compare_readme.md
#
# Usage:
#   bash src/scripts/compare_single_layer_ctpt_proj.sh
#
# Optional env knobs:
# - MOAI_BUILD_DIR (default: ./build)
# - MOAI_SINGLE_LAYER_OMP_THREADS (forwarded to binary)
# - MOAI_PRECOMPUTED_KEYS_DIR / MOAI_KEYS_BASE / MOAI_ALPHA (forwarded to binary)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${MOAI_BUILD_DIR:-${ROOT}/build}"
OUT_DIR="${ROOT}/output/single_layer_compare"
mkdir -p "${OUT_DIR}"

echo "[compare_single_layer] build dir: ${BUILD_DIR}"
echo "[compare_single_layer] out dir:   ${OUT_DIR}"

cmake --build "${BUILD_DIR}" --target test -j"$(nproc)"

run_once () {
  local label="$1"
  local legacy_scalar="$2" # 0/1
  local log="${OUT_DIR}/run_${label}.txt"
  local csv="${OUT_DIR}/single_layer_results_${label}.csv"

  # test_single_layer.cuh appends to "../single_layer_results.csv" relative to CWD.
  # We run from BUILD_DIR and then copy it out with a unique name.
  rm -f "${BUILD_DIR}/../single_layer_results.csv" || true

  echo "[compare_single_layer] running label=${label} legacy_scalar=${legacy_scalar}"
  (
    cd "${BUILD_DIR}"
    export MOAI_BENCH_MODE=single_layer
    # Keep single-layer config stable and avoid heavy 768-ct bootstrapping.
    # Variant>=1 skips pre-LN bootstrapping and uses internal LN var-branch bootstrap (1 ct),
    # which makes A/B runs much cheaper while preserving CT×PT matmul structure.
    export MOAI_LN_BOOTSTRAP_VARIANT="${MOAI_LN_BOOTSTRAP_VARIANT:-1}"
    # Reduce GPU memory pressure from OMP parallel sections (must be identical for both runs).
    export MOAI_SINGLE_LAYER_OMP_THREADS="${MOAI_SINGLE_LAYER_OMP_THREADS:-1}"
    if [[ "${legacy_scalar}" == "1" ]]; then
      export MOAI_SCALAR_ENCODE_LEGACY_VEC=1
    else
      unset MOAI_SCALAR_ENCODE_LEGACY_VEC || true
    fi
    ./test
  ) |& tee "${log}"

  if [[ ! -f "${BUILD_DIR}/../single_layer_results.csv" ]]; then
    echo "ERROR: expected ${BUILD_DIR}/../single_layer_results.csv was not produced" >&2
    exit 2
  fi
  cp -f "${BUILD_DIR}/../single_layer_results.csv" "${csv}"
  echo "[compare_single_layer] wrote ${csv}"
}

run_once "legacy" "1"
run_once "v2" "0"

python3 - <<'PY'
import csv, json, re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out_dir = root / "output" / "single_layer_compare"

def read_csv(p: Path):
  rows = []
  with p.open("r", encoding="utf-8", newline="") as f:
    r = csv.reader(f)
    for row in r:
      if len(row) < 2: continue
      name = row[0].strip()
      try:
        val = float(row[1])
      except ValueError:
        continue
      rows.append((name, val))
  return rows

def pick_last(rows, key):
  for name, val in reversed(rows):
    if name == key:
      return val
  return None

legacy_csv = out_dir / "single_layer_results_legacy.csv"
v2_csv = out_dir / "single_layer_results_v2.csv"
legacy = read_csv(legacy_csv)
v2 = read_csv(v2_csv)

keys = ["Attention Block", "SelfOutput", "Intermediate Linear", "Final Linear"]
data = {"legacy_csv": str(legacy_csv), "v2_csv": str(v2_csv), "stages": {}}
for k in keys:
  l = pick_last(legacy, k)
  n = pick_last(v2, k)
  data["stages"][k] = {"legacy_s": l, "v2_s": n, "speedup": (l/n) if (l and n) else None}

summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# markdown
md = []
md.append("# Single-layer CT×PT stage comparison (legacy vs v2)\n")
md.append("This report compares a single-layer inference run with identical parameters, differing only in scalar plaintext encoding mode.\n")
md.append("- **legacy**: `MOAI_SCALAR_ENCODE_LEGACY_VEC=1` (scalar encoded via full slot-vector; includes IFFT/NTT work)\n")
md.append("- **v2**: default scalar fast-path (uniform scalar encode + broadcast-plain fast-path)\n")
md.append("\n## Results (seconds)\n")
for k in keys:
  st = data["stages"][k]
  md.append(f"- **{k}**: legacy={st['legacy_s']:.6f}s v2={st['v2_s']:.6f}s speedup={st['speedup']:.3f}x" if st["speedup"] else f"- **{k}**: (missing)")
md.append("\n")
readme_path = out_dir / "compare_readme.md"
readme_path.write_text("\n".join(md), encoding="utf-8")

# plot
import matplotlib.pyplot as plt
labels = keys
legacy_vals = [data["stages"][k]["legacy_s"] for k in keys]
v2_vals = [data["stages"][k]["v2_s"] for k in keys]

x = range(len(labels))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar([i - width/2 for i in x], legacy_vals, width, label="legacy (vec encode)", color="#a5a5a5")
ax.bar([i + width/2 for i in x], v2_vals, width, label="v2 (fast scalar)", color="#4472c4")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Time (s)")
ax.set_title("Single-layer stage times (CT×PT-heavy stages)")
ax.legend(frameon=False)
fig.tight_layout()
png_path = out_dir / "compare_total.png"
fig.savefig(png_path, dpi=170)
plt.close(fig)

print("Wrote:", summary_path)
print("Wrote:", readme_path)
print("Wrote:", png_path)
PY

echo "[compare_single_layer] done."

