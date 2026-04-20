#!/usr/bin/env bash
# Regenerate hybrid_ks_profile.csv and write stacked + engine-compare PNGs.
# From MOAI_GPU repo root:
#   src/scripts/run_hybrid_ks_sweep_plots.sh
#
# Env overrides:
#   MOAI_SIM_HYBRID_KS_ALPHA_RANGE (default 1-35)
#   MOAI_SIM_HYBRID_KS_EXACT_PARTITION (default 1 — skip α where |Ql|≥α and |Ql| mod α≠0; set 0 for full range)
#   MOAI_SIM_HYBRID_KS_PROFILE_CSV

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${MOAI_SIM_HYBRID_KS_ALPHA_RANGE:=1-35}"
: "${MOAI_SIM_HYBRID_KS_EXACT_PARTITION:=1}"

export MOAI_TEST_OUTPUT_DISABLE=1
export MOAI_SIM_BACKEND=1
export MOAI_BENCH_MODE=sim_hybrid_ks_profile
export MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1
export MOAI_SIM_HYBRID_KS_ALPHA_RANGE
export MOAI_SIM_HYBRID_KS_EXACT_PARTITION

./build/test
python3 src/scripts/plot_hybrid_ks_profile.py --all-plots --csv "${MOAI_SIM_HYBRID_KS_PROFILE_CSV:-output/sim/hybrid_ks_profile.csv}"
