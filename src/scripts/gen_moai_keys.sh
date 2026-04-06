#!/usr/bin/env bash
# Pre-generate MOAI inference keys (same CKKS params as single_layer_test).
# Output: <base>/keys_dnum_<dnum>/ where dnum follows hybrid KS (alpha = special_modulus_size).
# Usage: ./gen_moai_keys.sh [base_dir] [alpha]
#   alpha defaults to 1, or set MOAI_ALPHA when omitting the second argument.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD="${BUILD_DIR:-build}"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD" -j --target moai_gen_keys
OUT="${1:-$ROOT/keys}"
ALPHA="${2:-${MOAI_ALPHA:-1}}"
mkdir -p "$OUT"
"$BUILD/moai_gen_keys" "$OUT" "$ALPHA"
echo "Done. Keys under: $OUT/keys_dnum_* (see log above)"
