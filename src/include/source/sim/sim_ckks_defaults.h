#pragma once

#include <cstddef>
#include <cstdint>

namespace moai {
namespace sim {

// Default CKKS shape for `single_layer_test()` (see `test/test_single_layer.cuh`).
// `kSingleLayerCoeffModulusCount()` is the **full** Phantom `coeff_modulus` prime count **|QP|**
// (data Q + special P). Estimator primitives treat MOAI_SIM_NUM_LIMBS as |QP| by default and use
// |Ql| = |QP| − MOAI_SIM_ALPHA for coarse ops (`sim_effective_rns_limbs_for_ct`); set
// MOAI_SIM_NUM_LIMBS_COUNTS_QP=0 if MOAI_SIM_NUM_LIMBS is already |Ql|.
//
// coeff_bit_vec layout there: [logq] + logp * remaining_level + logq * boot_level + [log_special_prime]
// => chain length = 1 + remaining_level + boot_level + 1.
inline constexpr int kSingleLayerLogN = 16;
inline constexpr int kSingleLayerRemainingLevel = 20;
inline constexpr int kSingleLayerBootLevel = 14;

inline constexpr std::size_t kSingleLayerCoeffModulusCount() {
  return static_cast<std::size_t>(1) + static_cast<std::size_t>(kSingleLayerRemainingLevel) +
         static_cast<std::size_t>(kSingleLayerBootLevel) + static_cast<std::size_t>(1);
}

inline constexpr uint64_t kSingleLayerPolyModulusDegree() {
  return 1ULL << static_cast<unsigned>(kSingleLayerLogN);
}

}  // namespace sim
}  // namespace moai
