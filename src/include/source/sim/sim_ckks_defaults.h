#pragma once

#include <cstddef>
#include <cstdint>

namespace moai {
namespace sim {

// Default CKKS shape for `single_layer_test()` (see `test/test_single_layer.cuh`).
// Primitive-only sim (`test_sim_primitives.cuh`) uses the same N and |coeff_modulus| count by default
// so coarse bytes/cycles match that recipe without setting MOAI_SIM_POLY_DEGREE / MOAI_SIM_NUM_LIMBS.
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
