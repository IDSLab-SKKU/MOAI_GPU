#pragma once

#include "source/sim/prng_evalkey_a.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace moai {
namespace sim {

inline void sim_prng_evalkey_a_unit_tests() {
  // Determinism test
  PrngGenerationDescriptor d{};
  for (size_t i = 0; i < d.master_seed.size(); ++i) d.master_seed[i] = static_cast<uint8_t>(i);
  d.key_id = 1;
  d.decomp_id = 2;
  d.limb_id = 3;
  d.poly_id = 4;
  d.lane_id = 0;
  d.num_coeffs = 256;

  const uint64_t q = 12289;  // small prime (NTT-friendly)
  std::vector<uint64_t> a1, a2, a3;
  RejectStats st1{}, st2{};
  shake128_uniform_poly(q, d, a1, &st1);
  shake128_uniform_poly(q, d, a2, &st2);
  assert(a1 == a2);

  // Domain separation: change one field => different output (with overwhelming probability)
  PrngGenerationDescriptor d2 = d;
  d2.limb_id ^= 1;
  shake128_uniform_poly(q, d2, a3, nullptr);
  assert(a1 != a3);

  // Range
  for (uint64_t v : a1) assert(v < q);

  // Reject sampling sanity: accepts == num_coeffs, rejects >= 0, words >= accepts
  assert(st1.accepts == d.num_coeffs);
  assert(st1.words >= st1.accepts);

  std::cout << "[MOAI_SIM_PRNG_EVK_A_TEST] determinism/range/reject tests passed. "
            << "words=" << st1.words << " rejects=" << st1.rejects << "\n";
}

inline void sim_prng_evalkey_a_microbench() {
  sim_prng_evalkey_a_unit_tests();

  PrngTimingConfig cfg;
  // Override knobs via env would be nicer, but keep microbench simple and self-contained.
  PrngEvalKeyAGenerator gen;
  gen.reset(cfg);

  const uint64_t coeffs = 65536;  // one poly
  const uint64_t t0 = 0;
  (void)gen.request(t0, coeffs, /*accept_prob_hint=*/1.0);

  // Consume in chunks to emulate downstream.
  uint64_t t = 0;
  uint64_t remain = coeffs;
  while (remain) {
    const uint64_t c = std::min<uint64_t>(cfg.chunk_size, remain);
    t = gen.consume(t, c);
    remain -= c;
  }

  const PrngStats &st = gen.stats();
  const uint64_t key_bytes_baseline = coeffs * 8ULL;        // one limb poly, bytes
  const uint64_t key_bytes_saved_est = key_bytes_baseline;  // model: 'a' fully generated
  std::cout << "[MOAI_SIM_PRNG_EVK_A_BENCH] coeffs=" << coeffs
            << " chunk=" << cfg.chunk_size
            << " bytes_saved_est=" << key_bytes_saved_est
            << " stall_cycles_due_to_prng_unavailability=" << st.stall_cycles_due_to_prng_unavailability
            << " first_request=" << st.first_request_cycle
            << " first_needed=" << st.first_needed_cycle
            << " first_ready=" << st.first_ready_cycle
            << " last_ready=" << st.last_ready_cycle
            << "\n";
}

}  // namespace sim
}  // namespace moai

