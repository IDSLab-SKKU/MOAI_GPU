#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "source/sim/engine_config.h"

namespace moai {
namespace sim {

// Extremely lightweight "backend" for early bring-up:
// - Controlled by env var MOAI_SIM_BACKEND=1
// - Accumulates coarse counters (bytes + "cycles")
// - Does not attempt correctness; intended for estimator-style runs.
struct SimTiming {
  enum class OpKind : uint8_t { Encode, MultiplyPlain, AddInplace, Rescale };

  struct OpStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> cycles{0};
    std::atomic<uint64_t> bytes{0};
  };

  OpStats encode{};
  OpStats encode_vec{};
  OpStats deep_copy_cipher{};
  OpStats ntt_fwd{};
  OpStats pt_mul{};
  OpStats ntt_inv{};
  OpStats add_inplace{};
  OpStats rescale{};
  OpStats ct_ct_mul{};

  std::atomic<uint64_t> cycles_total{0};
  std::atomic<uint64_t> bytes_total{0};

  // Coverage-gap counters: when sim is enabled, these indicate which Phantom ops
  // are still being invoked (i.e., not yet modeled / replaced).
  std::atomic<uint64_t> gap_encrypt_calls{0};
  std::atomic<uint64_t> gap_decrypt_calls{0};
  std::atomic<uint64_t> gap_rotate_calls{0};
  std::atomic<uint64_t> gap_relin_calls{0};
  std::atomic<uint64_t> gap_modswitch_calls{0};

  static SimTiming &instance() {
    static SimTiming inst;
    return inst;
  }

  static bool enabled() {
    if (const char *v = std::getenv("MOAI_SIM_BACKEND"); v && v[0] != '\0') {
      return std::strcmp(v, "0") != 0;
    }
    return false;
  }

  static bool strict() {
    if (const char *v = std::getenv("MOAI_SIM_STRICT"); v && v[0] != '\0')
      return std::strcmp(v, "0") != 0;
    return false;
  }

  enum class GapPolicy : uint8_t { Skip, Error, Model };

  static GapPolicy gap_policy() {
    // Default: skip (warn once); strict mode still aborts.
    const char *v = std::getenv("MOAI_SIM_GAP_POLICY");
    if (!v || v[0] == '\0') return GapPolicy::Skip;
    if (std::strcmp(v, "error") == 0) return GapPolicy::Error;
    if (std::strcmp(v, "model") == 0) return GapPolicy::Model;
    if (std::strcmp(v, "skip") == 0) return GapPolicy::Skip;
    return GapPolicy::Skip;
  }

  void record_gap(const char *name, std::atomic<uint64_t> &ctr) {
    ctr.fetch_add(1, std::memory_order_relaxed);
    // Print only on first occurrence to avoid spam.
    if (ctr.load(std::memory_order_relaxed) == 1) {
      std::cerr << "[MOAI_SIM_BACKEND][COVERAGE_GAP] " << name
                << " was called while MOAI_SIM_BACKEND=1\n";
    }
    if (gap_policy() == GapPolicy::Error) {
      std::cerr << "[MOAI_SIM_BACKEND][COVERAGE_GAP] gap policy=error; aborting.\n";
      std::abort();
    }
    if (strict()) {
      std::cerr << "[MOAI_SIM_BACKEND][COVERAGE_GAP] strict mode enabled; aborting.\n";
      std::abort();
    }
  }

  static uint64_t env_u64(const char *name, uint64_t def) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') return def;
    char *end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v) return def;
    return static_cast<uint64_t>(x);
  }

  // Model knobs (defaults are intentionally simple / conservative).
  // Users can override from env without recompiling.
  uint64_t encode_cycles_per_call() const { return env_u64("MOAI_SIM_ENC_CYC", 5000); }
  uint64_t encode_vec_cycles_per_slot() const { return env_u64("MOAI_SIM_ENC_VEC_CYC_PER_SLOT", 1); }
  // Legacy scalars (EngineModel uses pipeline+lanes; see estimate_* in engine_config.h).
  uint64_t vec_cycles_per_coeff_muladd() const { return env_u64("MOAI_SIM_VEC_CYC_PER_COEFF", 2); }
  uint64_t ntt_cycles_per_coeff() const { return env_u64("MOAI_SIM_NTT_CYC_PER_COEFF", 1); }
  uint64_t rescale_cycles_per_coeff() const { return env_u64("MOAI_SIM_RESCALE_CYC_PER_COEFF", 1); }
  uint64_t d2d_copy_cycles_per_byte() const { return env_u64("MOAI_SIM_D2D_CYC_PER_B", 1); }

  // If you want "no-NTT" steady component, set MOAI_SIM_NTT_STEADY_CYC_PER_COEFF=0 (and pipe depth 0).
  void record_encode() {
    const uint64_t c = encode_cycles_per_call();
    encode.calls.fetch_add(1, std::memory_order_relaxed);
    encode.cycles.fetch_add(c, std::memory_order_relaxed);
    cycles_total.fetch_add(c, std::memory_order_relaxed);
  }

  // Legacy-style scalar encode via full slot vector encode(values[slot_count])
  void record_encode_vec(uint64_t slot_count) {
    const uint64_t c = encode_vec_cycles_per_slot() * slot_count;
    encode_vec.calls.fetch_add(1, std::memory_order_relaxed);
    encode_vec.cycles.fetch_add(c, std::memory_order_relaxed);
    cycles_total.fetch_add(c, std::memory_order_relaxed);
  }

  void record_deep_copy_cipher(uint64_t ct_size, uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = ct_size * poly_degree * limbs;
    const uint64_t b = coeffs * sizeof(uint64_t);
    const uint64_t c = d2d_copy_cycles_per_byte() * b;

    deep_copy_cipher.calls.fetch_add(1, std::memory_order_relaxed);
    deep_copy_cipher.cycles.fetch_add(c, std::memory_order_relaxed);
    deep_copy_cipher.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void record_ntt_fwd(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t c = estimate_ntt_cycles(coeffs, poly_degree);
    const uint64_t b = 2ULL * coeffs * sizeof(uint64_t);

    ntt_fwd.calls.fetch_add(1, std::memory_order_relaxed);
    ntt_fwd.cycles.fetch_add(c, std::memory_order_relaxed);
    ntt_fwd.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void record_pt_mul(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t c = estimate_vec_mul_cycles(coeffs);
    const uint64_t b = 3ULL * coeffs * sizeof(uint64_t);

    pt_mul.calls.fetch_add(1, std::memory_order_relaxed);
    pt_mul.cycles.fetch_add(c, std::memory_order_relaxed);
    pt_mul.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void record_ntt_inv(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t c = estimate_ntt_cycles(coeffs, poly_degree);
    const uint64_t b = 2ULL * coeffs * sizeof(uint64_t);

    ntt_inv.calls.fetch_add(1, std::memory_order_relaxed);
    ntt_inv.cycles.fetch_add(c, std::memory_order_relaxed);
    ntt_inv.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  // Roughly: multiply_plain involves NTT-ish transforms and pointwise mul on (N * limbs).
  // We don't know the exact kernel mix here; this is a coarse proxy.
  void record_multiply_plain(uint64_t poly_degree, uint64_t limbs) {
    // Decompose into sub-ops so the output looks like kernel summaries.
    record_ntt_fwd(poly_degree, limbs);
    record_pt_mul(poly_degree, limbs);
    record_ntt_inv(poly_degree, limbs);
  }

  // CT×CT multiply (coarse): two RLWE inputs + one output @ ct_size=2; cycles from estimate_ct_ct_multiply_cycles.
  void record_ct_ct_multiply(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t ct_size = 2;
    const uint64_t bytes_ct = ct_size * coeffs * sizeof(uint64_t);
    const uint64_t c = estimate_ct_ct_multiply_cycles(coeffs, poly_degree);
    const uint64_t b = 3ULL * bytes_ct;

    ct_ct_mul.calls.fetch_add(1, std::memory_order_relaxed);
    ct_ct_mul.cycles.fetch_add(c, std::memory_order_relaxed);
    ct_ct_mul.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void record_add_inplace(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const EngineModelConfig cfg = EngineModelConfig::from_env();
    const uint64_t c = estimate_vec_pipeline_cycles(coeffs, cfg.vec_add_cyc_per_coeff);
    const uint64_t b = 3ULL * coeffs * sizeof(uint64_t);

    add_inplace.calls.fetch_add(1, std::memory_order_relaxed);
    add_inplace.cycles.fetch_add(c, std::memory_order_relaxed);
    add_inplace.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void record_rescale(uint64_t poly_degree, uint64_t limbs) {
    const uint64_t coeffs = poly_degree * limbs;
    const uint64_t c = estimate_vec_pipeline_cycles(coeffs, rescale_cycles_per_coeff());
    const uint64_t b = 2ULL * coeffs * sizeof(uint64_t);

    rescale.calls.fetch_add(1, std::memory_order_relaxed);
    rescale.cycles.fetch_add(c, std::memory_order_relaxed);
    rescale.bytes.fetch_add(b, std::memory_order_relaxed);

    cycles_total.fetch_add(c, std::memory_order_relaxed);
    bytes_total.fetch_add(b, std::memory_order_relaxed);
  }

  void reset() {
    encode.calls.store(0, std::memory_order_relaxed);
    encode.cycles.store(0, std::memory_order_relaxed);
    encode.bytes.store(0, std::memory_order_relaxed);

    encode_vec.calls.store(0, std::memory_order_relaxed);
    encode_vec.cycles.store(0, std::memory_order_relaxed);
    encode_vec.bytes.store(0, std::memory_order_relaxed);

    deep_copy_cipher.calls.store(0, std::memory_order_relaxed);
    deep_copy_cipher.cycles.store(0, std::memory_order_relaxed);
    deep_copy_cipher.bytes.store(0, std::memory_order_relaxed);

    ntt_fwd.calls.store(0, std::memory_order_relaxed);
    ntt_fwd.cycles.store(0, std::memory_order_relaxed);
    ntt_fwd.bytes.store(0, std::memory_order_relaxed);

    pt_mul.calls.store(0, std::memory_order_relaxed);
    pt_mul.cycles.store(0, std::memory_order_relaxed);
    pt_mul.bytes.store(0, std::memory_order_relaxed);

    ntt_inv.calls.store(0, std::memory_order_relaxed);
    ntt_inv.cycles.store(0, std::memory_order_relaxed);
    ntt_inv.bytes.store(0, std::memory_order_relaxed);

    add_inplace.calls.store(0, std::memory_order_relaxed);
    add_inplace.cycles.store(0, std::memory_order_relaxed);
    add_inplace.bytes.store(0, std::memory_order_relaxed);

    rescale.calls.store(0, std::memory_order_relaxed);
    rescale.cycles.store(0, std::memory_order_relaxed);
    rescale.bytes.store(0, std::memory_order_relaxed);

    ct_ct_mul.calls.store(0, std::memory_order_relaxed);
    ct_ct_mul.cycles.store(0, std::memory_order_relaxed);
    ct_ct_mul.bytes.store(0, std::memory_order_relaxed);

    cycles_total.store(0, std::memory_order_relaxed);
    bytes_total.store(0, std::memory_order_relaxed);

    gap_encrypt_calls.store(0, std::memory_order_relaxed);
    gap_decrypt_calls.store(0, std::memory_order_relaxed);
    gap_rotate_calls.store(0, std::memory_order_relaxed);
    gap_relin_calls.store(0, std::memory_order_relaxed);
    gap_modswitch_calls.store(0, std::memory_order_relaxed);
  }

  void print_summary(std::ostream &os) const {
    const auto load = [](const std::atomic<uint64_t> &v) { return v.load(std::memory_order_relaxed); };
    const uint64_t ctot = load(cycles_total);
    const uint64_t btot = load(bytes_total);
    os << "[MOAI_SIM_BACKEND] total cycles=" << ctot << " bytes=" << fmt_bytes_iec(btot) << " (" << btot << ")\n";

    auto row = [&](const char *name, const OpStats &s) {
      const uint64_t calls = load(s.calls);
      const uint64_t cyc = load(s.cycles);
      const uint64_t byt = load(s.bytes);
      const uint64_t cyc_per = calls ? (cyc / calls) : 0;
      const uint64_t byt_per = calls ? (byt / calls) : 0;
      os << "[MOAI_SIM_BACKEND] "
         << std::left << std::setw(24) << name
         << std::right << std::setw(10) << calls
         << std::setw(14) << cyc
         << std::setw(14) << fmt_bytes_iec(byt)
         << std::setw(14) << cyc_per
         << std::setw(14) << fmt_bytes_iec(byt_per)
         << "\n";
    };

    os << "[MOAI_SIM_BACKEND] "
       << std::left << std::setw(24) << "op"
       << std::right << std::setw(10) << "calls"
       << std::setw(14) << "cycles"
       << std::setw(14) << "bytes"
       << std::setw(14) << "cyc/call"
       << std::setw(14) << "bytes/call"
       << "\n";
    row("encode_scalar_uniform", encode);
    row("encode_scalar_legacy_vec", encode_vec);
    row("deep_copy_cipher_d2d", deep_copy_cipher);
    row("ntt_fwd_inplace", ntt_fwd);
    row("pt_mul_pointwise", pt_mul);
    row("ntt_inv_inplace", ntt_inv);
    row("add_inplace", add_inplace);
    row("rescale", rescale);
    row("ct_ct_multiply", ct_ct_mul);

    const uint64_t ge = load(gap_encrypt_calls);
    const uint64_t gd = load(gap_decrypt_calls);
    const uint64_t gr = load(gap_rotate_calls);
    const uint64_t gl = load(gap_relin_calls);
    const uint64_t gm = load(gap_modswitch_calls);
    if (ge || gd || gr || gl || gm) {
      os << "[MOAI_SIM_BACKEND] coverage_gaps"
         << " encrypt=" << ge
         << " decrypt=" << gd
         << " rotate=" << gr
         << " relinearize=" << gl
         << " modswitch=" << gm
         << "\n";
    }
  }
};

}  // namespace sim
}  // namespace moai

