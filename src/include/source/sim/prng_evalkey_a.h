#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace moai {
namespace sim {

// -------------------------
// Minimal SHAKE128 (Keccak-f[1600]) for deterministic simulator streams.
// Not tuned for performance; intended for reproducible PRNG streams + correct rejection sampling.
// -------------------------
namespace shake128_detail {

inline uint64_t rotl64(uint64_t x, uint32_t n) { return (x << n) | (x >> (64 - n)); }

inline uint64_t load64_le(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (static_cast<uint64_t>(p[i]) << (8 * i));
  return v;
}

inline void store64_le(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
}

inline void keccak_f1600(uint64_t s[25]) {
  static constexpr uint64_t RC[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
  };
  static constexpr uint8_t R[25] = {
      0,  1, 62, 28, 27,
      36, 44,  6, 55, 20,
      3, 10, 43, 25, 39,
      41, 45, 15, 21,  8,
      18,  2, 61, 56, 14,
  };
  static constexpr uint8_t PI[25] = {
      0, 10, 20,  5, 15,
     16,  1, 11, 21,  6,
      7, 17,  2, 12, 22,
     23,  8, 18,  3, 13,
     14, 24,  9, 19,  4,
  };

  for (int round = 0; round < 24; ++round) {
    // Theta
    uint64_t C[5];
    for (int x = 0; x < 5; ++x) C[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
    uint64_t D[5];
    for (int x = 0; x < 5; ++x) D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
    for (int i = 0; i < 25; i += 5) {
      s[i + 0] ^= D[0];
      s[i + 1] ^= D[1];
      s[i + 2] ^= D[2];
      s[i + 3] ^= D[3];
      s[i + 4] ^= D[4];
    }

    // Rho+Pi
    uint64_t B[25];
    for (int i = 0; i < 25; ++i) B[PI[i]] = rotl64(s[i], R[i]);

    // Chi
    for (int y = 0; y < 5; ++y) {
      const int o = 5 * y;
      const uint64_t b0 = B[o + 0], b1 = B[o + 1], b2 = B[o + 2], b3 = B[o + 3], b4 = B[o + 4];
      s[o + 0] = b0 ^ ((~b1) & b2);
      s[o + 1] = b1 ^ ((~b2) & b3);
      s[o + 2] = b2 ^ ((~b3) & b4);
      s[o + 3] = b3 ^ ((~b4) & b0);
      s[o + 4] = b4 ^ ((~b0) & b1);
    }

    // Iota
    s[0] ^= RC[round];
  }
}

class Shake128 {
 public:
  static constexpr size_t kRateBytes = 168;  // SHAKE128 rate = 1344 bits

  Shake128() { reset(); }

  void reset() {
    std::memset(m_state.data(), 0, sizeof(uint64_t) * m_state.size());
    m_pos = 0;
    m_finalized = false;
  }

  void absorb(const uint8_t *in, size_t len) {
    if (m_finalized) return;
    while (len) {
      const size_t take = std::min(len, kRateBytes - m_pos);
      for (size_t i = 0; i < take; ++i) {
        const size_t idx = m_pos + i;
        const size_t lane = idx / 8;
        const size_t off = idx % 8;
        m_state[lane] ^= (static_cast<uint64_t>(in[i]) << (8 * off));
      }
      m_pos += take;
      in += take;
      len -= take;
      if (m_pos == kRateBytes) {
        keccak_f1600(m_state.data());
        m_pos = 0;
      }
    }
  }

  void finalize() {
    if (m_finalized) return;
    // Domain separation for SHAKE: 0x1F, then 0x80 at end of rate.
    {
      const size_t idx = m_pos;
      const size_t lane = idx / 8;
      const size_t off = idx % 8;
      m_state[lane] ^= (static_cast<uint64_t>(0x1F) << (8 * off));
    }
    {
      const size_t idx = kRateBytes - 1;
      const size_t lane = idx / 8;
      const size_t off = idx % 8;
      m_state[lane] ^= (static_cast<uint64_t>(0x80) << (8 * off));
    }
    keccak_f1600(m_state.data());
    m_pos = 0;
    m_finalized = true;
  }

  void squeeze(uint8_t *out, size_t len) {
    if (!m_finalized) finalize();
    while (len) {
      if (m_pos == kRateBytes) {
        keccak_f1600(m_state.data());
        m_pos = 0;
      }
      const size_t take = std::min(len, kRateBytes - m_pos);
      // Export bytes from state
      for (size_t i = 0; i < take; ++i) {
        const size_t idx = m_pos + i;
        const size_t lane = idx / 8;
        const size_t off = idx % 8;
        out[i] = static_cast<uint8_t>((m_state[lane] >> (8 * off)) & 0xFF);
      }
      out += take;
      len -= take;
      m_pos += take;
    }
  }

 private:
  std::array<uint64_t, 25> m_state{};
  size_t m_pos = 0;
  bool m_finalized = false;
};

}  // namespace shake128_detail

// -------------------------
// Descriptor / domain separation
// -------------------------
enum class PrngOutputFormat : uint32_t {
  NTT_DIRECT_NATURAL = 0,
};

struct PrngGenerationDescriptor {
  std::array<uint8_t, 32> master_seed{};  // 256-bit master seed
  uint64_t key_id = 0;
  uint64_t decomp_id = 0;
  uint64_t limb_id = 0;
  uint64_t poly_id = 0;
  uint64_t lane_id = 0;
  uint64_t num_coeffs = 0;
  PrngOutputFormat output_format = PrngOutputFormat::NTT_DIRECT_NATURAL;
};

inline void append_le_u64(std::vector<uint8_t> &buf, uint64_t x) {
  for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
}
inline void append_le_u32(std::vector<uint8_t> &buf, uint32_t x) {
  for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
}

inline std::vector<uint8_t> serialize_descriptor(const PrngGenerationDescriptor &d) {
  std::vector<uint8_t> b;
  b.reserve(32 + 8 * 7 + 4 + 8);
  b.insert(b.end(), d.master_seed.begin(), d.master_seed.end());
  append_le_u64(b, d.key_id);
  append_le_u64(b, d.decomp_id);
  append_le_u64(b, d.limb_id);
  append_le_u64(b, d.poly_id);
  append_le_u64(b, d.lane_id);
  append_le_u64(b, d.num_coeffs);
  append_le_u32(b, static_cast<uint32_t>(d.output_format));
  // Context tag for domain separation across uses inside simulator.
  const char tag[] = "MOAI_GPU::PrngEvalKeyA::SHAKE128";
  b.insert(b.end(), tag, tag + sizeof(tag) - 1);
  return b;
}

// -------------------------
// Reject sampling (correct modulo-bias-free uniform in [0,q))
// -------------------------
struct RejectStats {
  uint64_t blocks = 0;
  uint64_t words = 0;
  uint64_t accepts = 0;
  uint64_t rejects = 0;
};

inline uint64_t reject_threshold_u64(uint64_t q) {
  // T = floor(2^64 / q) * q. Implemented as: T = (UINT64_MAX / q) * q, since 2^64-1 is max.
  // We want 2^64, not 2^64-1. Using UINT64_MAX biases threshold slightly; fix by using 128-bit division.
  // Compute floor(2^64 / q) via (__uint128_t(1)<<64)/q.
  const __uint128_t two64 = (static_cast<__uint128_t>(1) << 64);
  const __uint128_t k = two64 / static_cast<__uint128_t>(q);
  return static_cast<uint64_t>(k * static_cast<__uint128_t>(q));
}

inline uint64_t shake128_reject_sample_u64(shake128_detail::Shake128 &xof, uint64_t q, RejectStats &st) {
  // Consume 64-bit words until x < T, then return x % q.
  const uint64_t T = reject_threshold_u64(q);
  std::array<uint8_t, 8> buf{};
  for (;;) {
    xof.squeeze(buf.data(), buf.size());
    ++st.words;
    const uint64_t x = shake128_detail::load64_le(buf.data());
    if (x < T) {
      ++st.accepts;
      return (q == 0) ? 0 : (x % q);
    }
    ++st.rejects;
  }
}

inline void shake128_uniform_poly(uint64_t q,
                                  const PrngGenerationDescriptor &desc,
                                  std::vector<uint64_t> &out_coeffs,
                                  RejectStats *out_stats = nullptr) {
  out_coeffs.clear();
  out_coeffs.reserve(static_cast<size_t>(desc.num_coeffs));
  RejectStats st{};

  shake128_detail::Shake128 xof;
  const std::vector<uint8_t> seed = serialize_descriptor(desc);
  xof.absorb(seed.data(), seed.size());
  xof.finalize();

  // Count blocks at 168B boundaries (approx): squeeze is byte-granular; we track blocks conservatively.
  // For simulator stats, we'll approximate blocks = ceil(bytes_squeezed/168).
  const uint64_t T = reject_threshold_u64(q);
  (void)T;

  for (uint64_t i = 0; i < desc.num_coeffs; ++i) {
    const uint64_t v = shake128_reject_sample_u64(xof, q, st);
    out_coeffs.push_back(v);
  }

  if (out_stats) *out_stats = st;
}

// -------------------------
// Timing model scaffolding (FIFO/backpressure/chunk)
// -------------------------
struct PrngTimingConfig {
  // SHAKE lanes produce 168B blocks into bit FIFO.
  uint64_t num_prng_lanes = 4;
  uint64_t num_sampler_lanes = 4;
  uint64_t shake_startup_cycles = 50;
  uint64_t shake_block_cycles = 4;  // per lane per block
  uint64_t bit_fifo_capacity_blocks = 8;  // per lane
  uint64_t coeff_fifo_capacity = 256;      // global
  uint64_t chunk_size = 32;                // emit granularity

  // Seed/metadata DMA model (bytes read when issuing a request).
  uint64_t seed_metadata_bytes = 64;
};

struct PrngStats {
  uint64_t blocks_produced = 0;
  uint64_t words_consumed = 0;
  uint64_t accepts = 0;
  uint64_t rejects = 0;
  uint64_t coeffs_emitted = 0;
  uint64_t stall_cycles_due_to_prng_unavailability = 0;
  uint64_t fifo_bit_peak_blocks = 0;
  uint64_t fifo_coeff_peak = 0;
  uint64_t first_request_cycle = 0;
  uint64_t first_needed_cycle = 0;
  uint64_t first_ready_cycle = 0;
  uint64_t last_ready_cycle = 0;
  bool has_request = false;
  bool has_need = false;
  bool has_ready = false;
};

// A rough, deterministic producer/consumer timing model:
// - We don't simulate per-cycle FIFOs byte-accurately; we emulate throughput and bounded buffering.
// - We still expose "stall due to PRNG unavailability" and "hidden vs exposed" latency.
class PrngEvalKeyAGenerator {
 public:
  void reset(const PrngTimingConfig &cfg) {
    m_cfg = cfg;
    m_stats = {};
    m_available_coeffs = 0;
    m_produced_coeffs = 0;
    m_requested_coeffs = 0;
    m_ready_time = 0;
    m_request_time = 0;
  }

  const PrngStats &stats() const { return m_stats; }

  // Issue a generation request at time t. Returns the time when request DMA(seed/metadata) is "done"
  // (caller can overlap this with other work).
  uint64_t request(uint64_t t, uint64_t num_coeffs, double accept_prob_hint = 1.0) {
    m_requested_coeffs += num_coeffs;
    if (!m_stats.has_request) {
      m_stats.has_request = true;
      m_stats.first_request_cycle = t;
    }
    m_request_time = std::min<uint64_t>(m_request_time == 0 ? t : m_request_time, t);

    // Update ready time for total requested coeffs using a bounded-buffer throughput model.
    // Expected words per accept = 1/p (p≈T/2^64). words_needed ≈ coeffs/p.
    const double p = std::max(1e-9, std::min(1.0, accept_prob_hint));
    const double words_needed_d = std::ceil(static_cast<double>(m_requested_coeffs) / p);
    const uint64_t words_needed = static_cast<uint64_t>(std::max(0.0, words_needed_d));
    const uint64_t bytes_needed = words_needed * 8ULL;
    const uint64_t blocks_needed = (bytes_needed + shake128_detail::Shake128::kRateBytes - 1) /
                                   shake128_detail::Shake128::kRateBytes;

    // Producer: prng_lanes each make 1 block per shake_block_cycles after startup.
    const uint64_t lanes = std::max<uint64_t>(1, m_cfg.num_prng_lanes);
    const uint64_t prod_waves = (blocks_needed + lanes - 1) / lanes;
    const uint64_t prod_cycles = m_cfg.shake_startup_cycles + prod_waves * std::max<uint64_t>(1, m_cfg.shake_block_cycles);

    // Sampler: sampler_lanes convert 1 word/cycle/lane (rough) plus rejects included in words_needed.
    const uint64_t samp_lanes = std::max<uint64_t>(1, m_cfg.num_sampler_lanes);
    const uint64_t samp_cycles = (words_needed + samp_lanes - 1) / samp_lanes;

    // Buffering/backpressure: limit in-flight blocks and coeffs; approximate by adding penalty when queues saturate.
    // Very rough: if either FIFO is small relative to demand, add an extra "drain" term proportional to oversubscription.
    const uint64_t bit_cap = std::max<uint64_t>(1, m_cfg.bit_fifo_capacity_blocks) * lanes;
    const uint64_t coeff_cap = std::max<uint64_t>(1, m_cfg.coeff_fifo_capacity);
    uint64_t backpressure_penalty = 0;
    if (blocks_needed > bit_cap) backpressure_penalty += (blocks_needed - bit_cap) / lanes;
    if (m_requested_coeffs > coeff_cap) backpressure_penalty += (m_requested_coeffs - coeff_cap) / samp_lanes;

    const uint64_t gen_cycles = std::max(prod_cycles, samp_cycles) + backpressure_penalty;
    m_ready_time = std::max<uint64_t>(m_ready_time, t + gen_cycles);

    // Peak occupancy hints
    m_stats.fifo_bit_peak_blocks = std::max<uint64_t>(m_stats.fifo_bit_peak_blocks, std::min(blocks_needed, bit_cap));
    m_stats.fifo_coeff_peak = std::max<uint64_t>(m_stats.fifo_coeff_peak, std::min(m_requested_coeffs, coeff_cap));

    // Seed/metadata DMA completion time is modeled by caller (EngineModel DMA). Here we just return t.
    return t;
  }

  // Consume coeffs at time t. Returns the time when coeffs are available; if not ready, caller must stall.
  uint64_t consume(uint64_t t, uint64_t coeffs_needed) {
    if (!m_stats.has_need) {
      m_stats.has_need = true;
      m_stats.first_needed_cycle = t;
    }

    // If request wasn't made, treat as requested on demand.
    if (m_requested_coeffs < m_produced_coeffs + coeffs_needed) {
      const uint64_t add = (m_produced_coeffs + coeffs_needed) - m_requested_coeffs;
      (void)request(t, add);
    }

    const uint64_t ready = m_ready_time;
    if (ready > t) {
      m_stats.stall_cycles_due_to_prng_unavailability += (ready - t);
      t = ready;
    }
    m_produced_coeffs += coeffs_needed;
    m_available_coeffs = (m_requested_coeffs > m_produced_coeffs) ? (m_requested_coeffs - m_produced_coeffs) : 0;

    if (!m_stats.has_ready) {
      m_stats.has_ready = true;
      m_stats.first_ready_cycle = ready;
    }
    m_stats.last_ready_cycle = std::max<uint64_t>(m_stats.last_ready_cycle, ready);
    m_stats.coeffs_emitted += coeffs_needed;
    return t;
  }

 private:
  PrngTimingConfig m_cfg{};
  PrngStats m_stats{};
  uint64_t m_requested_coeffs = 0;
  uint64_t m_produced_coeffs = 0;
  uint64_t m_available_coeffs = 0;
  uint64_t m_ready_time = 0;
  uint64_t m_request_time = 0;
};

}  // namespace sim
}  // namespace moai

