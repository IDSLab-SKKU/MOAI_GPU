#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace moai {
namespace sim {

inline uint64_t env_u64(const char *name, uint64_t def) {
  const char *v = std::getenv(name);
  if (!v || v[0] == '\0') return def;
  char *end = nullptr;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (end == v) return def;
  return static_cast<uint64_t>(x);
}

inline bool env_bool(const char *name, bool def) {
  const char *v = std::getenv(name);
  if (!v || v[0] == '\0') return def;
  return std::strcmp(v, "0") != 0;
}

inline std::string fmt_bytes_iec(uint64_t bytes) {
  const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(bytes);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[64];
  // Keep it compact: 0 decimals for B/KiB, 2 decimals for MiB+
  if (u <= 1)
    std::snprintf(buf, sizeof(buf), "%.0f%s", v, units[u]);
  else
    std::snprintf(buf, sizeof(buf), "%.2f%s", v, units[u]);
  return std::string(buf);
}

struct EngineModelConfig {
  // Timebase
  uint64_t cycle_ns = 1;  // ns per cycle

  // DMA
  uint64_t dma_base_cyc = 200;
  uint64_t dma_bw_gbps = 1000;  // GB/s

  // Compute
  uint64_t ntt_cyc_per_coeff = 1;
  uint64_t vec_cyc_per_coeff = 2;
  uint64_t rescale_cyc_per_coeff = 1;
  uint64_t modswitch_cyc_per_coeff = 1;

  // Per-pass overheads (to make chunking hurt realistically)
  uint64_t ntt_pass_overhead_cyc = 0;
  uint64_t vec_pass_overhead_cyc = 0;

  // Keys (bytes)
  uint64_t galois_key_bytes = 0;
  uint64_t relin_key_bytes = 0;

  static EngineModelConfig from_env() {
    EngineModelConfig c;
    c.cycle_ns = env_u64("MOAI_SIM_CYCLE_NS", c.cycle_ns);
    c.dma_base_cyc = env_u64("MOAI_SIM_DMA_BASE_CYC", c.dma_base_cyc);
    c.dma_bw_gbps = env_u64("MOAI_SIM_DMA_BW_GBPS", c.dma_bw_gbps);
    c.ntt_cyc_per_coeff = env_u64("MOAI_SIM_NTT_CYC_PER_COEFF", c.ntt_cyc_per_coeff);
    c.vec_cyc_per_coeff = env_u64("MOAI_SIM_VEC_CYC_PER_COEFF", c.vec_cyc_per_coeff);
    c.rescale_cyc_per_coeff = env_u64("MOAI_SIM_RESCALE_CYC_PER_COEFF", c.rescale_cyc_per_coeff);
    c.modswitch_cyc_per_coeff = env_u64("MOAI_SIM_MODSWITCH_CYC_PER_COEFF", c.modswitch_cyc_per_coeff);
    c.ntt_pass_overhead_cyc = env_u64("MOAI_SIM_NTT_PASS_OVERHEAD_CYC", c.ntt_pass_overhead_cyc);
    c.vec_pass_overhead_cyc = env_u64("MOAI_SIM_VEC_PASS_OVERHEAD_CYC", c.vec_pass_overhead_cyc);
    c.galois_key_bytes = env_u64("MOAI_SIM_GALOIS_KEY_BYTES", c.galois_key_bytes);
    c.relin_key_bytes = env_u64("MOAI_SIM_RELIN_KEY_BYTES", c.relin_key_bytes);
    return c;
  }
};

struct OnChipConfig {
  // Global SPAD
  uint64_t gspad_bytes = 10ULL * 1024ULL * 1024ULL;  // default 10MB
  uint64_t gspad_bw_gbps = 5000;  // GB/s
  uint64_t gspad_base_cyc = 1;
  uint64_t gspad_banks = 1;

  // RFs
  uint64_t vec_rf_bytes = 0;
  uint64_t vec_rf_bw_gbps = 10000;
  uint64_t vec_rf_banks = 1;
  uint64_t ntt_rf_bytes = 0;
  uint64_t ntt_rf_bw_gbps = 10000;
  uint64_t ntt_rf_banks = 1;

  static OnChipConfig from_env() {
    OnChipConfig c;
    c.gspad_bytes = env_u64("MOAI_SIM_GSPAD_BYTES", c.gspad_bytes);
    c.gspad_bw_gbps = env_u64("MOAI_SIM_GSPAD_BW_GBPS", c.gspad_bw_gbps);
    c.gspad_base_cyc = env_u64("MOAI_SIM_GSPAD_BASE_CYC", c.gspad_base_cyc);
    c.gspad_banks = env_u64("MOAI_SIM_GSPAD_BANKS", c.gspad_banks);
    c.vec_rf_bytes = env_u64("MOAI_SIM_VEC_RF_BYTES", c.vec_rf_bytes);
    c.vec_rf_bw_gbps = env_u64("MOAI_SIM_VEC_RF_BW_GBPS", c.vec_rf_bw_gbps);
    c.vec_rf_banks = env_u64("MOAI_SIM_VEC_RF_BANKS", c.vec_rf_banks);
    c.ntt_rf_bytes = env_u64("MOAI_SIM_NTT_RF_BYTES", c.ntt_rf_bytes);
    c.ntt_rf_bw_gbps = env_u64("MOAI_SIM_NTT_RF_BW_GBPS", c.ntt_rf_bw_gbps);
    c.ntt_rf_banks = env_u64("MOAI_SIM_NTT_RF_BANKS", c.ntt_rf_banks);
    return c;
  }

  bool enabled() const { return gspad_bytes != 0; }
};

}  // namespace sim
}  // namespace moai

