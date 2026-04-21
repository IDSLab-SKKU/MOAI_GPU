#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "source/sim/sim_ckks_defaults.h"

#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/types.h>
#endif

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

// One relin / Galois keyswitch DMA chunk (Phantom layout), same as MOAI_SIM `compute_key_bytes.py`:
//   dnum * (2 * coeff_modulus_primes * N) * sizeof(uint64)
// with dnum = (T - alpha) / alpha, T = |coeff_modulus|, alpha = MOAI_SIM_ALPHA (default 1).
// Matches `build-genkeys/keys/keys_dnum_35/manifest.txt` for N=65536, T=36, alpha=1 -> 1321205760 bytes.
inline uint64_t moai_sim_default_key_switch_bytes_per_use() {
  const uint64_t N = env_u64("MOAI_SIM_POLY_DEGREE", kSingleLayerPolyModulusDegree());
  const uint64_t T = env_u64("MOAI_SIM_NUM_LIMBS", static_cast<uint64_t>(kSingleLayerCoeffModulusCount()));
  const uint64_t alpha = std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_ALPHA", 1));
  if (N == 0 || T == 0 || T <= alpha) return 0;
  if ((T - alpha) % alpha != 0) return 0;
  const uint64_t dnum = (T - alpha) / alpha;
  return dnum * 2ull * T * N * 8ull;
}

// Hybrid KS digit count for full modulus chain (Phantom / gen_moai_keys.cu): dnum = (T - alpha) / alpha.
inline uint64_t hybrid_ks_dnum(uint64_t T_chain_primes, uint64_t alpha) {
  if (alpha == 0 || T_chain_primes == 0 || T_chain_primes <= alpha) return 0;
  if ((T_chain_primes - alpha) % alpha != 0) return 0;
  return (T_chain_primes - alpha) / alpha;
}

// Optional positive float/double from env (MHz, GHz, etc.); returns false if unset/invalid.
inline bool env_positive_double(const char *name, double *out) {
  const char *v = std::getenv(name);
  if (!v || v[0] == '\0' || !out) return false;
  char *end = nullptr;
  const double x = std::strtod(v, &end);
  if (end == v || !std::isfinite(x) || x <= 0.0) return false;
  *out = x;
  return true;
}

// Optional non-negative float/double from env; returns false if unset/invalid.
inline bool env_nonneg_double(const char *name, double *out) {
  const char *v = std::getenv(name);
  if (!v || v[0] == '\0' || !out) return false;
  char *end = nullptr;
  const double x = std::strtod(v, &end);
  if (end == v || !std::isfinite(x) || x < 0.0) return false;
  *out = x;
  return true;
}

inline bool env_bool(const char *name, bool def) {
  const char *v = std::getenv(name);
  if (!v || v[0] == '\0') return def;
  return std::strcmp(v, "0") != 0;
}

// Phantom-style: MOAI_SIM_NUM_LIMBS counts |QP| (full coeff_modulus prime count). Ciphertext RNS size |Ql| for
// data at the top of the chain is |Q| = |QP| − |P|, and hybrid uses |P| = alpha (special_modulus_size).
// MOAI_SIM_NUM_LIMBS_COUNTS_QP=0 → treat MOAI_SIM_NUM_LIMBS as |Ql| already (legacy sim).
inline uint64_t sim_effective_rns_limbs_for_ct(uint64_t num_limbs_env, uint64_t alpha) {
  const uint64_t a = std::max<uint64_t>(1ULL, alpha);
  if (!env_bool("MOAI_SIM_NUM_LIMBS_COUNTS_QP", true)) return num_limbs_env;
  if (num_limbs_env <= a) return num_limbs_env;
  return num_limbs_env - a;
}

// floor(log2(x)) for x>=1; 0 for x<=1
inline uint64_t floor_log2_u64(uint64_t x) {
  uint64_t r = 0;
  while (x > 1) {
    x >>= 1;
    ++r;
  }
  return r;
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

// Decimal GB/s (1e9 bytes/s), matches bandwidth_gbps units elsewhere.
inline std::string fmt_gbps(double gbps) {
  if (!std::isfinite(gbps) || gbps <= 0.0) return std::string("-");
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f", gbps);
  return std::string(buf);
}

// Average transfer rate (decimal GB/s) = bytes / (cycles * period_ns as seconds) / 1e9.
inline double avg_throughput_gbps(uint64_t bytes, uint64_t cycles, double cycle_period_ns) {
  if (cycles == 0 || !std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double sec = static_cast<double>(cycles) * cycle_period_ns * 1e-9;
  return static_cast<double>(bytes) / sec / 1e9;
}

// NTT/VEC logical throughput: coefficient-elements charged per schedule step, over busy or wall time.
inline double avg_ops_per_sec(uint64_t ops, uint64_t cycles, double cycle_period_ns) {
  if (cycles == 0 || !std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double sec = static_cast<double>(cycles) * cycle_period_ns * 1e-9;
  return static_cast<double>(ops) / sec;
}

inline std::string fmt_ops_s(double opss) {
  if (!std::isfinite(opss) || opss <= 0.0) return std::string("-");
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g", opss);
  return std::string(buf);
}

// Default relative to process cwd (run from MOAI_GPU repo root): output/sim/...
inline const char *default_sim_report_path() { return "output/sim/moai_sim_report.txt"; }

inline void ensure_parent_dirs_for_file(const char *file_path) {
#if !defined(_WIN32)
  if (!file_path || file_path[0] == '\0') return;
  const std::string p(file_path);
  for (size_t i = 0; i < p.size(); ++i) {
    if (p[i] == '/') {
      const std::string sub = p.substr(0, i);
      if (!sub.empty()) (void)mkdir(sub.c_str(), 0755);
    }
  }
#else
  (void)file_path;
#endif
}

// Off-chip / HBM-style memory seen by the DMA engine (one logical transaction per enqueue_dma).
struct ExternalMemoryConfig {
  // Label for logs only (e.g. HBM2, HBM3, DDR5)
  std::string kind = "HBM2";

  // Streaming bandwidth (GB/s). Example: HBM2 class ~812 GB/s effective (design-dependent).
  uint64_t bandwidth_gbps = 812;

  // Fixed latency per DMA transaction:
  // - If latency_ns > 0: use ceil(latency_ns / sim_cycle_period_ns) (double ns/cycle).
  // - Else: use latency_cyc directly.
  uint64_t latency_cyc = 200;
  uint64_t latency_ns = 0;

  uint64_t fixed_latency_cycles(double sim_cycle_period_ns) const {
    if (latency_ns > 0) {
      const double per = (!std::isfinite(sim_cycle_period_ns) || sim_cycle_period_ns <= 0.0)
                             ? 1.0
                             : sim_cycle_period_ns;
      const double cyc = std::ceil(static_cast<double>(latency_ns) / per);
      return static_cast<uint64_t>(std::max(1.0, cyc));
    }
    return latency_cyc;
  }

  static ExternalMemoryConfig from_env(uint64_t /*sim_cycle_ns*/) {
    ExternalMemoryConfig c;
    if (const char *k = std::getenv("MOAI_SIM_EXT_MEM_KIND"); k != nullptr && k[0] != '\0')
      c.kind = k;

    // BW: explicit external > legacy DMA knob > struct default
    c.bandwidth_gbps =
        env_u64("MOAI_SIM_EXT_MEM_BW_GBPS", env_u64("MOAI_SIM_DMA_BW_GBPS", c.bandwidth_gbps));

    c.latency_ns = env_u64("MOAI_SIM_EXT_MEM_LATENCY_NS", c.latency_ns);

    // Cycles: explicit external > legacy DMA latency > legacy BASE > default
    c.latency_cyc = env_u64(
        "MOAI_SIM_EXT_MEM_LATENCY_CYC",
        env_u64("MOAI_SIM_DMA_LATENCY_CYC", env_u64("MOAI_SIM_DMA_BASE_CYC", c.latency_cyc)));

    return c;
  }
};

struct EngineModelConfig {
  // Timebase
  // - cycle_ns: integer ns/cycle (legacy); MOAI_SIM_CYCLE_NS overrides. Used as coarse display when
  //   no fractional period is set.
  // - sim_cycle_period_ns: ns/cycle for xfer + latency_ns + BW print. Set by from_env():
  //   MOAI_SIM_ENGINE_MHZ > MOAI_SIM_CYCLE_PERIOD_NS > MOAI_SIM_CYCLE_NS > these defaults.
  //   (Header defaults apply only when those env vars are unset.)
  uint64_t cycle_ns = 1;
  double sim_cycle_period_ns = 1.0;

  // External memory (models VRAM/HBM traffic for the DMA engine)
  ExternalMemoryConfig ext_mem{};

  // NTT engine (pipeline-abstract; streaming after fill)
  // - ntt_lanes: parallel coeff pipelines (>=1). Steady work scales as ceil(coeffs/lanes).
  // - ntt_pipe_depth_cyc: fill / first-result latency (cycles). 0 => use floor_log2(poly_degree)
  //   when enqueue passes poly_degree (e.g. N=2^16 -> 16 cycles).
  // - ntt_steady_cyc_per_coeff: per wavefront step after fill (often 1 when fully pipelined).
  // MOAI_SIM_NTT_CYC_PER_COEFF maps to ntt_steady_cyc_per_coeff (legacy name).
  // NOTE: 2^16 is (1<<16)==65536. Common mistake: (2<<16)==131072 (extra factor of two on lanes).
  uint64_t ntt_lanes =  8;
  uint64_t ntt_pipe_depth_cyc = 0;
  uint64_t ntt_steady_cyc_per_coeff = 1;

  // Vector engine — separate mul vs add; lanes parallelize coeff streams like NTT.
  uint64_t vec_lanes = 8;
  uint64_t vec_add_cyc_per_coeff = 1;
  // Vector MUL (double-buffered assumption): fill latency is overlapped — model only steady
  // throughput: ceil(coeffs/vec_lanes) * steady_cyc_per_wave (+ pass overhead).
  uint64_t vec_mul_steady_cyc_per_coeff = 1;
  // Vector fused multiply-add (MAC / FMA) on coeff streams — same wave shape as vec_mul.
  uint64_t vec_mac_steady_cyc_per_coeff = 1;
  uint64_t rescale_cyc_per_coeff = 1;
  uint64_t modswitch_cyc_per_coeff = 1;
  // Montgomery-style primitives (used when modeling BConv as MMUL/MADD matmul):
  // These are *effective* per-op cycles for one lane. Lanes parallelize ops similarly to coeff streams.
  uint64_t vec_mmul_cyc_per_op = 1;
  uint64_t vec_madd_cyc_per_op = 1;

  // Per-pass overheads (to make chunking hurt realistically)
  uint64_t ntt_pass_overhead_cyc = 0;
  uint64_t vec_pass_overhead_cyc = 0;

  // PRNG engine (on-the-fly random poly generation; e.g., SHAKE128+reject sampling).
  // Modeled as a coefficient stream engine like NTT/VEC: ceil(coeffs/lanes)*cyc + overhead.
  uint64_t prng_lanes = 8;
  uint64_t prng_cyc_per_coeff = 1;
  uint64_t prng_pass_overhead_cyc = 0;

  // Detailed on-the-fly eval-key 'a' generator (SHAKE128 + reject sampling) timing knobs.
  // These model buffering/backpressure/chunking beyond the simple prng_* stream model above.
  uint64_t otf_evk_a_num_prng_lanes = 4;
  uint64_t otf_evk_a_num_sampler_lanes = 4;
  uint64_t otf_evk_a_shake_startup_cycles = 50;
  uint64_t otf_evk_a_shake_block_cycles = 4;
  uint64_t otf_evk_a_bit_fifo_capacity_blocks = 8;
  uint64_t otf_evk_a_coeff_fifo_capacity = 256;
  uint64_t otf_evk_a_chunk_size = 32;
  uint64_t otf_evk_a_seed_metadata_bytes = 64;

  // Keys (bytes). `from_env()` sets from MOAI_SIM_GALOIS_KEY_BYTES / MOAI_SIM_RELIN_KEY_BYTES, or auto
  // `dnum * 2 * T * N * 8` (see `moai_sim_default_key_switch_bytes_per_use()`). Set env to 0 to omit key DMA.
  uint64_t galois_key_bytes = 0;
  uint64_t relin_key_bytes = 0;

  // On-the-fly keyswitch key generation:
  // Treat keyswitch key as (b,a), where a is a random poly generated on-chip. This reduces key DMA
  // volume by scaling the key bytes read, and charges the PRNG engine for generating a in NTT domain.
  bool otf_keygen = false;
  double otf_keygen_key_bytes_scale = 0.5;  // default: halve key DMA (skip 'a' fetch)

  // CT×CT coarse model (enqueue_ct_ct_multiply): extra vec_mul waves per chunk — proxy for tensor/RNS
  // dyadic work before relinearize (separate Evaluator call). Default 3.
  uint64_t ct_ct_vec_mul_passes = 3;

  // Keyswitch (Phantom eval_key_switch.cu keyswitch_inplace + rns_bconv DRNSTool::modup / moddown_from_NTT, CKKS).
  // kswitch_size_p = |P| primes appended for QlP (size_QlP = |Ql| + |P|). NOT the hybrid digit size.
  // Hybrid digit size = MOAI_SIM_ALPHA (kswitch_digit_alpha). beta = ceil(|Ql|/alpha) when MOAI_SIM_KSWITCH_BETA_MODE
  // is unset/phantom; legacy uses ceil(|Ql|/kswitch_size_p). Override: MOAI_SIM_KSWITCH_BETA > 0.
  uint64_t kswitch_size_p = 1;
  uint64_t kswitch_digit_alpha = 1;
  bool kswitch_beta_legacy = false;
  uint64_t kswitch_beta = 0;
  uint64_t kswitch_modup_bconv_cyc_per_coeff = 1;
  uint64_t kswitch_moddown_bconv_cyc_per_coeff = 1;
  // If enabled, model keyswitch BConv time from MMUL/MADD ops (matmul), instead of per-coeff constant.
  bool kswitch_bconv_use_montgomery_ops = false;
  // If enabled (and Montgomery BConv above), schedule matmul/accumulate as FMA/MAC (one wave per MAC) plus
  // extra MMUL-only waves when mmul_ops > madd_ops (BEHZ moddown phase1: mmul−madd = N·|P|).
  bool kswitch_bconv_use_fma_ops = false;
  // Cycles per fused MAC op on the vec lane (defaults to vec_mmul_cyc_per_op in from_env).
  uint64_t vec_mac_cyc_per_op = 1;
  // apply_galois_ntt on c0 then c1 (evaluate.cu apply_galois_inplace, CKKS).
  uint64_t galois_perm_cyc_per_coeff = 1;

  static EngineModelConfig from_env() {
    EngineModelConfig c;
    c.cycle_ns = env_u64("MOAI_SIM_CYCLE_NS", c.cycle_ns);
    c.sim_cycle_period_ns = static_cast<double>(c.cycle_ns);

    // Fractional ns/cycle (e.g. 3.3) without going through MHz.
    double period_ns = 0.0;
    if (env_positive_double("MOAI_SIM_CYCLE_PERIOD_NS", &period_ns)) {
      c.sim_cycle_period_ns = period_ns;
      c.cycle_ns = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(c.sim_cycle_period_ns)));
    }

    // Clock: MOAI_SIM_ENGINE_MHZ wins over MOAI_SIM_CYCLE_PERIOD_NS / integer cycle_ns.
    double mhz = 0.0;
    if (env_positive_double("MOAI_SIM_ENGINE_MHZ", &mhz)) {
      c.sim_cycle_period_ns = 1000.0 / mhz;
      c.cycle_ns = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(c.sim_cycle_period_ns)));
    }

    c.ext_mem = ExternalMemoryConfig::from_env(c.cycle_ns);
    c.ntt_lanes = env_u64("MOAI_SIM_NTT_LANES", c.ntt_lanes);
    c.ntt_pipe_depth_cyc = env_u64("MOAI_SIM_NTT_PIPE_DEPTH_CYC", c.ntt_pipe_depth_cyc);
    c.ntt_steady_cyc_per_coeff =
        env_u64("MOAI_SIM_NTT_STEADY_CYC_PER_COEFF", env_u64("MOAI_SIM_NTT_CYC_PER_COEFF", c.ntt_steady_cyc_per_coeff));
    c.vec_lanes = env_u64("MOAI_SIM_VEC_LANES", c.vec_lanes);
    c.vec_add_cyc_per_coeff = env_u64("MOAI_SIM_VEC_ADD_CYC_PER_COEFF", c.vec_add_cyc_per_coeff);
    c.vec_mul_steady_cyc_per_coeff = env_u64(
        "MOAI_SIM_VEC_MUL_STEADY_CYC_PER_COEFF",
        env_u64("MOAI_SIM_VEC_MUL_CYC_PER_COEFF", env_u64("MOAI_SIM_VEC_CYC_PER_COEFF", c.vec_mul_steady_cyc_per_coeff)));
    c.vec_mac_steady_cyc_per_coeff = std::max<uint64_t>(
        1ULL,
        env_u64("MOAI_SIM_VEC_MAC_STEADY_CYC_PER_COEFF",
                env_u64("MOAI_SIM_VEC_MAC_CYC_PER_COEFF", c.vec_mul_steady_cyc_per_coeff)));
    c.vec_mmul_cyc_per_op = env_u64("MOAI_SIM_VEC_MMUL_CYC_PER_OP", c.vec_mmul_cyc_per_op);
    c.vec_madd_cyc_per_op = env_u64("MOAI_SIM_VEC_MADD_CYC_PER_OP", c.vec_madd_cyc_per_op);
    c.rescale_cyc_per_coeff = env_u64("MOAI_SIM_RESCALE_CYC_PER_COEFF", c.rescale_cyc_per_coeff);
    c.modswitch_cyc_per_coeff = env_u64("MOAI_SIM_MODSWITCH_CYC_PER_COEFF", c.modswitch_cyc_per_coeff);
    c.ntt_pass_overhead_cyc = env_u64("MOAI_SIM_NTT_PASS_OVERHEAD_CYC", c.ntt_pass_overhead_cyc);
    c.vec_pass_overhead_cyc = env_u64("MOAI_SIM_VEC_PASS_OVERHEAD_CYC", c.vec_pass_overhead_cyc);

    c.prng_lanes = std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_PRNG_ENGINE_LANES", c.prng_lanes));
    c.prng_cyc_per_coeff = std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_PRNG_CYC_PER_COEFF", c.prng_cyc_per_coeff));
    c.prng_pass_overhead_cyc = env_u64("MOAI_SIM_PRNG_PASS_OVERHEAD_CYC", c.prng_pass_overhead_cyc);

    c.otf_evk_a_num_prng_lanes =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_NUM_PRNG_LANES", c.otf_evk_a_num_prng_lanes));
    c.otf_evk_a_num_sampler_lanes =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_NUM_SAMPLER_LANES", c.otf_evk_a_num_sampler_lanes));
    c.otf_evk_a_shake_startup_cycles =
        env_u64("MOAI_SIM_OTF_EVK_A_SHAKE_STARTUP_CYCLES", c.otf_evk_a_shake_startup_cycles);
    c.otf_evk_a_shake_block_cycles =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_SHAKE_BLOCK_CYCLES", c.otf_evk_a_shake_block_cycles));
    c.otf_evk_a_bit_fifo_capacity_blocks =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_BIT_FIFO_CAPACITY_BLOCKS", c.otf_evk_a_bit_fifo_capacity_blocks));
    c.otf_evk_a_coeff_fifo_capacity =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_COEFF_FIFO_CAPACITY", c.otf_evk_a_coeff_fifo_capacity));
    c.otf_evk_a_chunk_size =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_OTF_EVK_A_CHUNK_SIZE", c.otf_evk_a_chunk_size));
    c.otf_evk_a_seed_metadata_bytes =
        env_u64("MOAI_SIM_OTF_EVK_A_SEED_METADATA_BYTES", c.otf_evk_a_seed_metadata_bytes);

    c.galois_key_bytes = env_u64("MOAI_SIM_GALOIS_KEY_BYTES", moai_sim_default_key_switch_bytes_per_use());
    c.relin_key_bytes = env_u64("MOAI_SIM_RELIN_KEY_BYTES", moai_sim_default_key_switch_bytes_per_use());

    c.otf_keygen = env_bool("MOAI_SIM_OTF_KEYGEN", c.otf_keygen);
    {
      double scale = c.otf_keygen_key_bytes_scale;
      if (env_nonneg_double("MOAI_SIM_OTF_KEYGEN_KEY_BYTES_SCALE", &scale)) {
        // Clamp to [0,1] to avoid accidental amplification.
        if (scale > 1.0) scale = 1.0;
        c.otf_keygen_key_bytes_scale = scale;
      }
    }

    c.ct_ct_vec_mul_passes = env_u64("MOAI_SIM_CT_CT_VEC_MUL_PASSES", c.ct_ct_vec_mul_passes);
    c.ct_ct_vec_mul_passes = std::max<uint64_t>(1ULL, c.ct_ct_vec_mul_passes);
    c.kswitch_digit_alpha = std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_ALPHA", c.kswitch_digit_alpha));
    // Default |P| in QlP = alpha (Phantom special_modulus_size); set MOAI_SIM_KSWITCH_SIZE_P to override.
    if (const char *ksp = std::getenv("MOAI_SIM_KSWITCH_SIZE_P"); ksp != nullptr && ksp[0] != '\0')
      c.kswitch_size_p = std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_KSWITCH_SIZE_P", 1));
    else
      c.kswitch_size_p = std::max<uint64_t>(1ULL, c.kswitch_digit_alpha);
    if (const char *bm = std::getenv("MOAI_SIM_KSWITCH_BETA_MODE"); bm != nullptr && std::strcmp(bm, "legacy") == 0)
      c.kswitch_beta_legacy = true;
    else
      c.kswitch_beta_legacy = false;
    c.kswitch_beta = env_u64("MOAI_SIM_KSWITCH_BETA", c.kswitch_beta);
    c.kswitch_modup_bconv_cyc_per_coeff =
        env_u64("MOAI_SIM_KSWITCH_MODUP_BCONV_CYC_PER_COEFF", c.kswitch_modup_bconv_cyc_per_coeff);
    c.kswitch_moddown_bconv_cyc_per_coeff =
        env_u64("MOAI_SIM_KSWITCH_MODDOWN_BCONV_CYC_PER_COEFF", c.kswitch_moddown_bconv_cyc_per_coeff);
    c.kswitch_bconv_use_montgomery_ops =
        env_bool("MOAI_SIM_KSWITCH_BCONV_USE_MONTGOMERY_OPS", c.kswitch_bconv_use_montgomery_ops);
    c.kswitch_bconv_use_fma_ops = env_bool("MOAI_SIM_KSWITCH_BCONV_USE_FMA_OPS", c.kswitch_bconv_use_fma_ops);
    c.vec_mac_cyc_per_op =
        std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_VEC_MAC_CYC_PER_OP", c.vec_mmul_cyc_per_op));
    c.galois_perm_cyc_per_coeff = env_u64("MOAI_SIM_GALOIS_PERM_CYC_PER_COEFF", c.galois_perm_cyc_per_coeff);
    return c;
  }
};

// Ideal steady-state "coefficient stream" equivalent GB/s (decimal GB/s = 1e9 B/s): each logical coeff
// element counted as 8B (uint64 RNS limb), matching the ntt/vec row when byte counters are zero.
// Ignores pass overhead and pipe fill — same wave model as enqueue_ntt_coeffs / enqueue_vec_mul:
// ceil(coeffs/lanes) waves × steady_cyc per wave ⇒ long-run ~steady/lanes cycles per coeff.
inline double theoretical_ntt_steady_coeff_equiv_gbps(const EngineModelConfig &cfg, double cycle_period_ns) {
  if (!std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double lanes = static_cast<double>(std::max<uint64_t>(1ULL, cfg.ntt_lanes));
  const double steady = static_cast<double>(std::max<uint64_t>(1ULL, cfg.ntt_steady_cyc_per_coeff));
  const double coeffs_per_s = (lanes / steady) / (cycle_period_ns * 1e-9);
  return coeffs_per_s * 8.0 / 1e9;
}

inline double theoretical_vec_mul_steady_coeff_equiv_gbps(const EngineModelConfig &cfg, double cycle_period_ns) {
  if (!std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double lanes = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_lanes));
  const double steady = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_mul_steady_cyc_per_coeff));
  const double coeffs_per_s = (lanes / steady) / (cycle_period_ns * 1e-9);
  return coeffs_per_s * 8.0 / 1e9;
}

inline double theoretical_vec_add_steady_coeff_equiv_gbps(const EngineModelConfig &cfg, double cycle_period_ns) {
  if (!std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double lanes = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_lanes));
  const double steady = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_add_cyc_per_coeff));
  const double coeffs_per_s = (lanes / steady) / (cycle_period_ns * 1e-9);
  return coeffs_per_s * 8.0 / 1e9;
}

inline double theoretical_vec_mac_steady_coeff_equiv_gbps(const EngineModelConfig &cfg, double cycle_period_ns) {
  if (!std::isfinite(cycle_period_ns) || cycle_period_ns <= 0.0) return 0.0;
  const double lanes = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_lanes));
  const double steady = static_cast<double>(std::max<uint64_t>(1ULL, cfg.vec_mac_steady_cyc_per_coeff));
  const double coeffs_per_s = (lanes / steady) / (cycle_period_ns * 1e-9);
  return coeffs_per_s * 8.0 / 1e9;
}

// Same formulas as EngineModel ntt_service_cycles / vec lane waves (for SimTiming alignment).
inline uint64_t estimate_ntt_cycles(uint64_t coeffs, uint64_t poly_degree) {
  const EngineModelConfig cfg = EngineModelConfig::from_env();
  const uint64_t lanes = std::max<uint64_t>(1ULL, cfg.ntt_lanes);
  uint64_t fill = cfg.ntt_pipe_depth_cyc;
  if (fill == 0 && poly_degree >= 2) fill = floor_log2_u64(poly_degree);
  const uint64_t waves = (coeffs + lanes - 1) / lanes;
  return cfg.ntt_pass_overhead_cyc + fill + waves * cfg.ntt_steady_cyc_per_coeff;
}

inline uint64_t estimate_vec_pipeline_cycles(uint64_t coeffs, uint64_t cyc_per_coeff_elem) {
  const EngineModelConfig cfg = EngineModelConfig::from_env();
  const uint64_t lanes = std::max<uint64_t>(1ULL, cfg.vec_lanes);
  const uint64_t waves = (coeffs + lanes - 1) / lanes;
  return cfg.vec_pass_overhead_cyc + waves * cyc_per_coeff_elem;
}

inline uint64_t estimate_vec_mul_cycles(uint64_t coeffs) {
  const EngineModelConfig cfg = EngineModelConfig::from_env();
  const uint64_t lanes = std::max<uint64_t>(1ULL, cfg.vec_lanes);
  const uint64_t waves = (coeffs + lanes - 1) / lanes;
  return cfg.vec_pass_overhead_cyc + waves * cfg.vec_mul_steady_cyc_per_coeff;
}

inline uint64_t estimate_vec_mac_cycles(uint64_t coeffs) {
  const EngineModelConfig cfg = EngineModelConfig::from_env();
  const uint64_t lanes = std::max<uint64_t>(1ULL, cfg.vec_lanes);
  const uint64_t waves = (coeffs + lanes - 1) / lanes;
  return cfg.vec_pass_overhead_cyc + waves * cfg.vec_mac_steady_cyc_per_coeff;
}

// Coarse CT×CT multiply (matches enqueue_ct_ct_multiply off-chip shape: NTT + K×vec + NTT).
inline uint64_t estimate_ct_ct_multiply_cycles(uint64_t coeffs, uint64_t poly_degree) {
  const EngineModelConfig cfg = EngineModelConfig::from_env();
  const uint64_t K = std::max<uint64_t>(1ULL, cfg.ct_ct_vec_mul_passes);
  return 2 * estimate_ntt_cycles(coeffs, poly_degree) + K * estimate_vec_mul_cycles(coeffs);
}

struct OnChipConfig {
  // Global SPAD
  uint64_t gspad_bytes = 10ULL * 1024ULL * 1024ULL;  // default 10MB
  uint64_t gspad_bw_gbps = 5000;  // GB/s
  uint64_t gspad_base_cyc = 1;
  uint64_t gspad_banks = 8;

  // RFs (engine-local scratch); default ~64KiB each, 8 banks (effective BW = bw * banks)
  uint64_t vec_rf_bytes = 64ULL * 1024ULL;
  uint64_t vec_rf_bw_gbps = 10000;
  uint64_t vec_rf_banks = 8;
  uint64_t ntt_rf_bytes = 64ULL * 1024ULL;
  uint64_t ntt_rf_bw_gbps = 10000;
  uint64_t ntt_rf_banks = 8;

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

