#pragma once

// Structural op counts for Phantom CKKS keyswitch core: same formulas as
// EngineModel::enqueue_keyswitch_phantom_ckks (engine_model.h). This header does **not** sample the
// scheduler at runtime; it is an analytical mirror of that enqueue graph (for sweeps / CSV / plots).
// Phantom sets beta = ceil(|Ql| / alpha) with alpha = special_modulus_size; see thirdparty/phantom-fhe/src/rns.cu

#include "source/sim/engine_config.h"
#include "source/sim/engine_model.h"
#include "source/sim/sim_ckks_defaults.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace moai {
namespace sim {

// hybrid_ks_dnum: see engine_config.h

// beta for keyswitch: ceil(|Ql| / alpha_digit). Phantom DRNSTool (rns.cu): ceil(size_Ql / alpha).
inline uint64_t keyswitch_beta_phantom(uint64_t size_Ql, uint64_t alpha_digit) {
  const uint64_t a = std::max<uint64_t>(1ULL, alpha_digit);
  return std::max<uint64_t>(1ULL, (size_Ql + a - 1) / a);
}

// Legacy MOAI coarse rule (pre-phantom-beta): ceil(|Ql| / MOAI_SIM_KSWITCH_SIZE_P).
inline uint64_t keyswitch_beta_legacy(uint64_t size_Ql, uint64_t kswitch_size_p) {
  const uint64_t P = std::max<uint64_t>(1ULL, kswitch_size_p);
  return std::max<uint64_t>(1ULL, (size_Ql + P - 1) / P);
}

struct KeyswitchPhantomProfile {
  uint64_t beta = 0;
  uint64_t ntt_kernels = 0;           // total enqueue_ntt_coeffs (= ntt_fwd + ntt_inv)
  uint64_t ntt_fwd_kernels = 0;       // forward NTT (modup loop on QlP)
  uint64_t ntt_inv_kernels = 0;       // INTT: c2 on Ql + moddown head on QlP (×2)
  uint64_t bconv_modup_kernels = 0;   // enqueue_vec_coeffs modup
  uint64_t bconv_moddown_kernels = 0; // enqueue_vec_coeffs moddown
  uint64_t vec_mul_kernels = 0;
  uint64_t vec_add_kernels = 0;
  // Sum of logical coeff-elements (N * limbs) across all passes of each kind (for capacity scaling).
  uint64_t weighted_ntt_coeff_elems = 0;
  uint64_t weighted_ntt_fwd_coeff_elems = 0;
  uint64_t weighted_ntt_inv_coeff_elems = 0;
  uint64_t weighted_bconv_coeff_elems = 0;
  uint64_t weighted_vec_mul_coeff_elems = 0;
  uint64_t weighted_vec_add_coeff_elems = 0;
  // VEC op breakdown (Montgomery model; final reduction absorbed):
  // Counted as "ops" not coeff_elems.
  uint64_t bconv_modup_mmul_ops = 0;
  uint64_t bconv_modup_madd_ops = 0;
  uint64_t bconv_moddown_mmul_ops = 0;
  uint64_t bconv_moddown_madd_ops = 0;
  uint64_t bconv_mmul_ops = 0;
  uint64_t bconv_madd_ops = 0;
  uint64_t vec_mul_mmul_ops = 0;
  uint64_t vec_add_madd_ops = 0;
};

// Fills counts for one keyswitch with given beta (already resolved: env override or phantom/legacy).
inline KeyswitchPhantomProfile compute_keyswitch_phantom_profile(uint64_t poly_degree,
                                                                 uint64_t size_Ql,
                                                                 uint64_t num_P,
                                                                 uint64_t beta) {
  KeyswitchPhantomProfile p;
  p.beta = std::max<uint64_t>(1ULL, beta);
  const uint64_t coeffs_ql = poly_degree * size_Ql;
  const uint64_t size_QlP = size_Ql + num_P;
  const uint64_t coeffs_qlp = poly_degree * size_QlP;
  const uint64_t coeffs_p = poly_degree * num_P;

  // Match Phantom CKKS:
  // - modup: INTT(Ql) then for each digit: bconv + NTT_fwd(QlP excluding part-Ql range)
  // - moddown (x2): INTT(P only) + bconv(P->Ql) + NTT_fwd(Ql fused) + vec_mul
  p.ntt_fwd_kernels = p.beta + 2;
  p.ntt_inv_kernels = 3;
  p.ntt_kernels = p.ntt_fwd_kernels + p.ntt_inv_kernels;
  p.bconv_modup_kernels = p.beta;
  p.bconv_moddown_kernels = 2;
  p.vec_mul_kernels = p.beta + 2;
  p.vec_add_kernels = 2;

  // Forward NTT:
  // - modup: per digit NTT on QlP excluding digit part-Ql ranges (summed exclusion is |Ql|)
  // - moddown: fused forward on Ql (×2)
  p.weighted_ntt_fwd_coeff_elems = (p.beta * coeffs_qlp - coeffs_ql) + 2 * coeffs_ql;
  // INTT: modup backward on Ql + moddown backward on P (×2).
  p.weighted_ntt_inv_coeff_elems = coeffs_ql + 2 * coeffs_p;
  p.weighted_ntt_coeff_elems = p.weighted_ntt_fwd_coeff_elems + p.weighted_ntt_inv_coeff_elems;
  p.weighted_bconv_coeff_elems = p.beta * coeffs_qlp + 2 * coeffs_ql;
  p.weighted_vec_mul_coeff_elems = p.beta * coeffs_qlp + 2 * coeffs_ql;
  p.weighted_vec_add_coeff_elems = 2 * coeffs_ql;

  // ----
  // Montgomery-style op breakdown for VEC mapping (final reduce absorbed).
  // ----
  // We need alpha (digit size) to compute per-digit part size. In this analytic mirror, assume exact partition
  // (size_Ql % alpha == 0) which is the default sweep policy. Derive alpha from num_P when unset (Phantom profile uses num_P=alpha).
  const uint64_t alpha = std::max<uint64_t>(1ULL, num_P);
  for (uint64_t bi = 0; bi < p.beta; ++bi) {
    const uint64_t start = alpha * bi;
    const uint64_t part_limbs = (start < size_Ql) ? std::min<uint64_t>(alpha, size_Ql - start) : 0ULL;
    const uint64_t obase = size_QlP - part_limbs;
    const uint64_t ibase = part_limbs;
    const uint64_t mm = poly_degree * obase * ibase;
    const uint64_t ma = poly_degree * obase * ibase;
    p.bconv_modup_mmul_ops += mm;
    p.bconv_modup_madd_ops += ma;
  }
  // moddown: P->Ql bConv_BEHZ done twice
  {
    const uint64_t ibase = num_P;
    const uint64_t obase = size_Ql;
    const uint64_t mmul_one = poly_degree * (ibase + obase * ibase);
    const uint64_t madd_one = poly_degree * (obase * ibase);
    p.bconv_moddown_mmul_ops += 2 * mmul_one;
    p.bconv_moddown_madd_ops += 2 * madd_one;
  }
  p.bconv_mmul_ops = p.bconv_modup_mmul_ops + p.bconv_moddown_mmul_ops;
  p.bconv_madd_ops = p.bconv_modup_madd_ops + p.bconv_moddown_madd_ops;
  // vec arith
  p.vec_mul_mmul_ops = p.weighted_vec_mul_coeff_elems;  // 1 MMUL per coeff-element
  p.vec_add_madd_ops = p.weighted_vec_add_coeff_elems;  // 1 MADD per coeff-element
  return p;
}

inline void trim_in_place(std::string &s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
}

inline std::vector<uint64_t> parse_uint64_list_env(const char *name) {
  std::vector<uint64_t> out;
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return out;
  std::string str(raw);
  size_t start = 0;
  while (start < str.size()) {
    size_t comma = str.find(',', start);
    std::string tok =
        (comma == std::string::npos) ? str.substr(start) : str.substr(start, comma - start);
    trim_in_place(tok);
    if (!tok.empty()) {
      char *end = nullptr;
      unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
      if (end != tok.c_str()) out.push_back(static_cast<uint64_t>(v));
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

// "1-35" inclusive (for MOAI_SIM_HYBRID_KS_ALPHA_RANGE).
inline void parse_alpha_range_string(const char *range, std::vector<uint64_t> &out) {
  if (range == nullptr || range[0] == '\0') return;
  const char *dash = std::strchr(range, '-');
  if (dash == nullptr || dash == range) return;
  char *end = nullptr;
  unsigned long long lo = std::strtoull(range, &end, 10);
  if (end == range) return;
  unsigned long long hi = std::strtoull(dash + 1, &end, 10);
  if (end == dash + 1) return;
  if (hi < lo) std::swap(lo, hi);
  for (unsigned long long a = lo; a <= hi && a < 10000000ULL; ++a)
    out.push_back(static_cast<uint64_t>(a));
}

inline void unset_env_key(const char *key) {
#if !defined(_WIN32)
  ::unsetenv(key);
#else
  std::string z = std::string(key) + "=";
  (void)_putenv(z.c_str());
#endif
}

// Estimator-only: print table + CSV for hybrid KS op profile vs alpha (and optional alpha list).
// Env:
//   MOAI_SIM_POLY_DEGREE, MOAI_SIM_NUM_LIMBS — by default **|QP|** (see MOAI_SIM_NUM_LIMBS_COUNTS_QP in engine_config.h)
//   MOAI_SIM_KSWITCH_SIZE_P — |P| in QlP; **unset** defaults to **alpha** (same row’s sweep α).
//   MOAI_SIM_KSWITCH_SIZE_Ql — |Ql|; **0** defaults to **|QP|−α** for that row (Phantom top-level |Q|).
//   MOAI_SIM_HYBRID_KS_ALPHA_LIST — comma-separated alphas (e.g. 1,2,4,8). Overrides range if non-empty.
//   MOAI_SIM_HYBRID_KS_ALPHA_RANGE — inclusive range "1-35" (β=1 at α=35 when |Q|=1, |QP|=36). Used if LIST empty.
//   MOAI_SIM_KSWITCH_BETA — if >0, overrides beta for all rows (debug)
//   MOAI_SIM_KSWITCH_BETA_MODE — legacy | phantom (default phantom): how beta is derived when MOAI_SIM_KSWITCH_BETA unset
//   MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1 — after analytic row, run EngineModel::profile_keyswitch_phantom_ckks and record
//     schedule() counts (needs MOAI_SIM_BACKEND=1). eng_* columns; -1 if not measured.
//   MOAI_SIM_HYBRID_KS_PROFILE_CSV — output path (default output/sim/hybrid_ks_profile.csv)
//   MOAI_SIM_HYBRID_KS_EXACT_PARTITION — default 1: skip α when |Ql|≥α and |Ql| mod α ≠ 0 (no equal-size digit split).
//     If |Ql|<α, β=1 (single short digit) — row kept. Set 0 to keep all α with phantom ceil(β).
//   Memory columns (u64 count × 8B): Phantom keyswitch_inplace buffers — c2, t_mod_up (β×QlP), cx (2×QlP);
//     mem_working_peak_bytes_est = sum if all resident (upper bound).
inline int moai_sim_hybrid_ks_profile_run() {
  const uint64_t N = env_u64("MOAI_SIM_POLY_DEGREE", kSingleLayerPolyModulusDegree());
  const uint64_t T_chain = env_u64("MOAI_SIM_NUM_LIMBS", static_cast<uint64_t>(kSingleLayerCoeffModulusCount()));
  const char *ksp_env = std::getenv("MOAI_SIM_KSWITCH_SIZE_P");
  const bool ksp_explicit = (ksp_env != nullptr && ksp_env[0] != '\0');
  const uint64_t num_P_fixed =
      ksp_explicit ? std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_KSWITCH_SIZE_P", 1)) : 0;
  const uint64_t size_Ql_override = env_u64("MOAI_SIM_KSWITCH_SIZE_Ql", 0);

  const uint64_t beta_env = env_u64("MOAI_SIM_KSWITCH_BETA", 0);
  const char *mode_ev = std::getenv("MOAI_SIM_KSWITCH_BETA_MODE");
  const bool legacy_mode = (mode_ev != nullptr && std::strcmp(mode_ev, "legacy") == 0);

  std::vector<uint64_t> alphas = parse_uint64_list_env("MOAI_SIM_HYBRID_KS_ALPHA_LIST");
  if (alphas.empty()) {
    if (const char *rng = std::getenv("MOAI_SIM_HYBRID_KS_ALPHA_RANGE"); rng != nullptr && rng[0] != '\0')
      parse_alpha_range_string(rng, alphas);
  }
  if (alphas.empty()) {
    alphas.push_back(std::max<uint64_t>(1ULL, env_u64("MOAI_SIM_ALPHA", 1)));
  }

  const bool measure_engine = env_bool("MOAI_SIM_HYBRID_KS_MEASURE_ENGINE", false);
  const bool exact_partition_only = env_bool("MOAI_SIM_HYBRID_KS_EXACT_PARTITION", true);

  const char *csv_path = std::getenv("MOAI_SIM_HYBRID_KS_PROFILE_CSV");
  const std::string csv_file =
      (csv_path != nullptr && csv_path[0] != '\0') ? std::string(csv_path) : std::string("output/sim/hybrid_ks_profile.csv");
  ensure_parent_dirs_for_file(csv_file.c_str());

  std::ofstream csv(csv_file.c_str(), std::ios::out | std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "[MOAI_HYBRID_KS_PROFILE] failed to open " << csv_file << "\n";
    return 2;
  }

  csv << "alpha,t_qp,size_Ql,num_P,beta_mode,beta,dnum,ntt_kernels,ntt_fwd_kernels,ntt_inv_kernels,bconv_modup,bconv_moddown,vec_mul,vec_add,"
         "weighted_ntt_coeff_elems,weighted_ntt_fwd_coeff_elems,weighted_ntt_inv_coeff_elems,weighted_bconv_coeff_elems,weighted_vec_mul_coeff_elems,weighted_vec_add_coeff_elems,"
         "bconv_modup_mmul_ops,bconv_modup_madd_ops,bconv_moddown_mmul_ops,bconv_moddown_madd_ops,"
         "bconv_mmul_ops,bconv_madd_ops,vec_mul_mmul_ops,vec_add_madd_ops,"
         "key_bytes_est,mem_c2_bytes,mem_modup_buf_bytes,mem_cx_buf_bytes,mem_working_peak_bytes_est,"
         "eng_ntt_enq,eng_ntt_fwd_enq,eng_ntt_inv_enq,eng_vec_enq,eng_vec_bconv_enq,eng_vec_arith_enq,"
         "eng_ntt_coeff_ops,eng_ntt_fwd_coeff_ops,eng_ntt_inv_coeff_ops,eng_vec_coeff_ops,eng_vec_bconv_coeff_ops,eng_vec_arith_coeff_ops,"
         "eng_vec_bconv_mmul_ops,eng_vec_bconv_madd_ops,eng_vec_arith_mmul_ops,eng_vec_arith_madd_ops,eng_vec_arith_mac_ops,"
         "eng_makespan_cyc,eng_ntt_busy_cyc,eng_ntt_fwd_busy_cyc,eng_ntt_inv_busy_cyc,eng_vec_busy_cyc,eng_vec_bconv_busy_cyc,eng_vec_arith_busy_cyc\n";

  std::cout << "[MOAI_HYBRID_KS_PROFILE] analytic = same graph as enqueue_keyswitch_phantom_ckks (not sampled).\n";
  if (measure_engine)
    std::cout << "[MOAI_HYBRID_KS_PROFILE] eng_* = EngineModel schedule() on profile_keyswitch_phantom_ckks "
                 "(MOAI_SIM_BACKEND=1): enq = schedule() calls; *_coeff_ops = logical_ops sum; "
                 "*_busy_cyc = EngineStats::busy_cycles; eng_makespan_cyc = Summary::makespan_cycles (crit.path).\n";
  std::cout << "N=" << N << " T_chain(|QP|)=" << T_chain << " size_Ql=per-row(|Q| default T−α) num_P=per-row(|P| default α)"
            << " beta_mode=" << (legacy_mode ? "legacy" : "phantom");
  if (exact_partition_only)
    std::cout << " exact_partition=1 (skip α if |Ql|≥α and |Ql| mod α ≠ 0; else β=⌈|Ql|/α⌉)";
  std::cout << "\n";
  std::cout << "csv=" << csv_file << "\n\n";

  for (uint64_t alpha : alphas) {
    if (alpha == 0) continue;
    if (alpha >= T_chain) {
      std::cerr << "[MOAI_HYBRID_KS_PROFILE] skip alpha=" << alpha << " (require alpha < T_qp=" << T_chain << ")\n";
      continue;
    }
    const uint64_t size_Ql =
        (size_Ql_override > 0)
            ? size_Ql_override
            : ((T_chain > alpha) ? (T_chain - alpha) : T_chain);
    if (exact_partition_only && size_Ql >= alpha && (size_Ql % alpha) != 0) {
      std::cerr << "[MOAI_HYBRID_KS_PROFILE] skip alpha=" << alpha << " (|Ql|=" << size_Ql
                << " ≥ α and remainder≠0; set MOAI_SIM_HYBRID_KS_EXACT_PARTITION=0 to include)\n";
      continue;
    }
    const uint64_t num_P = ksp_explicit ? num_P_fixed : std::max<uint64_t>(1ULL, alpha);

    uint64_t beta = 0;
    if (beta_env > 0)
      beta = beta_env;
    else if (legacy_mode)
      beta = keyswitch_beta_legacy(size_Ql, num_P);
    else
      beta = keyswitch_beta_phantom(size_Ql, alpha);

    const KeyswitchPhantomProfile prof =
        compute_keyswitch_phantom_profile(N, size_Ql, num_P, beta);
    const uint64_t dnum = hybrid_ks_dnum(T_chain, alpha);
    const uint64_t key_bytes_est = (dnum > 0) ? (dnum * 2ull * T_chain * N * 8ull) : 0;

    // Phantom eval_key_switch.cu keyswitch_inplace: t_mod_up = beta * size_QlP_n u64; cx = 2 * size_QlP_n u64; c2 is size_Ql_n.
    const uint64_t size_QlP = size_Ql + num_P;
    const uint64_t n_coeffs_ql = N * size_Ql;
    const uint64_t n_coeffs_qlp = N * size_QlP;
    constexpr uint64_t k_u64 = sizeof(uint64_t);
    const uint64_t mem_c2_bytes = n_coeffs_ql * k_u64;
    const uint64_t mem_modup_buf_bytes = beta * n_coeffs_qlp * k_u64;
    const uint64_t mem_cx_buf_bytes = 2ull * n_coeffs_qlp * k_u64;
    const uint64_t mem_working_peak_bytes_est = mem_c2_bytes + mem_modup_buf_bytes + mem_cx_buf_bytes;

    int64_t eng_ntt = -1;
    int64_t eng_ntt_fwd = -1;
    int64_t eng_ntt_inv = -1;
    int64_t eng_vec = -1;
    int64_t eng_ntt_coeff_ops = -1;
    int64_t eng_ntt_fwd_coeff_ops = -1;
    int64_t eng_ntt_inv_coeff_ops = -1;
    int64_t eng_vec_coeff_ops = -1;
    int64_t eng_vec_bconv_enq = -1;
    int64_t eng_vec_arith_enq = -1;
    int64_t eng_vec_bconv_coeff_ops = -1;
    int64_t eng_vec_arith_coeff_ops = -1;
    int64_t eng_makespan_cyc = -1;
    int64_t eng_ntt_busy_cyc = -1;
    int64_t eng_ntt_fwd_busy_cyc = -1;
    int64_t eng_ntt_inv_busy_cyc = -1;
    int64_t eng_vec_busy_cyc = -1;
    int64_t eng_vec_bconv_busy_cyc = -1;
    int64_t eng_vec_arith_busy_cyc = -1;
    int64_t eng_vec_bconv_mmul_ops = -1;
    int64_t eng_vec_bconv_madd_ops = -1;
    int64_t eng_vec_arith_mmul_ops = -1;
    int64_t eng_vec_arith_madd_ops = -1;
    int64_t eng_vec_arith_mac_ops = -1;
    if (measure_engine && EngineModel::enabled()) {
      {
        std::string av = std::to_string(alpha);
        (void)::setenv("MOAI_SIM_ALPHA", av.c_str(), 1);
      }
      unset_env_key("MOAI_SIM_KSWITCH_SIZE_P");
      EngineModel::instance().reset();
      (void)EngineModel::instance().profile_keyswitch_phantom_ckks(N, size_Ql, 0);
      const EngineModel::Summary sum = EngineModel::instance().summary();
      eng_ntt = static_cast<int64_t>(sum.ntt.enqueue_calls);
      eng_ntt_fwd = static_cast<int64_t>(sum.ntt_fwd.enqueue_calls);
      eng_ntt_inv = static_cast<int64_t>(sum.ntt_inv.enqueue_calls);
      eng_vec = static_cast<int64_t>(sum.vec.enqueue_calls);
      eng_ntt_coeff_ops = static_cast<int64_t>(sum.ntt.logical_ops);
      eng_ntt_fwd_coeff_ops = static_cast<int64_t>(sum.ntt_fwd.logical_ops);
      eng_ntt_inv_coeff_ops = static_cast<int64_t>(sum.ntt_inv.logical_ops);
      eng_vec_coeff_ops = static_cast<int64_t>(sum.vec.logical_ops);
      eng_vec_bconv_enq = static_cast<int64_t>(sum.vec_bconv.enqueue_calls);
      eng_vec_arith_enq = static_cast<int64_t>(sum.vec_arith.enqueue_calls);
      eng_vec_bconv_coeff_ops = static_cast<int64_t>(sum.vec_bconv.logical_ops);
      eng_vec_arith_coeff_ops = static_cast<int64_t>(sum.vec_arith.logical_ops);
      eng_makespan_cyc = static_cast<int64_t>(sum.makespan_cycles);
      eng_ntt_busy_cyc = static_cast<int64_t>(sum.ntt.busy_cycles);
      eng_ntt_fwd_busy_cyc = static_cast<int64_t>(sum.ntt_fwd.busy_cycles);
      eng_ntt_inv_busy_cyc = static_cast<int64_t>(sum.ntt_inv.busy_cycles);
      eng_vec_busy_cyc = static_cast<int64_t>(sum.vec.busy_cycles);
      eng_vec_bconv_busy_cyc = static_cast<int64_t>(sum.vec_bconv.busy_cycles);
      eng_vec_arith_busy_cyc = static_cast<int64_t>(sum.vec_arith.busy_cycles);
      eng_vec_bconv_mmul_ops = static_cast<int64_t>(sum.vec_bconv.mmul_ops);
      eng_vec_bconv_madd_ops = static_cast<int64_t>(sum.vec_bconv.madd_ops);
      eng_vec_arith_mmul_ops = static_cast<int64_t>(sum.vec_arith.mmul_ops);
      eng_vec_arith_madd_ops = static_cast<int64_t>(sum.vec_arith.madd_ops);
      eng_vec_arith_mac_ops = static_cast<int64_t>(sum.vec_arith.mac_ops);
    }

    csv << alpha << "," << T_chain << "," << size_Ql << "," << num_P << "," << (legacy_mode ? "legacy" : "phantom") << ","
        << beta << "," << dnum << "," << prof.ntt_kernels << "," << prof.ntt_fwd_kernels << "," << prof.ntt_inv_kernels
        << "," << prof.bconv_modup_kernels << "," << prof.bconv_moddown_kernels << "," << prof.vec_mul_kernels << ","
        << prof.vec_add_kernels << "," << prof.weighted_ntt_coeff_elems << "," << prof.weighted_ntt_fwd_coeff_elems << ","
        << prof.weighted_ntt_inv_coeff_elems << "," << prof.weighted_bconv_coeff_elems << ","
        << prof.weighted_vec_mul_coeff_elems << "," << prof.weighted_vec_add_coeff_elems << ","
        << prof.bconv_modup_mmul_ops << "," << prof.bconv_modup_madd_ops << "," << prof.bconv_moddown_mmul_ops << ","
        << prof.bconv_moddown_madd_ops << ","
        << prof.bconv_mmul_ops << "," << prof.bconv_madd_ops << "," << prof.vec_mul_mmul_ops << "," << prof.vec_add_madd_ops << ","
        << key_bytes_est << ","
        << mem_c2_bytes << "," << mem_modup_buf_bytes << "," << mem_cx_buf_bytes << "," << mem_working_peak_bytes_est
        << "," << eng_ntt << "," << eng_ntt_fwd << "," << eng_ntt_inv         << "," << eng_vec << "," << eng_vec_bconv_enq << ","
        << eng_vec_arith_enq << "," << eng_ntt_coeff_ops << "," << eng_ntt_fwd_coeff_ops << "," << eng_ntt_inv_coeff_ops
        << "," << eng_vec_coeff_ops << "," << eng_vec_bconv_coeff_ops << "," << eng_vec_arith_coeff_ops
        << "," << eng_vec_bconv_mmul_ops << "," << eng_vec_bconv_madd_ops << "," << eng_vec_arith_mmul_ops << "," << eng_vec_arith_madd_ops << ","
        << eng_vec_arith_mac_ops << ","
        << eng_makespan_cyc << "," << eng_ntt_busy_cyc << "," << eng_ntt_fwd_busy_cyc << "," << eng_ntt_inv_busy_cyc << ","
        << eng_vec_busy_cyc << "," << eng_vec_bconv_busy_cyc << "," << eng_vec_arith_busy_cyc << "\n";

    std::cout << "alpha=" << alpha << " beta=" << beta << " dnum=" << dnum << " NTT=" << prof.ntt_kernels
              << " NTT_fwd=" << prof.ntt_fwd_kernels << " NTT_inv=" << prof.ntt_inv_kernels
              << " BConv_up=" << prof.bconv_modup_kernels << " BConv_down=" << prof.bconv_moddown_kernels
              << " vec_mul=" << prof.vec_mul_kernels << " vec_add=" << prof.vec_add_kernels
              << " w_ntt_coeffs=" << prof.weighted_ntt_coeff_elems
              << " mem_peak_B=" << mem_working_peak_bytes_est << " key_B=" << key_bytes_est;
    if (eng_ntt >= 0)
      std::cout << " eng_ntt=" << eng_ntt << " eng_ntt_fwd=" << eng_ntt_fwd << " eng_ntt_inv=" << eng_ntt_inv
                << " eng_vec=" << eng_vec << " eng_ntt_coeff_ops=" << eng_ntt_coeff_ops
                << " eng_ntt_fwd_ops=" << eng_ntt_fwd_coeff_ops << " eng_ntt_inv_ops=" << eng_ntt_inv_coeff_ops
                << " eng_vec_coeff_ops=" << eng_vec_coeff_ops << " ms_cyc=" << eng_makespan_cyc
                << " ntt_busy_cyc=" << eng_ntt_busy_cyc << " vec_busy_cyc=" << eng_vec_busy_cyc;
    std::cout << "\n";
  }

  std::cout << "[MOAI_HYBRID_KS_PROFILE] done.\n";
  return 0;
}

}  // namespace sim
}  // namespace moai
