#pragma once

#include "include.cuh"

#include "source/sim/engine_config.h"
#include "source/sim/engine_model.h"
#include "source/sim/sim_ckks_defaults.h"
#include "source/sim/sim_timing.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

namespace moai {
namespace test {
namespace sim_primitive_detail {

// One .txt per primitive tag (not a single shared moai_sim_report append).
// - MOAI_SIM_REPORT_PATH unset: output/sim/primitive_<tag>.txt
// - MOAI_SIM_REPORT_PATH ends with '/' or '\\': directory; writes <dir>primitive_<tag>.txt
// - else: treat as file path; strip extension to stem; writes <dir><stem>_<tag>.txt
//   e.g. output/sim/moai_sim_report.txt -> output/sim/moai_sim_report_mul_plain.txt
inline std::string primitive_sim_report_path(const char *tag) {
  std::string safe(tag ? tag : "unknown");
  for (char &c : safe) {
    if (c == '/' || c == '\\' || c == ' ')
      c = '_';
  }

  const char *ev = std::getenv("MOAI_SIM_REPORT_PATH");
  if (ev == nullptr || ev[0] == '\0') {
    std::string d(::moai::sim::default_sim_report_path());
    const size_t slash = d.find_last_of("/\\");
    if (slash != std::string::npos)
      return d.substr(0, slash + 1) + "primitive_" + safe + ".txt";
    return std::string("output/sim/primitive_") + safe + ".txt";
  }

  std::string p(ev);
  const bool trailing_dir = !p.empty() && (p.back() == '/' || p.back() == '\\');
  if (trailing_dir) {
    while (!p.empty() && (p.back() == '/' || p.back() == '\\'))
      p.pop_back();
    if (!p.empty())
      p.push_back('/');
    else
      p = "./";
    return p + "primitive_" + safe + ".txt";
  }

  const size_t slash = p.find_last_of("/\\");
  std::string dir = slash == std::string::npos ? std::string() : p.substr(0, slash + 1);
  std::string fname = slash == std::string::npos ? p : p.substr(slash + 1);
  const size_t dot = fname.find_last_of('.');
  const std::string stem =
      (dot != std::string::npos && dot > 0) ? fname.substr(0, dot) : fname;
  if (dir.empty())
    dir = "./";
  return dir + stem + "_" + safe + ".txt";
}

template <typename Fn>
inline void primitive_block(const char *tag, uint64_t N, uint64_t T, uint64_t loops, bool quiet, Fn &&fn) {
  using ::moai::sim::EngineModel;
  using ::moai::sim::SimTiming;
  SimTiming::instance().reset();
  if (EngineModel::enabled())
    EngineModel::instance().reset();
  uint64_t dep = 0;
  for (uint64_t i = 0; i < loops; ++i)
    fn(dep);

  const std::string report_path = primitive_sim_report_path(tag);
  ::moai::sim::ensure_parent_dirs_for_file(report_path.c_str());
  std::ofstream report_ofs(report_path.c_str(), std::ios::out | std::ios::trunc);
  const bool report_ok = report_ofs.is_open();
  std::ostream &report_os = report_ok ? report_ofs : std::cout;

  const std::time_t now = std::time(nullptr);
  report_os << "\n=== MOAI_SIM_PRIMITIVE " << tag << " N=" << N << " T=" << T << " loops=" << loops
            << " ts=" << static_cast<long long>(now) << " ===\n";
  report_os << "report_file=" << report_path << "\n";
  SimTiming::instance().print_summary(report_os);
  if (EngineModel::enabled())
    EngineModel::instance().print_summary(report_os, tag);
  if (report_ok)
    report_ofs.flush();

  if (!quiet) {
    if (report_ok)
      std::cout << "[MOAI_SIM_PRIMITIVE] wrote " << report_path << "\n";
    else
      std::cerr << "[MOAI_SIM_PRIMITIVE] warning: could not open report file " << report_path << "\n";
  }
}

}  // namespace sim_primitive_detail

// Estimator-only: one homomorphic primitive (or "all") with SimTiming + optional EngineModel.
// Requires MOAI_SIM_BACKEND=1. N from MOAI_SIM_POLY_DEGREE. MOAI_SIM_NUM_LIMBS defaults to |QP| (chain prime count);
// effective ciphertext |Ql| = |QP|−MOAI_SIM_ALPHA when MOAI_SIM_NUM_LIMBS_COUNTS_QP=1 (Phantom hybrid default).
// MOAI_SIM_NUM_LIMBS_COUNTS_QP=0 keeps legacy (T is already |Ql|). See engine_config.h sim_effective_rns_limbs_for_ct.
// op: mul_plain | mul_ct | add_inplace | rescale | rotate | relin | modswitch | all
inline void moai_sim_primitive_micro_bench(const char *op) {
  using ::moai::sim::EngineModel;
  using ::moai::sim::SimTiming;
  using sim_primitive_detail::primitive_block;

  if (!SimTiming::enabled()) {
    std::cerr << "[MOAI_SIM_PRIMITIVE] requires MOAI_SIM_BACKEND=1\n";
    std::exit(2);
  }

  const uint64_t N =
      SimTiming::env_u64("MOAI_SIM_POLY_DEGREE", ::moai::sim::kSingleLayerPolyModulusDegree());
  const uint64_t T_qp = SimTiming::env_u64(
      "MOAI_SIM_NUM_LIMBS", static_cast<uint64_t>(::moai::sim::kSingleLayerCoeffModulusCount()));
  const uint64_t alpha = std::max<uint64_t>(1ULL, SimTiming::env_u64("MOAI_SIM_ALPHA", 1));
  const uint64_t T = ::moai::sim::sim_effective_rns_limbs_for_ct(T_qp, alpha);
  const uint64_t loops = SimTiming::env_u64("MOAI_SIM_PRIMITIVE_LOOPS", 1);

  const char *quiet_ev = std::getenv("MOAI_SIM_REPORT_QUIET");
  const bool quiet =
      quiet_ev != nullptr && quiet_ev[0] != '\0' && std::strcmp(quiet_ev, "0") != 0;

#if !defined(_WIN32)
  {
    char cwd_buf[4096];
    if (getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr)
      std::cout << "[MOAI_SIM_PRIMITIVE] cwd=" << cwd_buf << "\n";
  }
#endif
  std::cout << "[MOAI_SIM_PRIMITIVE] per-primitive reports (see MOAI_SIM_REPORT_PATH rules in test_sim_primitives.cuh)\n";
  if (T != T_qp)
    std::cout << "[MOAI_SIM_PRIMITIVE] hybrid default: T_qp=" << T_qp << " alpha=" << alpha << " -> effective |Ql| T=" << T
              << " (set MOAI_SIM_NUM_LIMBS_COUNTS_QP=0 for legacy T=T_qp)\n";

  auto run_mul_plain = [&]() {
    primitive_block(
        "mul_plain", N, T, loops, quiet, [&](uint64_t &dep) {
          SimTiming::instance().record_multiply_plain(N, T);
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_multiply_plain(2, N, T, dep);
        });
  };
  auto run_mul_ct = [&]() {
    primitive_block(
        "mul_ct", N, T, loops, quiet, [&](uint64_t &dep) {
          SimTiming::instance().record_ct_ct_multiply(N, T);
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_ct_ct_multiply(2, N, T, dep);
        });
  };
  auto run_add = [&]() {
    primitive_block(
        "add_inplace", N, T, loops, quiet, [&](uint64_t &dep) {
          SimTiming::instance().record_add_inplace(N, T);
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_add_inplace(2, N, T, dep);
        });
  };
  auto run_rescale = [&]() {
    primitive_block(
        "rescale", N, T, loops, quiet, [&](uint64_t &dep) {
          SimTiming::instance().record_rescale(N, T);
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_rescale(2, N, T, dep);
        });
  };
  auto run_rotate = [&]() {
    primitive_block(
        "rotate", N, T, loops, quiet, [&](uint64_t &dep) {
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_rotate(2, N, T, dep);
        });
  };
  auto run_relin = [&]() {
    primitive_block(
        "relinearize", N, T, loops, quiet, [&](uint64_t &dep) {
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_relinearize(2, N, T, dep);
        });
  };
  auto run_modswitch = [&]() {
    primitive_block(
        "modswitch", N, T, loops, quiet, [&](uint64_t &dep) {
          if (EngineModel::enabled())
            dep = EngineModel::instance().enqueue_modswitch(2, N, T, dep);
        });
  };

  const bool all = (op == nullptr || op[0] == '\0' || std::strcmp(op, "all") == 0);

  if (all) {
    run_mul_plain();
    run_mul_ct();
    run_add();
    run_rescale();
    run_rotate();
    run_relin();
    run_modswitch();
  } else if (std::strcmp(op, "mul_plain") == 0 || std::strcmp(op, "ct_pt") == 0) {
    run_mul_plain();
  } else if (std::strcmp(op, "mul_ct") == 0 || std::strcmp(op, "ct_ct") == 0) {
    run_mul_ct();
  } else if (std::strcmp(op, "add_inplace") == 0 || std::strcmp(op, "add") == 0) {
    run_add();
  } else if (std::strcmp(op, "rescale") == 0) {
    run_rescale();
  } else if (std::strcmp(op, "rotate") == 0) {
    run_rotate();
  } else if (std::strcmp(op, "relin") == 0 || std::strcmp(op, "relinearize") == 0) {
    run_relin();
  } else if (std::strcmp(op, "modswitch") == 0) {
    run_modswitch();
  } else {
    std::cerr << "[MOAI_SIM_PRIMITIVE] unknown op '" << (op ? op : "") << "' — use mul_plain | mul_ct | add_inplace | "
                 "rescale | rotate | relin | modswitch | all\n";
    std::exit(2);
  }

  if (!quiet) {
    // Re-print last block only is confusing for "all"; mirror ct_ct: user can open file.
    if (!all) {
      SimTiming::instance().print_summary(std::cout);
      if (EngineModel::enabled())
        EngineModel::instance().print_summary(std::cout, op);
    }
  }

  std::cout << "[MOAI_SIM_PRIMITIVE] finished (estimator-only; no GPU).\n";
}

}  // namespace test
}  // namespace moai

inline void moai_sim_primitive_micro_bench(const char *op) {
  ::moai::test::moai_sim_primitive_micro_bench(op);
}
