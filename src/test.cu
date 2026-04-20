#include "include.cuh"
#include "source/sim/engine_config.h"
#include "source/sim/keyswitch_op_profile.h"

#include <cstdio>
#include <cstring>
#include <string>

#if !defined(_WIN32)
// Tee-style redirect: whole-process stdout goes under output/test_logs/<name>.txt
static void moai_redirect_test_stdout() {
  if (const char *d = std::getenv("MOAI_TEST_OUTPUT_DISABLE"); d != nullptr && d[0] != '\0' && std::strcmp(d, "0") != 0)
    return;

  const char *explicit_path = std::getenv("MOAI_TEST_OUTPUT_PATH");
  const char *bench = std::getenv("MOAI_BENCH_MODE");

  std::string path;
  if (explicit_path != nullptr && explicit_path[0] != '\0') {
    path = explicit_path;
  } else {
    std::string tag = "default";
    if (bench != nullptr && bench[0] != '\0') {
      tag.assign(bench);
      for (char &c : tag) {
        if (c == '/' || c == '\\')
          c = '_';
      }
    }
    path = std::string("output/test_logs/") + tag + ".txt";
  }

  const char *mode = "w";
  if (const char *a = std::getenv("MOAI_TEST_OUTPUT_APPEND"); a != nullptr && a[0] != '\0' && std::strcmp(a, "0") != 0)
    mode = "a";

  ::moai::sim::ensure_parent_dirs_for_file(path.c_str());
  std::fflush(stdout);
  if (std::freopen(path.c_str(), mode, stdout) == nullptr) {
    std::perror("[MOAI_TEST] freopen stdout failed");
  } else {
    std::fprintf(stderr, "[MOAI_TEST] stdout -> %s (mode=%s)\n", path.c_str(), mode);
  }
}
#endif
// #include "test_ct_pt_matrix_mul.cuh"
// #include "test_phantom_ckks.cuh"
// #include "test_batch_encode_encrypt.cuh"
// #include "test_ct_ct_matrix_mul.cuh"
// #include "test_gelu.cuh"
// #include "test_layernorm.cuh"
// #include "test_softmax.cuh"
// single layer test is included by include.cuh
#include <cuda_runtime.h>
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace moai;

int main()
{
#if !defined(_WIN32)
    moai_redirect_test_stdout();
#endif
    // Micro-benchmarks only (fixed CKKS params inside each test; no MOAI_ALPHA):
    //   Sim reports (MOAI_SIM_BACKEND=1): ct_pt/ct_ct append output/sim/moai_sim_report.txt by default.
    //   sim_* primitives: one file per op — output/sim/primitive_<tag>.txt (see test_sim_primitives.cuh MOAI_SIM_REPORT_PATH).
    //   You must set MOAI_BENCH_MODE below; plain ./build/test does not run ct_pt.
    //   CT×CT knobs: MOAI_SIM_CT_CT_VEC_MUL_PASSES (default 3, coarse tensor/RNS proxy in enqueue_ct_ct_multiply).
    //   Keyswitch (Phantom-aligned coarse): MOAI_SIM_KSWITCH_SIZE_P (|P| in QlP), MOAI_SIM_ALPHA (digit size),
    //   MOAI_SIM_KSWITCH_BETA (0=auto: phantom ceil(|Ql|/alpha), or legacy via MOAI_SIM_KSWITCH_BETA_MODE=legacy),
    //   MOAI_SIM_KSWITCH_MODUP_BCONV_CYC_PER_COEFF, MOAI_SIM_KSWITCH_MODDOWN_BCONV_CYC_PER_COEFF, MOAI_SIM_GALOIS_PERM_CYC_PER_COEFF.
    //   Rotate/relin through Evaluator use EngineModel only when MOAI_SIM_GAP_POLICY=model; ct_ct estimator calls
    //   enqueue_rotate / enqueue_relinearize directly so engine sees relin/rotate traffic regardless.
    //   MOAI_BENCH_MODE=boot | bootstrap_micro -> bootstrapping_test() (nsys: src/scripts/profile_bootstrap_micro.sh)
    //   MOAI_BENCH_MODE=ct_pt      -> ct_pt_matrix_mul_test() (wo_pre; nsys: profile_ct_pt_micro.sh)
    //   MOAI_BENCH_MODE=ct_pt_pre  -> ct_pt_matrix_mul_w_preprocess_test() (ecd W; nsys: profile_ct_pt_pre_micro.sh)
    //   MOAI_BENCH_MODE=ct_ct   -> ct_ct_matrix_mul_test()  (nsys: src/scripts/profile_ct_ct_micro.sh)
    //   MOAI_BENCH_MODE=softmax_micro -> softmax_test()+softmax_boot_test() (nsys: src/scripts/profile_softmax_micro.sh)
    //   MOAI_BENCH_MODE=softmax | softmax_boot -> single test only
    //   MOAI_BENCH_MODE=gelu      -> gelu_test() (nsys: src/scripts/profile_gelu_micro.sh)
    //   MOAI_BENCH_MODE=layernorm -> layernorm_test() (nsys: src/scripts/profile_layernorm_micro.sh)
    //   Per-primitive sim (estimator-only, MOAI_SIM_BACKEND=1): MOAI_BENCH_MODE=sim_primitive + MOAI_SIM_PRIMITIVE
    //   (mul_plain|mul_ct|add_inplace|rescale|rotate|relin|modswitch|all). Shortcuts: sim_mul_plain, sim_mul_ct,
    //   sim_add_inplace, sim_rescale, sim_rotate, sim_relin, sim_modswitch, sim_primitives (= all).
    //   N,T: MOAI_SIM_POLY_DEGREE, MOAI_SIM_NUM_LIMBS (defaults = single_layer: sim_ckks_defaults.h, N=65536 T=36);
    //   repeat: MOAI_SIM_PRIMITIVE_LOOPS (default 1).
    //   Hybrid KS op profile (no GPU): MOAI_BENCH_MODE=sim_hybrid_ks_profile — CSV output/sim/hybrid_ks_profile.csv;
    //   MOAI_SIM_HYBRID_KS_ALPHA_LIST=1,2,4,8 MOAI_SIM_KSWITCH_SIZE_Ql (default |Ql|=T), see keyswitch_op_profile.h.
    //   Stdout log: default output/test_logs/<MOAI_BENCH_MODE>.txt (or default.txt if unset). Override with
    //   MOAI_TEST_OUTPUT_PATH, disable with MOAI_TEST_OUTPUT_DISABLE=1, append with MOAI_TEST_OUTPUT_APPEND=1.
    //   rotate/relin/modswitch: SimTiming coarse rows stay 0; engine model carries traffic (keyswitch etc.).
    // Unset -> single_layer_test() below.
    if (const char *bench = std::getenv("MOAI_BENCH_MODE");
        bench != nullptr && bench[0] != '\0') {
        if (std::strcmp(bench, "boot") == 0 || std::strcmp(bench, "bootstrap_micro") == 0) {
            bootstrapping_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt") == 0) {
            ct_pt_matrix_mul_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt_sanity") == 0) {
            ct_pt_matrix_mul_sanity_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt_sanity_small") == 0) {
            ct_pt_matrix_mul_sanity_small_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt_pre") == 0) {
            ct_pt_matrix_mul_w_preprocess_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt_proj_compare") == 0) {
            ct_pt_proj_matmul_bench_single_layer_compare();
            return 0;
        }
        if (std::strcmp(bench, "ct_ct") == 0) {
            ct_ct_matrix_mul_test();
            return 0;
        }
        if (std::strcmp(bench, "softmax_micro") == 0) {
            softmax_micro_bench();
            return 0;
        }
        if (std::strcmp(bench, "softmax") == 0) {
            softmax_test();
            return 0;
        }
        if (std::strcmp(bench, "softmax_boot") == 0) {
            softmax_boot_test();
            return 0;
        }
        if (std::strcmp(bench, "gelu") == 0) {
            gelu_test();
            return 0;
        }
        if (std::strcmp(bench, "layernorm") == 0) {
            layernorm_test();
            return 0;
        }
        if (std::strcmp(bench, "sim_primitive") == 0) {
            const char *p = std::getenv("MOAI_SIM_PRIMITIVE");
            moai_sim_primitive_micro_bench((p != nullptr && p[0] != '\0') ? p : "all");
            return 0;
        }
        if (std::strcmp(bench, "sim_mul_plain") == 0) {
            moai_sim_primitive_micro_bench("mul_plain");
            return 0;
        }
        if (std::strcmp(bench, "sim_mul_ct") == 0) {
            moai_sim_primitive_micro_bench("mul_ct");
            return 0;
        }
        if (std::strcmp(bench, "sim_add_inplace") == 0) {
            moai_sim_primitive_micro_bench("add_inplace");
            return 0;
        }
        if (std::strcmp(bench, "sim_rescale") == 0) {
            moai_sim_primitive_micro_bench("rescale");
            return 0;
        }
        if (std::strcmp(bench, "sim_rotate") == 0) {
            moai_sim_primitive_micro_bench("rotate");
            return 0;
        }
        if (std::strcmp(bench, "sim_relin") == 0) {
            moai_sim_primitive_micro_bench("relin");
            return 0;
        }
        if (std::strcmp(bench, "sim_modswitch") == 0) {
            moai_sim_primitive_micro_bench("modswitch");
            return 0;
        }
        if (std::strcmp(bench, "sim_primitives") == 0) {
            moai_sim_primitive_micro_bench("all");
            return 0;
        }
        if (std::strcmp(bench, "sim_hybrid_ks_profile") == 0) {
            return moai::sim::moai_sim_hybrid_ks_profile_run();
        }
        if (std::strcmp(bench, "single_layer") == 0) {
            single_layer_test();
            return 0;
        }
        std::cerr << "MOAI_BENCH_MODE='" << bench
                  << "' — use boot | bootstrap_micro | ct_pt | ct_pt_sanity | ct_pt_pre | ct_ct | softmax_micro | softmax | "
                     "softmax_boot | gelu | "
                     "layernorm | single_layer | ct_pt_proj_compare | sim_primitive | sim_primitives | sim_hybrid_ks_profile | sim_mul_plain | sim_mul_ct | sim_add_inplace | "
                     "sim_rescale | sim_rotate | sim_relin | sim_modswitch | "
                     "(unset for single_layer)\n";
        return 2;
    }

    // cout << "test Phantom ckks" << endl;
    // phantom_ckks_test();
    // cout << "lib test passed!" << endl;

    // cout << "unit test: Batch encode and encrypt" << endl;
    // batch_input_test();
    // cout << "unit test Batch encode and encrypt passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication without preprocessing" << endl;
    // ct_pt_matrix_mul_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication with preprocessing" << endl;
    // ct_pt_matrix_mul_w_preprocess_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-ct matrix multiplication" << endl;
    // ct_ct_matrix_mul_test();
    // cout << "unit test Ct-ct matrix multiplication passed!" << endl;

    // cout << "unit test: GeLU" << endl;
    // gelu_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: LayerNorm" << endl;
    // layernorm_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Softmax" << endl;
    // softmax_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Bootstrapping" << endl;
    // bootstrapping_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: softmax with bootstrapping" << endl;
    // softmax_boot_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: BPmax test" << endl;
    // BPmax_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: BatchLN_test" << endl;
    // BatchLN_test();
    // cout << "unit test passed!" << endl;

    // Precomputed keys (optional): export MOAI_PRECOMPUTED_KEYS_DIR=/path/to/keys_dnum_35
    // or MOAI_KEYS_BASE=/path/to/keys and MOAI_ALPHA=1 (selects .../keys_dnum_<dnum>).
   //cout << "single layer test" << endl;
   // single_layer_test();
   // cout << "single layer test passed!" << endl;

    // cout << "Rotary Position Embedding test" << endl;
    // rotary_pos_embed_test();
    // cout << "Rotary Position Embedding test passed!" << endl;

    // cout << "Causal Masked Softmax test" << endl;
    // causal_masked_softmax_test();
    // cout << "Causal Masked Softmax test passed!" << endl;

    // cout << "SiLU coefficient export test" << endl;
    // silu_coeff_verify();
    // cout << "SiLU coefficient export test passed!" << endl;

    // cout << "SiLU coefficient export test" << endl;
    // silu_test();
    // cout << "SiLU coefficient export test passed!" << endl;

    // cout << "RMSNorm test" << endl;
    // RMSNorm_test();
    // cout << "RMSNorm test passed!" << endl;

    return 0;
}
