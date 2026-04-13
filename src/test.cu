#include "include.cuh"
#include <cstring>
// #include "test_ct_pt_matrix_mul.cuh"
// #include "test_phantom_ckks.cuh"
// #include "test_batch_encode_encrypt.cuh"
// #include "test_ct_ct_matrix_mul.cuh"
// #include "test_gelu.cuh"
// #include "test_layernorm.cuh"
// #include "test_softmax.cuh"
// #include "test_single_layer.cuh"
#include <cuda_runtime.h>
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace moai;

int main()
{
    // Micro-benchmarks only (fixed CKKS params inside each test; no MOAI_ALPHA):
    //   MOAI_BENCH_MODE=boot    -> bootstrapping_test()
    //   MOAI_BENCH_MODE=ct_pt   -> ct_pt_matrix_mul_test()
    //   MOAI_BENCH_MODE=ct_ct   -> ct_ct_matrix_mul_test()
    // Unset -> single_layer_test() below.
    if (const char *bench = std::getenv("MOAI_BENCH_MODE");
        bench != nullptr && bench[0] != '\0') {
        if (std::strcmp(bench, "boot") == 0) {
            bootstrapping_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_pt") == 0) {
            ct_pt_matrix_mul_test();
            return 0;
        }
        if (std::strcmp(bench, "ct_ct") == 0) {
            ct_ct_matrix_mul_test();
            return 0;
        }
        std::cerr << "MOAI_BENCH_MODE='" << bench
                  << "' — use boot | ct_pt | ct_ct | (unset for single_layer)\n";
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
