#include "include.cuh"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ctime>

#if defined(MOAI_HAVE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif
#include "source/sim/sim_timing.h"

using namespace std;
using namespace phantom;
using namespace moai;

void ct_pt_matrix_mul_sanity_test()
{
    cout << "Task: sanity check encode(double) fast path preserves functionality" << endl;

    // Small parameters for functional verification (keep runtime small/stable).
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    parms.set_sparse_slots(static_cast<long>(poly_modulus_degree / 2));
    double scale = pow(2.0, 40);

    PhantomContext context(parms);
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Encryptor encryptor(&context, &public_key);

    const size_t slot_count = encoder.slot_count();

    // Build a plaintext slot vector x and encrypt it.
    vector<double> x(slot_count, 0.0);
    for (size_t s = 0; s < std::min<size_t>(slot_count, 32); ++s)
        x[s] = 0.01 * static_cast<double>(s + 1); // small non-constant values

    PhantomPlaintext pt_x;
    encoder.encode(x, scale, pt_x);
    PhantomCiphertext ct_x;
    encryptor.encrypt(pt_x, ct_x);

    // Multiply by a broadcast scalar w via plaintext encode.
    const double w = -0.125;

    // legacy plaintext: explicit slot vector
    PhantomPlaintext pt_w_vec;
    vector<double> wvec(slot_count, w);
    encoder.encode(wvec, ct_x.params_id(), ct_x.scale(), pt_w_vec);

    // fast plaintext: scalar encode (uniform path)
    PhantomPlaintext pt_w_uni;
    encoder.encode(w, ct_x.params_id(), ct_x.scale(), pt_w_uni);

    Evaluator evaluator(&context, &phantom_encoder);
    PhantomCiphertext ct_y_vec, ct_y_uni;
    evaluator.multiply_plain(ct_x, pt_w_vec, ct_y_vec);
    evaluator.multiply_plain(ct_x, pt_w_uni, ct_y_uni);

    PhantomPlaintext p_y_vec, p_y_uni;
    decryptor.decrypt(ct_y_vec, p_y_vec);
    decryptor.decrypt(ct_y_uni, p_y_uni);

    vector<double> y_vec, y_uni;
    encoder.decode(p_y_vec, y_vec);
    encoder.decode(p_y_uni, y_uni);

    double max_err_vec = 0.0;
    double max_err_uni = 0.0;
    double max_diff = 0.0;
    for (size_t s = 0; s < std::min<size_t>(slot_count, 32); ++s)
    {
        const double expected = x[s] * w;
        max_err_vec = std::max(max_err_vec, std::fabs(y_vec[s] - expected));
        max_err_uni = std::max(max_err_uni, std::fabs(y_uni[s] - expected));
        max_diff = std::max(max_diff, std::fabs(y_uni[s] - y_vec[s]));
    }

    const double tol = 1e-3;
    cout << "[SANITY] w=" << w << " tol=" << tol << endl;
    cout << "[SANITY] max_abs_err_vs_expected (legacy vec) = " << max_err_vec << endl;
    cout << "[SANITY] max_abs_err_vs_expected (fast uni)   = " << max_err_uni << endl;
    cout << "[SANITY] max_abs_diff(fast,legacy)           = " << max_diff << endl;

    if (max_err_vec <= tol && max_err_uni <= tol && max_diff <= tol)
        cout << "[SANITY] PASS" << endl;
    else
        cout << "[SANITY] FAIL" << endl;
}

void ct_pt_matrix_mul_test()
{
    cout << "Task: test Ct-Pt matrix multiplication in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    const std::vector<int> coeff_bits = {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60};
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bits));
    //{60, 40, 40, 60}));
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    // ct_pt micro dims (W is num_col x col_W; X provides num_col ciphertexts)
    const int num_X = 256;
    const int num_row = 128;
    const int num_col = 768;
    const int col_W_bench = 64;

    if (::moai::sim::SimTiming::enabled())
    {
        // Optional: append reports to a file for easier sweep analysis.
        // Usage: MOAI_SIM_REPORT_PATH=/path/to/report.txt
        const char *report_path = std::getenv("MOAI_SIM_REPORT_PATH");
        const char *quiet_ev = std::getenv("MOAI_SIM_REPORT_QUIET");
        const bool quiet =
            quiet_ev != nullptr && quiet_ev[0] != '\0' && std::strcmp(quiet_ev, "0") != 0;
        std::ofstream report_ofs;
        if (report_path && report_path[0] != '\0')
            report_ofs.open(report_path, std::ios::out | std::ios::app);
        std::ostream &report_os = (report_ofs.is_open() ? report_ofs : std::cout);

        // Estimator-only path: avoid allocating GPU ciphertext/plaintext buffers.
        const uint64_t N = static_cast<uint64_t>(poly_modulus_degree);
        const uint64_t T = static_cast<uint64_t>(coeff_bits.size());
        const uint64_t slot_count = static_cast<uint64_t>(sparse_slots);
        const int row_W = num_col;

        // Mode:
        // - default: fast-path scalar encode (encode_uniform_real)
        // - MOAI_SIM_ENCODE_LEGACY=1: legacy scalar encode via full slot vector encode
        // - MOAI_SIM_COMPARE_LEGACY=1: print both back-to-back
        const char *cmp_ev = std::getenv("MOAI_SIM_COMPARE_LEGACY");
        const bool compare =
            cmp_ev != nullptr && cmp_ev[0] != '\0' && std::strcmp(cmp_ev, "0") != 0;
        const char *leg_ev = std::getenv("MOAI_SIM_ENCODE_LEGACY");
        const bool legacy_only =
            leg_ev != nullptr && leg_ev[0] != '\0' && std::strcmp(leg_ev, "0") != 0;

        auto run_once = [&](bool legacy) {
            ::moai::sim::SimTiming::instance().reset();
            if (::moai::sim::EngineModel::enabled())
                ::moai::sim::EngineModel::instance().reset();
            const char *once_ev = std::getenv("MOAI_SIM_ENCODE_ONCE");
            const bool encode_once =
                once_ev != nullptr && once_ev[0] != '\0' && std::strcmp(once_ev, "0") != 0;

            if (encode_once)
            {
                // Model "L=1 pre-encode then reuse on-chip" for the scalar/plaintext constant path.
                if (legacy)
                    ::moai::sim::SimTiming::instance().record_encode_vec(slot_count);
                else
                    ::moai::sim::SimTiming::instance().record_encode();

                if (::moai::sim::EngineModel::enabled())
                {
                    // Encode cost is VEC-side; represent legacy as slot_count work, uniform as 1 slot.
                    if (legacy)
                        ::moai::sim::EngineModel::instance().enqueue_vec_coeffs(slot_count, ::moai::sim::SimTiming::instance().encode_vec_cycles_per_slot());
                    else
                        ::moai::sim::EngineModel::instance().enqueue_vec_coeffs(1, ::moai::sim::SimTiming::instance().encode_cycles_per_call());

                    // If on-chip is enabled, treat the encoded PT constant as resident in GlobalSPAD.
                    const uint64_t coeffs = N * T;
                    ::moai::sim::EngineModel::instance().mark_pt_const_resident(coeffs * sizeof(uint64_t));
                }
            }

            for (int i = 0; i < col_W_bench; ++i)
            {
                for (int j = 0; j < row_W; ++j)
                {
                    if (!encode_once)
                    {
                        if (legacy)
                            ::moai::sim::SimTiming::instance().record_encode_vec(slot_count);
                        else
                            ::moai::sim::SimTiming::instance().record_encode();

                        if (::moai::sim::EngineModel::enabled())
                        {
                            if (legacy)
                                ::moai::sim::EngineModel::instance().enqueue_vec_coeffs(slot_count, ::moai::sim::SimTiming::instance().encode_vec_cycles_per_slot());
                            else
                                ::moai::sim::EngineModel::instance().enqueue_vec_coeffs(1, ::moai::sim::SimTiming::instance().encode_cycles_per_call());
                        }
                    }

                    ::moai::sim::SimTiming::instance().record_multiply_plain(N, T);
                    if (::moai::sim::EngineModel::enabled())
                        ::moai::sim::EngineModel::instance().enqueue_multiply_plain(/*ct_size=*/2, N, T);
                    if (j != 0)
                    {
                        ::moai::sim::SimTiming::instance().record_add_inplace(N, T);
                        if (::moai::sim::EngineModel::enabled())
                            ::moai::sim::EngineModel::instance().enqueue_add_inplace(/*ct_size=*/2, N, T);
                    }
                }
                ::moai::sim::SimTiming::instance().record_rescale(N, T);
                if (::moai::sim::EngineModel::enabled())
                    ::moai::sim::EngineModel::instance().enqueue_rescale(/*ct_size=*/2, N, T);
            }

            // Header separator for multi-run appends.
            const std::time_t now = std::time(nullptr);
            report_os << "\n=== MOAI_SIM_REPORT ct_pt_wo_pre ts=" << static_cast<long long>(now)
                      << " encode_model=" << (legacy ? "legacy_vec" : "uniform_real")
                      << " encode_once=" << (encode_once ? 1 : 0)
                      << " ===\n";
            ::moai::sim::SimTiming::instance().print_summary(report_os);
            if (::moai::sim::EngineModel::enabled())
                ::moai::sim::EngineModel::instance().print_summary(report_os, "ct_pt_wo_pre");

            if (!quiet)
            {
                cout << "[MOAI_SIM_BACKEND] ct_pt_wo_pre encode_model=" << (legacy ? "legacy_vec" : "uniform_real")
                     << " encode_once=" << (encode_once ? 1 : 0) << endl;
                if (report_ofs.is_open())
                    cout << "[MOAI_SIM_BACKEND] report appended to " << report_path << endl;
                ::moai::sim::SimTiming::instance().print_summary(std::cout);
                if (::moai::sim::EngineModel::enabled())
                    ::moai::sim::EngineModel::instance().print_summary(std::cout, "ct_pt_wo_pre");
            }
        };

        if (compare)
        {
            run_once(false);
            run_once(true);
        }
        else
        {
            run_once(legacy_only);
        }

        cout << "[MOAI_SIM_BACKEND] (ct_pt wo_pre) finished; skipping GPU run." << endl;
        return;
    }

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // PhantomKeyGenerator keygen(context);
    // PhantomSecretKey secret_key = keygen.secret_key();
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // PhantomDecryptor decryptor(context, secret_key);
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Encryptor encryptor(&context, &public_key);
    Evaluator evaluator(&context, &phantom_encoder);

    size_t slot_count = encoder.slot_count();

    // struct timeval tstart1, tend1;

    // construct input
    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_x[i][j][k] = (double)j + 1.0;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }
    }
    cout << "Matrix X size = " << num_row << " * " << num_col << endl;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    /*
    //decrypt


    for (int i = 0; i < num_col; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<"decrypt + decode result of "<<i+1<<"-th ciphertext: "<<endl;
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    */

    // construct W
    int col_W = 64;
    vector<vector<double>> W(num_col, vector<double>(col_W, 1.0 / 128.0));
    cout << "Matrix W size = " << num_col << " * " << col_W << endl;

    // Optional encoder equivalence check (no decrypt involved):
    // Compare plaintext coeffs produced by legacy slot-vector encode vs uniform fast path.
    // Enable: MOAI_ENC_EQ_CHECK=1
    if (const char *ev = std::getenv("MOAI_ENC_EQ_CHECK"); ev && std::strcmp(ev, "1") == 0)
    {
        cout << "[MOAI_ENC_EQ] Checking encode equivalence (legacy vec vs uniform scalar)..." << endl;
        const double w = W[0][0];
        PhantomPlaintext pt_vec, pt_uni;

        // (1) legacy: explicit slot vector
        vector<double> wvec(slot_count, w);
        encoder.encode(wvec, enc_ecd_x[0].params_id(), enc_ecd_x[0].scale(), pt_vec);

        // (2) uniform fast path: scalar encode (routes to encode_uniform_real)
        encoder.encode(w, enc_ecd_x[0].params_id(), enc_ecd_x[0].scale(), pt_uni);

        const size_t coeff_mod_size = context.get_context_data(enc_ecd_x[0].params_id()).parms().coeff_modulus().size();
        const size_t poly_degree = context.get_context_data(enc_ecd_x[0].params_id()).parms().poly_modulus_degree();
        const size_t n_u64 = coeff_mod_size * poly_degree;

        vector<uint64_t> h_vec(n_u64), h_uni(n_u64);
        cudaMemcpy(h_vec.data(), pt_vec.data(), n_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uni.data(), pt_uni.data(), n_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        size_t diff_cnt = 0;
        size_t first_diff = static_cast<size_t>(-1);
        for (size_t i = 0; i < n_u64; ++i)
        {
            if (h_vec[i] != h_uni[i])
            {
                if (diff_cnt == 0) first_diff = i;
                diff_cnt++;
            }
        }
        cout << "[MOAI_ENC_EQ] coeff_u64_count=" << n_u64
             << " diff_cnt=" << diff_cnt;
        if (diff_cnt)
        {
            cout << " first_diff_idx=" << first_diff
                 << " vec=" << h_vec[first_diff]
                 << " uni=" << h_uni[first_diff];
        }
        cout << endl;
    }

    /*
    //encode W
    vector<vector<Plaintext>> ecd_w(num_col,vector<Plaintext>(col_W));
    for (int i = 0; i < num_col; ++i){
        for (int j = 0 ; j < col_W ; ++j){
            encoder.encode(W[i][j], scale, ecd_w[i][j]);
        }
    }
    cout <<"encode W. "<<endl;
    */
    cout << "Encrypted col-packing X * ecd W = Encrypted col-packing XW. " << endl;

    // matrix multiplication
    //  gettimeofday(&tstart1,NULL);
    std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();
    cudaDeviceSynchronize();
#if defined(MOAI_SIM_BACKEND)
    (void)start;
#endif
#if defined(MOAI_HAVE_NVTX)
    nvtxRangePushA("moai:ct_pt_matrix_mul_wo_pre");
#endif
    if (::moai::sim::SimTiming::enabled())
    {
        ::moai::sim::SimTiming::instance().reset();
    }
    vector<PhantomCiphertext> ct_pt_mul = ct_pt_matrix_mul_wo_pre(enc_ecd_x, W, num_col, col_W, num_col, context);
#if defined(MOAI_HAVE_NVTX)
    cudaDeviceSynchronize();
    nvtxRangePop();
#endif

    // gettimeofday(&tend1,NULL);
    // double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    // cout <<"Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    std::chrono::_V2::system_clock::time_point end = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaDeviceSynchronize();
    cout << "Ct-Pt matrix multiplication time = " << duration.count() << " ms" << endl;
    if (::moai::sim::SimTiming::enabled())
    {
        ::moai::sim::SimTiming::instance().print_summary(std::cout);
        cout << "[MOAI_SIM_BACKEND] skipping decrypt/decode correctness checks." << endl;
        return;
    }

    cout << "Modulus chain index for the result: " << context.get_context_data(ct_pt_mul[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "ct_pt_matrix_mul_without_preprocessing", duration.count() / 1000.0);

    // Optional correctness check: compare fast-path (current) vs legacy scalar-encode path.
    // Enable: MOAI_CT_PT_CHECK=1
    if (const char *ev = std::getenv("MOAI_CT_PT_CHECK"); ev && std::strcmp(ev, "1") == 0)
    {
        cout << "[MOAI_CHECK] Running legacy scalar-encode reference..." << endl;
        vector<PhantomCiphertext> ct_pt_mul_legacy =
            ct_pt_matrix_mul_wo_pre_legacy_scalar_encode(enc_ecd_x, W, num_col, col_W, num_col, context);

        const double expected = 64.5; // sum_{r=1..128} r / 128 = 64.5 (given input_x[j][k][i] = (double)k+1, W=1/128)
        const double abs_tol = 1e-3;
        double max_abs_err_fast = 0.0;
        double max_abs_err_legacy = 0.0;
        double max_abs_diff = 0.0;

        // Check a few output columns to keep runtime reasonable.
        for (int out_i = 0; out_i < std::min(8, col_W); ++out_i)
        {
            PhantomPlaintext plain_fast, plain_legacy;
            decryptor.decrypt(ct_pt_mul[out_i], plain_fast);
            decryptor.decrypt(ct_pt_mul_legacy[out_i], plain_legacy);

            vector<double> v_fast, v_legacy;
            encoder.decode(plain_fast, v_fast);
            encoder.decode(plain_legacy, v_legacy);

            for (size_t s = 0; s < v_fast.size() && s < v_legacy.size(); ++s)
            {
                const double ef = std::fabs(v_fast[s] - expected);
                const double el = std::fabs(v_legacy[s] - expected);
                const double ed = std::fabs(v_fast[s] - v_legacy[s]);
                max_abs_err_fast = std::max(max_abs_err_fast, ef);
                max_abs_err_legacy = std::max(max_abs_err_legacy, el);
                max_abs_diff = std::max(max_abs_diff, ed);
            }
        }

        cout << "[MOAI_CHECK] max_abs_err_vs_expected (fast)   = " << max_abs_err_fast << endl;
        cout << "[MOAI_CHECK] max_abs_err_vs_expected (legacy) = " << max_abs_err_legacy << endl;
        cout << "[MOAI_CHECK] max_abs_diff(fast,legacy)        = " << max_abs_diff << endl;

        if (max_abs_diff > abs_tol)
        {
            cout << "[MOAI_CHECK] WARNING: fast vs legacy diff exceeds abs_tol=" << abs_tol << endl;
        }
        if (max_abs_err_fast > 1.0 || max_abs_err_legacy > 1.0)
        {
            cout << "[MOAI_CHECK] WARNING: decoded values are far from expected=" << expected
                 << " (check scale/params alignment)" << endl;
        }
    }

    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 32; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = col_W - 32; i < col_W; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
}

void ct_pt_matrix_mul_w_preprocess_test()
{
    cout << "Task: test Ct-Pt matrix multiplication with preprocess in CKKS scheme: " << endl;

    if (::moai::sim::SimTiming::enabled())
    {
        // Estimator-only path: avoid allocating GPU ciphertext/plaintext buffers.
        const uint64_t poly_modulus_degree = 65536;
        const uint64_t coeff_modulus_size = 4; // {60,40,40,60} in this test

        // Respect the same micro mode sizing knob (used to avoid OOM on real runs).
        const char *pre_micro = std::getenv("MOAI_CT_PT_PRE_MICRO");
        const bool use_pre_micro =
            pre_micro != nullptr && pre_micro[0] != '\0' && std::strcmp(pre_micro, "0") != 0;
        const int num_col = use_pre_micro ? 256 : 768;
        const int row_W = num_col;
        const int col_W = use_pre_micro ? 32 : 64;
        const int max_threads = omp_get_max_threads();
        const int nthreads = std::max(1, std::min(max_threads, col_W));
        const uint64_t ct_size = 2; // typical CKKS ciphertext size before relinearization
        const char *reuse_ev = std::getenv("MOAI_SIM_PREENCODE_REUSE");
        const bool reuse_preencoded =
            reuse_ev != nullptr && reuse_ev[0] != '\0' && std::strcmp(reuse_ev, "0") != 0;

        // Stage 1: encode W (nvtx: moai:ct_pt_pre_encode_w)
        if (!reuse_preencoded)
        {
            ::moai::sim::SimTiming::instance().reset();
            for (int i = 0; i < num_col; ++i)
            {
                for (int j = 0; j < col_W; ++j)
                {
                    ::moai::sim::SimTiming::instance().record_encode();
                }
            }
            cout << "[MOAI_SIM_BACKEND] stage=ct_pt_pre_encode_w" << endl;
            ::moai::sim::SimTiming::instance().print_summary(std::cout);
        }
        else
        {
            cout << "[MOAI_SIM_BACKEND] stage=ct_pt_pre_encode_w (cached; skipped)" << endl;
        }

        // Stage 2: matmul with pre-encoded W (nvtx: moai:ct_pt_matrix_mul_pre_encoded)
        ::moai::sim::SimTiming::instance().reset();
        if (::moai::sim::EngineModel::enabled())
            ::moai::sim::EngineModel::instance().reset();
        // Model X_local deep copies (see Ct_pt_matrix_mul.cuh: deep_copy_cipher per row_W per thread)
        for (int t = 0; t < nthreads; ++t)
        {
            for (int j = 0; j < row_W; ++j)
            {
                ::moai::sim::SimTiming::instance().record_deep_copy_cipher(ct_size, poly_modulus_degree, coeff_modulus_size);
                if (::moai::sim::EngineModel::enabled())
                {
                    const uint64_t coeffs = ct_size * poly_modulus_degree * coeff_modulus_size;
                    ::moai::sim::EngineModel::instance().enqueue_dma(coeffs * sizeof(uint64_t));
                }
            }
        }
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < row_W; ++j)
            {
                ::moai::sim::SimTiming::instance().record_multiply_plain(poly_modulus_degree, coeff_modulus_size);
                if (::moai::sim::EngineModel::enabled())
                    ::moai::sim::EngineModel::instance().enqueue_multiply_plain(ct_size, poly_modulus_degree, coeff_modulus_size);
                if (j != 0)
                {
                    ::moai::sim::SimTiming::instance().record_add_inplace(poly_modulus_degree, coeff_modulus_size);
                    if (::moai::sim::EngineModel::enabled())
                        ::moai::sim::EngineModel::instance().enqueue_add_inplace(ct_size, poly_modulus_degree, coeff_modulus_size);
                }
            }
            ::moai::sim::SimTiming::instance().record_rescale(poly_modulus_degree, coeff_modulus_size);
            if (::moai::sim::EngineModel::enabled())
                ::moai::sim::EngineModel::instance().enqueue_rescale(ct_size, poly_modulus_degree, coeff_modulus_size);
        }
        cout << "[MOAI_SIM_BACKEND] stage=ct_pt_matrix_mul_pre_encoded" << endl;
        ::moai::sim::SimTiming::instance().print_summary(std::cout);
        if (::moai::sim::EngineModel::enabled())
            ::moai::sim::EngineModel::instance().print_summary(std::cout, "ct_pt_pre_encoded");
        cout << "[MOAI_SIM_BACKEND] (ct_pt pre) finished; skipping GPU run." << endl;
        return;
    }

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 {60, 40, 40, 60}));
    //{60, 40, 40, 60}));
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);

    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // size_t slot_count = encoder.slot_count();

    // struct timeval tstart1, tend1;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // PhantomDecryptor decryptor(context, secret_key);
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Encryptor encryptor(&context, &public_key);
    Evaluator evaluator(&context, &phantom_encoder);

    size_t slot_count = encoder.slot_count();

    // construct input (full size OOMs on many GPUs when building ecd_w = num_col*col_W plaintexts;
    // MOAI_CT_PT_PRE_MICRO=1 shrinks for nsys; profile script sets it by default.)
    const char *pre_micro = std::getenv("MOAI_CT_PT_PRE_MICRO");
    const bool use_pre_micro =
        pre_micro != nullptr && pre_micro[0] != '\0' && std::strcmp(pre_micro, "0") != 0;
    int num_X = 256;
    int num_row = 128;
    int num_col = 768;
    if (use_pre_micro)
    {
        num_X = 4;
        num_row = 128;
        num_col = 256;
        cout << "[MOAI_CT_PT_PRE_MICRO] num_X=" << num_X << " num_row=" << num_row << " num_col=" << num_col << endl;
    }
    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_x[i][j][k] = (double)j + 1.0;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }
    }
    cout << "Matrix X size = " << num_row << " * " << num_col << endl;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    /*
    //decrypt


    for (int i = 0; i < num_col; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<"decrypt + decode result of "<<i+1<<"-th ciphertext: "<<endl;
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    */

    // construct W
    int col_W = use_pre_micro ? 32 : 64;
    vector<vector<double>> W(num_col, vector<double>(col_W, 1.0 / 128.0));
    cout << "Matrix W size = " << num_col << " * " << col_W << endl;

    // encode W (each encode runs CKKS slot encoding → IFFT on GPU; not part of ct_pt_matrix_mul)
    vector<vector<PhantomPlaintext>> ecd_w(num_col, vector<PhantomPlaintext>(col_W));

    // #pragma omp parallel for
    std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
    cudaDeviceSynchronize();
#if defined(MOAI_HAVE_NVTX)
    nvtxRangePushA("moai:ct_pt_pre_encode_w");
#endif
    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < col_W; ++j)
        {
            encoder.encode(W[i][j], scale, ecd_w[i][j]);
        }
    }
#if defined(MOAI_HAVE_NVTX)
    cudaDeviceSynchronize();
    nvtxRangePop();
#endif
    cout << "encode W. " << endl;

    cout << "Encrypted col-packing X * ecd W = Encrypted col-packing XW. " << endl;
    std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_encode = end_encode - start_encode;
    cudaDeviceSynchronize();
    cout << "[DEBUG]pre-encoding time: " << duration_encode.count() << " ms" << endl;
    // matrix multiplication (W already encoded to ecd_w; NVTX = mul only for nsys filter)
    // gettimeofday(&tstart1,NULL);
    cudaDeviceSynchronize();
#if defined(MOAI_HAVE_NVTX)
    nvtxRangePushA("moai:ct_pt_matrix_mul_pre_encoded");
#endif
    std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();
    if (::moai::sim::SimTiming::enabled())
    {
        ::moai::sim::SimTiming::instance().reset();
    }
    vector<PhantomCiphertext> ct_pt_mul = ct_pt_matrix_mul(enc_ecd_x, ecd_w, num_col, col_W, num_col, context);
    std::chrono::_V2::system_clock::time_point end = high_resolution_clock::now();
#if defined(MOAI_HAVE_NVTX)
    cudaDeviceSynchronize();
    nvtxRangePop();
#endif

    // gettimeofday(&tend1,NULL);
    // double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    // cout <<"Ct-Pt matrix multiplication time (pre process not included) = "<<ct_pt_matrix_mul_time<<endl;
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaDeviceSynchronize();
    cout << "Ct-Pt matrix multiplication time (pre process not included) = " << duration.count() << " ms" << endl;
    if (::moai::sim::SimTiming::enabled())
    {
        ::moai::sim::SimTiming::instance().print_summary(std::cout);
        cout << "[MOAI_SIM_BACKEND] skipping decrypt/decode correctness checks." << endl;
        return;
    }
    cout << "Modulus chain index for the result: " << context.get_context_data(ct_pt_mul[0].params_id()).chain_depth() << endl;

    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 5; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = col_W - 5; i < col_W; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
}