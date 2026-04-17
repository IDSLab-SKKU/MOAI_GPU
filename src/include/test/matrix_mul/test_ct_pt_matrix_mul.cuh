#include "include.cuh"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <fstream>
#include <ctime>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "source/sim/engine_config.h"

#if defined(MOAI_HAVE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif
#include "source/sim/sim_timing.h"

using namespace std;
using namespace phantom;
using namespace moai;

void ct_pt_matrix_mul_sanity_small_test()
{
    cout << "Task: sanity check encode(double) fast path (small poly_degree)" << endl;

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {50, 30, 30, 50}));
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

    vector<double> x(slot_count, 0.0);
    for (size_t s = 0; s < std::min<size_t>(slot_count, 32); ++s)
        x[s] = 0.01 * static_cast<double>(s + 1);

    PhantomPlaintext pt_x;
    encoder.encode(x, scale, pt_x);
    PhantomCiphertext ct_x;
    encryptor.encrypt(pt_x, ct_x);

    Evaluator evaluator(&context, &phantom_encoder);
    const double tol = 1e-3;
    bool ok = true;

    const vector<double> ws = {-0.125, 0.0, 0.25, -1.5};
    for (double w : ws) {
        PhantomPlaintext pt_w_vec;
        vector<double> wvec(slot_count, w);
        encoder.encode(wvec, ct_x.params_id(), ct_x.scale(), pt_w_vec);

        PhantomPlaintext pt_w_uni;
        encoder.encode(w, ct_x.params_id(), ct_x.scale(), pt_w_uni);

        // multiply_plain
        PhantomCiphertext ct_mul_vec, ct_mul_uni;
        evaluator.multiply_plain(ct_x, pt_w_vec, ct_mul_vec);
        evaluator.multiply_plain(ct_x, pt_w_uni, ct_mul_uni);

        PhantomPlaintext p_mul_vec, p_mul_uni;
        decryptor.decrypt(ct_mul_vec, p_mul_vec);
        decryptor.decrypt(ct_mul_uni, p_mul_uni);

        vector<double> y_mul_vec, y_mul_uni;
        encoder.decode(p_mul_vec, y_mul_vec);
        encoder.decode(p_mul_uni, y_mul_uni);

        // add_plain_inplace (acts on c0 only)
        PhantomCiphertext ct_add_vec = ct_x;
        PhantomCiphertext ct_add_uni = ct_x;
        evaluator.add_plain_inplace(ct_add_vec, pt_w_vec);
        evaluator.add_plain_inplace(ct_add_uni, pt_w_uni);

        PhantomPlaintext p_add_vec, p_add_uni;
        decryptor.decrypt(ct_add_vec, p_add_vec);
        decryptor.decrypt(ct_add_uni, p_add_uni);

        vector<double> y_add_vec, y_add_uni;
        encoder.decode(p_add_vec, y_add_vec);
        encoder.decode(p_add_uni, y_add_uni);

        // sub_plain_inplace (acts on c0 only)
        PhantomCiphertext ct_sub_vec = ct_x;
        PhantomCiphertext ct_sub_uni = ct_x;
        evaluator.sub_plain_inplace(ct_sub_vec, pt_w_vec);
        evaluator.sub_plain_inplace(ct_sub_uni, pt_w_uni);

        PhantomPlaintext p_sub_vec, p_sub_uni;
        decryptor.decrypt(ct_sub_vec, p_sub_vec);
        decryptor.decrypt(ct_sub_uni, p_sub_uni);

        vector<double> y_sub_vec, y_sub_uni;
        encoder.decode(p_sub_vec, y_sub_vec);
        encoder.decode(p_sub_uni, y_sub_uni);

        double max_diff_mul = 0.0, max_diff_add = 0.0, max_diff_sub = 0.0;
        double max_err_mul = 0.0, max_err_add = 0.0, max_err_sub = 0.0;
        for (size_t s = 0; s < std::min<size_t>(slot_count, 32); ++s) {
            const double exp_mul = x[s] * w;
            const double exp_add = x[s] + w;
            const double exp_sub = x[s] - w;
            max_err_mul = std::max(max_err_mul, std::fabs(y_mul_uni[s] - exp_mul));
            max_err_add = std::max(max_err_add, std::fabs(y_add_uni[s] - exp_add));
            max_err_sub = std::max(max_err_sub, std::fabs(y_sub_uni[s] - exp_sub));
            max_diff_mul = std::max(max_diff_mul, std::fabs(y_mul_uni[s] - y_mul_vec[s]));
            max_diff_add = std::max(max_diff_add, std::fabs(y_add_uni[s] - y_add_vec[s]));
            max_diff_sub = std::max(max_diff_sub, std::fabs(y_sub_uni[s] - y_sub_vec[s]));
        }

        cout << "[SANITY_SMALL] w=" << w << " tol=" << tol
             << " | mul(diff=" << max_diff_mul << ", err=" << max_err_mul << ")"
             << " add(diff=" << max_diff_add << ", err=" << max_err_add << ")"
             << " sub(diff=" << max_diff_sub << ", err=" << max_err_sub << ")" << endl;

        if (max_err_mul > tol || max_err_add > tol || max_err_sub > tol ||
            max_diff_mul > tol || max_diff_add > tol || max_diff_sub > tol) {
            ok = false;
        }
    }

    // Overflow/robustness: make sure scalar encode rejects values that cannot fit in int64 rounding.
    bool overflow_thrown = false;
    try {
        const long double max_ll = static_cast<long double>(std::numeric_limits<long long>::max());
        const long double s = static_cast<long double>(ct_x.scale());
        // Choose value so that value*scale exceeds max_ll.
        const double huge_w = static_cast<double>((max_ll / s) * 2.0L);
        PhantomPlaintext pt_over;
        encoder.encode(huge_w, ct_x.params_id(), ct_x.scale(), pt_over);
    } catch (const std::invalid_argument &) {
        overflow_thrown = true;
    }
    cout << "[SANITY_SMALL] overflow_reject=" << (overflow_thrown ? "PASS" : "FAIL") << endl;

    // Reject NaN/Inf explicitly
    bool nan_thrown = false;
    bool inf_thrown = false;
    try {
        PhantomPlaintext pt_nan;
        encoder.encode(std::numeric_limits<double>::quiet_NaN(), ct_x.params_id(), ct_x.scale(), pt_nan);
    } catch (const std::invalid_argument &) {
        nan_thrown = true;
    }
    try {
        PhantomPlaintext pt_inf;
        encoder.encode(std::numeric_limits<double>::infinity(), ct_x.params_id(), ct_x.scale(), pt_inf);
    } catch (const std::invalid_argument &) {
        inf_thrown = true;
    }
    cout << "[SANITY_SMALL] nan_reject=" << (nan_thrown ? "PASS" : "FAIL")
         << " inf_reject=" << (inf_thrown ? "PASS" : "FAIL") << endl;

    // Mismatch checks: scale mismatch and parms_id mismatch must throw
    bool scale_mismatch_thrown = false;
    try {
        PhantomPlaintext pt_bad_scale;
        encoder.encode(0.25, ct_x.params_id(), ct_x.scale() * 2.0, pt_bad_scale);
        PhantomCiphertext ct_tmp = ct_x;
        evaluator.add_plain_inplace(ct_tmp, pt_bad_scale);
    } catch (const std::invalid_argument &) {
        scale_mismatch_thrown = true;
    }
    bool chain_mismatch_thrown = false;
    try {
        size_t bad_chain = 0;
        if (context.context_data_.size() > 1) {
            bad_chain = (ct_x.chain_index() == 0) ? 1 : 0;
        }
        PhantomPlaintext pt_bad_chain;
        encoder.encode(0.25, bad_chain, ct_x.scale(), pt_bad_chain);
        PhantomCiphertext ct_tmp = ct_x;
        evaluator.add_plain_inplace(ct_tmp, pt_bad_chain);
    } catch (const std::invalid_argument &) {
        chain_mismatch_thrown = true;
    }
    cout << "[SANITY_SMALL] scale_mismatch_throw=" << (scale_mismatch_thrown ? "PASS" : "FAIL")
         << " chain_mismatch_throw=" << (chain_mismatch_thrown ? "PASS" : "FAIL") << endl;

    cout << "[SANITY_SMALL] "
         << (ok && overflow_thrown && nan_thrown && inf_thrown && scale_mismatch_thrown && chain_mismatch_thrown
                     ? "PASS"
                     : "FAIL")
         << endl;
}

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

    Evaluator evaluator(&context, &phantom_encoder);
    const double tol = 1e-3;
    bool ok = true;

    const vector<double> ws = {-0.125, 0.0, 0.25, -1.5};
    for (double w : ws) {
        // legacy plaintext: explicit slot vector
        PhantomPlaintext pt_w_vec;
        vector<double> wvec(slot_count, w);
        encoder.encode(wvec, ct_x.params_id(), ct_x.scale(), pt_w_vec);

        // fast plaintext: scalar encode (uniform path)
        PhantomPlaintext pt_w_uni;
        encoder.encode(w, ct_x.params_id(), ct_x.scale(), pt_w_uni);

        // multiply_plain
        PhantomCiphertext ct_mul_vec, ct_mul_uni;
        evaluator.multiply_plain(ct_x, pt_w_vec, ct_mul_vec);
        evaluator.multiply_plain(ct_x, pt_w_uni, ct_mul_uni);

        PhantomPlaintext p_mul_vec, p_mul_uni;
        decryptor.decrypt(ct_mul_vec, p_mul_vec);
        decryptor.decrypt(ct_mul_uni, p_mul_uni);

        vector<double> y_mul_vec, y_mul_uni;
        encoder.decode(p_mul_vec, y_mul_vec);
        encoder.decode(p_mul_uni, y_mul_uni);

        // add_plain_inplace
        PhantomCiphertext ct_add_vec = ct_x;
        PhantomCiphertext ct_add_uni = ct_x;
        evaluator.add_plain_inplace(ct_add_vec, pt_w_vec);
        evaluator.add_plain_inplace(ct_add_uni, pt_w_uni);

        PhantomPlaintext p_add_vec, p_add_uni;
        decryptor.decrypt(ct_add_vec, p_add_vec);
        decryptor.decrypt(ct_add_uni, p_add_uni);

        vector<double> y_add_vec, y_add_uni;
        encoder.decode(p_add_vec, y_add_vec);
        encoder.decode(p_add_uni, y_add_uni);

        // sub_plain_inplace
        PhantomCiphertext ct_sub_vec = ct_x;
        PhantomCiphertext ct_sub_uni = ct_x;
        evaluator.sub_plain_inplace(ct_sub_vec, pt_w_vec);
        evaluator.sub_plain_inplace(ct_sub_uni, pt_w_uni);

        PhantomPlaintext p_sub_vec, p_sub_uni;
        decryptor.decrypt(ct_sub_vec, p_sub_vec);
        decryptor.decrypt(ct_sub_uni, p_sub_uni);

        vector<double> y_sub_vec, y_sub_uni;
        encoder.decode(p_sub_vec, y_sub_vec);
        encoder.decode(p_sub_uni, y_sub_uni);

        double max_diff_mul = 0.0, max_diff_add = 0.0, max_diff_sub = 0.0;
        double max_err_mul = 0.0, max_err_add = 0.0, max_err_sub = 0.0;
        for (size_t s = 0; s < std::min<size_t>(slot_count, 32); ++s) {
            const double exp_mul = x[s] * w;
            const double exp_add = x[s] + w;
            const double exp_sub = x[s] - w;
            max_err_mul = std::max(max_err_mul, std::fabs(y_mul_uni[s] - exp_mul));
            max_err_add = std::max(max_err_add, std::fabs(y_add_uni[s] - exp_add));
            max_err_sub = std::max(max_err_sub, std::fabs(y_sub_uni[s] - exp_sub));
            max_diff_mul = std::max(max_diff_mul, std::fabs(y_mul_uni[s] - y_mul_vec[s]));
            max_diff_add = std::max(max_diff_add, std::fabs(y_add_uni[s] - y_add_vec[s]));
            max_diff_sub = std::max(max_diff_sub, std::fabs(y_sub_uni[s] - y_sub_vec[s]));
        }

        cout << "[SANITY] w=" << w << " tol=" << tol
             << " | mul(diff=" << max_diff_mul << ", err=" << max_err_mul << ")"
             << " add(diff=" << max_diff_add << ", err=" << max_err_add << ")"
             << " sub(diff=" << max_diff_sub << ", err=" << max_err_sub << ")" << endl;

        if (max_err_mul > tol || max_err_add > tol || max_err_sub > tol ||
            max_diff_mul > tol || max_diff_add > tol || max_diff_sub > tol) {
            ok = false;
        }
    }

    bool overflow_thrown = false;
    try {
        const long double max_ll = static_cast<long double>(std::numeric_limits<long long>::max());
        const long double s = static_cast<long double>(ct_x.scale());
        const double huge_w = static_cast<double>((max_ll / s) * 2.0L);
        PhantomPlaintext pt_over;
        encoder.encode(huge_w, ct_x.params_id(), ct_x.scale(), pt_over);
    } catch (const std::invalid_argument &) {
        overflow_thrown = true;
    }
    cout << "[SANITY] overflow_reject=" << (overflow_thrown ? "PASS" : "FAIL") << endl;

    bool nan_thrown = false;
    bool inf_thrown = false;
    try {
        PhantomPlaintext pt_nan;
        encoder.encode(std::numeric_limits<double>::quiet_NaN(), ct_x.params_id(), ct_x.scale(), pt_nan);
    } catch (const std::invalid_argument &) {
        nan_thrown = true;
    }
    try {
        PhantomPlaintext pt_inf;
        encoder.encode(std::numeric_limits<double>::infinity(), ct_x.params_id(), ct_x.scale(), pt_inf);
    } catch (const std::invalid_argument &) {
        inf_thrown = true;
    }
    cout << "[SANITY] nan_reject=" << (nan_thrown ? "PASS" : "FAIL")
         << " inf_reject=" << (inf_thrown ? "PASS" : "FAIL") << endl;

    bool scale_mismatch_thrown = false;
    try {
        PhantomPlaintext pt_bad_scale;
        encoder.encode(0.25, ct_x.params_id(), ct_x.scale() * 2.0, pt_bad_scale);
        PhantomCiphertext ct_tmp = ct_x;
        evaluator.add_plain_inplace(ct_tmp, pt_bad_scale);
    } catch (const std::invalid_argument &) {
        scale_mismatch_thrown = true;
    }
    bool chain_mismatch_thrown = false;
    try {
        size_t bad_chain = 0;
        if (context.context_data_.size() > 1) {
            bad_chain = (ct_x.chain_index() == 0) ? 1 : 0;
        }
        PhantomPlaintext pt_bad_chain;
        encoder.encode(0.25, bad_chain, ct_x.scale(), pt_bad_chain);
        PhantomCiphertext ct_tmp = ct_x;
        evaluator.add_plain_inplace(ct_tmp, pt_bad_chain);
    } catch (const std::invalid_argument &) {
        chain_mismatch_thrown = true;
    }
    cout << "[SANITY] scale_mismatch_throw=" << (scale_mismatch_thrown ? "PASS" : "FAIL")
         << " chain_mismatch_throw=" << (chain_mismatch_thrown ? "PASS" : "FAIL") << endl;

    cout << "[SANITY] "
         << (ok && overflow_thrown && nan_thrown && inf_thrown && scale_mismatch_thrown && chain_mismatch_thrown ? "PASS"
                                                                                                               : "FAIL")
         << endl;
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
        // Append reports under output/sim/ by default (cwd = MOAI_GPU repo root).
        // Override: MOAI_SIM_REPORT_PATH=/path/to/report.txt
        // Engine clock / BW: MOAI_SIM_ENGINE_MHZ > MOAI_SIM_CYCLE_PERIOD_NS (double ns/cyc) > MOAI_SIM_CYCLE_NS (int)
        const char *report_ev = std::getenv("MOAI_SIM_REPORT_PATH");
        const std::string report_file =
            (report_ev != nullptr && report_ev[0] != '\0') ? std::string(report_ev)
                                                         : std::string(::moai::sim::default_sim_report_path());
        ::moai::sim::ensure_parent_dirs_for_file(report_file.c_str());
        std::ofstream report_ofs(report_file.c_str(), std::ios::out | std::ios::app);
        const bool report_ok = report_ofs.is_open();
        std::ostream &report_os = report_ok ? report_ofs : std::cout;

        const char *quiet_ev = std::getenv("MOAI_SIM_REPORT_QUIET");
        const bool quiet =
            quiet_ev != nullptr && quiet_ev[0] != '\0' && std::strcmp(quiet_ev, "0") != 0;

#if !defined(_WIN32)
        {
          char cwd_buf[4096];
          if (getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr)
            cout << "[MOAI_SIM_BACKEND] cwd=" << cwd_buf << endl;
        }
#endif
        cout << "[MOAI_SIM_BACKEND] sim_report path=" << report_file
             << " opened=" << (report_ok ? "yes" : "NO") << endl;

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

            if (report_ok) report_ofs.flush();

            if (!quiet)
            {
                cout << "[MOAI_SIM_BACKEND] ct_pt_wo_pre encode_model=" << (legacy ? "legacy_vec" : "uniform_real")
                     << " encode_once=" << (encode_once ? 1 : 0) << endl;
                if (report_ok)
                    cout << "[MOAI_SIM_BACKEND] report appended to " << report_file << endl;
                else
                    cerr << "[MOAI_SIM_BACKEND] warning: could not open report file " << report_file << endl;
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
        if (pt_uni.is_ckks_broadcast_ntt()) {
            const int64_t c = pt_uni.broadcast_scalar_coeff();
            const uint64_t abs_c = static_cast<uint64_t>(c < 0 ? -static_cast<int64_t>(c) : c);
            const auto &coeff_modulus = context.get_context_data(enc_ecd_x[0].params_id()).parms().coeff_modulus();
            for (size_t j = 0; j < coeff_mod_size; ++j) {
                const uint64_t q = coeff_modulus[j].value();
                uint64_t r = static_cast<uint64_t>(abs_c % q);
                if (c < 0 && r) r = q - r;
                for (size_t k = 0; k < poly_degree; ++k) {
                    h_uni[j * poly_degree + k] = r;
                }
            }
        } else {
            cudaMemcpy(h_uni.data(), pt_uni.data(), n_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }

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

static double ct_pt_matmul_end2end_seconds(
    const char *tag,
    vector<PhantomCiphertext> &enc_X,
    const vector<vector<double>> &W,
    int col_X, int col_W, int row_W,
    PhantomContext &context)
{
    // Single-thread, default stream end-to-end timing including scalar encode + multiply/add + rescale.
    PhantomCKKSEncoder phantom_encoder(context);
    moai::Encoder encoder(&context, &phantom_encoder);
    moai::Evaluator evaluator(&context, &phantom_encoder);

    const double scale0 = enc_X[0].scale();
    vector<PhantomCiphertext> out(static_cast<size_t>(col_W));

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < col_W; ++i)
    {
        PhantomCiphertext acc;

        // j=0
        {
            PhantomPlaintext pt;
            encoder.encode(W[0][i], enc_X[0].chain_index(), enc_X[0].scale(), pt);
            evaluator.multiply_plain(enc_X[0], pt, acc);
        }

        for (int j = 1; j < row_W; ++j)
        {
            PhantomPlaintext pt;
            encoder.encode(W[j][i], enc_X[j].chain_index(), enc_X[j].scale(), pt);
            PhantomCiphertext tmp;
            evaluator.multiply_plain(enc_X[j], pt, tmp);
            evaluator.add_inplace(acc, tmp);
        }

        evaluator.rescale_to_next_inplace(acc);
        acc.scale() = scale0;
        out[static_cast<size_t>(i)] = std::move(acc);
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    cout << "[CT_PT_PROJ] " << tag << " time_s=" << sec << endl;
    return sec;
}

static double ct_pt_matmul_end2end_seconds_stream_inputs(
    const char *tag,
    int row_W, int col_W,
    const std::function<void(int /*j*/, PhantomCiphertext &/*out_ct*/)> &get_ct_j,
    const std::function<double(int /*j*/, int /*i*/)> &get_w_ji,
    size_t target_chain_depth,
    double scale,
    PhantomContext &context)
{
    // Memory-lean variant for very large row_W (e.g., FC2 row_W=3072):
    // keep only output accumulators (col_W cts) + one input ct at a time.
    PhantomCKKSEncoder phantom_encoder(context);
    moai::Encoder encoder(&context, &phantom_encoder);
    moai::Evaluator evaluator(&context, &phantom_encoder);

    vector<PhantomCiphertext> acc(static_cast<size_t>(col_W));

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();

    // j=0 initializes acc[i] = ct0 * w0i
    PhantomCiphertext ctj;
    get_ct_j(0, ctj);
    for (int i = 0; i < col_W; ++i) {
        PhantomPlaintext pt;
        encoder.encode(get_w_ji(0, i), ctj.chain_index(), ctj.scale(), pt);
        evaluator.multiply_plain(ctj, pt, acc[static_cast<size_t>(i)]);
    }

    // j>=1: acc[i] += ctj * wji
    for (int j = 1; j < row_W; ++j) {
        get_ct_j(j, ctj);
        for (int i = 0; i < col_W; ++i) {
            PhantomPlaintext pt;
            encoder.encode(get_w_ji(j, i), ctj.chain_index(), ctj.scale(), pt);
            PhantomCiphertext tmp;
            evaluator.multiply_plain(ctj, pt, tmp);
            evaluator.add_inplace(acc[static_cast<size_t>(i)], tmp);
        }
    }

    // rescale outputs (match ct_pt_matrix_mul_wo_pre behavior)
    for (int i = 0; i < col_W; ++i) {
        evaluator.rescale_to_next_inplace(acc[static_cast<size_t>(i)]);
        acc[static_cast<size_t>(i)].scale() = scale;
        (void)target_chain_depth;
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    cout << "[CT_PT_PROJ] " << tag << " time_s=" << sec << endl;
    return sec;
}

void ct_pt_proj_matmul_bench_single_layer_compare()
{
    cout << "Task: CT×PT projection microbench (single-layer parms), legacy-vs-v2 scalar encode" << endl;

    // Match single_layer_test() CKKS parms.
    const long logn = 15;
    const long sparse_slots = (1 << logn);
    const int logp = 46;
    const int logq = 51;
    const int log_special_prime = 58;
    const int remaining_level = moai::sim::kSingleLayerRemainingLevel; // 20
    const int boot_level = moai::sim::kSingleLayerBootLevel;           // 14

    EncryptionParameters parms(scheme_type::ckks);
    const size_t poly_modulus_degree = static_cast<size_t>(moai::sim::kSingleLayerPolyModulusDegree());
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_sparse_slots(sparse_slots);

    std::vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; ++i) coeff_bit_vec.push_back(logp);
    for (int i = 0; i < boot_level; ++i) coeff_bit_vec.push_back(logq);
    coeff_bit_vec.push_back(log_special_prime);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));

    const double scale = std::pow(2.0, logp);

    PhantomContext context(parms);
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // Single-layer-style batching shape: num_X x num_row packed into slots.
    // Use num_X=256 so (num_X*num_row)=256*128=32768 fully utilizes sparse_slots (no wasted slots).
    const int num_X = 256;
    const int num_row = 128;

    // Build encrypted packed columns without materializing a full [num_X][num_row][num_col] tensor.
    // This matches the packing used by batch_input(): vec[num_X * row + batch] = X[batch][row][col].
    auto encrypt_packed_columns = [&](int num_col) -> vector<PhantomCiphertext> {
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        moai::Encryptor encryptor(&context, &public_key);
        const size_t slot_count = encoder.slot_count();
        const size_t used = static_cast<size_t>(num_X) * static_cast<size_t>(num_row);
        vector<PhantomCiphertext> enc(static_cast<size_t>(num_col));

        for (int col = 0; col < num_col; ++col) {
            vector<double> vec(slot_count, 0.0);
            for (int r = 0; r < num_row; ++r) {
                for (int b = 0; b < num_X; ++b) {
                    const size_t pos = static_cast<size_t>(num_X) * static_cast<size_t>(r) + static_cast<size_t>(b);
                    // Deterministic non-zero pattern.
                    vec[pos] = 0.001 * (1.0 + double((b + 1) % 7)) + 0.0001 * double((r + col) % 13);
                }
            }
            // Remaining slots (if any) stay 0.
            (void)used;
            PhantomPlaintext pt;
            encoder.encode(vec, scale, pt);
            encryptor.encrypt(pt, enc[static_cast<size_t>(col)]);
        }
        return enc;
    };

    auto encrypt_one_packed_column = [&](int num_col, int col_idx) -> PhantomCiphertext {
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        moai::Encryptor encryptor(&context, &public_key);
        const size_t slot_count = encoder.slot_count();

        vector<double> vec(slot_count, 0.0);
        for (int r = 0; r < num_row; ++r) {
            for (int b = 0; b < num_X; ++b) {
                const size_t pos = static_cast<size_t>(num_X) * static_cast<size_t>(r) + static_cast<size_t>(b);
                vec[pos] = 0.001 * (1.0 + double((b + 1) % 7)) + 0.0001 * double((r + col_idx) % 13);
            }
        }
        PhantomPlaintext pt;
        encoder.encode(vec, scale, pt);
        PhantomCiphertext ct;
        encryptor.encrypt(pt, ct);
        return ct;
    };

    auto make_weights = [&](int row_W, int col_W) {
        vector<vector<double>> W(row_W, vector<double>(col_W, 0.0));
        // Scalar weights; value doesn't matter for complexity, but keep deterministic non-zero.
        for (int j = 0; j < row_W; ++j)
            for (int i = 0; i < col_W; ++i)
                W[j][i] = 1.0 / 128.0;
        return W;
    };

    auto modswitch_to_depth = [&](vector<PhantomCiphertext> &cts, size_t target_chain_depth) {
        PhantomCKKSEncoder phantom_encoder(context);
        moai::Evaluator evaluator(&context, &phantom_encoder);
        for (auto &ct : cts) {
            while (context.get_context_data(ct.params_id()).chain_depth() > target_chain_depth) {
                evaluator.mod_switch_to_next_inplace(ct);
            }
        }
    };

    auto v_branch_depth_cap = [&](const PhantomCiphertext &k_ref) -> size_t {
        if (const char *ev = std::getenv("MOAI_ATT_V_DEPTH_CAP")) {
            char *end = nullptr;
            unsigned long v = std::strtoul(ev, &end, 10);
            if (end != ev && *end == '\0' && v <= 64) {
                return std::max<size_t>(1, static_cast<size_t>(v));
            }
        }
        const size_t kd = context.get_context_data(k_ref.params_id()).chain_depth();
        if (kd <= 1) return 3;
        return std::min<size_t>(3, kd - 1);
    };

    auto run_op = [&](const char *name, int row_W, int col_W) -> double {
        cout << "[CT_PT_PROJ] begin " << name << " row_W=" << row_W << " col_W=" << col_W << endl;
        vector<PhantomCiphertext> enc_X = encrypt_packed_columns(row_W);
        auto W = make_weights(row_W, col_W);
        // Warmup (small subset) to stabilize JIT/caches
        {
            vector<PhantomCiphertext> enc_X_small(enc_X.begin(), enc_X.begin() + std::min<int>(row_W, 8));
            vector<vector<double>> W_small(std::min<int>(row_W, 8), vector<double>(std::min<int>(col_W, 8), 1.0 / 128.0));
            ct_pt_matmul_end2end_seconds("warmup", enc_X_small, W_small, (int)enc_X_small.size(), (int)W_small[0].size(), (int)enc_X_small.size(), context);
        }
        return ct_pt_matmul_end2end_seconds(name, enc_X, W, row_W, col_W, row_W, context);
    };

    // Projections (BERT-base like):
    // - Q/K/V per head: 768→64, repeated 12 heads => 36 CT×PT matmuls of shape (R^{128×768} × R^{768×64})
    // - out proj: 768→768
    // - FC1: 768→3072
    // - FC2: 3072→768
    const int d_model = 768;
    const int d_ff = 3072;
    const int num_head = 12;
    const int d_head = 64;

    // For OOM-safety, allow running exactly one op per process:
    //   MOAI_CT_PT_PROJ_OP=qkv|out|fc1|fc2
    // Default (unset): run all (mostly for debugging; scripts should set this).
    std::string op = "all";
    if (const char *ev = std::getenv("MOAI_CT_PT_PROJ_OP"); ev && ev[0] != '\0') {
        op = ev;
        for (char &c : op) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (op == "qkv" || op == "all") {
        // Match single-layer depth semantics from the paper table: 15→14 for QKV proj.
        // In our context chain_depth appears to be (Depth-1).
        const size_t target_depth_qkv = 14; // corresponds to Depth=15

        vector<PhantomCiphertext> enc_X_qk = encrypt_packed_columns(d_model);
        modswitch_to_depth(enc_X_qk, target_depth_qkv);

        // Q/K use the same enc_X after pre-attention drops; V uses a capped-depth branch relative to K.
        vector<vector<double>> W = make_weights(d_model, d_head);

        double t_sum = 0.0;
        for (int h = 0; h < num_head; ++h) {
            (void)h;
            t_sum += ct_pt_matmul_end2end_seconds("Q_proj", enc_X_qk, W, d_model, d_head, d_model, context);
            t_sum += ct_pt_matmul_end2end_seconds("K_proj", enc_X_qk, W, d_model, d_head, d_model, context);

            // V-branch pre-switch cap (same policy as single_att_block.cuh).
            // We approximate K_ref by reusing enc_X_qk[0] depth after K path; cap is derived from its depth.
            vector<PhantomCiphertext> enc_X_v = enc_X_qk;
            const size_t cap = v_branch_depth_cap(enc_X_qk[0]);
            modswitch_to_depth(enc_X_v, cap);
            t_sum += ct_pt_matmul_end2end_seconds("V_proj", enc_X_v, W, d_model, d_head, d_model, context);
        }
        cout << "[CT_PT_PROJ] QKV_proj time_s=" << t_sum << endl;
        if (op != "all") return;
    }
    if (op == "out" || op == "all") {
        // Paper table: 2→1 => chain_depth target 1
        vector<PhantomCiphertext> enc = encrypt_packed_columns(d_model);
        modswitch_to_depth(enc, 1);
        auto W = make_weights(d_model, d_model);
        (void)ct_pt_matmul_end2end_seconds("out_proj", enc, W, d_model, d_model, d_model, context);
        if (op != "all") return;
    }
    if (op == "fc1" || op == "all") {
        // Paper table: 10→9 => chain_depth target 9
        vector<PhantomCiphertext> enc = encrypt_packed_columns(d_model);
        modswitch_to_depth(enc, 9);
        auto W = make_weights(d_model, d_ff);
        (void)ct_pt_matmul_end2end_seconds("fc1", enc, W, d_model, d_ff, d_model, context);
        if (op != "all") return;
    }
    if (op == "fc2" || op == "all") {
        // Paper table: 2→1 => chain_depth target 1
        const size_t target_depth = 1;
        // FC2 modes (to control VRAM residency of the 3072 input ciphertexts):
        // - MOAI_CT_PT_FC2_MODE=full_vram : keep all 3072 CTs resident on GPU for compute (closest to "original" all-in-VRAM)
        // - MOAI_CT_PT_FC2_MODE=chunk_vram: keep `chunk_size` CTs resident on GPU at once (default)
        // - MOAI_CT_PT_FC2_MODE=stream   : keep 1 CT resident at a time
        //
        // Timing is split into:
        // - precompute: encrypt+modswitch+device->host copy for all input ciphertexts
        // - compute: host->device copy + multiply/add + final rescale
        std::string fc2_mode = "chunk_vram";
        if (const char *ev = std::getenv("MOAI_CT_PT_FC2_MODE"); ev && ev[0] != '\0') {
            fc2_mode = ev;
            for (char &c : fc2_mode) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        int chunk_size = 64;
        if (fc2_mode == "full_vram") {
            chunk_size = d_ff;
        } else if (fc2_mode == "stream") {
            chunk_size = 1;
        } else { // chunk_vram (default)
            if (const char *ev = std::getenv("MOAI_CT_PT_FC2_CHUNK"); ev && ev[0] != '\0') {
                char *end = nullptr;
                long v = std::strtol(ev, &end, 10);
                if (end != ev && *end == '\0' && v >= 1 && v <= d_ff) chunk_size = static_cast<int>(v);
            }
        }
        cout << "[CT_PT_PROJ] fc2 mode=" << fc2_mode << " chunk_size=" << chunk_size << endl;

        PhantomCKKSEncoder phantom_encoder(context);
        moai::Encoder encoder(&context, &phantom_encoder);
        moai::Evaluator evaluator(&context, &phantom_encoder);

        // Allocate output accumulators on device (768 ciphertexts).
        vector<PhantomCiphertext> acc(static_cast<size_t>(d_model));
        bool acc_init = false;

        // Precompute host cache for a chunk: store ciphertext payload + minimal metadata.
        struct HostCt {
            std::size_t chain_index{0};
            std::size_t size{0};
            std::size_t poly_degree{0};
            std::size_t coeff_mod_size{0};
            double scale{1.0};
            std::uint64_t correction_factor{1};
            bool is_ntt_form{true};
            std::vector<uint64_t> data;
        };

        auto export_to_host = [&](const PhantomCiphertext &ct) -> HostCt {
            HostCt h;
            h.chain_index = ct.chain_index();
            h.size = ct.size();
            h.poly_degree = ct.poly_modulus_degree();
            h.coeff_mod_size = ct.coeff_modulus_size();
            h.scale = ct.scale();
            h.correction_factor = ct.correction_factor();
            h.is_ntt_form = ct.is_ntt_form();
            const size_t words = h.size * h.poly_degree * h.coeff_mod_size;
            h.data.resize(words);
            PHANTOM_CHECK_CUDA(cudaMemcpy(h.data.data(), ct.data(), words * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            return h;
        };

        auto import_from_host = [&](const HostCt &h, PhantomCiphertext &ct) {
            // Allocate device buffer for this chain_index and size.
            ct.resize(context, h.chain_index, h.size, phantom::util::global_variables::default_stream->get_stream());
            ct.set_scale(h.scale);
            ct.set_correction_factor(h.correction_factor);
            ct.set_ntt_form(h.is_ntt_form);
            const size_t words = h.size * h.poly_degree * h.coeff_mod_size;
            PHANTOM_CHECK_CUDA(cudaMemcpy(ct.data(), h.data.data(), words * sizeof(uint64_t), cudaMemcpyHostToDevice));
        };

        cudaDeviceSynchronize();
        const auto t_pre0 = std::chrono::high_resolution_clock::now();

        // Precompute all inputs into host in chunks so compute phase can be isolated.
        std::vector<HostCt> host_chunk;
        host_chunk.reserve(static_cast<size_t>(chunk_size));

        // Compute phase timing: excludes precompute (encrypt/modswitch/D2H).
        double compute_sec = 0.0;

        for (int base = 0; base < d_ff; base += chunk_size) {
            const int end = std::min(d_ff, base + chunk_size);
            host_chunk.clear();

            // (1) Build encrypted inputs for this chunk and move them to host.
            for (int j = base; j < end; ++j) {
                PhantomCiphertext ct = encrypt_one_packed_column(d_ff, j);
                vector<PhantomCiphertext> tmp{ct};
                modswitch_to_depth(tmp, target_depth);
                ct = std::move(tmp[0]);
                host_chunk.push_back(export_to_host(ct));
            }

            // (2) Move this chunk to device and compute contributions.
            // This keeps `chunk_size` ciphertexts resident in VRAM at once, matching the desired
            // "hold half in VRAM, then swap" behavior (vs. 1-by-1 streaming).
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::high_resolution_clock::now();

            vector<PhantomCiphertext> dev_chunk;
            dev_chunk.resize(host_chunk.size());
            print_cuda_meminfo("[CT_PT_PROJ] fc2 before import_from_host(dev_chunk)");
            for (size_t jj = 0; jj < host_chunk.size(); ++jj) {
                import_from_host(host_chunk[jj], dev_chunk[jj]);
            }
            print_cuda_meminfo("[CT_PT_PROJ] fc2 after import_from_host(dev_chunk)");

            for (size_t jj = 0; jj < dev_chunk.size(); ++jj) {
                auto &ctj = dev_chunk[jj];
                for (int i = 0; i < d_model; ++i) {
                    PhantomPlaintext pt;
                    encoder.encode(1.0 / 128.0, ctj.chain_index(), ctj.scale(), pt);
                    if (!acc_init) {
                        evaluator.multiply_plain(ctj, pt, acc[static_cast<size_t>(i)]);
                    } else {
                        PhantomCiphertext tmp;
                        evaluator.multiply_plain(ctj, pt, tmp);
                        evaluator.add_inplace(acc[static_cast<size_t>(i)], tmp);
                    }
                }
                acc_init = true;
            }

            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();
            compute_sec += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        }

        // Final rescale once for each output ciphertext.
        cudaDeviceSynchronize();
        const auto t_rs0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < d_model; ++i) {
            evaluator.rescale_to_next_inplace(acc[static_cast<size_t>(i)]);
            acc[static_cast<size_t>(i)].scale() = scale;
        }
        cudaDeviceSynchronize();
        const auto t_rs1 = std::chrono::high_resolution_clock::now();
        compute_sec += std::chrono::duration_cast<std::chrono::duration<double>>(t_rs1 - t_rs0).count();

        const auto t_pre1 = std::chrono::high_resolution_clock::now();
        const double precompute_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t_pre1 - t_pre0).count();

        cout << "[CT_PT_PROJ] fc2_precompute time_s=" << precompute_sec << endl;
        cout << "[CT_PT_PROJ] fc2_compute time_s=" << compute_sec << endl;
        cout << "[CT_PT_PROJ] fc2 time_s=" << (precompute_sec + compute_sec) << endl;
        if (op != "all") return;
    }
}

void ct_pt_matrix_mul_w_preprocess_test()
{
    cout << "Task: test Ct-Pt matrix multiplication with preprocess in CKKS scheme: " << endl;

    if (::moai::sim::SimTiming::enabled())
    {
        const char *report_ev = std::getenv("MOAI_SIM_REPORT_PATH");
        const std::string report_file =
            (report_ev != nullptr && report_ev[0] != '\0') ? std::string(report_ev)
                                                         : std::string(::moai::sim::default_sim_report_path());
        ::moai::sim::ensure_parent_dirs_for_file(report_file.c_str());
        std::ofstream report_ofs(report_file.c_str(), std::ios::out | std::ios::app);
        const bool report_ok = report_ofs.is_open();
        std::ostream &report_os = report_ok ? report_ofs : std::cout;
        const char *quiet_ev = std::getenv("MOAI_SIM_REPORT_QUIET");
        const bool quiet =
            quiet_ev != nullptr && quiet_ev[0] != '\0' && std::strcmp(quiet_ev, "0") != 0;

#if !defined(_WIN32)
        {
          char cwd_buf[4096];
          if (getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr)
            cout << "[MOAI_SIM_BACKEND] cwd=" << cwd_buf << endl;
        }
#endif
        cout << "[MOAI_SIM_BACKEND] sim_report path=" << report_file
             << " opened=" << (report_ok ? "yes" : "NO") << endl;

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
            report_os << "\n=== MOAI_SIM_REPORT ct_pt_pre_encode_w ts=" << static_cast<long long>(std::time(nullptr))
                      << " ===\n";
            ::moai::sim::SimTiming::instance().print_summary(report_os);
            if (report_ok) report_ofs.flush();
            if (!quiet)
            {
                if (report_ok)
                    cout << "[MOAI_SIM_BACKEND] report appended to " << report_file << endl;
                else
                    cerr << "[MOAI_SIM_BACKEND] warning: could not open report file " << report_file << endl;
                ::moai::sim::SimTiming::instance().print_summary(std::cout);
            }
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
                    ::moai::sim::EngineModel::instance().enqueue_dma_d2d(coeffs * sizeof(uint64_t));
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
        report_os << "\n=== MOAI_SIM_REPORT ct_pt_matrix_mul_pre_encoded ts=" << static_cast<long long>(std::time(nullptr))
                  << " ===\n";
        ::moai::sim::SimTiming::instance().print_summary(report_os);
        if (::moai::sim::EngineModel::enabled())
            ::moai::sim::EngineModel::instance().print_summary(report_os, "ct_pt_pre_encoded");
        if (report_ok) report_ofs.flush();
        if (!quiet)
        {
            if (report_ok)
                cout << "[MOAI_SIM_BACKEND] report appended to " << report_file << endl;
            else
                cerr << "[MOAI_SIM_BACKEND] warning: could not open report file " << report_file << endl;
            ::moai::sim::SimTiming::instance().print_summary(std::cout);
            if (::moai::sim::EngineModel::enabled())
                ::moai::sim::EngineModel::instance().print_summary(std::cout, "ct_pt_pre_encoded");
        }
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