#include "include.cuh"
#include "keys/moai_precomputed_keys.cuh"
#include "source/sim/sim_ckks_defaults.h"

using namespace std;
using namespace phantom;
using namespace moai;

const int num_X = 256;
const int num_row = 128;
const int num_col = 768;
const int num_inter = 3072;
const double sqrt_d = 8.0;
int num_input = 11;

vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));

// paras for attention block
int col_W = 64;
int num_head = 12;
vector<vector<vector<double>>> WQ(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WK(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WV(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));

vector<vector<double>> bQ(num_head, vector<double>(col_W, 0.0));
vector<vector<double>> bK(num_head, vector<double>(col_W, 0.0));
vector<vector<double>> bV(num_head, vector<double>(col_W, 0.0));

vector<vector<double>> selfoutput(num_col, vector<double>(num_col, 0.0));
vector<double> selfoutput_bias(num_col, 0.0);
vector<double> layernorm1_gamma(num_col, 0.0);
vector<double> layernorm1_beta(num_col, 0.0);

vector<vector<double>> inter_weight(num_col, vector<double>(num_inter, 0.0));
vector<double> inter_bias(num_inter, 0.0);
vector<vector<double>> final_weight(num_inter, vector<double>(num_col, 0.0));
vector<double> final_bias(num_col, 0.0);
vector<double> layernorm2_gamma(num_col, 0.0);
vector<double> layernorm2_beta(num_col, 0.0);

void read_input()
{
    ifstream fin;
    fin.open("att_block_weights/embedded_inputs.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file embedded_inputs.txt" << endl;
    }
    char a;
    // the test file has 11 input vectors, length of each vector = 768

    for (int i = 0; i < num_input; ++i)
    {
        for (int j = 0; j < num_col - 1; ++j)
        {
            fin >> input_x[0][i][j];
            fin >> a;
        }
        fin >> input_x[0][i][num_col - 1];
    }
    fin.close();
    // for test
    // cout <<input_x[0][10][0]<<" "<<input_x[0][10][num_col-1]<<endl;
}

void read_weights()
{
    ifstream fin;
    // read matrix Q, size of Q = 12*64*768
    fin.open("att_block_weights/query_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file query_weight.txt" << endl;
    }
    char a;
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WQ[k][j][i];
                fin >> a;
            }
            fin >> WQ[k][num_col - 1][i];
        }
    }

    fin.close();
    // for test
    // cout <<"WQ last element: "<<WQ[num_head-1][num_col-1][col_W-1]<<endl;

    // Q = Q/sqrt(d')
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < num_col; ++i)
        {
            for (int j = 0; j < col_W; ++j)
            {
                WQ[k][i][j] = WQ[k][i][j] / sqrt_d;
            }
        }
    }

    // read matrix K
    fin.open("att_block_weights/key_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file key_weight.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WK[k][j][i];
                fin >> a;
            }
            fin >> WK[k][num_col - 1][i];
        }
    }
    fin.close();
    // for test
    // cout <<"WK last element: "<<WK[num_head-1][num_col-1][col_W-1]<<endl;

    // read matrix V
    fin.open("att_block_weights/value_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file value_weight.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WV[k][j][i];
                fin >> a;
            }
            fin >> WV[k][num_col - 1][i];
        }
    }
    fin.close();
    // for test
    // cout <<"WV last element: "<<WV[num_head-1][num_col-1][col_W-1]<<endl;

    // read self output weight
    fin.open("self_output_weights/self_output_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        for (int i = 0; i < num_col - 1; ++i)
        {
            fin >> selfoutput[i][k];
            fin >> a;
        }
        fin >> selfoutput[num_col - 1][k];
    }
    fin.close();
    // cout <<"selfoutput last element: "<<selfoutput[num_col-1][num_col-1]<<endl;

    // read layernorm1 weight
    fin.open("self_output_weights/self_output_LayerNorm_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_LayerNorm_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm1_gamma[k];
    }
    fin.close();
    // cout <<"LayerNorm1 last element: "<<layernorm1_gamma[num_col-1]<<endl;
}

void read_bias()
{
    ifstream fin;
    // read bias Q, size of Q = 64
    fin.open("att_block_weights/query_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file query_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bQ[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"Q bias last element: "<<bQ[num_head-1][col_W-1]<<endl;

    // bias Q = bias Q / sqrt_d'
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            bQ[k][i] = bQ[k][i] / sqrt_d;
        }
    }

    fin.open("att_block_weights/key_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file key_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bK[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"K bias last element: "<<bK[num_head-1][col_W-1]<<endl;

    fin.open("att_block_weights/value_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file value_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bV[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"v bias last element: "<<bV[num_head-1][col_W-1]<<endl;

    // read self output bias
    fin.open("self_output_weights/self_output_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> selfoutput_bias[k];
    }
    fin.close();
    // for test
    // cout <<"selfoutput bias last element: "<<selfoutput_bias[num_col-1]<<endl;

    // read layernorm1 weight
    fin.open("self_output_weights/self_output_LayerNorm_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_LayerNorm_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm1_beta[k];
    }
    fin.close();
    // cout <<"LayerNorm1 bias last element: "<<layernorm1_beta[num_col-1]<<endl;
}

void read_feed_forward_param()
{
    ifstream fin;
    char a;
    // read inter weight
    fin.open("feed_forward_weights/intermediate_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file intermediate_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_inter; ++k)
    {
        for (int i = 0; i < num_col - 1; ++i)
        {
            fin >> inter_weight[i][k];
            fin >> a;
        }
        fin >> inter_weight[num_col - 1][k];
    }
    fin.close();
    // cout <<"inter_weight last element: "<<inter_weight[num_col-1][num_inter-1]<<endl;

    // read inter bias
    fin.open("feed_forward_weights/intermediate_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file intermediate_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_inter; ++k)
    {
        fin >> inter_bias[k];
    }
    fin.close();
    // cout <<"inter_bias last element: "<<inter_bias[num_inter-1]<<endl;

    // read final weight
    fin.open("feed_forward_weights/final_output_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        for (int i = 0; i < num_inter - 1; ++i)
        {
            fin >> final_weight[i][k];
            fin >> a;
        }
        fin >> final_weight[num_inter - 1][k];
    }
    fin.close();
    // cout <<"final_weight last element: "<<final_weight[num_inter-1][num_col-1]<<endl;

    // read final bias
    fin.open("feed_forward_weights/final_output_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> final_bias[k];
    }
    fin.close();
    // cout <<"final_bias last element: "<<final_bias[num_col-1]<<endl;

    // read layernorm2 weight
    fin.open("feed_forward_weights/final_output_LayerNorm_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_LayerNorm_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm2_gamma[k];
    }
    fin.close();
    // cout <<"LayerNorm2 weights last element: "<<layernorm2_gamma[num_col-1]<<endl;

    // read layernorm2 bias
    fin.open("feed_forward_weights/final_output_LayerNorm_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_LayerNorm_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm2_beta[k];
    }
    fin.close();
    // cout <<"LayerNorm2 bias last element: "<<layernorm2_beta[num_col-1]<<endl;
}

void single_layer_test()
{
    cout << "Task: test one layer of BERT in CKKS scheme: " << endl;

    // LN bootstrapping placement variants:
    // 0: baseline (keep pre-LN 768-ct bootstrapping)
    // 1: remove pre-LN bootstrapping; do internal LN var-branch bootstrapping (1 ct)
    // 2: same as 1 + allow attention start depth bump via MOAI_LN_LEVEL_BUMP
    int moai_ln_bootstrap_variant = 0;
    if (const char *ev = std::getenv("MOAI_LN_BOOTSTRAP_VARIANT")) {
        char *end = nullptr;
        long v = std::strtol(ev, &end, 10);
        if (end != ev && *end == '\0' && v >= 0 && v <= 2) {
            moai_ln_bootstrap_variant = static_cast<int>(v);
        }
    }
    const bool moai_ln_enable_internal_var_bootstrap = (moai_ln_bootstrap_variant >= 1);
    const bool moai_ln_disable_preln_bootstrap = (moai_ln_bootstrap_variant >= 1);

    read_input();
    read_weights();
    read_bias();
    read_feed_forward_param();
    cout << "Read input, weights, bias from txt files. " << endl;

    // bootstrapping parameters
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    long logN = static_cast<long>(moai::sim::kSingleLayerLogN);
    long loge = 10;

    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 58;

    int secret_key_hamming_weight = 192;

    // Calculation required (defaults shared with `sim_ckks_defaults.h` for primitive sim).
    int boot_level = moai::sim::kSingleLayerBootLevel; // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    // remaining_level_att set after MOAI_ALPHA (must satisfy pre-attention mod-switch budget; see below)

    // Total primes T = 1 + remaining_level + boot_level + 1.
    // Original MOAI single-layer recipe uses remaining_level=20 (T=36 when boot_level=14).
    int remaining_level = moai::sim::kSingleLayerRemainingLevel;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
    {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < boot_level; i++)
    {
        coeff_bit_vec.push_back(logq);
    }
    coeff_bit_vec.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = (size_t)(1 << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    size_t moai_hybrid_alpha = 1;
    if (const char *a = std::getenv("MOAI_ALPHA")) {
        if (*a) {
            moai_hybrid_alpha = static_cast<size_t>(std::strtoul(a, nullptr, 10));
        }
    }
    parms.set_special_modulus_size(moai_hybrid_alpha);
    {
        size_t T = parms.coeff_modulus().size();
        if (moai_hybrid_alpha == 0 || moai_hybrid_alpha >= T || (T % moai_hybrid_alpha) != 0) {
            throw std::invalid_argument("MOAI_ALPHA invalid for hybrid KS (need T % alpha == 0)");
        }
    }

    // Pre-attention mod switches: N_used = 2*boot_level + (remaining_level - remaining_level_att) + 1.
    // Need N_used <= max_drops (tops - first_index). Tight equality leaves no room for QKV/rescale inside
    // single_att_block; for MOAI_ALPHA>1 add a small slack via higher remaining_level_att.
    // Match the original recipe (remaining_level=20): slack 0 (alpha==1) or 2 (alpha>1).
    const int att_chain_slack = (moai_hybrid_alpha > 1) ? 2 : 0;
    int remaining_level_att = boot_level + static_cast<int>(moai_hybrid_alpha) + att_chain_slack;
    // Variant2: bump attention start depth by reducing pre-attention mod-switches.
    // Starting depth before attention is approximately (remaining_level_att - 1) due to an extra mod_switch_to_next.
    if (moai_ln_bootstrap_variant == 2) {
        int bump = 0;
        if (const char *ev = std::getenv("MOAI_LN_LEVEL_BUMP")) {
            char *end = nullptr;
            long v = std::strtol(ev, &end, 10);
            if (end != ev && *end == '\0' && v >= 0 && v <= 16) {
                bump = static_cast<int>(v);
            }
        }
        const int old_att = remaining_level_att;
        remaining_level_att = std::min(remaining_level, remaining_level_att + bump);
        const int applied = remaining_level_att - old_att;
        cout << "MOAI LN Variant2: attention start depth bump requested=" << bump
             << " applied=" << applied << " (remaining_level_att=" << remaining_level_att
             << ", start_depth≈" << (remaining_level_att - 1) << ")." << endl;
        if (applied < bump) {
            cout << "MOAI LN Variant2: bump capped by remaining_level=" << remaining_level
                 << " (increase remaining_level in code if you need more headroom without changing start depth math)." << endl;
        }
    }
    if (remaining_level_att > remaining_level) {
        throw std::invalid_argument(
            "single_layer: remaining_level_att > remaining_level (reduce MOAI_ALPHA or att_chain_slack)");
    }
    const int total_level_att = remaining_level_att + boot_level;

    // Pre-attention modulus switching control (for enc_ecd_x path only).
    // Default drops match original recipe: boot_level + (remaining_level - remaining_level_att) + 1.
    // You can override with:
    // - MOAI_PRE_ATT_DROPS: absolute number of mod_switch_to_next drops applied to enc_ecd_x before attention
    // - MOAI_PRE_ATT_DROPS_DELTA: additive adjustment to the default (can be negative)
    int pre_att_drops = boot_level + (remaining_level - remaining_level_att) + 1;
    if (const char *ev = std::getenv("MOAI_PRE_ATT_DROPS")) {
        char *end = nullptr;
        long v = std::strtol(ev, &end, 10);
        if (end != ev && *end == '\0') {
            pre_att_drops = std::max(0, static_cast<int>(v));
        }
    } else if (const char *ev = std::getenv("MOAI_PRE_ATT_DROPS_DELTA")) {
        char *end = nullptr;
        long v = std::strtol(ev, &end, 10);
        if (end != ev && *end == '\0') {
            pre_att_drops = std::max(0, pre_att_drops + static_cast<int>(v));
        }
    }

    // Keys must match parms (same T and MOAI_ALPHA). E.g. MOAI_ALPHA=4, T=40 -> dnum=9 -> .../keys_dnum_9
    // (generate with gen_moai_keys / make_moai_inference_parms(4)).
    std::string moai_key_pack_dir;
    if (const char *p = std::getenv("MOAI_PRECOMPUTED_KEYS_DIR")) {
        if (*p) {
            moai_key_pack_dir = p;
        }
    } else if (const char *b = std::getenv("MOAI_KEYS_BASE")) {
        size_t T = parms.coeff_modulus().size();
        size_t dnum = (T - moai_hybrid_alpha) / moai_hybrid_alpha;
        moai_key_pack_dir = std::string(b) + "/keys_dnum_" + std::to_string(dnum);
    }
    const bool moai_use_precomputed = !moai_key_pack_dir.empty();

    double scale = pow(2.0, logp);
    // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
    //     40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    // double scale = pow(2.0,40);

    PhantomContext context(parms);

    {
        const size_t first = context.get_first_index();
        const size_t tops = context.get_context_data(first).parms().coeff_modulus().size();
        const size_t max_drops = tops - first;
        // Total pre-attention mod-switches include:
        // - enc_ecd_x_copy: boot_level drops
        // - enc_ecd_x: pre_att_drops drops (configurable via env vars above)
        const int pre_att_mod_switch = boot_level + pre_att_drops;
        if (static_cast<size_t>(pre_att_mod_switch) > max_drops) {
            throw std::invalid_argument(
                "single_layer: pre-attention mod_switch count exceeds chain budget (adjust MOAI_ALPHA, "
                "boot_level, or remaining_level/remaining_level_att)");
        }
        if (max_drops - static_cast<size_t>(pre_att_mod_switch) < 3) {
            cout << "single_layer: warning: pre-attention mod_switch budget has little slack (pre_att="
                 << pre_att_mod_switch << ", max_drops=" << max_drops
                 << "). If attention fails, increase remaining_level in steps of MOAI_ALPHA while keeping "
                    "T % MOAI_ALPHA == 0; sync gen_moai_keys.cu make_moai_inference_parms.\n";
        }
    }

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    if (moai_use_precomputed) {
        cout << "MOAI: loading precomputed keys from " << moai_key_pack_dir << endl;
    }

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key;
    PhantomRelinKey relin_keys;
    PhantomGaloisKey gal_keys_boot;

    if (moai_use_precomputed) {
        moai::load_precomputed_keys_from_directory(context, moai_key_pack_dir, secret_key, public_key, relin_keys,
                                                   gal_keys_boot);
    } else {
        public_key = secret_key.gen_publickey(context);
        relin_keys = secret_key.gen_relinkey(context);
    }

    // end

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // Evaluator evaluator(context, encoder);
    // size_t slot_count = encoder.slot_count();
    // cout <<slot_count<<endl;
    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    // prepare for bootstrapping
    //  Bootstrapper bootstrapper(
    //    loge,
    //    logn,
    //    logN - 1,
    //    total_level,
    //    scale,
    //    boundary_K,
    //    deg,
    //    scale_factor,
    //    inverse_deg,
    //    context,
    //    keygen,
    //    encoder,
    //    encryptor,
    //    decryptor,
    //    evaluator,
    //    relin_keys,
    //    gal_keys_boot);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &phantom_encoder, &relin_keys, &gal_keys_boot, scale);

    // add on Aug 18, save GPU memory
    // vector<int> gal_vector;
    // gal_vector.push_back(0);
    // for (int i = 0; i < sparse_slots/num_X; ++i)
    // {
    //     gal_vector.push_back((i * num_X));
    //     // cout << (i * num_X) << " ";
    // }

    // gal_vector.push_back(0); // NEXUS
    // for (int i = 0; i < logN - 1; i++)
    // {
    //     gal_vector.push_back((1 << i));
    // }

    // // keygen.create_galois_keys(gal_vector, gal_keys);
    // ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_vector, gal_keys);
    // gal_keys = secret_key.create_galois_keys(context);

    Bootstrapper bootstrapper(
        loge,
        logn,
        logN - 1,
        total_level,
        scale,
        boundary_K,
        deg,
        scale_factor,
        inverse_deg,
        &ckks_evaluator);

    cout << "preparing bootstrapping..." << endl;
    bootstrapper.prepare_mod_polynomial();

    cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    // ct-ct rotate steps
    // vector<int> gal_vector;
    // gal_steps_vector.push_back(0);
    // for (int i = 0; i < sparse_slots/num_X; ++i)
    // {
    //     gal_steps_vector.push_back((i * num_X));
    //     // cout << (i * num_X) << " ";
    // }

    // keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);

    // ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    if (!moai_use_precomputed) {
        gal_keys_boot = secret_key.create_galois_keys(context);
    }
    std::cout << (moai_use_precomputed ? "Galois keys loaded from disk.\n" : "Galois key generated from steps vector.\n");

    bootstrapper.slot_vec.push_back(logn);

    cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();

    

    struct timeval tstart1, tend1;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);
    vector<int> input_len(num_X, 0);
    input_len[0] = 11;
    vector<int> b_vec = bias_vec(input_len, num_X, num_row);
    // cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    // cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<vector<double>>>().swap(input_x);

    vector<PhantomCiphertext> enc_ecd_x_copy(num_col);
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }

    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }

    // Pre-attention mod-switch drops (configurable via MOAI_PRE_ATT_DROPS / MOAI_PRE_ATT_DROPS_DELTA).
    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < pre_att_drops; ++j)
        {
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    // Debug: report remaining depth right before attention.
    const bool moai_depth_debug = (std::getenv("MOAI_DEPTH_DEBUG") != nullptr);
    if (moai_depth_debug) {
        size_t min_depth = context.get_context_data(enc_ecd_x[0].params_id()).chain_depth();
        size_t max_depth = min_depth;
        for (int i = 1; i < num_col; ++i) {
            const size_t d = context.get_context_data(enc_ecd_x[i].params_id()).chain_depth();
            min_depth = std::min(min_depth, d);
            max_depth = std::max(max_depth, d);
        }
        const size_t start_depth = static_cast<size_t>(total_level);
        const size_t d0 = context.get_context_data(enc_ecd_x[0].params_id()).chain_depth();
        const size_t drops0 = (start_depth >= d0) ? (start_depth - d0) : 0;
        cout << "[MOAI_DEPTH_DEBUG] before attention: depth(enc_ecd_x[0])="
             << d0
             << " min_depth=" << min_depth << " max_depth=" << max_depth
             << " chain_index(enc_ecd_x[0])=" << enc_ecd_x[0].chain_index()
             << " start_depth(total_level)=" << start_depth
             << " drops(enc_ecd_x[0])=" << drops0
             << " pre_att_drops(enc_ecd_x)=" << pre_att_drops
             << " (expected before_att_depth≈" << (remaining_level_att - 1) << ")" << endl;
    }

    cout << "Modulus chain before attention: chain_depth=" << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()
         << " chain_index=" << enc_ecd_x[0].chain_index() << " (set MOAI_CHAIN_DEBUG=1 for per-head traces)" << endl;

    vector<vector<PhantomCiphertext>> att_block(num_head);

    gettimeofday(&tstart1, NULL);

    for (int i = 0; i < num_head; ++i)
    {
        att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], bQ[i], bK[i], bV[i],
                                        b_vec, num_input, context, relin_keys, gal_keys_boot, bootstrapper, num_X, secret_key, 16, 10);
        /*
        cout <<"Decrypt + decode result of ";
        cout <<i+1<<"-th head: "<<endl;
        for (int j = 0; j < att_block[i].size(); ++j){
            Plaintext plain_result;
            decryptor.decrypt(att_block[i][j], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            cout <<j+1<<"-th ciphertext: ";
            for (int ind = 0 ; ind < slot_count ; ++ind){
                if(b_vec[ind] == 1){
                    cout <<result[ind]<<" ";
                }
            }
            cout <<endl;
        }
    */
    }

    gettimeofday(&tend1, NULL);
    double att_block_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "Attention block time = " << att_block_time << endl;
    append_csv_row("../single_layer_results.csv", "Attention Block", att_block_time);
    // cout <<"Modulus chain index for the result: "<< context.get_context_data(att_block[2][0].params_id()).chain_depth()<<endl;
// }
/*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int k = 0; k < num_head; ++k){
        //cout <<k+1<<"-th head: "<<endl;
        for (int i = 0; i < att_block[k].size(); ++i){
            Plaintext plain_result;
            decryptor.decrypt(att_block[k][i], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            cout <<i+1<<"-th ciphertext: ";
            for (int ind = 0 ; ind < slot_count ; ++ind){
                if(b_vec[ind] == 1){
                    cout <<result[ind]<<", ";
                }
            }
            cout <<endl;
        }
    }


    cout <<endl;

*/
// delete enc_ecd_x
    vector<PhantomCiphertext>().swap(enc_ecd_x);

    int output_size = att_block[0].size();

    vector<PhantomCiphertext> att_output(num_head*output_size);

    for (int i = 0; i < num_head; ++i){
        for (int j = 0 ; j < output_size ; ++j){
            att_output[i*output_size+j] = att_block[i][j];
            // att_output[i*output_size+j] = att_block[0][j];
        }
    }

    cout <<"Concatenation. size of output of attention block = "<<num_head<<" * "<<output_size<<" = "<<att_output.size()<<endl;

    vector<vector<PhantomCiphertext>>().swap(att_block);
    double selfoutput_time;
    //att_output * selfoutput + selfoutput_bias

    vector<PhantomCiphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre_large_single(att_output, selfoutput, num_col, num_col, num_col, context, selfoutput_time);

    int att_selfoutput_size = att_selfoutput.size();
    //cout <<"num of ct in att_selfoutput = "<<att_selfoutput_size<<endl;
    for (int i = 0; i < num_col; ++i){
        PhantomPlaintext ecd_self_bias;
        vector<double> self_bias_vec(slot_count,0);
        for (int j = 0; j < slot_count; ++j){
            if(b_vec[j] == 1){
                self_bias_vec[j] = selfoutput_bias[i];
            }
        }
        encoder.encode(self_bias_vec, att_selfoutput[i].params_id(), att_selfoutput[i].scale(), ecd_self_bias);
        evaluator.mod_switch_to_inplace(ecd_self_bias, att_selfoutput[i].params_id());
        att_selfoutput[i].scale() = scale;
        ecd_self_bias.scale() = scale;
        evaluator.add_plain_inplace(att_selfoutput[i],ecd_self_bias);
    }

    cout <<"selfoutput time = "<<selfoutput_time<<endl;
    append_csv_row("../single_layer_results.csv", "SelfOutput", selfoutput_time);
    cout <<"Modulus chain index for the result: "<< context.get_context_data(att_selfoutput[0].params_id()).chain_depth()<<endl;

/*
    cout <<"Decrypt + decode result of selfoutput: "<<endl;
    //decrypt and decode
    for (int k = 0; k < att_selfoutput.size(); ++k){
        cout <<k+1<<"-th ciphertext: ";
        Plaintext plain_result;
        decryptor.decrypt(att_selfoutput[k], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }
*/
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < att_selfoutput_size; ++i){
        while(context.get_context_data(att_selfoutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(att_selfoutput[i]);
        }
    }

    vector<PhantomCiphertext> rtn(att_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for

    if (!moai_ln_disable_preln_bootstrap) {
        for(int i = 0 ; i < 128 ; ++i){
            for(int j = 0 ; j < 6 ; ++j){
                bootstrapper.bootstrap_3(rtn[i*6+j],att_selfoutput[i*6+j]);
            }
        }
    } else {
        // Variant 1/2: skip pre-LN bootstrapping; keep ciphertexts as-is for LN.
        rtn = att_selfoutput;
    }

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout << (moai_ln_disable_preln_bootstrap ? "[SKIP pre-LN1 BTS] " : "") << "bootstrapping time = " << boot_time << endl;
    append_csv_row("../single_layer_results.csv",
                   moai_ln_disable_preln_bootstrap ? "1st Bootstrapping (skipped)" : "1st Bootstrapping",
                   boot_time);
    append_csv_row("../single_layer_results.csv", "pre-LN1 bootstrapping calls", moai_ln_disable_preln_bootstrap ? 0.0 : 1.0);
    append_csv_row("../single_layer_results.csv", "pre-LN1 bootstrapped ciphertexts", moai_ln_disable_preln_bootstrap ? 0.0 : 768.0);
    cout <<"Modulus chain index after (pre-LN1) bootstrapping stage: "<< context.get_context_data(rtn[0].params_id()).chain_depth()<<endl;

    //for (int i = 0; i < rtn.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn[i]);
    //}
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;

    vector<PhantomCiphertext>().swap(att_selfoutput);

    /*
    //decrypt and decode
    cout <<"Decrypt + decode result of bootstrapping: "<<endl;
    for (int i = 0; i < rtn.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
    */
    //LayerNorm
    //cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);

    //rtn+enc_ecd_x_copy
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].params_id());
        evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);
    }

    double ln1_internal_var_bs_time = 0.0;
    vector<PhantomCiphertext> layernorm_selfoutput =
        layernorm(rtn, layernorm1_gamma, layernorm1_beta, b_vec, context, relin_keys, secret_key,
                 moai_ln_enable_internal_var_bootstrap ? &bootstrapper : nullptr,
                 moai_ln_enable_internal_var_bootstrap,
                 &ln1_internal_var_bs_time);

    gettimeofday(&tend1,NULL);
    double layernorm_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time<<endl;
    append_csv_row("../single_layer_results.csv", "LayerNorm1", layernorm_time);
    if (moai_ln_enable_internal_var_bootstrap) {
        append_csv_row("../single_layer_results.csv", "LayerNorm1 internal var bootstrap", ln1_internal_var_bs_time);
        append_csv_row("../single_layer_results.csv", "LN1 internal var bootstrap calls", 1.0);
        append_csv_row("../single_layer_results.csv", "LN1 internal bootstrapped ciphertexts", 1.0);
    } else {
        append_csv_row("../single_layer_results.csv", "LN1 internal var bootstrap calls", 0.0);
        append_csv_row("../single_layer_results.csv", "LN1 internal bootstrapped ciphertexts", 0.0);
    }
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_selfoutput[0].params_id()).chain_depth()<<endl;
    vector<PhantomCiphertext>().swap(enc_ecd_x_copy);
/*
    cout <<"Decrypt + decode result of layernorm: "<<endl;
    for (int i = 0; i < layernorm_selfoutput.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(layernorm_selfoutput[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.0001){
                cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                continue;
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //bootstrapping
    int layernorm_selfoutput_size = layernorm_selfoutput.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < layernorm_selfoutput_size; ++i){
        while(context.get_context_data(layernorm_selfoutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_selfoutput[i]);
        }
    }

    vector<PhantomCiphertext> boot_layer(layernorm_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
    rtn = vector<PhantomCiphertext>(layernorm_selfoutput_size);
    // vector<PhantomCiphertext>().swap(rtn);
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],layernorm_selfoutput[i*6+j]);
            boot_layer[i*6+j] = rtn[i*6+j];
        }
    }

    // #pragma omp parallel for

    for (int i = 0; i < rtn.size(); ++i) {
        for (int j = 0; j < 11; ++j){
            evaluator.mod_switch_to_next_inplace(rtn[i]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time2<<endl;
    append_csv_row("../single_layer_results.csv", "2nd Bootstrapping", boot_time2);
    //cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(boot_layer[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(layernorm_selfoutput);

/*
    //decrypt and decode
    cout <<"Decrypt + decode result of bootstrapping: "<<endl;
    for (int i = 0; i < 10; ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //rtn * inter_weight + inter_bias

    cout <<"Modulus chain index before intermediate linear: "<< context.get_context_data(rtn[0].params_id()).chain_depth()<<endl;
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> inter_output = ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight, num_col, num_inter, num_col, context);
    int inter_output_size = inter_output.size();
    //cout <<"num of ct in inter_output = "<<inter_output_size<<endl;
    /*
    cout <<"scale of inter_output = "<<log2(inter_output[0].scale())<<endl;
    cout <<"Decrypt + decode result of intermediate_linear wo bias: "<<endl;
    for (int i = 0; i < inter_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(inter_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                cout <<"( "<<ind<<", "<<result[ind]<<"). ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/

    const int max_threads = omp_get_max_threads();
    // GeLU / post-intermediate paths use up to this many OpenMP threads, each with a CUDA stream and
    // heavy temp ciphertexts in gelu_v2 — cap to reduce VRAM (see MOAI_SINGLE_LAYER_OMP_THREADS).
    int thread_cap = 32;
    if (const char *ev = std::getenv("MOAI_SINGLE_LAYER_OMP_THREADS")) {
        char *end = nullptr;
        long v = std::strtol(ev, &end, 10);
        if (end != ev && *end == '\0' && v >= 1 && v <= 128) {
            thread_cap = static_cast<int>(v);
        }
    }
    const int nthreads = std::max(1, std::min(max_threads, thread_cap));
    // std::cout << "nums of thread: " << nthreads << std::endl;


    if (stream_pool.size() < static_cast<size_t>(nthreads))
    {
        stream_pool.reserve(nthreads);
        for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
        {
            stream_pool.emplace_back();
        }
    }

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];

#pragma omp for schedule(static)    
        for (int i = 0; i < num_inter; ++i){
            PhantomPlaintext ecd_inter_bias;
            vector<double> inter_bias_vec(slot_count,0);
            for (int j = 0; j < slot_count; ++j){
                if(b_vec[j] == 1){
                    inter_bias_vec[j] = inter_bias[i];
                }
            }
            encoder_local.encode(inter_bias_vec, inter_output[i].params_id(), inter_output[i].scale(), ecd_inter_bias, stream);
            bridge_to_default(stream); 
            evaluator_local.mod_switch_to_inplace(ecd_inter_bias, inter_output[i].params_id());
            inter_output[i].scale() = scale;
            ecd_inter_bias.scale() = scale;
            evaluator_local.add_plain_inplace(inter_output[i],ecd_inter_bias);

        }
    cudaStreamSynchronize(stream.get_stream());
    }

    // for (int i = 0; i < num_inter; ++i){
    //     PhantomPlaintext ecd_inter_bias;
    //     vector<double> inter_bias_vec(slot_count,0);
    //     for (int j = 0; j < slot_count; ++j){
    //         if(b_vec[j] == 1){
    //             inter_bias_vec[j] = inter_bias[i];
    //         }
    //     }
    //     encoder.encode(inter_bias_vec, inter_output[i].params_id(), inter_output[i].scale(), ecd_inter_bias);
    //     evaluator.mod_switch_to_inplace(ecd_inter_bias, inter_output[i].params_id());
    //     inter_output[i].scale() = scale;
    //     ecd_inter_bias.scale() = scale;
    //     evaluator.add_plain_inplace(inter_output[i],ecd_inter_bias);

    // }

    gettimeofday(&tend1,NULL);
    double inter_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Inter layer time = "<<inter_time<<endl;
    append_csv_row("../single_layer_results.csv", "Intermediate Linear", inter_time);
    cout <<"Modulus chain index after inter layer: "<< context.get_context_data(inter_output[0].params_id()).chain_depth()<<endl;

/*
    cout <<"Decrypt + decode result of intermediate_linear: "<<endl;
    for (int i = 0; i < inter_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(inter_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                if(iscout == 0){
                    cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                    iscout ++;
                }
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/

    //2025.09.08, change to omp
    // const int max_threads = omp_get_max_threads();
    // const int nthreads = std::max(1, std::min(max_threads, 4));

    if (stream_pool.size() < static_cast<size_t>(nthreads))
    {
        stream_pool.reserve(nthreads);
        for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
        {
            stream_pool.emplace_back();
        }
    }

    vector<PhantomCiphertext> gelu_output(num_inter);

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
    {   
        // PhantomSecretKey secret_key_local(context);
        // PhantomRelinKey relin_keys_local = secret_key_local.gen_relinkey(context);

        const int tid = omp_get_thread_num();
        auto &stream = stream_pool[tid]; 
#pragma omp for schedule(static)
        for (int i = 0; i < 96; ++i)
        {
            for (int j = 0 ; j < 32; ++j)
            {
                gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key, stream);
            }
        }
        cudaStreamSynchronize(stream.get_stream());
    }

    gettimeofday(&tend1,NULL);
    double gelu_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"gelu time = "<<gelu_time<<endl;
    append_csv_row("../single_layer_results.csv", "GELU", gelu_time);
    cout <<"Modulus chain index for gelu: "<< context.get_context_data(gelu_output[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(inter_output);
/*
    cout <<"Decrypt + decode result of intermediate_gelu: "<<endl;
    for (int i = 0; i < gelu_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(gelu_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }

        //    else if(result[ind] >= 0.001){
        //        if(iscout == 0){
        //            cout <<"( "<<ind<<", "<<result[ind]<<"). ";
        //            iscout ++;
        //        }
        //    }

        }
        cout <<endl;
    }

    cout <<endl;
*/
    //gelu * final_weight + final_bias
    gettimeofday(&tstart1,NULL);
    double final_time;
    vector<PhantomCiphertext> final_output = ct_pt_matrix_mul_wo_pre_w_mask_single(gelu_output, final_weight,b_vec, num_inter, num_col, num_inter, context, final_time);
    int final_output_size = final_output.size();
    //cout <<"num of ct in final_output = "<<final_output_size<<endl;

#pragma omp parallel num_threads(nthreads)
    {   
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Encoder encoder_local(&context, &phantom_encoder_local);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

        const int tid = omp_get_thread_num();
        auto &stream = stream_pool[tid];
#pragma omp for schedule(static)
        for (int i = 0; i < num_col; ++i){
            PhantomPlaintext ecd_final_bias;
            vector<double> final_bias_vec(slot_count,0);
            for (int j = 0; j < slot_count; ++j){
                if(b_vec[j] == 1){
                    final_bias_vec[j] = final_bias[i];
                }
            }
            encoder_local.encode(final_bias_vec, final_output[i].params_id(), final_output[i].scale(), ecd_final_bias, stream);
            bridge_to_default(stream);
            evaluator_local.mod_switch_to_inplace(ecd_final_bias, final_output[i].params_id());
            final_output[i].scale() = scale;
            ecd_final_bias.scale() = scale;
            evaluator_local.add_plain_inplace(final_output[i],ecd_final_bias);

        }
    cudaStreamSynchronize(stream.get_stream());
    }

    // for (int i = 0; i < num_col; ++i){
    //     PhantomPlaintext ecd_final_bias;
    //     vector<double> final_bias_vec(slot_count,0);
    //     for (int j = 0; j < slot_count; ++j){
    //         if(b_vec[j] == 1){
    //             final_bias_vec[j] = final_bias[i];
    //         }
    //     }
    //     encoder.encode(final_bias_vec, final_output[i].params_id(), final_output[i].scale(), ecd_final_bias);
    //     evaluator.mod_switch_to_inplace(ecd_final_bias, final_output[i].params_id());
    //     final_output[i].scale() = scale;
    //     ecd_final_bias.scale() = scale;
    //     evaluator.add_plain_inplace(final_output[i],ecd_final_bias);

    // }

    gettimeofday(&tend1,NULL);
    // double final_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Final layer time = "<<final_time<<endl;
    append_csv_row("../single_layer_results.csv", "Final Linear", final_time);
    cout <<"Modulus chain index after final layer: "<< context.get_context_data(final_output[0].params_id()).chain_depth()<<endl;
/*
    cout <<"Decrypt + decode result of intermediate_final: "<<endl;
    for (int i = 0; i < final_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(final_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                if(iscout == 0){
                    cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                    iscout ++;
                }
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //bootstrapping
    //int final_output_size = final_output.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < final_output_size; ++i){
        while(context.get_context_data(final_output[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(final_output[i]);
        }
    }

    //cout<<"bootstrapping start. "<<endl;
    vector<PhantomCiphertext> rtn2(768);
    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for

    if (!moai_ln_disable_preln_bootstrap) {
        for(int i = 0 ; i < 128 ; ++i){
            for(int j = 0 ; j < 6 ; ++j){
                bootstrapper.bootstrap_3(rtn2[i*6+j],final_output[i*6+j]);
            }
        }
    } else {
        // Variant 1/2: skip pre-LN bootstrapping; keep ciphertexts as-is for LN.
        rtn2 = final_output;
    }

    gettimeofday(&tend1,NULL);
    double boot_time3 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout << (moai_ln_disable_preln_bootstrap ? "[SKIP pre-LN2 BTS] " : "") << "bootstrapping time = " << boot_time3 << endl;
    append_csv_row("../single_layer_results.csv",
                   moai_ln_disable_preln_bootstrap ? "3rd Bootstrapping (skipped)" : "3rd Bootstrapping",
                   boot_time3);
    append_csv_row("../single_layer_results.csv", "pre-LN2 bootstrapping calls", moai_ln_disable_preln_bootstrap ? 0.0 : 1.0);
    append_csv_row("../single_layer_results.csv", "pre-LN2 bootstrapped ciphertexts", moai_ln_disable_preln_bootstrap ? 0.0 : 768.0);
    cout <<"Modulus chain index after (pre-LN2) bootstrapping stage: "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;

   // for (int i = 0; i < rtn2.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn2[i]);
   // }
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;

    vector<PhantomCiphertext>().swap(final_output);

    cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);

    //rtn+enc_ecd_x_copy
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].params_id());
        evaluator.add_inplace(rtn2[i],boot_layer[i]);
    }
/*
    cout <<"Decrypt + decode result before layernorm: "<<endl;
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn2[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    double ln2_internal_var_bs_time = 0.0;
    vector<PhantomCiphertext> layernorm_finaloutput =
        layernorm2(rtn2, layernorm2_gamma, layernorm2_beta, b_vec, context, relin_keys, secret_key,
                  moai_ln_enable_internal_var_bootstrap ? &bootstrapper : nullptr,
                  moai_ln_enable_internal_var_bootstrap,
                  &ln2_internal_var_bs_time);

    gettimeofday(&tend1,NULL);
    double layernorm_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time2<<endl;
    append_csv_row("../single_layer_results.csv", "LayerNorm2", layernorm_time2);
    if (moai_ln_enable_internal_var_bootstrap) {
        append_csv_row("../single_layer_results.csv", "LayerNorm2 internal var bootstrap", ln2_internal_var_bs_time);
        append_csv_row("../single_layer_results.csv", "LN2 internal var bootstrap calls", 1.0);
        append_csv_row("../single_layer_results.csv", "LN2 internal bootstrapped ciphertexts", 1.0);
    } else {
        append_csv_row("../single_layer_results.csv", "LN2 internal var bootstrap calls", 0.0);
        append_csv_row("../single_layer_results.csv", "LN2 internal bootstrapped ciphertexts", 0.0);
    }
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_finaloutput[0].params_id()).chain_depth()<<endl;
    vector<PhantomCiphertext>().swap(rtn);

    // cout <<"Decrypt + decode result of one layer: "<<endl;
    // for (int i = 0; i < layernorm_finaloutput.size(); ++i){
    //     PhantomPlaintext plain_result;
    //     decryptor.decrypt(layernorm_finaloutput[i], plain_result);
    //     vector<double> result;
    //     encoder.decode(plain_result, result);
    //     cout <<i+1<<"-th ciphertext: ";
    //     for (int ind = 0 ; ind < slot_count ; ++ind){
    //         if(b_vec[ind] == 1){
    //             cout <<result[ind]<<", ";
    //         }
    //     }
    //     cout <<endl;
    // }

    // cout <<endl;

    //bootstrapping
    int layernorm2_size = layernorm_finaloutput.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < layernorm2_size; ++i){
        while(context.get_context_data(layernorm_finaloutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_finaloutput[i]);
        }
    }

    //cout<<"bootstrapping start. "<<endl;
    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
    rtn2 = vector<PhantomCiphertext>(layernorm2_size);
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],layernorm_finaloutput[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time4 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time4<<endl;
    append_csv_row("../single_layer_results.csv", "4th Bootstrapping", boot_time4);
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(layernorm_finaloutput);
/*
    cout <<"Decrypt + decode result of one layer: "<<endl;
    for (int i = 0; i < rtn2.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn2[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    double total_time = att_block_time+selfoutput_time+layernorm_time+inter_time+gelu_time+final_time+layernorm_time2
    +boot_time+boot_time2+boot_time3+boot_time4;
    cout <<"Total time for one layer: "<<total_time<<", amortized time: "<<total_time/256.0<<endl;
    append_csv_row("../single_layer_results.csv", "Total time for one layer", total_time);
    append_csv_row("../single_layer_results.csv", "Amortized time", total_time/256.0);

}
