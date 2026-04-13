/**
 * Pre-generate CKKS keys matching `single_layer_test` parameters (same coeff bit-length recipe),
 * with configurable hybrid key-switching alpha (= parms.special_modulus_size()).
 *
 * Usage: moai_gen_keys [base_output_dir] [alpha]
 *   base_output_dir — parent directory (default: "keys" relative to cwd)
 *   alpha           — special modulus size (default: 1). Must divide T = |coeff_modulus|;
 *                     dnum = (T - alpha) / alpha. Output: base_output_dir/keys_dnum_<dnum>/.
 *                     Example: T=40 and alpha=4 -> dnum=9 (match MOAI_ALPHA=4 for single_layer_test).
 *
 * Format (see keys/README.md):
 *   secret_key.bin     — coeff_mod_size * N uint64 (NTT form), row-major
 *   public_key.bin     — 2 * coeff_mod_size * N uint64 (pk ciphertext)
 *   relin/tower_XX.bin — one file per decomposition tower
 *   galois/KK/tower_XX.bin — Galois index KK, same tower layout as relin
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom.h"

namespace fs = std::filesystem;

using namespace phantom;
using namespace phantom::arith;

/** `PhantomRelinKey::public_keys_ptr()` points to a **device** array of per-tower GPU pointers; index it on host only after D2H copy. */
static std::vector<uint64_t *> copy_gpu_tower_ptrs(const uint64_t *const *d_ptr_table, size_t dnum) {
    std::vector<uint64_t *> host(dnum);
    cudaError_t err =
        cudaMemcpy(host.data(), d_ptr_table, dnum * sizeof(uint64_t *), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy tower ptr table: ") + cudaGetErrorString(err));
    }
    return host;
}

static void write_u64_file(const std::string &path, const uint64_t *gpu_ptr, size_t word_count) {
    std::vector<uint64_t> host(word_count);
    cudaMemcpy(host.data(), gpu_ptr, word_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open for write: " + path);
    }
    out.write(reinterpret_cast<const char *>(host.data()),
              static_cast<std::streamsize>(word_count * sizeof(uint64_t)));
}

static void write_manifest(const std::string &path, const std::string &text) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write manifest: " + path);
    }
    out << text;
}

/** Same CKKS parameter recipe as `test_single_layer.cuh` → `single_layer_test()`.
 *  Keep remaining_level / boot_level / coeff_bit_vec in lockstep when either file changes. */
static EncryptionParameters make_moai_inference_parms(size_t alpha) {
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 46;
    int logq = 51;
    int log_special_prime = 58;

    int secret_key_hamming_weight = 192;

    int remaining_level = 24;  // must match single_layer_test() (test_single_layer.cuh)
    int boot_level = 14;
    std::vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < boot_level; i++) {
        coeff_bit_vec.push_back(logq);
    }
    coeff_bit_vec.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = static_cast<size_t>(1ULL << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(static_cast<size_t>(secret_key_hamming_weight));
    parms.set_sparse_slots(static_cast<size_t>(sparse_slots));
    parms.set_special_modulus_size(alpha);

    return parms;
}

/** Hybrid KS: T = |coeff_modulus|, dnum = (T - alpha) / alpha requires T % alpha == 0 and alpha < T. */
static void check_hybrid_alpha(size_t T, size_t alpha) {
    if (alpha == 0) {
        throw std::invalid_argument("alpha (special_modulus_size) must be >= 1");
    }
    if (alpha >= T) {
        throw std::invalid_argument("alpha must be smaller than coeff_modulus prime count T");
    }
    if (T % alpha != 0) {
        throw std::invalid_argument(
            "hybrid KS: coeff_modulus prime count T must be divisible by alpha (got T=" + std::to_string(T) +
            " alpha=" + std::to_string(alpha) + ")");
    }
    const size_t dnum = (T - alpha) / alpha;
    if (dnum == 0) {
        throw std::invalid_argument("hybrid KS: dnum would be zero for this alpha");
    }
}

int main(int argc, char **argv) {
    try {
        const std::string base_dir = (argc >= 2) ? argv[1] : "keys";
        size_t alpha = 1;
        if (argc >= 3) {
            char *end = nullptr;
            unsigned long v = std::strtoul(argv[2], &end, 10);
            if (end == argv[2] || *end != '\0' || v == 0 || v > (1UL << 20)) {
                throw std::invalid_argument("alpha must be a positive decimal integer");
            }
            alpha = static_cast<size_t>(v);
        }

        EncryptionParameters parms = make_moai_inference_parms(alpha);
        const size_t T = parms.coeff_modulus().size();
        check_hybrid_alpha(T, alpha);

        const size_t coeff_mod_size = parms.coeff_modulus().size();
        const size_t size_P = parms.special_modulus_size();
        const size_t size_Q = coeff_mod_size - size_P;
        const size_t dnum = size_Q / size_P;

        const std::string out_root = (fs::path(base_dir) / ("keys_dnum_" + std::to_string(dnum))).string();

        PhantomContext context(parms);

        auto &key0 = context.get_context_data(0);
        auto &key_parms = key0.parms();
        const size_t N = key_parms.poly_modulus_degree();

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);
        PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
        PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

        cudaDeviceSynchronize();

        const size_t sk_words = coeff_mod_size * N;
        const size_t pk_words = 2U * coeff_mod_size * N;
        const size_t tower_words = 2U * coeff_mod_size * N;

        fs::create_directories(out_root);
        fs::create_directories(out_root + "/relin");
        fs::create_directories(out_root + "/galois");

        write_u64_file(out_root + "/secret_key.bin", secret_key.device_sk_ntt(), sk_words);

        const PhantomCiphertext &pkct = public_key.cipher();
        write_u64_file(out_root + "/public_key.bin", pkct.data(),
                       pkct.size() * pkct.coeff_modulus_size() * pkct.poly_modulus_degree());

        std::vector<uint64_t *> relin_towers = copy_gpu_tower_ptrs(relin_keys.public_keys_ptr(), dnum);
        for (size_t t = 0; t < dnum; t++) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s/relin/tower_%02zu.bin", out_root.c_str(), t);
            write_u64_file(name, relin_towers[t], tower_words);
        }

        const size_t galois_count = galois_keys.galois_key_count();
        for (size_t g = 0; g < galois_count; g++) {
            char gdir[256];
            std::snprintf(gdir, sizeof(gdir), "%s/galois/key_%02zu", out_root.c_str(), g);
            fs::create_directories(gdir);
            const PhantomRelinKey &grk = galois_keys.get_relin_keys(g);
            std::vector<uint64_t *> galois_towers = copy_gpu_tower_ptrs(grk.public_keys_ptr(), dnum);
            for (size_t t = 0; t < dnum; t++) {
                char name[512];
                std::snprintf(name, sizeof(name), "%s/tower_%02zu.bin", gdir, t);
                write_u64_file(name, galois_towers[t], tower_words);
            }
        }

        std::ostringstream man;
        man << "MOAI pre-generated keys (single_layer_test-compatible parameters)\n"
            << "poly_modulus_degree=" << N << "\n"
            << "coeff_modulus_primes=" << coeff_mod_size << "\n"
            << "alpha_special_modulus_size=" << alpha << "\n"
            << "special_modulus_size=" << size_P << "\n"
            << "dnum=" << dnum << "\n"
            << "galois_key_count=" << galois_count << "\n"
            << "secret_key_words=" << sk_words << "\n"
            << "public_key_words=" << pk_words << "\n"
            << "relin_tower_words=" << tower_words << "\n"
            << "sparse_slots=" << parms.sparse_slots() << "\n"
            << "secret_key_hamming_weight=" << parms.secret_key_hamming_weight() << "\n";

        man << "coeff_bit_lengths=";
        for (size_t i = 0; i < key_parms.coeff_modulus().size(); i++) {
            if (i) {
                man << ",";
            }
            man << key_parms.coeff_modulus()[i].bit_count();
        }
        man << "\n";

        write_manifest(out_root + "/manifest.txt", man.str());

        std::cout << "Wrote keys to: " << fs::absolute(out_root).string() << std::endl;
        std::cout << "  alpha=" << alpha << " dnum=" << dnum << " galois_keys=" << galois_count << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "gen_moai_keys: " << e.what() << std::endl;
        return 1;
    }
}
