#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom.h"

namespace moai {

/** Read manifest key=value lines written by moai_gen_keys. */
inline size_t read_manifest_u64_value(const std::string &manifest_path, const char *key) {
    std::ifstream in(manifest_path);
    if (!in) {
        throw std::runtime_error("cannot open manifest: " + manifest_path);
    }
    const std::string prefix = std::string(key) + "=";
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, prefix.size(), prefix) == 0) {
            return static_cast<size_t>(std::stoull(line.substr(prefix.size())));
        }
    }
    throw std::runtime_error(std::string("manifest missing key: ") + key);
}

inline void read_u64_file(const std::string &path, std::vector<uint64_t> &out, size_t expected_words) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open key file: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    if (bytes % sizeof(uint64_t) != 0) {
        throw std::runtime_error("bad u64 file size: " + path);
    }
    const size_t words = bytes / sizeof(uint64_t);
    if (expected_words != 0 && words != expected_words) {
        throw std::runtime_error("unexpected word count in " + path);
    }
    out.resize(words);
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
}

/**
 * Load keys from a directory produced by moai_gen_keys (secret_key.bin, public_key.bin, relin/, galois/).
 * @param key_pack_dir e.g. .../keys_dnum_35
 */
inline void load_precomputed_keys_from_directory(const PhantomContext &context, const std::string &key_pack_dir,
                                                 PhantomSecretKey &secret_key, PhantomPublicKey &public_key,
                                                 PhantomRelinKey &relin_keys, PhantomGaloisKey &gal_keys_boot) {
    const std::string manifest_path = key_pack_dir + "/manifest.txt";
    const size_t manifest_dnum = read_manifest_u64_value(manifest_path, "dnum");
    const size_t galois_count = read_manifest_u64_value(manifest_path, "galois_key_count");

    const auto &s = phantom::util::global_variables::default_stream->get_stream();
    auto &kp = context.get_context_data(0).parms();
    const size_t N = kp.poly_modulus_degree();
    const size_t coeff_mod = kp.coeff_modulus().size();
    const size_t size_P = kp.special_modulus_size();
    const size_t dnum = (coeff_mod - size_P) / size_P;

    if (dnum != manifest_dnum) {
        throw std::runtime_error("dnum mismatch: context vs manifest (check MOAI_ALPHA / key folder)");
    }

    std::vector<uint64_t> buf;

    read_u64_file(key_pack_dir + "/secret_key.bin", buf, N * coeff_mod);
    secret_key.load_secret_key_ntt_from_host(context, buf.data(), buf.size(), s);

    read_u64_file(key_pack_dir + "/public_key.bin", buf, 2U * N * coeff_mod);
    public_key.load_public_key_from_host_ntt(context, buf.data(), buf.size(), s);

    std::vector<std::vector<uint64_t>> relin_host(dnum);
    std::vector<const uint64_t *> relin_rows(dnum);
    for (size_t t = 0; t < dnum; ++t) {
        char name[512];
        std::snprintf(name, sizeof(name), "%s/relin/tower_%02zu.bin", key_pack_dir.c_str(), t);
        read_u64_file(name, relin_host[t], 2U * N * coeff_mod);
        relin_rows[t] = relin_host[t].data();
    }
    relin_keys.load_relin_towers_from_host(context, relin_rows, s);

    std::vector<PhantomRelinKey> galois_vec;
    galois_vec.reserve(galois_count);
    for (size_t g = 0; g < galois_count; ++g) {
        std::vector<std::vector<uint64_t>> gh(dnum);
        std::vector<const uint64_t *> rows(dnum);
        for (size_t t = 0; t < dnum; ++t) {
            char name[512];
            std::snprintf(name, sizeof(name), "%s/galois/key_%02zu/tower_%02zu.bin", key_pack_dir.c_str(), g, t);
            read_u64_file(name, gh[t], 2U * N * coeff_mod);
            rows[t] = gh[t].data();
        }
        PhantomRelinKey rk;
        rk.load_relin_towers_from_host(context, rows, s);
        galois_vec.push_back(std::move(rk));
    }
    gal_keys_boot.load_from_relin_keys(std::move(galois_vec));
}

} // namespace moai
