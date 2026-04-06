# Pre-generated CKKS keys

`gen_moai_keys` (see `src/scripts/`) writes **raw uint64 little-endian** blobs produced from Phantom-FHE on GPU. Parameters match `single_layer_test()` in `test_single_layer.cuh` (poly degree 65536, long modulus chain for bootstrapping, etc.). See `manifest.txt` after generation.

## Layout

| File | Words (uint64) | Description |
|------|------------------|-------------|
| `secret_key.bin` | `coeff_modulus_primes * N` | Secret key in NTT form (same layout as `PhantomSecretKey::load_secret_key` text format, but binary). |
| `public_key.bin` | `2 * coeff_modulus_primes * N` | Public key ciphertext (two polynomials). |
| `relin/tower_XX.bin` | `2 * coeff_modulus_primes * N` each | Relinearization key, one file per decomposition tower (`dnum` files). |
| `galois/key_KK/tower_XX.bin` | same as relin | Galois keys: index `KK` matches internal Phantom order (0 … `galois_key_count-1`). |

Loading these back into Phantom types is not implemented in-tree yet; this export is for **offline storage** and future loader glue.

## Generate

From repo root:

```bash
chmod +x src/scripts/gen_moai_keys.sh
./src/scripts/gen_moai_keys.sh          # writes to ./keys
./src/scripts/gen_moai_keys.sh /path/to/out
```

Or after configuring CMake:

```bash
cmake --build build --target moai_gen_keys
./build/moai_gen_keys ./keys
```

**Note:** Galois key generation uses the default Phantom set (`galois_elts` empty → full rotation set). This can take a long time and a lot of GPU memory.
