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

## Hybrid key-switching (`MOAI_ALPHA` / `special_modulus_size`)

`single_layer_test()` reads **`MOAI_ALPHA`** (default `1`) and sets Phantom’s `special_modulus_size`. The relin decomposition degree is **`dnum = (T - alpha) / alpha`** with `T = |coeff_modulus|`. Keys must match that `alpha`:

```bash
# Example: alpha = 4 → under `<base>/` you get `keys_dnum_9/` when T = 40 (remaining_level=24 in single_layer)
./build/moai_gen_keys /path/to/key_base 4
export MOAI_KEYS_BASE=/path/to/key_base
export MOAI_ALPHA=4
./build/test
```

If QK^T runs out of GPU memory, lower **parallelism in `ct_ct_matrix_mul_colpacking`** (each OpenMP thread uses its own CUDA stream and temporary ciphertexts):

- `export OMP_NUM_THREADS=4` (or `2` / `1`) — caps how many rows of the colpacking outer parallel loop run at once.
- `export MOAI_CT_CT_COLPACK_MAX_THREADS=4` — additional cap only for QK^T colpacking (`Ct_ct_matrix_mul.cuh`); try `4`, then `2`, then `1` if still OOM.

Also free other processes using the same GPU (`nvidia-smi`).

**OOM after attention (GeLU / final linear):** intermediate+GeLU use OpenMP with up to 32 streams; each `gelu_v2` holds many temporary ciphertexts. Cap parallelism:

- `export MOAI_SINGLE_LAYER_OMP_THREADS=4` (try `2` or `1` if still OOM)
- or lower `OMP_NUM_THREADS` globally (combines with the cap above)

For the V-branch mod-switch stop rule, optional **`MOAI_ATT_V_DEPTH_CAP`** (positive integer) overrides the default cap derived from `K`’s tier (see `single_att_block.cuh`).
