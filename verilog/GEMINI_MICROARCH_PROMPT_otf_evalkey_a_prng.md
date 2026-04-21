## Gemini 이미지 생성용 프롬프트 (MOAI OTF EvalKey‑A PRNG 엔진 마이크로아키텍처)

아래 내용을 그대로 Gemini(이미지 생성) 프롬프트로 넣어줘. 결과물은 “논문/아키텍처 리뷰”에 쓰기 좋은 **클린한 블록 다이어그램 1장**이 목표야.

---

**Prompt:**

Create a clean, professional microarchitecture block diagram for a hardware “On-the-fly EvalKey ‘a’ PRNG engine” implemented in SystemVerilog. The top module is `otf_evalkey_a_top.sv`. Style: modern flat vector diagram, white background, thin dark lines, subtle gray module boxes, readable labels, consistent arrows. Use a 16:9 landscape layout. Show clocked pipeline / valid-ready backpressure explicitly with `valid/ready` labels on key streams. Use concise annotations.

#### Overall function
This engine generates random polynomial coefficients `a` for FHE eval-keys on-chip using SHAKE128 XOF + reject sampling. Coefficients are generated directly in NTT domain (random uniform mod q). The design is lane-parallel: **NUM_SHAKE_LANES = 2**, **NUM_SAMPLER_LANES = 8**. Output is **vector output: 8 coefficients per cycle** (each lane has independent valid/ready). One request is processed at a time.

#### Top-level I/O (left and right)
On the left, show **Request Interface**:
- `req_valid`, `req_ready`
- `req` struct fields: `master_seed[255:0]`, `key_id[63:0]`, `decomp_id[63:0]`, `limb_id[63:0]`, `poly_id[63:0]`, `q[63:0]`, `threshold_T[63:0]`, `num_coeffs[63:0]`
Also show internal state registers: `cur`, `have_req`, `remaining`.

On the right, show **Vector Coeff Output**:
- `coeff_valid[7:0]`
- `coeff_ready[7:0]`
- `coeff_data[7:0][63:0]` (u64 coefficients)

#### Internal modules and datapath (center)
Depict these modules and connections:

1) **Seed Pack / Domain Separation (per SHAKE lane)**
- Takes `req.master_seed`, IDs, and lane index.
- Packs a 512-bit seed blob: `[master_seed(256)] + [key_id] + [decomp_id] + [limb_id] + [poly_id XOR lane_index]`.
- Outputs `seed_valid_l[i] / seed_ready_l[i]`, `seed_bits_l[i][511:0]`, `seed_bytes_l[i]=64`.

2) **Two SHAKE128 Block Lanes (i = 0..1)**
Module name: `shake128_xof_block`.
- Inputs: `seed_valid/ready`, `seed_bits`, `seed_bytes`, `squeeze_en`.
- Outputs: `block_valid/block_ready`, `block_data[1343:0]` (= 21×64-bit words).
- Show that `squeeze_en` is enabled while `have_req=1`.
- Note: block generation is throttled by downstream backpressure via `block_ready`.

3) **Per-lane Block→Multiword Issuer**
Module name: `block_word_issuer`.
- Input stream: `block_valid/block_ready + 1344b block`.
- Internal: small FIFO for blocks (use label “block FIFO depth ~ BITFIFO_DEPTH_WORDS/21”).
- Output each cycle: **ISSUE_W_PER_SHAKE = 4** candidate words:
  - `out_valid[3:0]`, `out_ready[3:0]`, `out_word[3:0][63:0]`.
- Mention “issues up to 4×64b words per cycle per SHAKE lane”.

4) **Static Distributor / Mapping**
Show a simple wiring mapping:
- SHAKE lane 0 issuer slots 0..3 → sampler lanes 0..3
- SHAKE lane 1 issuer slots 0..3 → sampler lanes 4..7
This mapping carries `valid/ready` backpressure upstream (sampler input readiness drives issuer `out_ready`).

5) **Eight Reject Sampler Lanes**
Module name: `reject_sampler_lane` (8 instances).
- Inputs: `q`, `threshold_T`, `in_valid/in_ready`, `in_word[63:0]`.
- Outputs: `out_valid/out_ready`, `out_coeff[63:0]`.
- Reject logic: accept if `in_word < T` where `T = floor(2^64/q)*q` (use `threshold_T` if provided).
- Output coefficient = `in_word % q` on accept; no output on reject.
- Show per-lane statistics counters as optional small boxes: `stat_words`, `stat_accepts`, `stat_rejects`.

6) **Output Vector Interface**
- `coeff_valid[i] = sampler_out_valid[i]`
- `coeff_ready[i]` drives sampler `out_ready[i]`
- `coeff_data[i] = out_coeff[i]`

7) **Completion / Remaining Counter Logic**
Show logic that decrements `remaining` by the number of accepted outputs each cycle:
- `dec = popcount(out_valid & out_ready)`
- `remaining -= dec`
- when `remaining` reaches 0: `have_req=0` and `req_ready=1`

#### Visual requirements
- Group repeated structures with braces and labels (e.g., “SHAKE lanes ×2”, “Sampler lanes ×8”).
- Use clear bus labels (1344b, 64b×4, 64b).
- Mark the key handshake points with `valid/ready`.
- Include a small legend: “backpressure propagates right→left via ready”.
- Title at top: “OTF EvalKey ‘a’ PRNG Engine (2 SHAKE lanes → 8 Reject Samplers → 8‑wide coeff vector)”.

Generate the final image only (no extra text).

