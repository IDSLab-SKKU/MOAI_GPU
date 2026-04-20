# 하이브리드 키스위칭 — 시뮬레이터 연산 횟수 정의

이 문서는 **MOAI GPU 시뮬레이터**가 하이브리드 CKKS 키스위칭 **코어**에 대해 세는 **구조적 호출 횟수**를 정리한 것이다. Phantom·SEAL 쪽의 수학적 정의(`alpha`, `dnum`, RNS 크기)는 [HYBRID_KEY_SWITCHING.md](HYBRID_KEY_SWITCHING.md)를 참고하고, 여기서는 **`EngineModel::enqueue_keyswitch_phantom_ckks`** 구현과 1:1로 맞춘 **`keyswitch_op_profile.h`의 해석적 공식**만 다룬다.

---

## 1. 코드 상의 “진실”

| 항목 | 위치 |
|------|------|
| 스케줄 그래프 (실제 `schedule()` 호출) | `src/include/source/sim/engine_model.h` — `enqueue_keyswitch_phantom_ckks` |
| CSV/표에 쓰는 **해석(analytic)** 카운트 | `src/include/source/sim/keyswitch_op_profile.h` — `compute_keyswitch_phantom_profile` |
| 엔진 측정 (`eng_ntt_enq`, `eng_vec_enq`) | `profile_keyswitch_phantom_ckks` → 위 enqueue와 동일 그래프, `MOAI_SIM_BACKEND=1`일 때 `EngineStats::enqueue_calls` |
| 엔진 **coeff 합산** (`eng_ntt_coeff_ops`, `eng_vec_coeff_ops`) | 동일 실행에서 `EngineStats::logical_ops` — 각 `enqueue_ntt_coeffs` / `enqueue_vec_*`에 넘긴 `coeffs`(= \(N\times\)해당 패스 limb 수)를 **패스마다 더한 값**. 해석 열의 `weighted_*`와는 **다른 출처**(엔진이 실제로 스케줄에 넣은 합). |

**추상(analytic) vs 실제 엔진:** `ntt_kernels` 등은 공식 복제. α 스윕마다 **실제 config로 엔진이 몇 번·몇 coeff-op을 잡았는지**를 쓰려면 `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1` + `MOAI_SIM_BACKEND=1` 로 생성되는 **`eng_*` 열**을 본다.

해석 프로파일은 **런타임 샘플링이 아니라** enqueue 순서를 따라 **종이로 세는 것**과 같다.

---

## 2. 기호 (이 시뮬 한정)

- \(N\) = `poly_degree` (`MOAI_SIM_POLY_DEGREE`)
- `limbs` 인자 = **\|Ql\|** (데이터 쪽 RNS 길이; 키스위치 입력 다항식이 올라가 있는 Q 쪽)
- `m_cfg.kswitch_size_p` = **\|P\|** (특수 모듈러스 쪽 소수 개수; env `MOAI_SIM_KSWITCH_SIZE_P`, 최소 1)
- \(\texttt{coeffs\_ql} = N \times |Q_l|\), \(\texttt{coeffs\_qlp} = N \times (|Q_l| + |P|)\)

**β (digit 개수)** — `keyswitch_beta_choose` / 프로파일의 `keyswitch_beta_phantom` 등:

- **Phantom 모드 (기본):** \(\beta = \max\left(1,\left\lceil \dfrac{|Q_l|}{\alpha_{\mathrm{digit}}} \right\rceil\right)\), 여기서 \(\alpha_{\mathrm{digit}} =\) `MOAI_SIM_ALPHA` (최소 1).
- **Legacy 모드:** \(\beta = \max\left(1,\left\lceil \dfrac{|Q_l|}{|P|_{\mathrm{cfg}}} \right\rceil\right)\) (`MOAI_SIM_KSWITCH_BETA_MODE=legacy`, `kswitch_size_p` 사용).
- **`MOAI_SIM_KSWITCH_BETA > 0`** 이면 모든 행에 그 값으로 고정(디버그).

하이브리드 KS CSV 스윕(`moai_sim_hybrid_ks_profile_run`)에서는 행마다 `|Q_l|`, `|P|`, \(\beta\)를 env에 맞게 넣은 뒤 위 공식과 동일하게 `compute_keyswitch_phantom_profile`을 호출한다.

**`MOAI_SIM_HYBRID_KS_EXACT_PARTITION` (기본 `1`):** 기본 \|Ql\|`=T_{qp}-\alpha` 일 때, **\|Ql\| ≥ α** 이고 **\|Ql\| mod α ≠ 0** 이면 해당 α는 스킵한다(완전한 digit 분할이 아님). **\|Ql\| < α** 인 행(예: α가 커서 β=1만 남는 경우)은 유지한다. 전 α·`ceil` β를 쓰려면 `=0`.

**메모리 열:** Phantom `keyswitch_inplace`와 같이 **c2**, **modup 버퍼** \(\beta\times|Q_l P|\), **cx** \(2\times|Q_l P|\) (uint64 개수×8B) 및 상한 **`mem_working_peak_bytes_est`**. `plot_hybrid_ks_profile.py --memory` / `--all-plots` 로 `hybrid_ks_profile_memory.png`.

**시뮬 사이클 열:** `eng_makespan_cyc`, `eng_ntt_*_busy_cyc`, `eng_vec_busy_cyc` — `EngineModel::summary()`의 makespan·`busy_cycles`. **`--cycles-plots`** 로 스택/라인 그래프 (`hybrid_ks_profile_*_cycles.png`). 개수 기반 플롯과 **동일 CSV·동일 env**에서 나온다.

---

## 3. `enqueue_keyswitch_phantom_ckks` 단계별 대응

아래는 `engine_model.h`에 있는 순서 그대로이다. (주석: *INTT Ql → modup ×β → inner prod ×β → moddown ×2 → add ×2*.)

### 3.1 초기 INTT (c₂, Ql)

- `enqueue_ntt_coeffs(coeffs_ql, …)` **1회** → **NTT 카운터 +1**

### 3.2 (선택) 온칩 스크래치

- `m_onchip.enabled()` 이고 \(\beta \neq 0\) 일 때 `enqueue_onchip_xfer(scratch, …)` 가 있을 수 있다.
- **구조적 CSV의 `ntt_kernels` / `bconv_*` / `vec_*`에는 포함하지 않는다** (별도 엔진/onchip 통계).

### 3.3 Mod-up 루프 — `bi = 0 … β−1`

각 반복:

1. `enqueue_vec_coeffs(coeffs_qlp, kswitch_modup_bconv_cyc_per_coeff, …)` — **BConv mod-up**으로 집계 → **`bconv_modup` += 1**
2. `enqueue_ntt_coeffs(coeffs_qlp, …)` → **`ntt_kernels` += 1**

총 **β**번 반복이므로: **mod-up BConv = β**, **NTT(QlP) = β**.

### 3.4 Inner product 루프 — `bi = 0 … β−1`

- `enqueue_vec_mul(coeffs_qlp, …)` → **`vec_mul` += 1** (β번)

### 3.5 Mod-down 루프 — `part = 0, 1` (고정 2번)

각 `part`마다:

1. `enqueue_ntt_coeffs(coeffs_qlp, …)` → **NTT += 1** (총 2)
2. `enqueue_vec_coeffs(coeffs_ql, kswitch_moddown_bconv_cyc_per_coeff, …)` → **`bconv_moddown` += 1** (총 2)
3. `enqueue_vec_mul(coeffs_ql, …)` → **`vec_mul` += 1** (총 2)

### 3.6 ct에 더하기

- `enqueue_vec_add(coeffs_ql, …)` **2회** → `enqueue_vec_add`는 내부적으로 `enqueue_vec_coeffs`를 쓰므로 **`vec_add` = 2** (CSV 열 의미: “vec_add **호출** 횟수”)

---

## 4. β에 대한 닫힌 식 (해석 프로파일과 동일)

`compute_keyswitch_phantom_profile`에서 \(\beta \ge 1\)로 두면:

| CSV / 필드 | 식 |
|------------|-----|
| `ntt_kernels` | \(\beta + 3\) (= `ntt_fwd_kernels` + `ntt_inv_kernels`) |
| `ntt_fwd_kernels` | \(\beta\) (modup 루프의 순방향 NTT on QlP) |
| `ntt_inv_kernels` | \(3\) (Ql에서 INTT 1 + moddown 앞 INTT on QlP ×2) |
| `bconv_modup` | \(\beta\) |
| `bconv_moddown` | \(2\) |
| `vec_mul` | \(\beta + 2\) |
| `vec_add` | \(2\) |

**예:** \(\beta = 1\) → NTT 4, BConv_up 1, BConv_down 2, vec_mul 3, vec_add 2.

**예:** \(\beta = 35\) → NTT 38, BConv_up 35, BConv_down 2, vec_mul 37, vec_add 2.

---

## 5. 가중치 열 (coeff-element 합)

용량·대역 거칠게 잡을 때, 각 패스마다 **\(N \times\) (해당 단계의 limb 수)** 를 더한다.

| 필드 | 식 (코드와 동일) |
|------|------------------|
| `weighted_ntt_coeff_elems` | `weighted_ntt_fwd` + `weighted_ntt_inv` |
| `weighted_ntt_fwd_coeff_elems` | \(\beta\cdot\texttt{coeffs\_qlp}\) |
| `weighted_ntt_inv_coeff_elems` | \(\texttt{coeffs\_ql} + 2\cdot\texttt{coeffs\_qlp}\) |
| `weighted_bconv_coeff_elems` | \(\beta\cdot\texttt{coeffs\_qlp} + 2\cdot\texttt{coeffs\_ql}\) |
| `weighted_vec_mul_coeff_elems` | \(\beta\cdot\texttt{coeffs\_qlp} + 2\cdot\texttt{coeffs\_ql}\) |
| `weighted_vec_add_coeff_elems` | \(2\cdot\texttt{coeffs\_ql}\) |

---

## 6. `EngineModel`의 NTT / VEC `enqueue_calls`와의 관계

- **`enqueue_ntt_coeffs(..., ntt_inverse)`** → `m_ntt` (시간) + **`m_ntt_fwd` / `m_ntt_inv`** (집계) → **`eng_ntt_fwd_enq`**, **`eng_ntt_inv_enq`** 및 합이 **`eng_ntt_enq`** (\(\beta+3\)).
- **`enqueue_vec_coeffs`** 와 **`enqueue_vec_mul`** 모두 **`m_vec`** → **`eng_vec_enq`** 는 “VEC 스케줄” 전부.

VEC 스케줄 합(해석과 비교용):

\[
\underbrace{\beta}_{\text{modup BConv}} + \underbrace{(\beta+2)}_{\text{vec\_mul}} + \underbrace{2}_{\text{moddown BConv}} + \underbrace{2}_{\text{vec\_add→vec\_coeffs}} = 2\beta + 6
\]

CSV에서 검산: `bconv_modup + bconv_moddown + vec_mul + vec_add` 와 `eng_vec_enq` 가 일치해야 한다 (vec_add 2가 각각 `enqueue_vec_coeffs` 1회이므로 `vec_add` 열이 그 2를 이미 센다).

---

## 7. 이 모델이 **포함하지 않는** 것

- 키스위칭 **앞뒤 DMA**, Galois, rotate 본체, relin에서 c₀/c₁ 읽기 등 **`enqueue_keyswitch_phantom_ckks` 밖**의 비용.
- Phantom 소스와 **완전 동일한 커널 분해**를 주장하지 않는다. MOAI는 **한 개의 거친 스케줄 그래프**로 처리량을 잡는다. Phantom 쪽 세부 NTT/BConv 호출이 다르면 이 카운트는 “MOAI enqueue 모델” 기준이다.

---

## 8. 검산 체크리스트

1. \(\beta\)를 Phantom과 동일하게 쓰는지 (`ceil(|Q_l|/alpha)` vs legacy).
2. `ntt_kernels` = \(\beta + 3\) 인지.
3. `vec_mul` = \(\beta + 2\) 인지.
4. `eng_ntt_enq` = `ntt_kernels`, `eng_vec_enq` = `bconv_modup + bconv_moddown + vec_mul + vec_add` 인지 (`MOAI_SIM_BACKEND=1`, `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1`).

이 중 하나라도 어긋나면 env 조합(`MOAI_SIM_KSWITCH_SIZE_P`, `MOAI_SIM_ALPHA`, `|QP|` vs `|Q_l|`) 또는 엔진 쪽 버킷 매핑을 의심하면 된다.
