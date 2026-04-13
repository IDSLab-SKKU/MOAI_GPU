# MOAI_GPU 분석 — 가속기·시뮬레이터 설계용

대상 코드베이스: `MOAI_GPU` (CKKS 기반 GPU FHE, PhantomFHE 백엔드).  
목적: **하드웨어 가속기를 가정한 사이클·메모리·파이프라인 시뮬레이터**를 만들 때, MOAI가 실제로 무엇을 얼마나 호출하는지 구조적으로 파악하기.

관련 기존 문서(같은 `doc/` 디렉터리):

- [LayerNorm 경로 요약](layernorm_moai_summary.md)
- [GELU / HMult·BSGS](gelu_hmult_bsgs.md)

---

## 1. 한 줄 요약

MOAI_GPU는 **BERT 스타일 트랜스포머의 “한 레이어”**를 CKKS 암호문 위에서 돌리는 연구/벤치 코드이며, 핵심은 **대규모 ct–pt 행렬곱**, **ct–ct 패킹 행렬곱**, **softmax(부트스트랩 포함 경로)**, **LayerNorm(분산·역제곱근)**, **GELU 다항 근사**, 그리고 **여러 번의 부트스트랩(`bootstrap_3`)**으로 구성된다. PhantomFHE가 **NTT·키스위치·GPU 커널**을 담당하고, MOAI 레이어는 이를 **`moai::Encoder` / `moai::Evaluator` 래퍼**로 조합한다.

---

## 2. 저장소·빌드 구조

| 구분 | 경로 | 비고 |
|------|------|------|
| 실행 엔트리 | `src/test.cu` | 기본은 `single_layer_test()`; `MOAI_BENCH_MODE=boot\|ct_ct` 시 마이크로 벤치 |
| 단일 레이어 통합 | `src/include/test/test_single_layer.cuh` | CKKS 파라미터, 키, 부트스트랩 준비, 어텐션 12헤드, LN, FFN, 타이밍·CSV |
| “소스” 연산 모듈 | `src/include/source/**` | matmul, non-linear, att_block, bootstrapping |
| 단위 테스트 | `src/include/test/**` | 각 연산별 검증·벤치 |
| FHE 코어 | `thirdparty/phantom-fhe/` | CKKS, NTT, evaluate 등 |
| 빌드 | `CMakeLists.txt` | `test` 타깃, CUDA + OpenMP, `CMAKE_CUDA_ARCHITECTURES` 기본 120 등 |

`include.cuh`가 소스·테스트 헤더를 한꺼번에 끌어온다. 가속기 시뮬레이터는 **“include 그래프”보다 `single_layer_test` + `single_att_block` + `softmax`/`layernorm`/`gelu` 본체**를 1차 타깃으로 두면 된다.

---

## 3. 런타임 엔트리와 워크로드

### 3.1 `src/test.cu`

- 환경 변수 `MOAI_BENCH_MODE`:
  - `boot` → `bootstrapping_test()`
  - `ct_ct` → `ct_ct_matrix_mul_test()`
- 미설정 시 **`single_layer_test()`** 실행 (BERT 한 레이어 end-to-end에 가까운 경로).

### 3.2 `single_layer_test()` (`test_single_layer.cuh`)

대표 상수(시뮬레이터 dimension 입력으로 쓰기 좋음):

- 배치·시퀀스·히든: `num_X`, `num_row`, `num_col` (기본 히든 768)
- FFN 중간: `num_inter` (기본 3072)
- 어텐션: `num_head = 12`, 헤드당 `col_W = 64` (즉 \(12 \times 64 = 768\))
- 가중치·입력은 `att_block_weights/`, `self_output_weights/`, `feed_forward_weights/` 텍스트에서 로드

시뮬레이터 관점에서 이 함수는 다음을 **한 번에** 고정한다.

- CKKS **계수 모듈러스 비트열** (`logq`, `logp` 반복, `boot_level`, `log_special_prime`)
- **sparse_slots** (`logn = 15` → \(2^{15}\))
- **하이브리드 special modulus** 크기 `MOAI_ALPHA` → `parms.set_special_modulus_size(moai_hybrid_alpha)`
- **어텐션 이전** `enc_ecd_x`에 대한 `mod_switch_to_next` 횟수 (`pre_att_drops` 및 복사본 `enc_ecd_x_copy`의 `boot_level`만큼 드롭 등)
- **부트스트랩 키·갈루아 스텝** 준비 (`Bootstrapper`, `generate_LT_coefficient_3` 등)

즉, **“연산 그래프”뿐 아니라 “모듈러스 체인 예산”**이 이 파일에 집중되어 있어 시뮬레이터의 **전역 스케줄러/depth 모델** 입력이 된다.

---

## 4. 암호문 데이터 모델 (시뮬레이터 추상화)

### 4.1 “768개의 암호문 = 한 토큰·한 행의 히든 차원”

BERT 히든 \(768\)는 **서로 다른 `PhantomCiphertext` 벡터 길이 768**로 표현되는 패턴이 반복된다 (`enc_ecd_x`, LayerNorm 입출력 등).  
시뮬레이터는 이를 다음 중 하나로 모델링할 수 있다.

- **벡터 레지스터 파일**: 길이 768의 ct 배열, 슬롯 마스크 `bias_vec`와 함께 SIMD 스타일 연산
- **슬롯 패킹**: `num_X` 등으로 같은 ct 안에 여러 “논리 배치”를 넣는 구조 (`batch_input` 계열)

### 4.2 메모리 추정 훅

`include.cuh`에 **디바이스 상 ct 바이트 수**를 합산하는 유틸(`ciphertext_device_bytes`, `print_output_mem`)이 있어, 시뮬레이터의 **DRAM/HBM 모델**과 맞추기 쉽다.

---

## 5. 한 레이어 연산 그래프 (추상 DAG)

아래는 `single_layer_test` + `single_att_block` 기준의 **논리적 순서**이다 (세부 분기·환경 변수는 §7).

1. **입력 인코딩·암호화** — `batch_input(...)`
2. **체인 조정 (어텐션 전)**  
   - `enc_ecd_x_copy`: `boot_level`회 `mod_switch_to_next`  
   - `enc_ecd_x`: `pre_att_drops`회 `mod_switch_to_next`  
   → 어텐션 시작 시 **남은 depth / chain_index**가 결정됨 (`MOAI_DEPTH_DEBUG`로 출력 가능)
3. **멀티헤드 어텐션** — 헤드 인덱스 `i = 0..11`에 대해 순차 호출  
   `att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], ...)`
4. **헤드 결합 + self-output 선형** — (파일 후반) ct–pt matmul 및 재배열
5. **(옵션) 대량 부트스트랩** — 기본은 **768 ct**에 대해 `bootstrap_3` (pre-LN1 전); `MOAI_LN_BOOTSTRAP_VARIANT`로 스킵·이전 가능
6. **LayerNorm 1** — `layernorm(...)`; 내부에서 **분산 브랜치 단일 ct 부트스트랩** 옵션 (`bootstrap_var_branch`)
7. **추가 부트스트랩** — LN 이후 단계에서 다시 `bootstrap_3` 반복 (코드상 2차 부트스트랩 구간 존재)
8. **FFN 상·하선형** — ct–pt matmul, 중간 차원 `3072`
9. **GELU** — `gelu_v2` (다항·BSGS류; `gelu_hmult_bsgs.md` 참고)
10. **최종 선형 + (선택적) LN2 등** — 파일 후반부의 weight/bias 적용

시뮬레이터는 각 단계에 대해 최소한 다음 **이벤트 카운터**를 잡을 수 있어야 한다.

- `mod_switch_to_next` / `mod_switch_to_inplace`
- `multiply`, `multiply_plain`, `square`, `add`, `sub`, `relinearize`, `rescale`
- **부트스트랩** (`bootstrap_3` 호출 횟수·입력 ct 수·내부 서브루틴은 `Bootstrapper.cu` 쪽 분해 필요)
- **회전 / keyswitch** (Galois; ct–ct diag/col packing 경로)

Phantom 내부의 **NTT 호출**은 `Evaluator` 구현에 숨겨져 있으므로, 저수준 시뮬레이터는 (a) Phantom을 **블랙박스 latency**로 두거나, (b) Phantom 소스에 훅을 넣어 **NTT 횟수**를 집계하는 두 층 중 하나를 선택하면 된다.

---

## 6. `single_att_block` 세부 (어텐션 서브그래프)

파일: `src/include/source/att_block/single_att_block.cuh`

대표 순서:

1. **Q, K, V** — `ct_pt_matrix_mul_wo_pre(enc_X, W*)` 후 **슬롯 마스크된 bias** `add_plain_inplace`
2. **V 브랜치 depth 캡** — `MOAI_ATT_V_DEPTH_CAP` 또는 휴리스틱 `moai_v_branch_depth_cap`: `enc_X_v`에 대해 `chain_depth > cap` 동안 `mod_switch_to_next`
3. **\(Q K^\top\)** — `ct_ct_matrix_mul_colpacking(Q, K, RotK, relin_keys, ...)`
4. **Softmax** — **`softmax_boot(QK, ...)`** (`softmax.cuh`):  
   - 행별 마스킹·`exp` 근사(스케일·`square` 루프)  
   - logits에서 **합 ct**를 만들고 **최저 레벨까지 mod_switch** 후 **`bootstrapper_att.bootstrap_3`**
   - 합에 대한 **`inverse` 반복**으로 역수 근사 후 각 행과 곱
5. **Softmax 결과와 V의 ct–ct 곱** — 주석에 softmax 결과를 쓰지 않고 `QK`를 `V[0]` 체인에 맞춘 뒤 `ct_ct_matrix_mul_diagpacking(QK, V, ...)` 호출 (구현·논문 정합 시 주의)

가속기 시뮬레이터 포인트:

- **Q/K와 V의 체인 분기**가 명시적으로 존재 → **비대칭 메모리·depth** 모델 필요
- **Softmax 경로만 부트스트랩 1회(합)** + inverse 반복 **iter** (`single_layer_test`에서 `16` 등으로 전달되는 패턴) → **부트스트랩 비용 vs. multiplicative depth** 트레이드오프 분석에 적합
- `append_csv_row("../results.csv", ...)` 로 서브블록 타이밍이 기록됨 → 시뮬레이터 검증용 **골든 레퍼런스**로 활용 가능

---

## 7. 환경 변수·스위치 (시뮬레이터 “시나리오” 입력)

`test_single_layer.cuh` 및 `single_att_block.cuh`에서 읽는 대표 변수:

| 변수 | 의미 (시뮬레이터) |
|------|-------------------|
| `MOAI_ALPHA` | 하이브리드 키스위칭용 `special_modulus_size`; `T % MOAI_ALPHA == 0` 제약 |
| `MOAI_PRECOMPUTED_KEYS_DIR` / `MOAI_KEYS_BASE` | 키 생성 생략·재현성 |
| `MOAI_PRE_ATT_DROPS` / `MOAI_PRE_ATT_DROPS_DELTA` | 어텐션 직전 체인 드롭 수 → **시작 depth** |
| `MOAI_LN_BOOTSTRAP_VARIANT` (0–2) | pre-LN 768-ct 부트스트랩 유지 vs. 생략 + LN 내부 var 부트스트랩 등 |
| `MOAI_LN_LEVEL_BUMP` | variant 2에서 어텐션 시작 depth 조정 |
| `MOAI_DEPTH_DEBUG` / `MOAI_CHAIN_DEBUG` | depth·chain 로그 (검증·캘리브레이션) |
| `MOAI_ATT_V_DEPTH_CAP` | V 브랜치 `mod_switch` 상한 |
| `MOAI_SINGLE_LAYER_OMP_THREADS` | GELU 등 OpenMP 병렬도 (VRAM 절약과 연동, 주석 참고) |
| `MOAI_BENCH_MODE` (`test.cu`) | 부트스트랩 단독 / ct–ct 단독 벤치 |

시뮬레이터는 **“기본 recipe + env 오버라이드”**를 시나리오로 저장·재생하는 형태가 MOAI와 잘 맞는다.

---

## 8. 비선형·고비용 블록 (시뮬레이터 서브모듈)

### 8.1 Softmax (`softmax.cuh`)

- **exp**: `multiply_plain`로 스케일 후 `add_plain`, 이어서 **log2(128)번 square + relin + rescale** — 고정 depth의 다항형 근사 구조
- **inverse**: `iter`번 루프, 매 회 `square`, `multiply`, `relinearize`, `rescale` 등
- **softmax_boot**: 합 ct를 **체인 끝까지 내린 뒤** `bootstrap_3`, 이후 다시 `inverse` 및 행별 곱셈

→ 시뮬레이터에 **“Inverse(iter)” + “Bootstrap_3” + “Exp(고정 제곱 횟수)”** 세 모듈을 분리해 두면, `softmax` / `softmax_boot` 조합을 재구성하기 쉽다.

### 8.2 LayerNorm (`layernorm.cuh`)

- 768 ct 합, 분산용 제곱·부분합, `invert_sqrt` (Newton + Goldschmidt; 반복 횟수 하드코딩 4+2)
- **옵션**: 분산 단일 ct에 대한 부트스트랩 (`bootstrap_var_branch`, `Bootstrapper*`)

자세한 depth 논의는 [layernorm_moai_summary.md](layernorm_moai_summary.md).

### 8.3 GELU (`gelu_other.cuh` + 문서)

다항 평가·HMult·BSGS 최적화는 [gelu_hmult_bsgs.md](gelu_hmult_bsgs.md) 참고.

### 8.4 기타 (시뮬레이터 확장 시)

`include.cuh`에 **causal masked softmax**, **SiLU**, **RMSNorm**, **Chebyshev** 등이 포함되어 있어, 동일 Phantom API 위에 **모듈 라이브러리**를 확장할 수 있다.

---

## 9. 병렬성 모델

1. **CUDA (Phantom)**  
   대부분의 homomorphic 연산이 GPU에서 실행. `stream_pool` + `deep_copy_cipher`로 **OMP 스레드별 CUDA stream**을 쓰는 패턴이 `softmax.cuh` 등에 존재한다.

2. **OpenMP**  
   `softmax`의 exp 계산 루프 등에 `#pragma omp parallel` 사용. `MOAI_SINGLE_LAYER_OMP_THREADS`로 스레드 수를 제한하는 코드가 `single_layer_test`에 있다.

3. **헤드 루프**  
   `single_layer_test`의 12헤드는 **현재 for 루프 순차** 호출이다. 가속기 시뮬레이터에서 **헤드 간 독립 실행**을 가정하면 이상적인 상한(optimistic)을, **순차**를 그대로 두면 현재 코드와 일치하는 하한(sequential)을 얻는다.

---

## 10. 시뮬레이터 설계 체크리스트 (MOAI 맞춤)

1. **입력**: `N`, `logN`, 모듈러스 비트 벡터 길이 `T`, `MOAI_ALPHA`, `pre_att_drops`, 헤드 수·`col_W`.
2. **연산 테이블**: Phantom `Evaluator` 연산별 **“multiplicative depth 소비”**와 **relin 횟수**를 이상화(문서화된 수식이 없으면 **프로파일링으로 역산**).
3. **부트스트랩 모델**: `bootstrap_3`를 **고정 사이클 블랙박스**로 두고, 입력·출력 **체인 인덱스**만 맞추는 1단계와, `Bootstrapper.cu` 내부를 펼친 **미세 모델** 2단계를 분리.
4. **메모리**: ct 크기 × 최대 동시 live ct (GELU 주변 임시 ct가 많다는 주석 참고) + 키·rotation 데이터.
5. **검증**: `results.csv`, `single_layer_results.csv`의 구간별 시간과 시뮬레이터 사이클 합을 비교.

---

## 11. 한계·주의 (분석 문서로서 명시)

- MOAI_GPU는 **프로덕션 추론 프레임워크**라기보다 **연구용 단일 바이너리 + 헤더 중심** 구조이다.
- 일부 블록(예: `single_att_block`의 softmax 출력과 `ct_ct_matrix_mul_diagpacking` 입력)은 **주석과 실제 데이터 플로우를 함께 읽어야** 논문식 Attention과 정확히 일치하는지 판단해야 한다.
- 최종 정확도는 **CKKS 스케일·노이즈·근사 다항식**에 의해 결정되므로, 사이클-only 시뮬레이터와 별도로 **오차 시뮬레이터**가 필요할 수 있다.

---

## 12. 마이크로 `ct_pt` 커널 비중 (Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=ct_pt` → [`../src/test.cu`](../src/test.cu)에서 [`ct_pt_matrix_mul_test()`](../src/include/test/matrix_mul/test_ct_pt_matrix_mul.cuh) 실행.
- **NVTX 구간**: CMake에서 `CUDA::nvtx3`를 링크할 수 있으면 `MOAI_HAVE_NVTX`가 켜지고, `ct_pt_matrix_mul_wo_pre(...)` 호출이 **`moai:ct_pt_matrix_mul_wo_pre`** 범위로 표시된다. `nsys stats --filter-nvtx moai:ct_pt_matrix_mul_wo_pre`로 **곱셈만** 커널 합을 뽑을 수 있다(범위 밖 커널은 전체 리포트로).
- **스크립트**: [`../src/scripts/profile_ct_pt_micro.sh`](../src/scripts/profile_ct_pt_micro.sh) — `nsys profile` + `nsys stats --report cuda_gpu_kern_sum` TSV 저장, 인덱스 TSV에 요약 행 기록. 기본 산출 디렉터리는 **`output/ct_pt/`** (`MOAI_PROFILE_DIR`로 변경 가능). `MOAI_KERN_EXPORT=1`(기본)이면 [`export_ct_pt_kern_sum.py`](../src/scripts/export_ct_pt_kern_sum.py)로 같은 폴더에 `*_clean.tsv`·막대/파이 차트 PNG를 추가한다. `MOAI_NSYS_STATS_FILTER_NVTX=0`이면 NVTX 필터 통계 생략.
- **NTT 휴리스틱 파서**: [`../src/scripts/parse_nsys_cuda_kern_sum.py`](../src/scripts/parse_nsys_cuda_kern_sum.py) — 위 TSV에서 커널 이름이 `fnwt|inwt|radix|nwt_2d` 등에 매칭되는 행의 **Total Time** 합 / 전체 합.
- **주의**: 이 마이크로의 CKKS 파라미터는 `single_layer_test`와 다르다(문서 앞부분 및 스크립트 주석 참고). **커널 믹스 경향**용으로 쓰고, forward와 동일한 절대 비율로 읽지 말 것.

예시:

```bash
CONDA_PREFIX="$CONDA_PREFIX" cmake -S MOAI_GPU -B MOAI_GPU/build
cmake --build MOAI_GPU/build --target test
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_ct_pt_micro.sh
nsys-ui MOAI_GPU/output/ct_pt/ct_pt_micro_run1.nsys-rep
```

**사전 인코딩 가중치 경로** (`ct_pt_matrix_mul(enc_X, ecd_w, …)`, 프로덕션에 가까움):

- **진입**: `MOAI_BENCH_MODE=ct_pt_pre` → [`ct_pt_matrix_mul_w_preprocess_test()`](../src/include/test/matrix_mul/test_ct_pt_matrix_mul.cuh).
- **NVTX** (`MOAI_HAVE_NVTX`): **`moai:ct_pt_pre_encode_w`** — 스칼라 `W[i][j]`마다 `encoder.encode` 하는 **사전 인코드 루프**(CKKS 슬롯 인코드 → **IFFT 비중 큼**). **`moai:ct_pt_matrix_mul_pre_encoded`** — 그 다음 **`ct_pt_matrix_mul`만**(내부는 `multiply_plain` 등, **새 IFFT 없음**).
- **스크립트**: [`../src/scripts/profile_ct_pt_pre_micro.sh`](../src/scripts/profile_ct_pt_pre_micro.sh) — 기본 **`output/ct_pt_pre/`**, 필터 TSV **`*.kern_sum.nvtx_encode_w.tsv`**, **`*.kern_sum.pre_enc_nvtx.tsv`**. 전체 `kern_sum` 차트는 **X 배치 인코드 + W 인코드 + 곱** 이 합쳐져 IFFT가 상위에 오는 것이 정상이다 → [`../output/ct_pt_pre/ct_pt_pre_micro_kernel_reference.md`](../output/ct_pt_pre/ct_pt_pre_micro_kernel_reference.md).
- **메모리**: 전체 `768×64` 가중치를 모두 `encode`하면 GPU 메모리가 부족한 환경이 많다. 스크립트는 기본 **`MOAI_CT_PT_PRE_MICRO=1`** 로 문제 크기를 줄인다(테스트가 `[MOAI_CT_PT_PRE_MICRO]` 로그 출력). 원래 크기는 `MOAI_CT_PT_PRE_MICRO=0` 과 함께 수동 실행.

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_ct_pt_pre_micro.sh
nsys-ui MOAI_GPU/output/ct_pt_pre/ct_pt_pre_micro_run1.nsys-rep
```

---

## 13. 마이크로 `ct_ct` 커널 비중 (Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=ct_ct` → [`../src/test.cu`](../src/test.cu)에서 [`ct_ct_matrix_mul_test()`](../src/include/test/matrix_mul/test_ct_ct_matrix_mul.cuh) 실행.
- **NVTX 구간** (`MOAI_HAVE_NVTX` 빌드 시): **`moai:ct_ct_matrix_mul_colpacking`**, **`moai:ct_ct_matrix_mul_diagpacking`** — 각각 열 패킹 / 대각 패킹 Ct×Ct 한 단계.
- **스크립트**: [`../src/scripts/profile_ct_ct_micro.sh`](../src/scripts/profile_ct_ct_micro.sh) — 한 번의 `nsys profile` 후 전체 `kern_sum.tsv`와, 위 두 NVTX 이름으로 필터한 **`*.kern_sum.nvtx_col.tsv`**, **`*.kern_sum.nvtx_diag.tsv`** 를 생성. 기본 산출 디렉터리는 **`output/ct_ct/`** (`MOAI_PROFILE_DIR`로 변경 가능). `MOAI_KERN_EXPORT=1`(기본)이면 같은 폴더에 `*_clean.tsv`·차트 PNG 추가.
- **커널 참고(정적)**: [`../output/ct_ct/ct_ct_micro_kernel_reference.md`](../output/ct_ct/ct_ct_micro_kernel_reference.md) — Ct–Ct가 Ct–Pt와 다르게 keyswitch·Galois·relin 쪽 커널을 많이 쓸 수 있다는 요약.
- **NTT 휴리스틱 파서**: [`../src/scripts/parse_nsys_cuda_kern_sum.py`](../src/scripts/parse_nsys_cuda_kern_sum.py) — §12와 동일.

예시:

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_ct_ct_micro.sh
nsys-ui MOAI_GPU/output/ct_ct/ct_ct_micro_run1.nsys-rep
```

---

## 14. 마이크로 Softmax (Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=softmax_micro` → [`../src/test.cu`](../src/test.cu)에서 [`softmax_micro_bench()`](../src/include/test/non_linear_func/test_softmax.cuh) (`softmax_test` 후 `softmax_boot_test`). 단독으로는 `MOAI_BENCH_MODE=softmax` / `softmax_boot` 도 지원한다.
- **NVTX 구간** (`MOAI_HAVE_NVTX` 빌드 시): **`moai:softmax_without_boot`**, **`moai:softmax_boot`** — 각각 `softmax(...)` / `softmax_boot(...)` 호출 구간.
- **스크립트**: [`../src/scripts/profile_softmax_micro.sh`](../src/scripts/profile_softmax_micro.sh) — 한 번의 `nsys profile` 후 전체 `kern_sum.tsv`와, 위 두 NVTX로 필터한 **`*.kern_sum.nvtx_no_boot.tsv`**, **`*.kern_sum.nvtx_boot.tsv`**. 기본 산출 디렉터리는 **`output/softmax/`**. `MOAI_KERN_EXPORT=1`(기본)이면 `export_ct_pt_kern_sum.py`로 `*_clean.tsv`·PNG 추가.
- **참고**: [`../output/softmax/softmax_micro_kernel_reference.md`](../output/softmax/softmax_micro_kernel_reference.md)

예시:

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_softmax_micro.sh
nsys-ui MOAI_GPU/output/softmax/softmax_micro_run1.nsys-rep
```

---

## 15. 마이크로 GeLU (`gelu_v2`, Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=gelu` → [`../src/test.cu`](../src/test.cu)에서 [`gelu_test()`](../src/include/test/non_linear_func/test_gelu.cuh).
- **NVTX** (`MOAI_HAVE_NVTX`): **`moai:gelu_v2_batch`** — OpenMP 배치 루프 안에서 `gelu_v2(...)` 호출 구간 (`test_gelu.cuh`).
- **스크립트**: [`../src/scripts/profile_gelu_micro.sh`](../src/scripts/profile_gelu_micro.sh) — 전체 `kern_sum.tsv` + `--filter-nvtx moai:gelu_v2_batch` → `*.kern_sum.nvtx.tsv` → export로 `*_clean.tsv`·PNG. 기본 **`output/gelu/`**.
- **참고**: [`../output/gelu/gelu_micro_kernel_reference.md`](../output/gelu/gelu_micro_kernel_reference.md)
- **공통 절차**: [`micro_profile_playbook.md`](micro_profile_playbook.md)

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_gelu_micro.sh
nsys-ui MOAI_GPU/output/gelu/gelu_micro_run1.nsys-rep
```

---

## 16. 마이크로 LayerNorm (Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=layernorm` → [`layernorm_test()`](../src/include/test/non_linear_func/test_layernorm.cuh).
- **NVTX**: **`moai:layernorm`** — `layernorm(...)` 호출 전후 (`test_layernorm.cuh`).
- **스크립트**: [`../src/scripts/profile_layernorm_micro.sh`](../src/scripts/profile_layernorm_micro.sh) — 동일 패턴, 기본 **`output/layernorm/`**.
- **참고**: [`../output/layernorm/layernorm_micro_kernel_reference.md`](../output/layernorm/layernorm_micro_kernel_reference.md)

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_layernorm_micro.sh
nsys-ui MOAI_GPU/output/layernorm/layernorm_micro_run1.nsys-rep
```

---

## 17. 새 마이크로 벤치 공통 플레이북

연산을 추가할 때 **NVTX → `profile_*_micro.sh` → `kern_sum` TSV → export 차트 → `output/<op>/` 커널 분석 MD → 본 문서에 짧은 절** 순서를 따르면 된다. 상세 체크리스트는 [`micro_profile_playbook.md`](micro_profile_playbook.md) 를 본문으로 삼는다.

---

## 18. 마이크로 Bootstrapping (`bootstrap_3`, Nsight Systems)

- **진입**: `MOAI_BENCH_MODE=bootstrap_micro` 또는 **`boot`** → [`bootstrapping_test()`](../src/include/test/bootstrapping/bootstrapping.cuh) (동일 테스트).
- **NVTX** (`MOAI_HAVE_NVTX`): **`moai:bootstrap_prepare`** (`prepare_mod_polynomial` ~ `generate_LT_coefficient_3`), **`moai:bootstrap_3`** (`Bootstrapper::bootstrap_3` 한 번).
- **스크립트**: [`../src/scripts/profile_bootstrap_micro.sh`](../src/scripts/profile_bootstrap_micro.sh) — 전체 `kern_sum.tsv` + 위 두 NVTX로 필터한 **`*.kern_sum.nvtx_prepare.tsv`**, **`*.kern_sum.nvtx_bootstrap3.tsv`**. 기본 **`output/bootstrap/`**. `MOAI_KERN_EXPORT=1`(기본)이면 `*_clean.tsv`·PNG.
- **참고**: [`../output/bootstrap/bootstrap_micro_kernel_reference.md`](../output/bootstrap/bootstrap_micro_kernel_reference.md)

```bash
MOAI_BUILD_DIR=MOAI_GPU/build ./MOAI_GPU/src/scripts/profile_bootstrap_micro.sh
nsys-ui MOAI_GPU/output/bootstrap/bootstrap_micro_run1.nsys-rep
```

---

*작성: 저장소 정적 리딩 기준 (`test.cu`, `test_single_layer.cuh`, `single_att_block.cuh`, `include.cuh`, `CMakeLists.txt`, 기존 `doc/*.md`). 실행 프로파일 수치는 환경·GPU에 의존하므로 본 문서에 고정값으로 적지 않았다.*
