# MOAI_GPU 시뮬레이터 가이드

MOAI_GPU에는 **Phantom CKKS 호출을 대체하지 않는** 두 층의 런타임 추정/스케줄 모델이 있다.

| 층 | 코드 위치 | 역할 |
|----|-------------|------|
| **SimTiming** (coarse estimator) | `src/include/source/sim/sim_timing.h` | 연산 종류별 대략적인 cycles·bytes 누적 (`MOAI_SIM_BACKEND=1`일 때 `record_*`가 쌓임). |
| **EngineModel** (dependency-aware scheduler) | `src/include/source/sim/engine_model.h` | DMA / NTT / VEC(및 on-chip 경로)를 **의존성 있는 이벤트 스트림**으로 스케줄. `enqueue_multiply_plain`, `enqueue_ct_ct_multiply`, `enqueue_rotate` 등 **primitive 단위** API. |

둘 다 **정확한 FHE 정합성을 보장하지 않으며**, 가속기 설계·용량 산정·병목 감각을 잡기 위한 **조악한 프록시**다. 설계 배경·워크로드 맥락은 [moai_accelerator_simulator_analysis.md](moai_accelerator_simulator_analysis.md)를 참고한다.

---

## 1. 빌드와 실행 전제

### 1.1 `engine_config.h` 등을 고쳤을 때

`src/include/source/sim/engine_config.h`는 **`engine_model.h`·`sim_timing.h`·여러 `.cu` / `.cuh`** 에서 include 된다. **헤더 내용을 바꾸면**(기본 상수, `from_env()` 파싱, `estimate_*` 함수 등) 그 헤더를 끌어가는 번역 단위가 다시 컴파일되므로 **`test`(및 같은 그래프에 묶인 타깃)를 다시 빌드해야 한다.**

반대로, 이미 코드에 연결해 둔 **환경 변수만 바꿔서 튜닝**하는 경우(`MOAI_SIM_NTT_LANES` 등)는 **재빌드 없이** 바이너리를 다시 실행하면 된다.

### 1.2 빌드 방법

`MOAI_GPU` 저장소 루트에서:

```bash
# 최초 또는 CMake 옵션을 바꾼 뒤
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

이미 `build/` 디렉터리가 있고 `CMakeCache.txt`만 유효하면, 소스·헤더 수정 후에는 아래만으로 충분하다.

```bash
cmake --build build -j"$(nproc)"
```

`test` 실행 파일만 빌드하고 싶다면(환경에 따라 타깃 이름 확인):

```bash
cmake --build build -j"$(nproc)" --target test
```

### 1.3 실행 관습

- 시뮬 관련 로그는 **프로세스 cwd** 기준 경로가 많다. 관습적으로 **`MOAI_GPU` 루트에서 `./build/test`** 를 실행한다.
- 기본 리포트 append 경로: **`output/sim/moai_sim_report.txt`** (`default_sim_report_path()`).
- **`test` 표준 출력:** Linux에서 기본으로 **`output/test_logs/<MOAI_BENCH_MODE>.txt`** 에 `stdout`을 넘긴다(`MOAI_BENCH_MODE` 없으면 `default.txt`). 경로는 **`MOAI_TEST_OUTPUT_PATH`** 로 덮어쓰기, **`MOAI_TEST_OUTPUT_DISABLE=1`** 로 끄기, **`MOAI_TEST_OUTPUT_APPEND=1`** 로 이어 붙이기. 실제 저장 경로는 **`stderr`** 한 줄로 알려 준다.
- 빌드 구조·CUDA 아키텍처 등은 저장소 루트의 **`CMakeLists.txt`** 를 본다.

---

## 2. 켜고 끄기

| 환경 변수 | 의미 |
|-----------|------|
| **`MOAI_SIM_BACKEND=1`** | SimTiming 활성. Evaluator 등에서 `record_*` / gap 정책이 동작한다. `0`이면 비활성. |
| **`MOAI_SIM_ENGINE_MODEL`** | `1`(기본, `MOAI_SIM_BACKEND=1`일 때) 또는 생략 시 EngineModel 스케줄링 켜짐. **`0`** 이면 EngineModel만 끔(SimTiming만 쌓일 수 있음). |
| **`MOAI_SIM_STRICT=1`** | 아직 모델에 없는 Phantom 경로가 호출되면(coverage gap) **abort**할 수 있음. |
| **`MOAI_SIM_GAP_POLICY`** | `skip`(기본) / `error` / `model`. Evaluator에서 아직 coarse 모델이 없는 op는 gap 카운터를 올린다. `model`이면 일부 op가 EngineModel에 매핑된다(코드 경로별 상이). |

---

## 3. 리포트 출력

| 환경 변수 | 의미 |
|-----------|------|
| **`MOAI_SIM_REPORT_PATH`** | **대부분의 GPU 시뮬 벤치**(예: `ct_pt`, `ct_ct`): SimTiming·EngineModel 요약을 **append**할 파일. 미설정 시 `output/sim/moai_sim_report.txt`. **`sim_*` primitive 마이크로 벤치**(`MOAI_BENCH_MODE=sim_primitives` 등)는 **primitive 태그마다 별도 파일**을 쓴다(`trunc`). 규칙은 `test/sim/test_sim_primitives.cuh` 의 `primitive_sim_report_path`: 미설정이면 `output/sim/primitive_<tag>.txt`; 값이 디렉터리로 끝나면 그 아래 `primitive_<tag>.txt`; 그 외는 파일 경로의 **stem**에 `_<tag>.txt`를 붙인 경로(예: `.../moai_sim_report.txt` → `.../moai_sim_report_mul_plain.txt`). |
| **`MOAI_SIM_REPORT_QUIET`** | 설정되어 있고 값이 `0`이 **아니면**, 요약을 콘솔에 한 번 더 찍지 않는 등 조용한 모드(각 벤치 구현 참고). |

로그 태그:

- **`[MOAI_SIM_BACKEND]`** — SimTiming `print_summary()` (op별 calls / cycles / bytes).
- **`[MOAI_SIM_ENGINE]`** — EngineModel `print_summary()` (makespan, DMA/NTT/VEC busy, tier별 바이트, `bound_hint` 등). 엔진 표에 **`coeff_ops`**(NTT/VEC만: 각 `enqueue_*` 스텝에 넘긴 **계수 원소 개수** `coeffs ≈ N×T`)와 **`op/s@busy` / `op/s@span`**(그 원소를 busy·makespan으로 나눈 처리율)가 붙는다. DMA·on-chip·RF xfer 행은 `coeff_ops=0`, `op/s`는 `-`. **NTT/VEC** 행은 바이트 카운터가 0일 때 **`GB/s@busy` / `GB/s@span`**에 **`8B×coeff_ops`** 등가 처리량을 쓴다(`op/s`와 동일 스케일). 그 위에 **`steady_peak_coeff_equiv_GB/s`** 한 줄로 설정 기준 이론 피크(NTT·vec_mul·vec_add 각 steady, fill/오버헤드 제외)와 **`ntt_rf` / `vec_rf`의 cfg eff GB/s**(대역×뱅크)를 같이 찍는다.

---

## 4. EngineModel / On-chip 관련 주요 env

시간축은 우선순위 **`MOAI_SIM_ENGINE_MHZ` > `MOAI_SIM_CYCLE_PERIOD_NS` > `MOAI_SIM_CYCLE_NS`** 로 해석된다(`EngineModelConfig::from_env`).

### 4.1 외부 메모리(DMA)

| 변수 | 설명 |
|------|------|
| `MOAI_SIM_EXT_MEM_KIND` | 로그용 라벨 |
| `MOAI_SIM_EXT_MEM_BW_GBPS` | 대역폭 (GB/s); 미설정 시 레거시 `MOAI_SIM_DMA_BW_GBPS` 등 |
| `MOAI_SIM_EXT_MEM_LATENCY_NS` / `MOAI_SIM_EXT_MEM_LATENCY_CYC` | 트랜잭션 고정 지연 |

### 4.2 NTT / VEC

| 변수 | 설명 |
|------|------|
| `MOAI_SIM_NTT_LANES`, `MOAI_SIM_NTT_PIPE_DEPTH_CYC`, `MOAI_SIM_NTT_STEADY_CYC_PER_COEFF` (별칭 `MOAI_SIM_NTT_CYC_PER_COEFF`) | NTT 파이프라인·레인 모델 |
| `MOAI_SIM_VEC_LANES`, `MOAI_SIM_VEC_ADD_CYC_PER_COEFF` | 벡터 덧셈 등 |
| `MOAI_SIM_VEC_MUL_STEADY_CYC_PER_COEFF` (별칭 `MOAI_SIM_VEC_MUL_CYC_PER_COEFF`, `MOAI_SIM_VEC_CYC_PER_COEFF`) | 벡터 곱 steady |
| `MOAI_SIM_NTT_PASS_OVERHEAD_CYC`, `MOAI_SIM_VEC_PASS_OVERHEAD_CYC` | 패스 오버헤드 |
| `MOAI_SIM_RESCALE_CYC_PER_COEFF`, `MOAI_SIM_MODSWITCH_CYC_PER_COEFF` | rescale / modswitch vec 단계 |

`vec_lanes` / `ntt_lanes` 를 헤더 기본값으로 쓸 때: **\(2^{16}=65536\)** 은 `1ull << 16` 이다. **`2 << 16` 은 131072**라서 `⌈(N·T)/\text{lanes}⌉` 가 **절반**으로 줄어든다(“0.5가 곱해진 것처럼” 보임).

### 4.3 CT×CT·키스위치·회전

| 변수 | 설명 |
|------|------|
| `MOAI_SIM_CT_CT_VEC_MUL_PASSES` | `enqueue_ct_ct_multiply` 안 vec_mul 반복 횟수(조악한 RNS/텐서 프록시). 기본 3. |
| `MOAI_SIM_KSWITCH_SIZE_P` | QlP 확장 시 **\|P\|**(특수 소수 개수). `size_QlP = \|Ql\| + \|P\|` (β 산정과 별개). |
| `MOAI_SIM_KSWITCH_BETA` | 수동 β. **`0`**이면 `MOAI_SIM_KSWITCH_BETA_MODE`에 따라 자동: **`phantom`(기본)** → `β = ⌈\|Ql\|/α⌉` (`MOAI_SIM_ALPHA`); **`legacy`** → `β = ⌈\|Ql\|/\|P\|⌉`(구버전). |
| `MOAI_SIM_KSWITCH_MODUP_BCONV_CYC_PER_COEFF`, `MOAI_SIM_KSWITCH_MODDOWN_BCONV_CYC_PER_COEFF` | Phantom CKKS keyswitch coarse 모델의 BConv 단계 사이클 |
| `MOAI_SIM_GALOIS_PERM_CYC_PER_COEFF` | Galois(회전) 전 perm 단계 |
| `MOAI_SIM_GALOIS_KEY_BYTES`, `MOAI_SIM_RELIN_KEY_BYTES` | 키 DMA 바이트. **미설정 시** `MOAI_SIM_POLY_DEGREE`·`MOAI_SIM_NUM_LIMBS`·`MOAI_SIM_ALPHA`(기본 1)로 `dnum=(T-\alpha)/\alpha` 를 두고 **`dnum·2·T·N·8`** ( `keys_dnum_35` / `compute_key_bytes.py` 와 동일)을 relin·Galois 각각에 사용. **`0`으로 명시**하면 키 트래픽 끔. |
| `MOAI_SIM_ALPHA` | 하이브리드 special modulus 크기(Phantom `MOAI_ALPHA` 와 맞춤). 기본 `1`. `T % alpha == 0` 등이 안 맞으면 자동 키 바이트는 0. |

### 4.4 On-chip (GlobalSPAD / RF)

`OnChipConfig::from_env()` — **`MOAI_SIM_GSPAD_BYTES=0`** 이면 on-chip 계층이 꺼지고, 예전 스타일의 단순 DMA+엔진 경로에 가깝게 동작한다.

| 변수 | 설명 |
|------|------|
| `MOAI_SIM_GSPAD_BYTES`, `MOAI_SIM_GSPAD_BW_GBPS`, `MOAI_SIM_GSPAD_BANKS`, `MOAI_SIM_GSPAD_BASE_CYC` | Global SPAD |
| `MOAI_SIM_NTT_RF_BYTES`, `MOAI_SIM_NTT_RF_BW_GBPS`, `MOAI_SIM_NTT_RF_BANKS` | NTT RF |
| `MOAI_SIM_VEC_RF_BYTES`, `MOAI_SIM_VEC_RF_BW_GBPS`, `MOAI_SIM_VEC_RF_BANKS` | Vec RF |

---

## 5. SimTiming 전용(레거시 스칼라) env

`SimTiming` 내부의 일부 op는 여전히 아래 **고정 스칼라**를 쓴다(EngineModel과 완전 동기는 아닐 수 있음).

| 변수 | 용도 |
|------|------|
| `MOAI_SIM_ENC_CYC`, `MOAI_SIM_ENC_VEC_CYC_PER_SLOT` | encode |
| `MOAI_SIM_D2D_CYC_PER_B` | deep copy |
| `MOAI_SIM_RESCALE_CYC_PER_COEFF` | rescale (SimTiming 쪽; Engine은 `EngineModelConfig` 키 사용) |

NTT/vec 곱 등은 최근 경로에서 **`estimate_*` + `EngineModelConfig::from_env()`** 와 맞추는 코드가 많다(`engine_config.h` 주석 참고).

---

## 6. CKKS 파라미터 (N·체인 길이)와 시뮬

시뮬레이터는 **비트 정확한 `parms.set_coeff_modulus({…})` 표현을 따로 들고 있지 않는다.** 대신 연산량·바이트 스케일을 잡기 위해 **다항식 차수 \(N\)** 과 **RNS limb 개수(Phantom의 `coeff_modulus_size()`에 해당하는 정수)** 를 쓴다.

### 6.1 Evaluator를 타는 경로 (`MOAI_SIM_BACKEND=1` + 실제 GPU 벤치)

`ckks_evaluator_parallel.cuh` 등에서 Sim/Engine 호출 시 **암호문 객체에서 그대로 읽는다.**

| 값 | Phantom / MOAI 쪽 소스 | 시뮬에서의 의미 |
|----|-------------------------|-----------------|
| **\(N\)** | `ct.poly_modulus_degree()` | 스케일링 차원(슬롯 수와 연관된 정책은 테스트·Phantom 설정 따름). |
| **체인 길이(림bs)** | `ct.coeff_modulus_size()` | 현재 암호문이 올라가 있는 **모듈러스 체인 길이**(레벨에 따라 줄어든 값이 들어옴). |

즉 **레벨을 바꾸려면** 테스트 안의 `EncryptionParameters` / `CoeffModulus::Create` / `mod_switch` 횟수 등 **실제 CKKS 파라미터와 데이터플로를 바꿔야** 하고, 시뮬 전용 env만으로 Phantom의 체인 비트열을 바꾸지는 않는다.

### 6.2 Estimator-only primitive (`sim_*`)

`test_sim_primitives.cuh` 등에서는 Phantom을 돌리지 않으므로, **환경 변수로 조악한 \(N, T\)를 넣는다.**

| 환경 변수 | 기본 | 의미 |
|-----------|------|------|
| `MOAI_SIM_POLY_DEGREE` | 65536 | \(N\) (poly modulus degree). 기본값은 `sim_ckks_defaults.h`의 `kSingleLayerPolyModulusDegree()` (= `single_layer_test` 의 `logN=16`). |
| `MOAI_SIM_NUM_LIMBS` | 36 | 기본은 **\|QP\|**(체인 전체 소수 개수, `kSingleLayerCoeffModulusCount()`). primitive·엔진은 **`MOAI_SIM_NUM_LIMBS_COUNTS_QP=1`(기본)** 일 때 암호문 **\|Ql\| = \|QP\| − `MOAI_SIM_ALPHA`** 로 변환(Phantom: α개가 특수 P, 나머지가 Q). |
| `MOAI_SIM_NUM_LIMBS_COUNTS_QP` | `1` | `0`이면 `MOAI_SIM_NUM_LIMBS`를 이미 **\|Ql\|**로 본다(레거시). |

개별 모듈러스 **비트 폭(예: 40비트×14 + 60비트)** 은 시뮬에 직접 들어가지 않는다. 바이트·사이클은 대략 **계수 개수 \(\propto N \times \|Ql\|\)** 로만 반영된다.

### 6.3 `ct_pt` / `ct_ct` estimator 블록

각 테스트 헤더에 **고정된 `N`, `T`** 가 박혀 있는 경우가 많다(예: colpacking estimator의 `65536`, `15`). primitive `sim_*` 기본 \(N,T\)는 **`sim_ckks_defaults.h`** 로 single_layer와 공유한다. 다른 벤치 전용 상수를 바꾸려면 **해당 `.cuh`를 수정**하고 재빌드한다.

---

## 7. 무엇을 돌릴지: `MOAI_BENCH_MODE`

`src/test.cu`에서 `MOAI_BENCH_MODE`로 분기한다. **시뮬만** 쓰려면 아래처럼 `MOAI_SIM_BACKEND=1`을 함께 주는 패턴이 일반적이다.

### 7.1 매트릭스·애플리케이션 벤치 (일부는 GPU 스킵)

| `MOAI_BENCH_MODE` | 동작 요약 |
|-------------------|-----------|
| `ct_pt` | Ct–Pt matmul; Sim 켜면 **estimator 경로로 GPU 생략** 가능. |
| `ct_pt_pre` | 사전 인코드 W 후 matmul; Sim 시 GPU 생략 가능. |
| `ct_ct` | Ct–Ct colpacking 등; Sim 켜면 **colpacking estimator**로 GPU 생략 가능. |
| `boot` / `bootstrap_micro` | 부트스트랩 마이크로 (GPU 실행). |
| `gelu`, `layernorm`, `softmax*` 등 | 해당 테스트 (Nsight 마이크로와 연계 시 [micro_profile_operations_index.md](micro_profile_operations_index.md)). |

`ct_ct` + Sim일 때 엔진 쪽 태그 예: **`ct_ct_colpacking_est`** (`EngineModel::print_summary` 두 번째 인자).

### 7.2 Primitive 단독 마이크로 (항상 estimator-only)

`src/include/test/sim/test_sim_primitives.cuh` — **`EngineModel::enqueue_*`** 를 **한 종류씩**(또는 `all` 순차) 호출한다.

**필수:** `MOAI_SIM_BACKEND=1` (아니면 종료 코드 2).

| 진입 | 설명 |
|------|------|
| `MOAI_BENCH_MODE=sim_primitive` + `MOAI_SIM_PRIMITIVE` | `mul_plain`, `mul_ct`, `add_inplace`, `rescale`, `rotate`, `relin`, `modswitch`, `all`(미설정 시 `all`). 별칭: `ct_pt`→`mul_plain`, `ct_ct`→`mul_ct`, `add`→`add_inplace`, `relinearize`→`relin`. |
| `sim_primitives` | 위의 `all` 과 동일. |
| `sim_mul_plain`, `sim_mul_ct`, `sim_add_inplace`, `sim_rescale`, `sim_rotate`, `sim_relin`, `sim_modswitch` | 단일 primitive 단축 모드. |

차원(조악 파라미터):

| 변수 | 기본 | 의미 |
|------|------|------|
| `MOAI_SIM_POLY_DEGREE` | 65536 | \(N\) (`sim_ckks_defaults.h` / single_layer) |
| `MOAI_SIM_NUM_LIMBS` | 36 | 기본 **\|QP\|**; 실제 primitive에 쓰는 \|Ql\|는 `MOAI_SIM_NUM_LIMBS_COUNTS_QP`·`MOAI_SIM_ALPHA`로 결정 |
| `MOAI_SIM_ALPHA` | 1 | 하이브리드 digit / 특수 소수 개수(Phantom `special_modulus_size`와 맞춤). |
| `MOAI_SIM_NUM_LIMBS_COUNTS_QP` | `1` | `0`이면 `MOAI_SIM_NUM_LIMBS` = \|Ql\| 직접. |
| `MOAI_SIM_PRIMITIVE_LOOPS` | 1 | 같은 primitive 반복 횟수 |

**주의:** `rotate` / `relin` / `modswitch`는 현재 **SimTiming 표에 전용 행이 없을 수 있어** 대부분의 coarse 카운터는 0에 가깝고, **`[MOAI_SIM_ENGINE]`** 블록이 실질적인 트래픽·makespan을 본다.

### 7.3 하이브리드 키스위칭 연산량 프로파일 (GPU 없음)

`MOAI_BENCH_MODE=sim_hybrid_ks_profile` — `keyswitch_op_profile.h`의 **`enqueue_keyswitch_phantom_ckks`와 동일 구조**로, α·\|Ql\|·β·dnum에 따른 **NTT/BConv/vec_mul 호출 수(및 가중 coeff 합)** 를 CSV로 내보낸다. 보안 λ는 없다. `plot_hybrid_ks_profile.py` 기본 **`--metric kernels`**이면 BConv(=modup+down)가 α와 함께 감소하는 추세를 보기 쉽다.

- **해석(analytic) 열** (`ntt_kernels`, `bconv_*`, `vec_*`, `weighted_*`): `enqueue_keyswitch_phantom_ckks` 그래프를 **공식으로 복제**한 값.
- **엔진 열** (`eng_ntt_enq`, `eng_vec_enq`): `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1` 이고 **`MOAI_SIM_BACKEND=1`** 일 때 **`profile_keyswitch_phantom_ckks`** 한 번 실행 후 **`schedule()` 호출 횟수**(`enqueue_calls`). 해석 `ntt_kernels` 등과 맞춰 검산 가능. 측정 불가면 `-1`.
- **엔진 coeff 합산** (`eng_ntt_coeff_ops`, `eng_vec_coeff_ops`): 위와 동일 실행에서 **`EngineStats::logical_ops`** — 각 패스에 넘긴 coeff 개수(\(N\times\)limbs)의 **합**. “모든 limb를 합산한” 시뮬레이터 측 작업량에 해당. 해석 `weighted_*`와는 정의가 다르니 비교 시 용도를 구분한다.
- **메모리(바이트 추정)** (`mem_c2_bytes`, `mem_modup_buf_bytes`, `mem_cx_buf_bytes`, `mem_working_peak_bytes_est`): Phantom `eval_key_switch.cu` `keyswitch_inplace` 의 **`c2`**, **`t_mod_up`**(\(\beta\times\)|QlP|), **`cx`**(2×|QlP|) u64 버퍼 크기와, 동시 상주 가정 시 **합(`mem_working_peak_bytes_est`)**. `plot_hybrid_ks_profile.py --memory` 또는 `--all-plots` 로 `hybrid_ks_profile_memory.png` 생성.
- **시뮬 사이클** (`eng_makespan_cyc`, `eng_ntt_busy_cyc`, `eng_ntt_fwd_busy_cyc`, `eng_ntt_inv_busy_cyc`, `eng_vec_busy_cyc`): 같은 `profile_keyswitch_phantom_ckks` 실행의 **`Summary::makespan_cycles`** 및 **`EngineStats::busy_cycles`**. 구조적 개수 대신 **모델 사이클**로 α 스윕 비교할 때 사용. `plot_hybrid_ks_profile.py --cycles-plots` → `hybrid_ks_profile_stacked_cycles.png`, `hybrid_ks_profile_engine_cycles.png`. 스택 합은 엔진별 일한 양의 합이며 **벽시계 makespan과 다를 수 있음**(겹침·크리티컬 패스).

| 변수 | 기본 | 의미 |
|------|------|------|
| `MOAI_SIM_POLY_DEGREE`, `MOAI_SIM_NUM_LIMBS` | single_layer | \(N\), **\|QP\|** (`t_qp` 열; dnum·키 바이트 추정). |
| `MOAI_SIM_KSWITCH_SIZE_Ql` | `0` → 각 행 **\|QP\|−α** | Phantom top-level **\|Q\|** 프록시. |
| `MOAI_SIM_KSWITCH_SIZE_P` | **미설정 시 각 행 α** | QlP의 \|P\|; 명시 시 모든 행 동일. |
| `MOAI_SIM_HYBRID_KS_ALPHA_LIST` | (비어 있으면 RANGE 또는 `MOAI_SIM_ALPHA`) | 쉼표로 구분된 α 목록 (예: `1,2,4,8`). 비어 있으면 RANGE 사용. |
| `MOAI_SIM_HYBRID_KS_ALPHA_RANGE` | (미설정) | LIST가 비어 있을 때만 사용. **양 끝 포함** 범위 문자열 (예: `1-35` — α=35일 때 \|Q\|=1이면 β=1). |
| `MOAI_SIM_HYBRID_KS_MEASURE_ENGINE` | `0` | `1`이면 CSV에 `eng_ntt_enq`, `eng_vec_enq`, `eng_*_coeff_ops` 기록(**`MOAI_SIM_BACKEND=1` 필요**). |
| `MOAI_SIM_HYBRID_KS_EXACT_PARTITION` | `1` | `1`이면 **\|Ql\| ≥ α** 이고 **\|Ql\| mod α ≠ 0** 인 α만 스킵(완전한 digit 분할). **\|Ql\| < α** (예: β=1 한 덩어리)는 유지. `0`이면 Phantom `ceil(\|Ql\|/α)` 로 모든 α 유지. |
| `MOAI_SIM_KSWITCH_BETA_MODE` | `phantom` | `legacy`면 β를 구버전 규칙으로 계산(프로파일만). |
| `MOAI_SIM_HYBRID_KS_PROFILE_CSV` | `output/sim/hybrid_ks_profile.csv` | 출력 CSV 경로. |

---

## 8. 실행 예시

저장소 루트(`MOAI_GPU/`)에서:

```bash
# Primitive 하나: ct×pt 엔진 경로 + SimTiming
MOAI_SIM_BACKEND=1 MOAI_BENCH_MODE=sim_mul_plain ./build/test

# Primitive 전부 순차 리포트
MOAI_SIM_BACKEND=1 MOAI_BENCH_MODE=sim_primitives ./build/test

# 하이브리드 KS: α 스윕 연산량 CSV (해석만 — BACKEND 불필요)
MOAI_TEST_OUTPUT_DISABLE=1 MOAI_BENCH_MODE=sim_hybrid_ks_profile \
  MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35 ./build/test

# 동일 CSV에 엔진 schedule() 횟수(NTT/VEC enqueue)까지 — BACKEND 필수
MOAI_TEST_OUTPUT_DISABLE=1 MOAI_SIM_BACKEND=1 MOAI_BENCH_MODE=sim_hybrid_ks_profile \
  MOAI_SIM_HYBRID_KS_MEASURE_ENGINE=1 MOAI_SIM_HYBRID_KS_ALPHA_RANGE=1-35 ./build/test

# 스택 막대 + 해석 vs eng_* 비교 PNG 한 번에
python3 src/scripts/plot_hybrid_ks_profile.py --all-plots

# 또는: CSV 재생성 + 두 PNG (엔진 측정·기본 exact_partition 포함)
# src/scripts/run_hybrid_ks_sweep_plots.sh

# 리포트 파일 지정, 콘솔 중복 출력 줄이기
MOAI_SIM_BACKEND=1 MOAI_BENCH_MODE=sim_rescale \
  MOAI_SIM_REPORT_PATH=output/sim/rescale_once.txt MOAI_SIM_REPORT_QUIET=1 \
  ./build/test
```

Ct–Ct colpacking estimator (매트릭스 루프 모양):

```bash
MOAI_SIM_BACKEND=1 MOAI_BENCH_MODE=ct_ct ./build/test
```

클럭만 바꿔서 makespan 초 단위 스케일 확인:

```bash
MOAI_SIM_BACKEND=1 MOAI_SIM_ENGINE_MHZ=1000 MOAI_BENCH_MODE=sim_mul_ct ./build/test
```

---

## 9. 해석 시 자주 나오는 점

- **primitive “전체” 시간:** 엔진 블록 첫 줄의 **`makespan_cycles` / `makespan_s`** 가 **모델이 잡은 벽시계(크리티컬 패스)** 이다. **HBM DMA·on-chip·NTT·VEC·RF** 를 의존성에 맞게 겹쳐 스케줄한 뒤의 **끝나는 시각**이며, **각 엔진 `busy_cycles`의 합**과 같지 않다(겹치면 합이 makespan보다 클 수 있음).
- **`ext_load` 등 바이트가 크다:** 한 번의 정적 CT 크기가 아니라, **루프·청크·여러 op 누적**이 합산된 값일 수 있다.
- **`bound_hint` / `critical_tail`:** makespan 시점에 **어느 엔진이 꼬리를 당기는지**에 대한 조악한 힌트. 여러 엔진이 겹치면 busy/makespan 비율이 1을 넘을 수 있다.
- **`util_pct@ms` / `util_agg@ms`:** 각 엔진 **`busy_cycles / makespan`** 을 퍼센트로 찍은 것(벽시계 대비 그 엔진이 일한 비율). **겹치면 합이 100%를 넘을 수 있음.** `util_agg@ms`는 **data_move**(dma+onchip+RF xfer) vs **compute**(ntt+vec) 묶음으로, 메모리 이동 대 연산이 makespan 구간에 얼마나 “바쁘게” 잡혔는지 대략 비교용이다.
- **SimTiming vs EngineModel 숫자가 어긋날 수 있음:** 서로 다른 근사식·레거시 env를 공존시키기 때문. 트렌드 비교에는 같은 env 세트로 둘 다 찍는 것이 안전하다.

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| [moai_accelerator_simulator_analysis.md](moai_accelerator_simulator_analysis.md) | 워크로드·DAG·메모리 관점 설계 메모 |
| [micro_profile_operations_index.md](micro_profile_operations_index.md) | `MOAI_BENCH_MODE` 전체 표, Nsight 스크립트 매핑 |
| [moai_neo_style_simulator_design.md](moai_neo_style_simulator_design.md) | Neo 스타일 시뮬 설계 메모(별축) |

코드 기준 최신 env 이름은 **`engine_config.h`**, **`engine_model.h`**, **`sim_timing.h`** 를 우선한다.
